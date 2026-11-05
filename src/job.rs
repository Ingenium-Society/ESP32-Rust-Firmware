//! One generation task owns the `Runner` and its 3 MB KV cache.
//!
//! The serial REPL and the HTTP handler both submit jobs here. There can only
//! ever be one `Runner` alive: two of them plus the 2 MB model copy would be
//! 8 MB of PSRAM, which is all of it.
//!
//! It also owns `VIEW`: the one story every connected phone is looking at. The
//! task appends to it as tokens arrive, and the HTTP handlers read snapshots.
//! That is what makes the page a shared screen rather than a private chat.

use crate::engine;
use crate::flash;
use crate::generate::{self, Stats};
use crate::runner::Runner;
use crate::selfcheck;
use crate::tokenizer::Tokenizer;
use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TrySendError};
use std::sync::{Mutex, OnceLock};

/// How many prompts may wait behind the one being generated. One: a second
/// visitor can line up a story while the current one plays, a third is told no.
pub const QUEUE_DEPTH: usize = 1;

/// The story everyone is watching.
pub struct View {
    pub id: u64,
    pub prompt: String,
    pub text: String,
    pub generating: bool,
}

static VIEW: Mutex<View> = Mutex::new(View {
    id: 0,
    prompt: String::new(),
    text: String::new(),
    generating: false,
});

fn view_begin(prompt: &str) {
    let mut v = VIEW.lock().unwrap();
    v.id += 1;
    v.prompt.clear();
    v.prompt.push_str(prompt);
    v.text.clear();
    v.generating = true;
}

fn view_push(bytes: &[u8]) {
    if let Ok(s) = std::str::from_utf8(bytes) {
        VIEW.lock().unwrap().text.push_str(s);
    }
}

fn view_finish() {
    VIEW.lock().unwrap().generating = false;
}

/// `(id, prompt, text, generating)` for the HTTP handlers.
pub fn snapshot() -> (u64, String, String, bool) {
    let v = VIEW.lock().unwrap();
    (v.id, v.prompt.clone(), v.text.clone(), v.generating)
}

/// Progress reports sent back to whoever submitted the job.
pub enum Msg {
    Text(Vec<u8>),
    Generated(Stats),
    Failed(String),
}

enum Task {
    Generate {
        prompt: String,
        max_new: usize,
        seed: u64,
        out: Sender<Msg>,
    },
    SelfCheck {
        done: Sender<Result<(), String>>,
    },
}

static QUEUE: OnceLock<SyncSender<Task>> = OnceLock::new();

/// Spawn the task. It does the expensive setup (2 MB model copy into PSRAM, 3 MB
/// KV cache) before it starts accepting jobs, so the first request is not slow.
pub fn start() {
    let (tx, rx) = mpsc::sync_channel::<Task>(QUEUE_DEPTH);
    if QUEUE.set(tx).is_err() {
        return;
    }
    let _ = std::thread::Builder::new()
        .stack_size(32 * 1024)
        .spawn(move || serve(rx));
}

pub fn submit_generate(
    prompt: String,
    max_new: usize,
    seed: u64,
) -> Result<Receiver<Msg>, String> {
    let q = QUEUE.get().ok_or("generation task is not running")?;
    let (tx, rx) = mpsc::channel();
    match q.try_send(Task::Generate {
        prompt,
        max_new,
        seed,
        out: tx,
    }) {
        Ok(()) => Ok(rx),
        Err(TrySendError::Full(_)) => Err(format!(
            "busy -- {} prompts are already queued",
            QUEUE_DEPTH
        )),
        Err(TrySendError::Disconnected(_)) => Err("generation task stopped".into()),
    }
}

pub fn submit_selfcheck() -> Result<Result<(), String>, String> {
    let q = QUEUE.get().ok_or("generation task is not running")?;
    let (tx, rx) = mpsc::channel();
    match q.try_send(Task::SelfCheck { done: tx }) {
        Ok(()) => rx.recv().map_err(|_| "generation task stopped".to_string()),
        Err(TrySendError::Full(_)) => Err("busy -- try again when the queue clears".into()),
        Err(TrySendError::Disconnected(_)) => Err("generation task stopped".into()),
    }
}

fn serve(rx: Receiver<Task>) {
    let (m, _) = match flash::model().and_then(engine::Model::parse) {
        Ok(v) => v,
        Err(e) => {
            println!("job task: model: {}", e);
            return;
        }
    };
    let tok = match Tokenizer::parse(flash::TOKENIZER_BIN) {
        Ok(t) => t,
        Err(e) => {
            println!("job task: tokenizer: {}", e);
            return;
        }
    };
    let vocab = m.cfg.vocab;
    let ctx = m.cfg.ctx;
    if tok.vocab() != vocab {
        println!("job task: tokenizer and model vocab sizes differ");
        return;
    }
    let mut r = Runner::new(m);

    for task in rx {
        match task {
            Task::Generate {
                prompt,
                max_new,
                seed,
                out,
            } => {
                view_begin(&prompt);
                let res = generate::run(&mut r, &tok, &prompt, max_new, seed, ctx, |bytes| {
                    view_push(bytes);
                    let _ = out.send(Msg::Text(bytes.to_vec()));
                });
                view_finish();
                let _ = match res {
                    Ok(stats) => out.send(Msg::Generated(stats)),
                    Err(e) => out.send(Msg::Failed(e)),
                };
            }
            Task::SelfCheck { done } => {
                let _ = done.send(selfcheck::run(&mut r, vocab));
            }
        }
    }
}
