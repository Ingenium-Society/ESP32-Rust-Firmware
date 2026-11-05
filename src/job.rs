//! One generation task owns the `Runner` and its 3 MB KV cache.
//!
//! The serial REPL and the HTTP handler both submit jobs here. There can only
//! ever be one `Runner` alive: two of them plus the 2 MB model copy would be
//! 8 MB of PSRAM, which is all of it.

use crate::engine;
use crate::flash;
use crate::generate::{self, Stats};
use crate::runner::Runner;
use crate::selfcheck;
use crate::tokenizer::Tokenizer;
use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TrySendError};
use std::sync::OnceLock;

/// How many prompts may wait behind the one being generated.
pub const QUEUE_DEPTH: usize = 3;

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
                let res = generate::run(&mut r, &tok, &prompt, max_new, seed, ctx, |bytes| {
                    let _ = out.send(Msg::Text(bytes.to_vec()));
                });
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
