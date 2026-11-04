//! Sampling and the generation loop. Same sampler as the export notebook's `sample_next`:
//! repetition penalty over the last 64 tokens, temperature, top-k, then a random draw.

use crate::runner::Runner;
use crate::tokenizer::Tokenizer;
use std::io::Write;

const TEMPERATURE: f32 = 0.7;
const TOP_K: usize = 40;
const REP_PENALTY: f32 = 1.15;
const REP_WINDOW: usize = 64;

/// xorshift64* : tiny and good enough for sampling.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1) // the state must never be zero
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }
}

fn sample(logits: &[f32], history: &[usize], rng: &mut Rng) -> usize {
    let mut l = logits.to_vec();

    // repetition penalty on every distinct token seen in the last REP_WINDOW positions
    let mut seen = vec![false; l.len()];
    for &t in &history[history.len().saturating_sub(REP_WINDOW)..] {
        if !seen[t] {
            seen[t] = true;
            l[t] = if l[t] > 0.0 {
                l[t] / REP_PENALTY
            } else {
                l[t] * REP_PENALTY
            };
        }
    }
    for v in l.iter_mut() {
        *v /= TEMPERATURE;
    }

    // top-k: keep everything >= the k-th largest logit
    let k = TOP_K.min(l.len());
    let mut tmp = l.clone();
    let idx = tmp.len() - k;
    let (_, kth, _) = tmp.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    let kth = *kth;
    let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut cand: Vec<(usize, f32)> = Vec::with_capacity(k + 4);
    let mut sum = 0.0f32;
    for (i, &v) in l.iter().enumerate() {
        if v >= kth {
            let e = (v - mx).exp();
            cand.push((i, e));
            sum += e;
        }
    }
    let mut r = rng.next_f32() * sum;
    for &(i, e) in &cand {
        r -= e;
        if r <= 0.0 {
            return i;
        }
    }
    cand.last().map(|c| c.0).unwrap_or(0)
}

/// Print whatever part of `buf` is complete UTF-8; keep an unfinished character for later.
fn flush_utf8(buf: &mut Vec<u8>, force: bool) {
    loop {
        match std::str::from_utf8(buf) {
            Ok(s) => {
                print!("{}", s);
                buf.clear();
                break;
            }
            Err(e) => {
                let v = e.valid_up_to();
                print!("{}", std::str::from_utf8(&buf[..v]).unwrap_or(""));
                match e.error_len() {
                    Some(bad) => {
                        print!("\u{FFFD}");
                        buf.drain(..v + bad);
                    }
                    None => {
                        buf.drain(..v);
                        if force {
                            print!("\u{FFFD}");
                            buf.clear();
                        }
                        break;
                    }
                }
            }
        }
    }
    let _ = std::io::stdout().flush();
}

/// Let FreeRTOS run the idle task, otherwise the task watchdog complains during long runs.
fn yield_cpu() {
    std::thread::sleep(std::time::Duration::from_millis(1));
}

/// Continue `prompt` and stream the text to the console. An empty prompt starts a fresh story.
pub fn generate(
    r: &mut Runner,
    tok: &Tokenizer,
    prompt: &str,
    max_new: usize,
    seed: u64,
    ctx: usize,
) -> Result<(), String> {
    let mut ids = tok.encode(prompt);
    if ids.is_empty() {
        ids.push(tok.eos());
    }
    if ids.len() >= ctx {
        return Err(format!(
            "prompt is {} tokens; the limit is {}",
            ids.len(),
            ctx - 1
        ));
    }

    let mut rng = Rng::new(seed);
    let t0 = std::time::Instant::now();

    // feed the prompt, keep only the logits after its last token
    let mut last: Vec<f32> = Vec::new();
    for (i, &t) in ids.iter().enumerate() {
        let lg = r.step(t, i);
        if i + 1 == ids.len() {
            last = lg.to_vec();
        }
        yield_cpu();
    }
    let prompt_ms = t0.elapsed().as_millis();

    print!("{}", prompt);
    let mut pending: Vec<u8> = Vec::new();
    let mut produced = 0usize;
    let t1 = std::time::Instant::now();

    while ids.len() < ctx && produced < max_new {
        let nxt = sample(&last, &ids, &mut rng);
        if nxt == tok.eos() {
            break;
        }
        pending.extend_from_slice(tok.token_bytes(nxt));
        flush_utf8(&mut pending, false);

        last.clear();
        last.extend_from_slice(r.step(nxt, ids.len()));
        ids.push(nxt);
        produced += 1;
        yield_cpu();
    }
    flush_utf8(&mut pending, true);

    let gen_ms = t1.elapsed().as_millis().max(1);
    println!();
    println!(
        "[prompt {} tokens in {} ms | generated {} tokens in {} ms = {:.1} tok/s | context {}/{}]",
        ids.len() - produced,
        prompt_ms,
        produced,
        gen_ms,
        produced as f32 * 1000.0 / gen_ms as f32,
        ids.len(),
        ctx
    );
    Ok(())
}
