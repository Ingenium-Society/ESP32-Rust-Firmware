//! Compares the integer engine against PyTorch's logits, using the reference
//! vectors baked into the firmware. This is what `test` runs.
//!
//! Takes a borrowed `Runner` rather than building its own: the generation task
//! owns the only one, because two 3 MB KV caches would not fit in PSRAM.

use crate::flash::{self, rd_f32, rd_u32};
use crate::runner::Runner;

pub fn run(r: &mut Runner, vocab: usize) -> Result<(), String> {
    let tv = flash::TEST_VECTORS;
    let n = rd_u32(tv, 0) as usize;
    let v = rd_u32(tv, 4) as usize;
    if v != vocab || tv.len() != 8 + 4 * n + 4 * n * v {
        return Err("test_vectors.bin has an unexpected size".into());
    }

    let (mut max_d, mut sum_d, mut agree) = (0f32, 0f64, 0usize);
    let t0 = std::time::Instant::now();
    for i in 0..n {
        let tok = rd_u32(tv, 8 + 4 * i) as usize;
        let logits = r.step(tok, i);
        let base = 8 + 4 * n + 4 * i * v;
        let (mut best, mut best_ref) = (0usize, 0usize);
        let (mut bv, mut bvr) = (f32::MIN, f32::MIN);
        for j in 0..v {
            let want = rd_f32(tv, base + 4 * j);
            let d = (logits[j] - want).abs();
            max_d = max_d.max(d);
            sum_d += d as f64;
            if logits[j] > bv {
                bv = logits[j];
                best = j;
            }
            if want > bvr {
                bvr = want;
                best_ref = j;
            }
        }
        if best == best_ref {
            agree += 1;
        }
    }
    let ms = t0.elapsed().as_millis();
    println!(
        "tokens: {}  time: {} ms ({} ms/token)",
        n,
        ms,
        ms / n as u128
    );
    println!("argmax agreement: {}/{}", agree, n);
    println!(
        "max |diff|: {:.5}  mean |diff|: {:.6}",
        max_d,
        sum_d / (n * v) as f64
    );
    Ok(())
}
