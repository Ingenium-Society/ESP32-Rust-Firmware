//! Forward pass, one token at a time, with a KV cache.
//! Mirrors `RefModel` from the export notebook: ternary weights, int8 activations,
//! i32 accumulation, one rescale at the end of every matrix multiply.
//!
//! The heavy loops (matrix multiplies, attention heads, output layer) are split in two
//! halves that run at the same time, one on each core. Every output value is still computed
//! by exactly the same arithmetic as before, so results are bit-for-bit identical.
#![allow(clippy::needless_range_loop)]

use crate::engine::{Model, Tern};
use std::sync::{mpsc, Mutex, OnceLock};

pub struct Runner<'a> {
    m: Model<'a>,
    hd: usize,
    cos: Vec<f32>, // [ctx][hd/2]
    sin: Vec<f32>,
    kc: Vec<f32>, // KV cache: [layer][pos][dim]
    vc: Vec<f32>,
    x: Vec<f32>,  // residual stream (dim)
    h: Vec<f32>,  // normed input (dim)
    t: Vec<f32>,  // matmul output that gets added to the residual (dim)
    q: Vec<f32>,  // dim
    k: Vec<f32>,  // dim
    v: Vec<f32>,  // dim
    y: Vec<f32>,  // attention output (dim)
    g: Vec<f32>,  // MLP gate (hidden)
    u: Vec<f32>,  // MLP up (hidden)
    a: Vec<f32>,  // silu(g) * u (hidden)
    an: Vec<f32>, // normed a (hidden)
    sc: Vec<f32>, // attention scores, one ctx-sized buffer per core (2 * ctx)
    xq: Vec<i8>,  // quantized activations (max(dim, hidden))
    logits: Vec<f32>,
}

/// Round to nearest, ties to even (what numpy and PyTorch do), without needing libm.
/// Adding then subtracting 1.5 * 2^23 forces the CPU's own rounding. Valid for |x| < 2^22.
#[inline]
fn round_even(x: f32) -> f32 {
    const M: f32 = 12_582_912.0;
    (x + M) - M
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let r = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    for i in 0..x.len() {
        out[i] = (x[i] * r) * w[i];
    }
}

/// Per-token absmax int8 quantization. Returns the scale `s` (x_q = round(x * s)).
fn quantize(x: &[f32], xq: &mut [i8]) -> f32 {
    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }
    let s = 127.0 / amax.max(1e-5);
    for i in 0..x.len() {
        xq[i] = round_even(x[i] * s).clamp(-128.0, 127.0) as i8;
    }
    s
}

type Job = Box<dyn FnOnce() + Send + 'static>;

/// One long-lived worker thread pinned to core 1. Starting a new thread for every matrix
/// multiply ran the chip out of memory, so jobs are handed to this single thread instead.
struct Worker {
    jobs: Mutex<mpsc::Sender<Job>>,
    done: Mutex<mpsc::Receiver<()>>,
}

fn worker() -> &'static Worker {
    static WORKER: OnceLock<Worker> = OnceLock::new();
    WORKER.get_or_init(|| {
        // The next thread we start runs on core 1 (the main task lives on core 0).
        #[cfg(target_os = "espidf")]
        {
            use esp_idf_svc::hal::cpu::Core;
            use esp_idf_svc::hal::task::thread::ThreadSpawnConfiguration;
            let _ = ThreadSpawnConfiguration {
                stack_size: 8192,
                priority: 5,
                pin_to_core: Some(Core::Core1),
                ..Default::default()
            }
            .set();
        }
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (done_tx, done_rx) = mpsc::channel::<()>();
        std::thread::spawn(move || {
            for job in job_rx {
                job();
                if done_tx.send(()).is_err() {
                    break;
                }
            }
        });
        Worker {
            jobs: Mutex::new(job_tx),
            done: Mutex::new(done_rx),
        }
    })
}

/// Run `a` on this core and `b` on the other core at the same time, then wait for both.
fn par2<A: FnOnce() + Send, B: FnOnce() + Send>(a: A, b: B) {
    let w = worker();
    let job: Box<dyn FnOnce() + Send + '_> = Box::new(b);
    // SAFETY: we do not return before the worker has reported that `b` finished, so
    // everything `b` borrows is still alive for as long as the worker uses it.
    // (This firmware builds with panic=abort, so `a` cannot unwind past the wait.)
    let job = unsafe { std::mem::transmute::<Box<dyn FnOnce() + Send + '_>, Job>(job) };
    w.jobs.lock().unwrap().send(job).unwrap();
    a();
    w.done.lock().unwrap().recv().unwrap();
}

/// out[k] = (row (row0 + k) of W_ternary * xq) * (gamma / s)
/// Weights are 2 bits each, 4 per byte, lowest bits first, stored as w + 1 (0 -> -1, 1 -> 0, 2 -> +1).
fn matvec_rows(w: &Tern, xq: &[i8], s: f32, row0: usize, out: &mut [f32]) {
    let scale = w.gamma / s;
    let bytes_per_row = w.cols / 4;
    for (k, o) in out.iter_mut().enumerate() {
        let r = row0 + k;
        let row = &w.packed[r * bytes_per_row..(r + 1) * bytes_per_row];
        let mut acc: i32 = 0;
        for (j, &b) in row.iter().enumerate() {
            let x = &xq[j * 4..j * 4 + 4];
            acc += ((b & 3) as i32 - 1) * x[0] as i32
                + (((b >> 2) & 3) as i32 - 1) * x[1] as i32
                + (((b >> 4) & 3) as i32 - 1) * x[2] as i32
                + ((b >> 6) as i32 - 1) * x[3] as i32;
        }
        *o = acc as f32 * scale;
    }
}

/// Attention for heads h0..h1 at position `pos`. `y` and `sc` belong to this call only.
#[allow(clippy::too_many_arguments)]
fn attn_heads(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    li: usize,
    ctx: usize,
    d: usize,
    hd: usize,
    pos: usize,
    h0: usize,
    h1: usize,
    y: &mut [f32],
    sc: &mut [f32],
) {
    let inv_sqrt = 1.0 / (hd as f32).sqrt();
    for hh in h0..h1 {
        let o = hh * hd;
        let yo = (hh - h0) * hd;
        let mut mx = f32::NEG_INFINITY;
        for t in 0..=pos {
            let kb = (li * ctx + t) * d + o;
            let mut dot = 0.0f32;
            for i in 0..hd {
                dot += q[o + i] * kc[kb + i];
            }
            let score = dot * inv_sqrt;
            sc[t] = score;
            if score > mx {
                mx = score;
            }
        }
        let mut sum = 0.0f32;
        for t in 0..=pos {
            let e = (sc[t] - mx).exp();
            sc[t] = e;
            sum += e;
        }
        for i in 0..hd {
            y[yo + i] = 0.0;
        }
        for t in 0..=pos {
            let p = sc[t] / sum;
            let vb = (li * ctx + t) * d + o;
            for i in 0..hd {
                y[yo + i] += p * vc[vb + i];
            }
        }
    }
}

/// Tied output layer for vocabulary entries v0 .. v0 + out.len().
fn logit_rows(emb: &[i8], emb_scale: &[f32], h: &[f32], d: usize, v0: usize, out: &mut [f32]) {
    for (k, o) in out.iter_mut().enumerate() {
        let v = v0 + k;
        let erow = &emb[v * d..(v + 1) * d];
        let mut acc = 0.0f32;
        for i in 0..d {
            acc += erow[i] as f32 * h[i];
        }
        *o = acc * emb_scale[v];
    }
}

impl<'a> Runner<'a> {
    pub fn new(m: Model<'a>) -> Self {
        let c = &m.cfg;
        let (d, hid, ctx, nl) = (c.dim, c.hidden, c.ctx, c.layers);
        let hd = d / c.heads;
        let half = hd / 2;

        // RoPE tables, same formula as the notebook (split-half pairing).
        let mut cos = vec![0.0f32; ctx * half];
        let mut sin = vec![0.0f32; ctx * half];
        for i in 0..half {
            let freq = 1.0 / c.rope_theta.powf((2 * i) as f32 / hd as f32);
            for p in 0..ctx {
                let ang = p as f32 * freq;
                cos[p * half + i] = ang.cos();
                sin[p * half + i] = ang.sin();
            }
        }

        let vocab = c.vocab;
        Runner {
            hd,
            cos,
            sin,
            kc: vec![0.0; nl * ctx * d], // 1.5 MB each: lands in PSRAM automatically
            vc: vec![0.0; nl * ctx * d],
            x: vec![0.0; d],
            h: vec![0.0; d],
            t: vec![0.0; d],
            q: vec![0.0; d],
            k: vec![0.0; d],
            v: vec![0.0; d],
            y: vec![0.0; d],
            g: vec![0.0; hid],
            u: vec![0.0; hid],
            a: vec![0.0; hid],
            an: vec![0.0; hid],
            sc: vec![0.0; 2 * ctx],
            xq: vec![0; d.max(hid)],
            logits: vec![0.0; vocab],
            m,
        }
    }

    /// Process one token at position `pos` (0, 1, 2, ... in order) and return the logits
    /// for the next token. The KV cache keeps everything seen so far.
    pub fn step(&mut self, tok: usize, pos: usize) -> &[f32] {
        let m = &self.m;
        let c = &m.cfg;
        let (d, hid, nh, ctx) = (c.dim, c.hidden, c.heads, c.ctx);
        let hd = self.hd;
        let half = hd / 2;
        let eps = c.norm_eps;
        let (dh, hh2, nh2) = (d / 2, hid / 2, nh / 2);
        assert!(pos < ctx, "position beyond context");
        assert!(tok < c.vocab, "token id out of range");

        // Embedding lookup (int8 row * per-row scale).
        let es = m.emb_scale[tok];
        let row = &m.emb[tok * d..(tok + 1) * d];
        for i in 0..d {
            self.x[i] = row[i] as f32 * es;
        }

        for (li, l) in m.layers.iter().enumerate() {
            // ---- attention ----
            rmsnorm(&self.x, l.attn_norm, eps, &mut self.h);
            let s = quantize(&self.h, &mut self.xq[..d]);
            {
                let xq = &self.xq[..d];
                let (qa, qb) = self.q.split_at_mut(dh);
                let (ka, kb) = self.k.split_at_mut(dh);
                let (va, vb) = self.v.split_at_mut(dh);
                par2(
                    || {
                        matvec_rows(&l.wq, xq, s, 0, qa);
                        matvec_rows(&l.wk, xq, s, 0, ka);
                        matvec_rows(&l.wv, xq, s, 0, va);
                    },
                    || {
                        matvec_rows(&l.wq, xq, s, dh, qb);
                        matvec_rows(&l.wk, xq, s, dh, kb);
                        matvec_rows(&l.wv, xq, s, dh, vb);
                    },
                );
            }

            // RoPE on q and k, per head, first half paired with second half.
            for hh in 0..nh {
                let o = hh * hd;
                for i in 0..half {
                    let cs = self.cos[pos * half + i];
                    let sn = self.sin[pos * half + i];
                    let (q1, q2) = (self.q[o + i], self.q[o + half + i]);
                    self.q[o + i] = q1 * cs - q2 * sn;
                    self.q[o + half + i] = q1 * sn + q2 * cs;
                    let (k1, k2) = (self.k[o + i], self.k[o + half + i]);
                    self.k[o + i] = k1 * cs - k2 * sn;
                    self.k[o + half + i] = k1 * sn + k2 * cs;
                }
            }

            // Store this token's k and v in the cache.
            let base = (li * ctx + pos) * d;
            self.kc[base..base + d].copy_from_slice(&self.k);
            self.vc[base..base + d].copy_from_slice(&self.v);

            // Causal attention over positions 0..=pos: first half of the heads on this core,
            // second half on the other one.
            {
                let (sc0, sc1) = self.sc.split_at_mut(ctx);
                let (ya, yb) = self.y.split_at_mut(nh2 * hd);
                let (q, kc, vc) = (&self.q[..], &self.kc[..], &self.vc[..]);
                par2(
                    || attn_heads(q, kc, vc, li, ctx, d, hd, pos, 0, nh2, ya, sc0),
                    || attn_heads(q, kc, vc, li, ctx, d, hd, pos, nh2, nh, yb, sc1),
                );
            }

            rmsnorm(&self.y, l.wo_norm, eps, &mut self.h);
            let s = quantize(&self.h, &mut self.xq[..d]);
            {
                let xq = &self.xq[..d];
                let (ta, tb) = self.t.split_at_mut(dh);
                par2(
                    || matvec_rows(&l.wo, xq, s, 0, ta),
                    || matvec_rows(&l.wo, xq, s, dh, tb),
                );
            }
            for i in 0..d {
                self.x[i] += self.t[i];
            }

            // ---- MLP (SwiGLU) ----
            rmsnorm(&self.x, l.mlp_norm, eps, &mut self.h);
            let s = quantize(&self.h, &mut self.xq[..d]);
            {
                let xq = &self.xq[..d];
                let (ga, gb) = self.g.split_at_mut(hh2);
                let (ua, ub) = self.u.split_at_mut(hh2);
                par2(
                    || {
                        matvec_rows(&l.w_gate, xq, s, 0, ga);
                        matvec_rows(&l.w_up, xq, s, 0, ua);
                    },
                    || {
                        matvec_rows(&l.w_gate, xq, s, hh2, gb);
                        matvec_rows(&l.w_up, xq, s, hh2, ub);
                    },
                );
            }
            for i in 0..hid {
                let g = self.g[i];
                self.a[i] = (g / (1.0 + (-g).exp())) * self.u[i];
            }
            rmsnorm(&self.a, l.down_norm, eps, &mut self.an);
            let s = quantize(&self.an, &mut self.xq[..hid]);
            {
                let xq = &self.xq[..hid];
                let (ta, tb) = self.t.split_at_mut(dh);
                par2(
                    || matvec_rows(&l.w_down, xq, s, 0, ta),
                    || matvec_rows(&l.w_down, xq, s, dh, tb),
                );
            }
            for i in 0..d {
                self.x[i] += self.t[i];
            }
        }

        // Final norm, then the tied output layer (embedding table used as the output matrix).
        rmsnorm(&self.x, m.final_norm, eps, &mut self.h);
        {
            let vh = c.vocab / 2;
            let (emb, esc, hv) = (m.emb, m.emb_scale, &self.h[..]);
            let (la, lb) = self.logits.split_at_mut(vh);
            par2(
                || logit_rows(emb, esc, hv, d, 0, la),
                || logit_rows(emb, esc, hv, d, vh, lb),
            );
        }
        &self.logits
    }
}
