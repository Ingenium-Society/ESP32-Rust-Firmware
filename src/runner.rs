//! Forward pass, one token at a time, with a KV cache.
//! Mirrors `RefModel` from the export notebook: ternary weights, int8 activations,
//! i32 accumulation, one rescale at the end of every matrix multiply.
#![allow(clippy::needless_range_loop)]

use crate::engine::{Model, Tern};

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
    sc: Vec<f32>, // attention scores for one head (ctx)
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

/// out = (W_ternary * xq) * (gamma / s)
/// Weights are 2 bits each, 4 per byte, lowest bits first, stored as w + 1 (0 -> -1, 1 -> 0, 2 -> +1).
fn matvec(w: &Tern, xq: &[i8], s: f32, out: &mut [f32]) {
    let scale = w.gamma / s;
    let bytes_per_row = w.cols / 4;
    for r in 0..w.rows {
        let row = &w.packed[r * bytes_per_row..(r + 1) * bytes_per_row];
        let mut acc: i32 = 0;
        for (j, &b) in row.iter().enumerate() {
            let x = &xq[j * 4..j * 4 + 4];
            acc += ((b & 3) as i32 - 1) * x[0] as i32
                + (((b >> 2) & 3) as i32 - 1) * x[1] as i32
                + (((b >> 4) & 3) as i32 - 1) * x[2] as i32
                + ((b >> 6) as i32 - 1) * x[3] as i32;
        }
        out[r] = acc as f32 * scale;
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
            sc: vec![0.0; ctx],
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
        assert!(pos < ctx, "position beyond context");
        assert!(tok < c.vocab, "token id out of range");

        // Embedding lookup (int8 row * per-row scale).
        let es = m.emb_scale[tok];
        let row = &m.emb[tok * d..(tok + 1) * d];
        for i in 0..d {
            self.x[i] = row[i] as f32 * es;
        }

        let inv_sqrt = 1.0 / (hd as f32).sqrt();

        for (li, l) in m.layers.iter().enumerate() {
            // ---- attention ----
            rmsnorm(&self.x, l.attn_norm, eps, &mut self.h);
            let s = quantize(&self.h, &mut self.xq[..d]);
            matvec(&l.wq, &self.xq[..d], s, &mut self.q);
            matvec(&l.wk, &self.xq[..d], s, &mut self.k);
            matvec(&l.wv, &self.xq[..d], s, &mut self.v);

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

            // Causal attention over positions 0..=pos, head by head.
            for hh in 0..nh {
                let o = hh * hd;
                let mut mx = f32::NEG_INFINITY;
                for t in 0..=pos {
                    let kb = (li * ctx + t) * d + o;
                    let mut dot = 0.0f32;
                    for i in 0..hd {
                        dot += self.q[o + i] * self.kc[kb + i];
                    }
                    let score = dot * inv_sqrt;
                    self.sc[t] = score;
                    if score > mx {
                        mx = score;
                    }
                }
                let mut sum = 0.0f32;
                for t in 0..=pos {
                    let e = (self.sc[t] - mx).exp();
                    self.sc[t] = e;
                    sum += e;
                }
                for i in 0..hd {
                    self.y[o + i] = 0.0;
                }
                for t in 0..=pos {
                    let p = self.sc[t] / sum;
                    let vb = (li * ctx + t) * d + o;
                    for i in 0..hd {
                        self.y[o + i] += p * self.vc[vb + i];
                    }
                }
            }

            rmsnorm(&self.y, l.wo_norm, eps, &mut self.h);
            let s = quantize(&self.h, &mut self.xq[..d]);
            matvec(&l.wo, &self.xq[..d], s, &mut self.t);
            for i in 0..d {
                self.x[i] += self.t[i];
            }

            // ---- MLP (SwiGLU) ----
            rmsnorm(&self.x, l.mlp_norm, eps, &mut self.h);
            let s = quantize(&self.h, &mut self.xq[..d]);
            matvec(&l.w_gate, &self.xq[..d], s, &mut self.g);
            matvec(&l.w_up, &self.xq[..d], s, &mut self.u);
            for i in 0..hid {
                let g = self.g[i];
                self.a[i] = (g / (1.0 + (-g).exp())) * self.u[i];
            }
            rmsnorm(&self.a, l.down_norm, eps, &mut self.an);
            let s = quantize(&self.an, &mut self.xq[..hid]);
            matvec(&l.w_down, &self.xq[..hid], s, &mut self.t);
            for i in 0..d {
                self.x[i] += self.t[i];
            }
        }

        // Final norm, then the tied output layer (embedding table used as the output matrix).
        rmsnorm(&self.x, m.final_norm, eps, &mut self.h);
        for v in 0..c.vocab {
            let erow = &m.emb[v * d..(v + 1) * d];
            let mut acc = 0.0f32;
            for i in 0..d {
                acc += erow[i] as f32 * self.h[i];
            }
            self.logits[v] = acc * m.emb_scale[v];
        }
        &self.logits
    }
}
