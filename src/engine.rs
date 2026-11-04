//! BTLM model file: a zero-copy view over the bytes in flash.
//! Layout is defined in the export notebook (section 5). Nothing is copied or unpacked here;
//! every field is just a slice that points into the original buffer.

#![allow(dead_code)] // fields get used in the next steps; keeps clippy -D warnings quiet

pub struct Config {
    pub vocab: usize,
    pub ctx: usize,
    pub layers: usize,
    pub heads: usize,
    pub dim: usize,
    pub hidden: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
}

/// One ternary matrix (rows x cols): a scale plus weights packed 4 per byte, row by row.
pub struct Tern<'a> {
    pub gamma: f32,
    pub packed: &'a [u8],
    pub rows: usize,
    pub cols: usize,
}

pub struct Layer<'a> {
    pub attn_norm: &'a [f32],
    pub wo_norm: &'a [f32],
    pub mlp_norm: &'a [f32],
    pub down_norm: &'a [f32],
    pub wq: Tern<'a>,
    pub wk: Tern<'a>,
    pub wv: Tern<'a>,
    pub wo: Tern<'a>,
    pub w_gate: Tern<'a>,
    pub w_up: Tern<'a>,
    pub w_down: Tern<'a>,
}

pub struct Model<'a> {
    pub cfg: Config,
    pub emb_scale: &'a [f32],
    pub emb: &'a [i8],
    pub final_norm: &'a [f32],
    pub layers: Vec<Layer<'a>>,
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).ok_or("size overflow")?;
        if end > self.buf.len() {
            return Err(format!(
                "file too short: need {} bytes, have {}",
                end,
                self.buf.len()
            ));
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn f32s(&mut self, n: usize) -> Result<&'a [f32], String> {
        let b = self.bytes(n * 4)?;
        // The ESP32 and the file are both little-endian, so f32 data can be read in place.
        let (pre, mid, post) = unsafe { b.align_to::<f32>() };
        if !pre.is_empty() || !post.is_empty() {
            return Err(format!(
                "f32 data not 4-byte aligned near offset {}",
                self.pos
            ));
        }
        Ok(mid)
    }

    fn i8s(&mut self, n: usize) -> Result<&'a [i8], String> {
        let b = self.bytes(n)?;
        Ok(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const i8, n) })
    }

    fn tern(&mut self, rows: usize, cols: usize) -> Result<Tern<'a>, String> {
        let gamma = self.f32s(1)?[0];
        let packed = self.bytes(rows * cols / 4)?;
        Ok(Tern {
            gamma,
            packed,
            rows,
            cols,
        })
    }
}

impl<'a> Model<'a> {
    /// Returns the model and the number of bytes the file occupies.
    pub fn parse(buf: &'a [u8]) -> Result<(Model<'a>, usize), String> {
        if buf.len() < 64 || &buf[0..4] != b"BTLM" {
            return Err("bad magic (expected BTLM)".into());
        }
        let ver = u32_at(buf, 4);
        if ver != 1 {
            return Err(format!("unsupported version {}", ver));
        }
        let cfg = Config {
            vocab: u32_at(buf, 8) as usize,
            ctx: u32_at(buf, 12) as usize,
            layers: u32_at(buf, 16) as usize,
            heads: u32_at(buf, 20) as usize,
            dim: u32_at(buf, 24) as usize,
            hidden: u32_at(buf, 28) as usize,
            rope_theta: f32_at(buf, 32),
            norm_eps: f32_at(buf, 36),
        };
        if cfg.layers == 0 || cfg.layers > 64 || cfg.heads == 0 || cfg.dim % cfg.heads != 0 {
            return Err("implausible header".into());
        }
        if cfg.dim % 4 != 0 || cfg.hidden % 4 != 0 || (cfg.dim / cfg.heads) % 2 != 0 {
            return Err("dims must be multiples of 4 (head_dim even)".into());
        }

        let (d, h) = (cfg.dim, cfg.hidden);
        let mut r = Reader { buf, pos: 64 };
        let emb_scale = r.f32s(cfg.vocab)?;
        let emb = r.i8s(cfg.vocab * d)?;
        let final_norm = r.f32s(d)?;

        let mut layers = Vec::with_capacity(cfg.layers);
        for _ in 0..cfg.layers {
            layers.push(Layer {
                attn_norm: r.f32s(d)?,
                wo_norm: r.f32s(d)?,
                mlp_norm: r.f32s(d)?,
                down_norm: r.f32s(h)?,
                wq: r.tern(d, d)?,
                wk: r.tern(d, d)?,
                wv: r.tern(d, d)?,
                wo: r.tern(d, d)?,
                w_gate: r.tern(h, d)?,
                w_up: r.tern(h, d)?,
                w_down: r.tern(d, h)?,
            });
        }
        let used = r.pos;
        Ok((
            Model {
                cfg,
                emb_scale,
                emb,
                final_norm,
                layers,
            },
            used,
        ))
    }
}
