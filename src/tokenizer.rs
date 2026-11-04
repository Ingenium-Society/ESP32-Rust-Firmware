//! Byte-level BPE tokenizer (GPT-2 style), reading the compact `tokenizer.bin`
//! produced by `make_tokenizer.py`. No regex crate: the GPT-2 split rule is hand-coded.

use std::collections::HashMap;

pub struct Tokenizer<'a> {
    toks: Vec<&'a [u8]>,                     // id -> raw bytes of the token
    byte_id: [u16; 256],                     // single byte -> token id
    merges: HashMap<(u16, u16), (u16, u16)>, // (left, right) -> (rank, merged id)
    eos: usize,
}

fn u32_at(b: &[u8], o: usize) -> usize {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize
}
fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes(b[o..o + 2].try_into().unwrap())
}

impl<'a> Tokenizer<'a> {
    pub fn parse(b: &'a [u8]) -> Result<Tokenizer<'a>, String> {
        if b.len() < 16 || &b[0..4] != b"BTOK" {
            return Err("tokenizer.bin: bad magic".into());
        }
        let vocab = u32_at(b, 4);
        let n_merges = u32_at(b, 8);
        let eos = u32_at(b, 12);
        let mut pos = 16;

        if b.len() < pos + 512 {
            return Err("tokenizer.bin: truncated".into());
        }
        let mut byte_id = [0u16; 256];
        for (i, slot) in byte_id.iter_mut().enumerate() {
            *slot = u16_at(b, pos + 2 * i);
        }
        pos += 512;

        let mut toks = Vec::with_capacity(vocab);
        for _ in 0..vocab {
            let len = *b.get(pos).ok_or("tokenizer.bin: truncated")? as usize;
            pos += 1;
            toks.push(b.get(pos..pos + len).ok_or("tokenizer.bin: truncated")?);
            pos += len;
        }

        if b.len() < pos + 6 * n_merges {
            return Err("tokenizer.bin: truncated merges".into());
        }
        let mut merges = HashMap::with_capacity(n_merges);
        for rank in 0..n_merges {
            let o = pos + 6 * rank;
            merges.insert(
                (u16_at(b, o), u16_at(b, o + 2)),
                (rank as u16, u16_at(b, o + 4)),
            );
        }
        Ok(Tokenizer {
            toks,
            byte_id,
            merges,
            eos,
        })
    }

    pub fn vocab(&self) -> usize {
        self.toks.len()
    }

    pub fn eos(&self) -> usize {
        self.eos
    }

    pub fn token_bytes(&self, id: usize) -> &[u8] {
        self.toks[id]
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let chars: Vec<char> = text.chars().collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let j = next_piece(&chars, i);
            let piece: String = chars[i..j].iter().collect();
            self.bpe(piece.as_bytes(), &mut out);
            i = j;
        }
        out
    }

    /// Standard BPE: repeatedly merge the adjacent pair with the lowest merge rank.
    fn bpe(&self, bytes: &[u8], out: &mut Vec<usize>) {
        let mut ids: Vec<u16> = bytes.iter().map(|&b| self.byte_id[b as usize]).collect();
        loop {
            let mut best: Option<(u16, usize, u16)> = None; // (rank, position, merged id)
            for w in 0..ids.len().saturating_sub(1) {
                if let Some(&(rank, merged)) = self.merges.get(&(ids[w], ids[w + 1])) {
                    let better = match best {
                        None => true,
                        Some((r0, _, _)) => rank < r0,
                    };
                    if better {
                        best = Some((rank, w, merged));
                    }
                }
            }
            match best {
                Some((_, w, merged)) => {
                    ids[w] = merged;
                    ids.remove(w + 1);
                }
                None => break,
            }
        }
        out.extend(ids.iter().map(|&x| x as usize));
    }
}

/// End (exclusive) of the pre-token that starts at `i`. This is the GPT-2 pattern
///   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
fn next_piece(c: &[char], i: usize) -> usize {
    let n = c.len();

    // contractions
    if c[i] == '\'' && i + 1 < n {
        let a = c[i + 1];
        if matches!(a, 's' | 't' | 'm' | 'd') {
            return i + 2;
        }
        if i + 2 < n {
            let b = c[i + 2];
            if matches!((a, b), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
                return i + 3;
            }
        }
    }

    // optional single space, then a run of letters / digits / other symbols
    fn class(x: char) -> u8 {
        if x.is_alphabetic() {
            1
        } else if x.is_numeric() {
            2
        } else if !x.is_whitespace() {
            3
        } else {
            0
        }
    }
    let start = if c[i] == ' ' { i + 1 } else { i };
    if start < n {
        let k = class(c[start]);
        if k != 0 {
            let mut j = start + 1;
            while j < n && class(c[j]) == k {
                j += 1;
            }
            return j;
        }
    }

    // whitespace: leave the last space for the next word unless the text ends
    let mut j = i;
    while j < n && c[j].is_whitespace() {
        j += 1;
    }
    if j < n && j - i > 1 {
        j -= 1;
    }
    j.max(i + 1)
}
