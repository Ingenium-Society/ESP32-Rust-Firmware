"""Convert a HuggingFace byte-level BPE tokenizer.json into the compact tokenizer.bin
that the ESP32 firmware embeds.   usage: make_tokenizer.py tokenizer.json tokenizer.bin

Layout (little endian):
  "BTOK", u32 vocab, u32 n_merges, u32 eos_id
  u16[256]          byte value -> token id of that single byte
  vocab x           u8 length, raw bytes of the token
  n_merges x        u16 a, u16 b, u16 merged   (merge rank = position in this list)
"""
import json
import struct
import sys

def bytes_to_unicode():            # GPT-2's byte <--> printable character table
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))

def merge_pair(merge):             # tokenizers <0.20 uses "a b", newer uses ["a", "b"]
    if isinstance(merge, str):
        a, b = merge.split(" ")
    else:
        a, b = merge
    return a, b

if len(sys.argv) != 3:
    sys.exit(f"usage: {sys.argv[0]} tokenizer.json tokenizer.bin")

src, dst = sys.argv[1], sys.argv[2]
t = json.load(open(src))
pre = t.get("pre_tokenizer") or {}
assert t["model"]["type"] == "BPE" and pre.get("type") == "ByteLevel", \
    "expected a ByteLevel BPE tokenizer"
vocab, merges = t["model"]["vocab"], t["model"]["merges"]
V = len(vocab)
assert sorted(vocab.values()) == list(range(V)), "ids must be 0..V-1"
assert V <= 0xFFFF, "vocab does not fit in u16 token ids"

added = {a["content"]: a["id"] for a in t.get("added_tokens", [])}
special_by_id = {a["id"]: a["content"] for a in t.get("added_tokens", [])}
eos_content = t["model"].get("eos_token") or "<|endoftext|>"
assert eos_content in added, f"{eos_content!r} is not an added token"
eos = added[eos_content]

b2u = bytes_to_unicode(); u2b = {c: b for b, c in b2u.items()}
by_id = [None] * V
for s, i in vocab.items():
    raw = special_by_id[i].encode("utf-8") if i in special_by_id else bytes(u2b[ch] for ch in s)
    assert len(raw) <= 255, f"token {i} is longer than 255 bytes"
    by_id[i] = raw

pairs = [merge_pair(m) for m in merges]
for a, b in pairs:
    assert a in vocab and b in vocab and a + b in vocab, f"merge {a!r} + {b!r} is not in vocab"

out = bytearray(b"BTOK" + struct.pack("<3I", V, len(pairs), eos))
for b in range(256):
    assert b2u[b] in vocab, f"no single-byte token for 0x{b:02x}"
    out += struct.pack("<H", vocab[b2u[b]])
for raw in by_id:
    out += bytes([len(raw)]) + raw
for a, b in pairs:
    out += struct.pack("<3H", vocab[a], vocab[b], vocab[a + b])
open(dst, "wb").write(out)
print(f"wrote {dst}: {len(out):,} bytes | vocab {V} | merges {len(pairs)} | eos id {eos}")
