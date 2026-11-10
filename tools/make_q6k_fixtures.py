"""Q6_K decode fixtures from an independent implementation (the `gguf` Python package).

Format (little endian): u32 n_blocks, n_blocks*210 bytes of Q6_K blocks, n_blocks*256 float32 expected values.
Fixtures:
  q6k_embd.bin  - 4 rows of token_embd.weight (the tied output head), real bytes from the pinned Q4_K_M GGUF
  q6k_down.bin  - 2 rows of blk.3.ffn_down.weight, real bytes
  q6k_random.bin - random bytes with finite fp16 d: every ql/qh/scale combination is a valid block

Regenerate the real inputs first with tools/gguf_probe.py --dump (see the README of this tool's usage).
"""
import struct
import sys

import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

BLOCK = 210


def write(path, blocks_u8):
    n = blocks_u8.shape[0]
    expected = dequantize(blocks_u8.reshape(n, BLOCK), GGMLQuantizationType.Q6_K).astype(np.float32)
    assert expected.size == n * 256
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(blocks_u8.tobytes())
        f.write(expected.reshape(-1).tobytes())
    print(f"{path}: {n} blocks, |expected| max {np.abs(expected).max():.4g}")


def from_dump(dump, out):
    raw = open(dump, "rb").read()
    typ, nd, cols, rows, _ = struct.unpack_from("<4sIIIQ", raw, 0)
    assert typ.rstrip(b"\0") == b"Q6_K", typ
    payload = np.frombuffer(raw, dtype=np.uint8, offset=struct.calcsize("<4sIIIQ"))
    assert payload.size == rows * cols // 256 * BLOCK
    write(out, payload.reshape(-1, BLOCK).copy())
    return cols


def main():
    from_dump("tests/fixtures/_embd.bin", "tests/fixtures/q6k_embd.bin")
    from_dump("tests/fixtures/_down.bin", "tests/fixtures/q6k_down.bin")
    rng = np.random.default_rng(777)
    n = 64
    blocks = rng.integers(0, 256, size=(n, BLOCK), dtype=np.uint8)
    for i in range(n):
        d = np.float16(rng.uniform(1e-4, 5e-3))
        blocks[i, 208:210] = np.frombuffer(d.tobytes(), dtype=np.uint8)
    write("tests/fixtures/q6k_random.bin", blocks)


if __name__ == "__main__":
    sys.exit(main())
