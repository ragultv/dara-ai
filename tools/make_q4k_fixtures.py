"""Q4_K decode fixtures produced by an independent implementation (the `gguf` Python package).

Fixture format (little endian): u32 n_blocks, n_blocks*144 bytes of Q4_K blocks, n_blocks*256 float32
expected dequantized values.

Two fixtures:
  q4k_real.bin   - 4 rows of a real tensor from the pinned Q4_K_M GGUF (tools/gguf_probe.py --dump)
  q4k_random.bin - random bytes with finite fp16 d/dmin: any bit pattern is a valid Q4_K block, so this
                   exercises every scale/min/nibble combination, including the high-bit scale packing.
"""
import struct
import sys

import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

BLOCK = 144


def write(path, blocks_u8):
    n = blocks_u8.shape[0]
    expected = dequantize(blocks_u8.reshape(n, BLOCK), GGMLQuantizationType.Q4_K).astype(np.float32)
    assert expected.size == n * 256
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(blocks_u8.tobytes())
        f.write(expected.reshape(-1).tobytes())
    print(f"{path}: {n} blocks, |expected| max {np.abs(expected).max():.4g}")


def main():
    raw = open("tests/fixtures/real_q4k_rows.bin", "rb").read()
    typ, nd, cols, rows, _ = struct.unpack_from("<4sIIIQ", raw, 0)
    assert typ.rstrip(b"\0") == b"Q4_K"
    payload = np.frombuffer(raw, dtype=np.uint8, offset=struct.calcsize("<4sIIIQ"))
    assert payload.size == rows * cols // 256 * BLOCK
    write("tests/fixtures/q4k_real.bin", payload.reshape(-1, BLOCK).copy())

    rng = np.random.default_rng(12345)
    n = 64
    blocks = rng.integers(0, 256, size=(n, BLOCK), dtype=np.uint8)
    for i in range(n):  # finite fp16 d and dmin in a sane range
        d = np.float16(rng.uniform(1e-4, 2e-2))
        m = np.float16(rng.uniform(1e-4, 2e-2))
        blocks[i, 0:2] = np.frombuffer(d.tobytes(), dtype=np.uint8)
        blocks[i, 2:4] = np.frombuffer(m.tobytes(), dtype=np.uint8)
    write("tests/fixtures/q4k_random.bin", blocks)


if __name__ == "__main__":
    sys.exit(main())
