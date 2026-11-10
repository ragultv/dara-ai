"""Q5_K and Q8_0 decode fixtures from an independent implementation (the `gguf` Python package).

Format (little endian): u32 n_blocks, n_blocks*block_bytes of raw blocks, n_blocks*block_weights float32 expected.
Real inputs come from tools/gguf_probe.py --dump on the pinned Q4_K_M GGUF:
  _qkv.bin   blk.0.attn_qkv.weight  (Q5_K, 4 rows)      _ssmout.bin  blk.0.ssm_out.weight (Q5_K, 2 rows)
  _alpha.bin blk.0.ssm_alpha.weight (Q8_0, 8 rows)
Outputs: q5k_qkv.bin, q5k_ssmout.bin, q5k_random.bin, q80_alpha.bin, q80_random.bin
"""
import struct
import sys

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf.quants import dequantize

FMT = {"Q5_K": (T.Q5_K, 176, 256), "Q8_0": (T.Q8_0, 34, 32)}


def write(path, blocks_u8, kind):
    qt, bsz, bw = FMT[kind]
    n = blocks_u8.shape[0]
    expected = dequantize(blocks_u8.reshape(n, bsz), qt).astype(np.float32)
    assert expected.size == n * bw
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(blocks_u8.tobytes())
        f.write(expected.reshape(-1).tobytes())
    print(f"{path}: {n} blocks, |expected| max {np.abs(expected).max():.4g}")


def from_dump(dump, out, kind):
    raw = open(dump, "rb").read()
    typ, nd, cols, rows, _ = struct.unpack_from("<4sIIIQ", raw, 0)
    assert typ.rstrip(b"\0").decode() == kind, typ
    _, bsz, bw = FMT[kind]
    payload = np.frombuffer(raw, dtype=np.uint8, offset=struct.calcsize("<4sIIIQ"))
    assert payload.size == rows * cols // bw * bsz
    write(out, payload.reshape(-1, bsz).copy(), kind)


def main():
    d = "tests/fixtures/"
    from_dump(d + "_qkv.bin", d + "q5k_qkv.bin", "Q5_K")
    from_dump(d + "_ssmout.bin", d + "q5k_ssmout.bin", "Q5_K")
    from_dump(d + "_alpha.bin", d + "q80_alpha.bin", "Q8_0")
    rng = np.random.default_rng(2024)
    n = 64
    b = rng.integers(0, 256, size=(n, 176), dtype=np.uint8)
    for i in range(n):  # finite fp16 d and dmin
        b[i, 0:2] = np.frombuffer(np.float16(rng.uniform(1e-4, 2e-2)).tobytes(), dtype=np.uint8)
        b[i, 2:4] = np.frombuffer(np.float16(rng.uniform(1e-4, 2e-2)).tobytes(), dtype=np.uint8)
    write(d + "q5k_random.bin", b, "Q5_K")
    n = 256
    b = rng.integers(0, 256, size=(n, 34), dtype=np.uint8)
    for i in range(n):
        b[i, 0:2] = np.frombuffer(np.float16(rng.uniform(1e-4, 5e-2)).tobytes(), dtype=np.uint8)
    write(d + "q80_random.bin", b, "Q8_0")


if __name__ == "__main__":
    sys.exit(main())
