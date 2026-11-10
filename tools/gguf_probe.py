"""Read a remote GGUF's header (HTTP range requests only) and report per-tensor quantization types.
Optionally save the first rows of one tensor as a binary fixture.

Usage:
  python -I tools/gguf_probe.py --repo unsloth/Qwen3.5-4B-GGUF --rev <sha> --file Qwen3.5-4B-Q4_K_M.gguf \
      [--dump blk.3.ffn_gate.weight --rows 4 --out tests/fixtures/real_q4k_rows.bin] [--json out.json]
"""
import argparse
import collections
import http.client
import json
import struct
import sys
import time
import urllib.request

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K",
         12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16"}
# (elements per block, bytes per block)
BLOCK = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22),
         "Q5_1": (32, 24), "Q8_0": (32, 34), "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144),
         "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_K": (256, 292)}


def get(url, lo, hi, tries=6):
    req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()  # http.client raises IncompleteRead itself if the body is shorter than Content-Length
        # URLError / timeouts / "unreachable host" (OSError) and mid-body truncation (HTTPException) are transient here.
        except (OSError, http.client.HTTPException):
            if attempt == tries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


class Buf:
    def __init__(self, data):
        self.d, self.i = data, 0

    def take(self, fmt):
        n = struct.calcsize(fmt)
        if self.i + n > len(self.d):
            raise EOFError
        v = struct.unpack_from(fmt, self.d, self.i)
        self.i += n
        return v[0] if len(v) == 1 else v

    def string(self):
        n = self.take("<Q")
        if self.i + n > len(self.d):
            raise EOFError
        s = self.d[self.i:self.i + n].decode("utf-8", "replace")
        self.i += n
        return s

    def value(self, t):
        sizes = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        if t in sizes:
            return self.take(sizes[t])
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.take("<I"), self.take("<Q")
            if et == 8:
                for _ in range(n):
                    self.string()
            else:
                self.i += n * struct.calcsize(sizes[et])
                if self.i > len(self.d):
                    raise EOFError
            return f"[array of {n}]"
        raise ValueError(f"unknown value type {t}")


def parse(data):
    b = Buf(data)
    if data[:4] != b"GGUF":
        raise ValueError("bad magic")
    b.i = 4
    version = b.take("<I")
    n_tensors, n_kv = b.take("<Q"), b.take("<Q")
    kv, align = {}, 32
    for _ in range(n_kv):
        k, t = b.string(), b.take("<I")
        kv[k] = b.value(t)
        if k == "general.alignment":
            align = kv[k]
    tensors = []
    for _ in range(n_tensors):
        name, nd = b.string(), b.take("<I")
        dims = [b.take("<Q") for _ in range(nd)]
        t, off = b.take("<I"), b.take("<Q")
        tensors.append({"name": name, "dims": dims, "type": TYPES.get(t, f"type{t}"), "offset": off})
    data_off = (b.i + align - 1) // align * align
    return version, kv, tensors, data_off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--rev", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--dump")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--out")
    ap.add_argument("--json")
    a = ap.parse_args()
    url = f"https://huggingface.co/{a.repo}/resolve/{a.rev}/{a.file}"

    size = 8 << 20
    while True:
        data = get(url, 0, size - 1)
        try:
            version, kv, tensors, data_off = parse(data)
            break
        except EOFError:
            size *= 2
            if size > (256 << 20):
                sys.exit("header larger than 256 MiB")
    print(f"GGUF v{version}, {len(tensors)} tensors, arch={kv.get('general.architecture')}, data at {data_off}")

    by_type = collections.OrderedDict()
    for t in tensors:
        elems = 1
        for d in t["dims"]:
            elems *= d
        be, bb = BLOCK[t["type"]]
        assert elems % be == 0, f"{t['name']}: {elems} not a multiple of block {be}"
        t["bytes"] = elems // be * bb
        by_type.setdefault(t["type"], [0, 0])
        by_type[t["type"]][0] += 1
        by_type[t["type"]][1] += t["bytes"]
    for ty, (n, by) in by_type.items():
        print(f"  {ty:5s} {n:4d} tensors {by / 2**20:9.1f} MiB")
    total = sum(t["bytes"] for t in tensors)
    print(f"  total {total / 2**20:.1f} MiB")
    for t in tensors:
        if t["name"].startswith(("blk.3.", "token_embd", "output")):
            print(f"    {t['name']:34s} {t['type']:5s} dims={t['dims']}")

    if a.json:
        json.dump({"source": {"repo": a.repo, "revision": a.rev, "file": a.file}, "tensors": tensors},
                  open(a.json, "w"), indent=1)
    if a.dump:
        t = next(t for t in tensors if t["name"] == a.dump)
        be, bb = BLOCK[t["type"]]
        row_bytes = t["dims"][0] // be * bb
        lo = data_off + t["offset"]
        raw = get(url, lo, lo + row_bytes * a.rows - 1)
        with open(a.out, "wb") as f:
            f.write(struct.pack("<4sIIIQ", t["type"].encode().ljust(4, b"\0"), len(t["dims"]), t["dims"][0], a.rows, 0))
            f.write(raw)
        print(f"saved {a.rows} rows of {t['name']} ({t['type']}, {t['dims']}) -> {a.out}")


if __name__ == "__main__":
    main()
