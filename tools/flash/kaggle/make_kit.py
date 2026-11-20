"""Builds the Kaggle dataset ("kit") for the Dara Flash distillation notebook, from this repository and the local Dara data.

Usage: python -I tools/flash/kaggle/make_kit.py [--out build/flash/kaggle_kit] [--pack-size 12]
Then zip/upload the folder as a PRIVATE Kaggle dataset named  dara-flash-kit  (Kaggle > Datasets > New Dataset), and attach it to the notebook.

Contents:
  flash_kaggle.py, smoke_test.py, train_drafter.py, self_instruct.py   the code the notebook runs
  hf/                      config + tokenizer files of google/gemma-4-E2B-it (no weights: they come from the GGUF)
  seed_prompts.jsonl       the 4,249 prompts of the existing v2 drafter data
  validation_pack/         a few sequences recorded by Dara itself (tokens + features + prompt text): the notebook checks the HF path against them
"""
import argparse
import json
import os
import shutil
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
HIDDEN = 1536


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "flash", "kaggle_kit"))
    ap.add_argument("--hf-dir", default=os.path.join(ROOT, "models", "gemma4-e2b"))
    ap.add_argument("--data", default=os.path.join(ROOT, "build", "flash", "nano_si"))
    ap.add_argument("--prompts", default=os.path.join(ROOT, "build", "flash", "prompts_si.jsonl"))
    ap.add_argument("--pack-size", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    for src in (os.path.join(HERE, "flash_kaggle.py"), os.path.join(HERE, "smoke_test.py"), os.path.join(ROOT, "tools", "flash", "train_drafter.py"),
                os.path.join(ROOT, "tools", "flash", "self_instruct.py")):
        shutil.copy2(src, a.out)
    hf = os.path.join(a.out, "hf")
    os.makedirs(hf, exist_ok=True)
    for n in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy2(os.path.join(a.hf_dir, n), hf)

    # seed prompts: every prompt of the existing data, once
    seen, seeds = set(), []
    for p in (os.path.join(ROOT, "build", "flash", "prompts.jsonl"), a.prompts):
        for j in read_jsonl(p):
            if j["prompt"].lower() not in seen:
                seen.add(j["prompt"].lower())
                seeds.append({"id": len(seeds), "prompt": j["prompt"], "category": j.get("category", "")})
    with open(os.path.join(a.out, "seed_prompts.jsonl"), "w", encoding="utf-8") as f:
        for j in seeds:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    # validation pack: evenly spaced sequences of the existing Dara data (different categories: email, code, Tamil, ...)
    prompts = {j["id"]: j["prompt"] for j in read_jsonl(a.prompts)}
    idx = read_jsonl(os.path.join(a.data, "index.jsonl"))
    raw = open(os.path.join(a.data, "seqs.bin"), "rb").read()
    feats = np.memmap(os.path.join(a.data, "feats.bin"), dtype=np.float16, mode="r")
    offs, off = [], 0
    for ix in idx:
        offs.append(off)
        off += 8 + 4 * ix["n_total"]
    pk = os.path.join(a.out, "validation_pack")
    os.makedirs(pk, exist_ok=True)
    step = max(1, len(idx) // a.pack_size)
    chosen = list(range(0, len(idx), step))[: a.pack_size]
    out_idx, foff = [], 0
    with open(os.path.join(pk, "seqs.bin"), "wb") as fs, open(os.path.join(pk, "feats.bin"), "wb") as ff, \
            open(os.path.join(pk, "prompts.jsonl"), "w", encoding="utf-8") as fp:
        for k in chosen:
            ix = idx[k]
            n, p = struct.unpack_from("ii", raw, offs[k])
            assert n == ix["n_total"] and p == ix["n_prompt"]
            fs.write(raw[offs[k]: offs[k] + 8 + 4 * n])
            f = np.asarray(feats[ix["feat_offset"]: ix["feat_offset"] + n * HIDDEN])
            ff.write(f.tobytes())
            fp.write(json.dumps({"id": ix["id"], "prompt": prompts[ix["id"]]}, ensure_ascii=False) + "\n")
            out_idx.append({"id": ix["id"], "n_total": n, "n_prompt": p, "feat_offset": foff})
            foff += n * HIDDEN
    with open(os.path.join(pk, "index.jsonl"), "w", encoding="utf-8") as f:
        for j in out_idx:
            f.write(json.dumps(j) + "\n")
    total = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(a.out) for fn in fns)
    print(f"kit -> {a.out}: {len(seeds)} seed prompts, validation pack of {len(out_idx)} sequences ({sum(j['n_total'] for j in out_idx)} tokens), {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
