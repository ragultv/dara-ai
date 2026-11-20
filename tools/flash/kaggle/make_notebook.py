"""Writes dara_flash_kaggle.ipynb (the notebook is generated so that its code cells stay plain, testable Python).
Usage: python -I tools/flash/kaggle/make_notebook.py [--out tools/flash/kaggle/dara_flash_kaggle.ipynb]
"""
import argparse
import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md("""
# Dara Flash drafter: distill + train on Kaggle (Nano = Gemma 4 E2B)

Makes the training data for the Flash drafter with batched generation on the Kaggle GPUs (the model's own answers + its hidden states),
then trains the drafter on it, in this one session. Output: `drafter_nano_v3.pt` (about 140 MB), which you export and test on your own PC.

**Before you run it**
1. Upload the folder made by `python -I tools/flash/kaggle/make_kit.py` as a *private* dataset named `dara-flash-kit`, and add it to this notebook (Add Input).
2. Settings: Accelerator **GPU T4 x2**, Internet **On**, Persistence: any.
3. Run the cells top to bottom. **Do not skip the validation cell**: it compares this notebook's model against sequences Dara recorded itself.

The data lives in the session's temporary disk. It is gone when the session ends, so distill and train in the same session (about 12 h limit).
If a cell is interrupted, run it again: generation resumes where it stopped.
""")

code("""
# ---- settings (the defaults suit a 12 h session on 2 x T4) -------------------------------------------------------------------
TARGET_TOKENS = 6_000_000   # tokens to distill (prompt + answer). The v2 drafter saw 0.99 M. Features take about 3 KB per token on disk.
MAX_NEW       = 256         # answer length (same as the local data)
SAMPLED_FRAC  = 0.4         # share of answers sampled like the app does (temperature 0.7, top-k 40, top-p 0.95, min-p 0.05); the rest greedy
PROMPT_SAMPLES = 16         # per category and call: this many sampled completions of the same request (one batch), each listing 20 requests
PROMPT_REPEATS = 1          # calls per category. Unique prompts needed: TARGET_TOKENS / ~230 (about 26 000 for 6 M tokens)
GEN_HOURS     = 6.0         # time budget for the distillation step (it stops cleanly at the budget or at TARGET_TOKENS)
DTYPE         = "fp16"      # the T4 has no bf16 hardware; if validation shows non-finite features or low agreement, try "bf16" (slower) or "fp32" (needs both GPUs' memory: no)
GEN_BATCH     = 48
EPOCHS        = 6
TRAIN_BATCH   = 16          # sequences per step
LR            = 5e-4        # v2 used 3e-4 at batch 4; larger batches take a higher rate
""")

code("""
import glob, hashlib, json, os, random, re, shutil, subprocess, sys, time

KIT = os.path.dirname(next(iter(glob.glob("/kaggle/input/**/flash_kaggle.py", recursive=True)), ""))
assert KIT, "attach the dataset dara-flash-kit to this notebook (Add Input)"
assert "KIT_VERSION = 4" in open(os.path.join(KIT, "flash_kaggle.py")).read(), (
    "the dataset still holds an OLD flash_kaggle.py: publish a new dataset version with the current file, re-add it under Add Input, restart the session")
print("kit:", KIT)

# transformers with the Gemma 4 modules (the repo's reference used 5.5.4); checked in a subprocess so this kernel stays clean
def has_gemma4():
    return subprocess.run([sys.executable, "-c", "import transformers.models.gemma4.modeling_gemma4"], capture_output=True).returncode == 0
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf"], check=True)
if not has_gemma4():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"], check=True)
assert has_gemma4(), "this transformers has no Gemma 4: pip install a newer one and restart the kernel"
print(subprocess.run([sys.executable, "-c", "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'gpus', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"],
                     capture_output=True, text=True).stdout)

# where the data goes: the disk with the most free space
cands = [d for d in ("/kaggle/temp", "/kaggle/working", "/tmp") if os.path.isdir(d)]
free = {d: shutil.disk_usage(d).free for d in cands}
ROOT = os.path.join(max(free, key=free.get), "dara_flash")
os.makedirs(ROOT, exist_ok=True)
OUT = "/kaggle/working"
room = int((free[os.path.dirname(ROOT)] - 8e9) * 0.8 / 3100)   # 8 GB for the GGUF, embedding table and logs; 3.1 KB per token
TARGET_TOKENS = min(TARGET_TOKENS, room)
NG = max(1, int(subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"], capture_output=True, text=True).stdout.strip() or 1))
print(f"data dir {ROOT}: {free[os.path.dirname(ROOT)] / 1e9:.0f} GB free -> target {TARGET_TOKENS:,} tokens; {NG} GPU(s)")
LIB = os.path.join(KIT, "flash_kaggle.py")
HF = os.path.join(KIT, "hf")
""")

code("""
def run_parallel(cmds, logs, every=60, tail_lines=1):
    \"\"\"One process per GPU (CUDA_VISIBLE_DEVICES=i); blocks until all finish, printing the last log line of each now and then.\"\"\"
    procs = []
    for i, (cmd, log) in enumerate(zip(cmds, logs)):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), PYTHONUNBUFFERED="1")
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT))   # fresh log per run: no stale errors
    t0 = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(every)
        for i, log in enumerate(logs):
            lines = open(log, errors="replace").read().strip().split("\\n")
            print(f"[{(time.time() - t0) / 60:6.1f} min] gpu{i}: " + " | ".join(lines[-tail_lines:])[:220])
    bad = [(i, p.returncode) for i, p in enumerate(procs) if p.returncode != 0]
    if bad:
        for i, _ in bad:
            print(f"--- log of process {i} ---\\n" + open(logs[i], errors="replace").read()[-3000:])
        raise RuntimeError(f"process(es) failed: {bad}")

def tokens_done(d):
    n = 0
    for p in glob.glob(os.path.join(d, "shard*", "index.jsonl")):
        n += sum(json.loads(l)["n_total"] for l in open(p) if l.strip())
    return n
""")

md("""
## 1. Self-test without weights (about 1 minute)
Tiny random model: checks batched generation, that the features really are the final normed hidden state, the writer, resume, and the trainer's data loader.
""")
code("""
r = subprocess.run([sys.executable, os.path.join(KIT, "smoke_test.py"), "--hf-dir", HF, "--device", "cuda"], capture_output=True, text=True)
print(r.stdout[-2500:], r.stderr[-1500:] if r.returncode else "")
assert r.returncode == 0, "smoke test failed"
""")

md("""
## 2. The weights: the exact Q4_K_M GGUF Dara runs
Downloaded at the revision and checked against the SHA-256 Dara's catalog uses (or taken from an attached input, if you added the GGUF as a dataset).
""")
code("""
REPO, REV, FILE = "unsloth/gemma-4-E2B-it-GGUF", "0314792d7f1f7e229411f620751375812bb9faf2", "gemma-4-E2B-it-Q4_K_M.gguf"
SIZE, SHA = 3106738272, "740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8"
found = glob.glob(f"/kaggle/input/**/{FILE}", recursive=True)
if found:
    GGUF = found[0]
else:
    from huggingface_hub import hf_hub_download
    GGUF = hf_hub_download(REPO, FILE, revision=REV, local_dir=os.path.join(ROOT, "gguf"))
assert os.path.getsize(GGUF) == SIZE, f"unexpected size {os.path.getsize(GGUF)}"
h = hashlib.sha256()
with open(GGUF, "rb") as f:
    for blk in iter(lambda: f.read(1 << 24), b""):
        h.update(blk)
assert h.hexdigest() == SHA, "GGUF checksum differs from the one Dara's catalog expects"
print("GGUF ok:", GGUF)
""")

md("""
## 3. Validation: this notebook's model vs sequences recorded by Dara itself
Loads the GGUF dequantized into the real transformers Gemma 4 modules, re-runs 12 recorded sequences, and compares prompt tokens, hidden states and top-1 predictions.
It also runs the generation path (equal-length batches, greedy and sampled, NaN guard on). Expect: all prompts identical, feature cosine about 0.99+, top-1 well above 90%, no generation error. **If it says FAILED, stop**: data from a mismatched model would train a drafter that does not accept.
""")
code("""
cmd = [sys.executable, LIB, "validate", "--gguf", GGUF, "--hf-dir", HF, "--pack", os.path.join(KIT, "validation_pack"), "--dtype", DTYPE]
r = subprocess.run(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"), capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:] if r.returncode else "")
assert r.returncode == 0, "validation failed: see the message above"
""")

md("""
## 4. Prompts
The model writes diverse user requests per category (the repo's self-instruct categories), sampled, one process per GPU. Merged with the 4,249 seed prompts of the v2 data.
""")
code("""
logs = [os.path.join(ROOT, f"prompts_{i}.log") for i in range(NG)]
cmds = [[sys.executable, LIB, "prompts", "--gguf", GGUF, "--hf-dir", HF, "--kit", KIT, "--out", os.path.join(ROOT, f"prompts_{i}.jsonl"),
         "--samples", str(PROMPT_SAMPLES), "--repeats", str(PROMPT_REPEATS), "--shard", str(i), "--nshards", str(NG), "--dtype", DTYPE] for i in range(NG)]
run_parallel(cmds, logs)

seen, allp = set(), []
def add(j):
    k = j["prompt"].strip().lower()
    if k not in seen:
        seen.add(k)
        allp.append({"id": len(allp), "prompt": j["prompt"], "category": j.get("category", "")})
for l in open(os.path.join(KIT, "seed_prompts.jsonl"), encoding="utf-8"):
    add(json.loads(l))
for i in range(NG):
    for l in open(os.path.join(ROOT, f"prompts_{i}.jsonl"), encoding="utf-8"):
        if l.strip():
            add(json.loads(l))
random.Random(0).shuffle(allp)
for n, j in enumerate(allp):   # ids stay stable between re-runs because the merge order is fixed
    j["id"] = n
PROMPTS = os.path.join(ROOT, "prompts_all.jsonl")
with open(PROMPTS, "w", encoding="utf-8") as f:
    for j in allp:
        f.write(json.dumps(j, ensure_ascii=False) + "\\n")
print(f"{len(allp)} unique prompts -> about {len(allp) * 230:,} tokens available (target {TARGET_TOKENS:,})")
""")

md("""
## 5. Distill: answers + hidden states
Greedy answers (what verification compares against) plus `SAMPLED_FRAC` sampled like the app. One process per GPU, each stops at its share of `TARGET_TOKENS` or at the time budget.
""")
code("""
DATA = os.path.join(ROOT, "data")
logs = [os.path.join(ROOT, f"gen_{i}.log") for i in range(NG)]
cmds = [[sys.executable, LIB, "gen", "--gguf", GGUF, "--hf-dir", HF, "--prompts", PROMPTS, "--out-dir", os.path.join(DATA, f"shard{i}"),
         "--shard", str(i), "--nshards", str(NG), "--max-new", str(MAX_NEW), "--batch", str(GEN_BATCH), "--sampled-frac", str(SAMPLED_FRAC),
         "--max-tokens", str(TARGET_TOKENS // NG), "--max-seconds", str(int(GEN_HOURS * 3600)), "--dtype", DTYPE] for i in range(NG)]
run_parallel(cmds, logs, every=120)
print(f"distilled {tokens_done(DATA):,} tokens in {len(glob.glob(os.path.join(DATA, 'shard*')))} shard(s)")
""")

md("""
## 6. Train the drafter
The same architecture as v2 (34.6 M parameters: the runtime only runs this shape), fp16 autocast, on GPU 0.
`held-out next-token accuracy` is the number to watch: v2 reached **0.554**. Per-token acceptance is about that, and 2x needs roughly 0.8.
""")
code("""
EMBD = os.path.join(ROOT, "embd.f16")
if not os.path.exists(EMBD):
    r = subprocess.run([sys.executable, LIB, "embd", "--gguf", GGUF, "--out", EMBD], capture_output=True, text=True)
    print(r.stdout[-500:], r.stderr[-1500:] if r.returncode else "")
    assert r.returncode == 0

CKPT = os.path.join(OUT, "drafter_nano_v3.pt")
cmd = [sys.executable, os.path.join(KIT, "train_drafter.py"), "--data", *sorted(glob.glob(os.path.join(DATA, "shard*"))), "--embd", EMBD, "--out", CKPT,
       "--hidden", "1536", "--heads", "8", "--kv-heads", "2", "--head-dim", "256", "--rot", "64", "--theta", "1e7", "--inter", "4096", "--vocab-k", "32768",
       "--epochs", str(EPOCHS), "--batch", str(TRAIN_BATCH), "--lr", str(LR), "--noise", "0.2", "--holdout", "300", "--amp"]
p = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
logf = open(os.path.join(OUT, "train_nano_v3.log"), "w")
for line in p.stdout:
    logf.write(line)
    m = re.search(r"step ([0-9]+)/", line)
    if any(k in line for k in ("accuracy", "sequences", "parameters", "saved", "rror")) or (m and int(m.group(1)) % 500 == 0):
        print(line.rstrip())
p.wait()
assert p.returncode == 0, "training failed (see train_nano_v3.log)"
print("checkpoint:", CKPT, os.path.getsize(CKPT) // 1_000_000, "MB")
""")

md("""
## 7. Use it on your PC
Download `drafter_nano_v3.pt` and `train_nano_v3.log` from this notebook's Output, then in the repo:

```
python -I tools/flash/export_drafter.py --ckpt build/flash/drafter_nano_v3.pt --out build/flash/drafter_nano_v3.dflash
build\\cuda\\dara.exe --gguf models\\gemma4-e2b-gguf\\gemma-4-E2B-it-Q4_K_M.gguf --bench --chat "Write a fast Python merge sort algorithm" --new 400 ^
    --flash build\\flash\\drafter_nano_v3.dflash --spec 4 --flash-min-p 0.6
```
Compare `speculation: ... accepted (NN%)` and `warm decode` with the v2 drafter (`drafter_nano_v2.dflash`) on the same prompts (code and prose).
Install it only if it is clearly better: replace `drafter_nano_v2.dflash` in the models folder and the release's `flash\\` folder.
""")


def build(out):
    cells = []
    for kind, text in CELLS:
        src = text.split("\n")
        src = [l + "\n" for l in src[:-1]] + [src[-1]]
        if kind == "code":
            ast.parse(text)   # every code cell must at least parse
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})
        else:
            cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"wrote {out}: {len(cells)} cells")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "dara_flash_kaggle.ipynb"))
    build(ap.parse_args().out)
