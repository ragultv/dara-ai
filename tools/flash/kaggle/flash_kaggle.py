"""Dara Flash drafter data distillation on a Kaggle GPU (Nano = Gemma 4 E2B).

The same data the local tools/flash_data_gemma.cu writes (seqs.bin, feats.bin, index.jsonl), made much faster with batched HF generation:
  * weights: the SAME Q4_K_M GGUF Dara runs, dequantized into the real transformers Gemma4 modules (fp16 on the GPU), so the target's
    hidden states match what the runtime verifies against (differences: int8 activations in Dara's fast path, fp16 here);
  * every sequence = Gemma 4 chat template (no thinking) + the model's own answer; features = final normed hidden state of every position;
  * `validate` compares all of this against sequences recorded by Dara itself (the validation pack) before any long run.

Sub-commands (the notebook runs them; each can also be run by hand):
  validate  --gguf G --hf-dir D --pack P            template / feature / top-1 agreement against Dara's own data
  embd      --gguf G --out F                        token_embd as fp16 [vocab][hidden] (the trainer's --embd file)
  prompts   --gguf G --hf-dir D --out F ...         self-instruct prompts, batched (the model writes the requests)
  gen       --gguf G --hf-dir D --prompts F --out-dir D ...   answers + features, resumable, one process per GPU
"""
import argparse
import json
import math
import os
import random
import re
import struct
import sys
import time

import numpy as np
import torch

KIT_VERSION = 4   # the notebook refuses an older kit (a stale dataset version gives confusing errors)
HIDDEN = 1536
EOS_IDS = (1, 106)   # <eos>, <turn|>: Dara stops on either (the token itself stays in the sequence)
APP_SAMPLING = dict(temperature=0.7, top_k=40, top_p=0.95, min_p=0.05)   # the engine's SampleParams defaults (what the app sends)


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------------------------------------------------------------------
# GGUF -> HF Gemma4ForCausalLM
# --------------------------------------------------------------------------------------------------------------------------------------
class Gguf:
    def __init__(self, path):
        from gguf import GGUFReader
        from gguf.quants import dequantize
        self.reader = GGUFReader(path)
        self.t = {t.name: t for t in self.reader.tensors}
        self.deq = dequantize

    def dims(self, name):
        """(rows, cols) of a matrix, or (n,) of a vector, in HF order ([out][in])."""
        s = [int(x) for x in self.t[name].shape]
        return (s[1], s[0]) if len(s) == 2 else (s[0],)

    def f32(self, name):
        t = self.t[name]
        if t.tensor_type.name == "F32":
            return np.asarray(t.data, dtype=np.float32)
        if t.tensor_type.name == "F16":
            return np.asarray(t.data, dtype=np.float16).astype(np.float32)
        return self.deq(t.data, t.tensor_type).astype(np.float32)

    def tensor(self, name):
        d = self.dims(name)
        return torch.from_numpy(self.f32(name).reshape(d).copy())

    def row_chunks(self, name, step=16384):
        """A large matrix in row chunks (float32), never fully expanded."""
        t = self.t[name]
        n = t.data.shape[0]
        for r in range(0, n, step):
            m = min(step, n - r)
            raw = t.data[r:r + m]
            if t.tensor_type.name == "F32":
                a = np.asarray(raw, dtype=np.float32)
            elif t.tensor_type.name == "F16":
                a = np.asarray(raw, dtype=np.float16).astype(np.float32)
            else:
                a = self.deq(raw, t.tensor_type).astype(np.float32)
            yield r, torch.from_numpy(a.reshape(m, -1).copy())


def hf_param_map(n_layers, n_shared):
    """HF parameter name -> (GGUF tensor name, kind). The tied lm_head is not listed. Mirrors tools/ref_forward_gemma.py (validated against Dara)."""
    m = {
        "model.embed_tokens.weight": ("token_embd.weight", "big"),
        "model.embed_tokens_per_layer.weight": ("per_layer_token_embd.weight", "big"),
        "model.per_layer_model_projection.weight": ("per_layer_model_proj.weight", "mat"),
        "model.per_layer_projection_norm.weight": ("per_layer_proj_norm.weight", "mat"),
        "model.norm.weight": ("output_norm.weight", "mat"),
    }
    per = {
        "input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "post_attention_norm.weight",
        "pre_feedforward_layernorm.weight": "ffn_norm.weight", "post_feedforward_layernorm.weight": "post_ffw_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight", "mlp.down_proj.weight": "ffn_down.weight",
        "self_attn.q_proj.weight": "attn_q.weight", "self_attn.o_proj.weight": "attn_output.weight", "self_attn.q_norm.weight": "attn_q_norm.weight",
        "per_layer_input_gate.weight": "inp_gate.weight", "per_layer_projection.weight": "proj.weight",
        "post_per_layer_input_norm.weight": "post_norm.weight", "layer_scalar": "layer_output_scale.weight",
    }
    own_kv = {"self_attn.k_proj.weight": "attn_k.weight", "self_attn.v_proj.weight": "attn_v.weight", "self_attn.k_norm.weight": "attn_k_norm.weight"}
    first_shared = n_layers - n_shared
    for i in range(n_layers):
        for k, g in per.items():
            m[f"model.layers.{i}.{k}"] = (f"blk.{i}.{g}", "mat")
        if i < first_shared:   # shared layers read another layer's K/V; the GGUF still stores unused k/v tensors for them
            for k, g in own_kv.items():
                m[f"model.layers.{i}.{k}"] = (f"blk.{i}.{g}", "mat")
    return m


def hf_text_config(hf_dir):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    raw = json.load(open(os.path.join(hf_dir, "config.json")))
    cfg = Gemma4TextConfig(**raw.get("text_config", raw))
    cfg._attn_implementation = "sdpa"
    return cfg


def check_mapping(gguf, cfg):
    """Problems (empty = fine) between the HF tensor list/shapes (meta device, no memory) and the GGUF tensors."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    with torch.device("meta"):
        model = Gemma4ForCausalLM(cfg)
    params = {n: tuple(t.shape) for n, t in model.state_dict().items()}   # parameters and persistent buffers (layer_scalar)
    pmap = hf_param_map(cfg.num_hidden_layers, cfg.num_kv_shared_layers)
    bad = []
    for n, shp in params.items():
        if n == "lm_head.weight":
            continue
        if n not in pmap:
            bad.append(f"HF parameter without a GGUF source: {n}")
            continue
        g = pmap[n][0]
        if g not in gguf.t:
            bad.append(f"GGUF tensor missing: {g} (for {n})")
            continue
        d = gguf.dims(g)
        if tuple(d) != tuple(shp) and not (len(shp) == 1 and math.prod(d) == shp[0]):
            bad.append(f"shape {n}: HF {shp} vs GGUF {d}")
    for n in pmap:
        if n not in params:
            bad.append(f"mapped name is not an HF parameter: {n}")
    return bad


@torch.no_grad()
def build_model(gguf_path, hf_dir, device="cuda", dtype=torch.float16):
    """Gemma4ForCausalLM with the GGUF's weights dequantized into it (born in `dtype` on `device`: no fp32 copy of the 5 B parameters)."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    cfg = hf_text_config(hf_dir)
    g = Gguf(gguf_path)
    bad = check_mapping(g, cfg)
    if bad:
        raise RuntimeError("GGUF does not match the HF model:\n  " + "\n  ".join(bad[:20]))
    t0 = time.time()
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = Gemma4ForCausalLM(cfg)
    finally:
        torch.set_default_dtype(prev)
    params = model.state_dict()   # tensors share storage with the parameters / buffers: copy_ writes into the model
    for name, (gname, kind) in hf_param_map(cfg.num_hidden_layers, cfg.num_kv_shared_layers).items():
        p = params[name]
        if kind == "big":
            for r, rows in g.row_chunks(gname):
                p.data[r:r + rows.shape[0]].copy_(rows.to(device=p.device, dtype=p.dtype))
        else:
            p.data.copy_(g.tensor(gname).reshape(p.shape).to(device=p.device, dtype=p.dtype))
    model.tie_weights()
    model.eval()
    log(f"model ready on {device} in {time.time() - t0:.0f}s ({sum(p.numel() for p in model.parameters()) / 1e9:.2f} B parameters, {dtype})")
    return model


def load_tokenizer(hf_dir):
    sys.modules.setdefault("torchao", None)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(hf_dir)


def render_prompt(tok, text):
    """Token ids of one user message with the Gemma 4 chat template, generation prompt, no thinking (identical to Dara's tokenizer: checked)."""
    s = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
    return tok(s, add_special_tokens=False)["input_ids"]


# --------------------------------------------------------------------------------------------------------------------------------------
# generation + features
# --------------------------------------------------------------------------------------------------------------------------------------
class NanGuard:
    """LogitsProcessor that stops on non-finite logits. Without it greedy decoding turns NaN into garbage tokens silently."""

    def __call__(self, input_ids, scores):
        if not bool(torch.isfinite(scores).all()):
            raise RuntimeError("non-finite logits during generation (fp16 overflow or NaN): use DTYPE='bf16', or check the validation cell")
        return scores


def _guards():
    from transformers import LogitsProcessorList
    return LogitsProcessorList([NanGuard()])


@torch.no_grad()
def generate_batch(model, prompt_ids, max_new, sample, pad_id):
    """Answers for a batch of prompts. All prompts must have the SAME length: there is no padding at all, because in fp16 the pad positions of a
    left-padded batch can turn into NaN and poison the real tokens (and greedy decoding would hide it). Returns the generated ids of each,
    cut after the first end token (included)."""
    from transformers import GenerationConfig
    dev = next(model.parameters()).device
    L = len(prompt_ids[0])
    if any(len(p) != L for p in prompt_ids):
        raise ValueError("generate_batch needs prompts of equal length")
    ids = torch.tensor(prompt_ids, dtype=torch.long)
    att = torch.ones_like(ids)
    kw = dict(max_new_tokens=max_new, eos_token_id=list(EOS_IDS), pad_token_id=pad_id, use_cache=True)
    gc = GenerationConfig(do_sample=True, **APP_SAMPLING, **kw) if sample else GenerationConfig(do_sample=False, **kw)
    out = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), generation_config=gc, logits_processor=_guards())[:, L:].cpu().tolist()
    res = []
    for row in out:
        cut = len(row)
        for j, t in enumerate(row):
            if t in EOS_IDS:
                cut = j + 1
                break
        res.append(row[:cut])
    return res


@torch.no_grad()
def hidden_states(model, seqs, pad_id, chunk=16):
    """Final normed hidden state of every position of every sequence (fp16 numpy [n][hidden]), teacher forced, right padded."""
    dev = next(model.parameters()).device
    out = []
    for a in range(0, len(seqs), chunk):
        part = seqs[a:a + chunk]
        T = max(len(s) for s in part)
        ids = torch.full((len(part), T), pad_id, dtype=torch.long)
        att = torch.zeros((len(part), T), dtype=torch.long)
        for i, s in enumerate(part):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        h = model.model(input_ids=ids.to(dev), attention_mask=att.to(dev)).last_hidden_state   # after the final norm, before the head
        for i, s in enumerate(part):
            out.append(h[i, :len(s)].to(torch.float16).cpu().numpy())
    return out


class Shard:
    """One output directory in the trainer's format. Resumable: a killed run is cut back to its last complete index line."""

    def __init__(self, d, hidden=HIDDEN):
        self.d, self.hidden = d, hidden
        os.makedirs(d, exist_ok=True)
        self.p_seq, self.p_feat, self.p_idx = (os.path.join(d, n) for n in ("seqs.bin", "feats.bin", "index.jsonl"))
        self.done, self.tokens, self.feat_off = set(), 0, 0
        seq_bytes = 0
        if os.path.exists(self.p_idx):
            good = []
            for line in open(self.p_idx, encoding="utf-8"):
                try:
                    j = json.loads(line)
                except ValueError:
                    break
                good.append(j)
                seq_bytes += 8 + 4 * j["n_total"]
                self.feat_off = j["feat_offset"] + j["n_total"] * hidden
                self.tokens += j["n_total"]
                self.done.add(j["id"])
            for path, size in ((self.p_seq, seq_bytes), (self.p_feat, self.feat_off * 2)):
                if os.path.exists(path) and os.path.getsize(path) > size:
                    with open(path, "r+b") as f:
                        f.truncate(size)
            with open(self.p_idx, "w", encoding="utf-8") as f:
                for j in good:
                    f.write(json.dumps(j) + "\n")
        self.fs, self.ff, self.fi = open(self.p_seq, "ab"), open(self.p_feat, "ab"), open(self.p_idx, "a", encoding="utf-8")

    def add(self, pid, seq, n_prompt, feats):
        assert feats.shape == (len(seq), self.hidden) and feats.dtype == np.float16
        self.fs.write(struct.pack("ii", len(seq), n_prompt) + np.asarray(seq, dtype=np.int32).tobytes())
        self.ff.write(feats.tobytes())
        self.fi.write(json.dumps({"id": pid, "n_total": len(seq), "n_prompt": n_prompt, "feat_offset": self.feat_off}) + "\n")
        self.feat_off += len(seq) * self.hidden
        self.tokens += len(seq)
        self.done.add(pid)

    def flush(self):
        for f in (self.fs, self.ff):   # index last: a line in it means the sequence and features are complete
            f.flush()
            os.fsync(f.fileno())
        self.fi.flush()

    def close(self):
        self.flush()
        for f in (self.fs, self.ff, self.fi):
            f.close()


def run_gen(model, tok, prompts, out_dir, max_new=256, batch=48, sampled_frac=0.4, max_tokens=0, hidden=HIDDEN, shard_name="shard", max_seconds=0):
    """prompts: [{"id", "prompt"}]. Every prompt gets one answer (greedy, or sampled like the app for `sampled_frac` of them) and its features."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    sh = Shard(out_dir, hidden)
    log(f"[{shard_name}] resuming with {len(sh.done)} sequences, {sh.tokens} tokens")
    todo = []
    for p in prompts:
        if p["id"] in sh.done:
            continue
        ids = render_prompt(tok, p["prompt"])
        if len(ids) + max_new + 1 > 2048 or len(ids) > 700:
            continue
        todo.append((p["id"], ids, (p["id"] % 100) < int(sampled_frac * 100)))
    log(f"[{shard_name}] {len(todo)} prompts to do")
    t0, t_tokens = time.time(), 0
    # batches share the mode and the EXACT prompt length (no padding: see generate_batch); the rare lengths make small batches
    groups = {}
    for t in todo:
        groups.setdefault((t[2], len(t[1])), []).append(t)
    batches = []
    for (sample, _), items in sorted(groups.items()):
        for a in range(0, len(items), batch):
            batches.append((sample, items[a:a + batch]))
    random.Random(1234 + len(sh.done)).shuffle(batches)   # mixed modes and lengths all along, so any cap (tokens/time) still leaves a balanced set
    log(f"[{shard_name}] {len(batches)} batches (largest {batch}, mean size {len(todo) / max(1, len(batches)):.1f})")
    for sample, part in batches:
        if (max_tokens and sh.tokens >= max_tokens) or (max_seconds and time.time() - t0 >= max_seconds):
            break
        gen = generate_batch(model, [t[1] for t in part], max_new, sample, pad)
        seqs = [t[1] + g for t, g in zip(part, gen)]
        feats = hidden_states(model, seqs, pad)
        for t, s_, f in zip(part, seqs, feats):
            sh.add(t[0], s_, len(t[1]), f)
        sh.flush()
        t_tokens += sum(len(x) for x in seqs)
        el = time.time() - t0
        log(f"[{shard_name}] {'sampled' if sample else 'greedy '} {len(sh.done)} seqs, {sh.tokens} tokens total, {t_tokens / el:.0f} tok/s this run")
    sh.close()
    log(f"[{shard_name}] done: {len(sh.done)} sequences, {sh.tokens} tokens")


# --------------------------------------------------------------------------------------------------------------------------------------
# prompts (self-instruct, batched)
# --------------------------------------------------------------------------------------------------------------------------------------
def parse_requests(text):
    items = []
    for line in text.split("\n"):   # numbered items; unnumbered lines continue the previous item (included data, code, tables)
        m = re.match(r"^\s*\d+[.)]\s*(.+)$", line)
        if m:
            items.append(m.group(1))
        elif items and line.strip():
            items[-1] += "\n" + line.strip()
    out = []
    for q in items:
        q = q.strip().strip('"').strip()
        if 15 <= len(q) <= 1500:
            out.append(q)
    return out


def run_prompts(model, tok, kit_dir, out_path, samples, repeats, per, shard, nshards, start_id, seed=1):
    """For every category: one generate() call with `samples` different sampled completions of the same request (so there is no padding), each
    listing `per` user requests. `repeats` calls per category. Finished (repeat, category) jobs are remembered in <out>.jobs."""
    sys.path.insert(0, kit_dir)
    from self_instruct import ASK, CATEGORIES
    from transformers import GenerationConfig
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    dev = next(model.parameters()).device
    cats = [c for i, c in enumerate(CATEGORIES) if i % nshards == shard]
    seen, n_id = set(), start_id
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            if line.strip():
                j = json.loads(line)
                seen.add(j["prompt"].lower())
                n_id = max(n_id, j["id"] + 1)
    jobs_path = out_path + ".jobs"
    done_jobs = set(open(jobs_path, encoding="utf-8").read().split("\n")) if os.path.exists(jobs_path) else set()
    torch.manual_seed(seed + shard)
    gc = GenerationConfig(do_sample=True, temperature=1.0, top_p=0.95, top_k=64, max_new_tokens=40 * per, num_return_sequences=samples,
                          eos_token_id=list(EOS_IDS), pad_token_id=pad)
    jobs = [(r, c) for r in range(repeats) for c in cats]
    t0, made = time.time(), 0
    with open(out_path, "a", encoding="utf-8") as fo, open(jobs_path, "a", encoding="utf-8") as fj:
        for k, (r, c) in enumerate(jobs):
            key = f"{r}|{c}"
            if key in done_jobs:
                continue
            ids = torch.tensor([render_prompt(tok, ASK.format(n=per, c=c))], dtype=torch.long)
            with torch.no_grad():
                out = model.generate(input_ids=ids.to(dev), attention_mask=torch.ones_like(ids).to(dev), generation_config=gc,
                                     logits_processor=_guards())[:, ids.shape[1]:].cpu().tolist()
            for row in out:
                cut = next((j for j, x in enumerate(row) if x in EOS_IDS), len(row))
                for q in parse_requests(tok.decode(row[:cut], skip_special_tokens=True)):
                    if q.lower() in seen:
                        continue
                    seen.add(q.lower())
                    fo.write(json.dumps({"id": n_id, "prompt": q, "category": c}, ensure_ascii=False) + "\n")
                    n_id += 1
                    made += 1
            fo.flush()
            fj.write(key + "\n")
            fj.flush()
            log(f"[prompts {shard}] job {k + 1}/{len(jobs)}, {made} new prompts this run, {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------------------------------------------------------------------
# validation against Dara's own recorded data
# --------------------------------------------------------------------------------------------------------------------------------------
def load_pack(pack):
    prompts = {}
    for line in open(os.path.join(pack, "prompts.jsonl"), encoding="utf-8"):
        if line.strip():
            j = json.loads(line)
            prompts[j["id"]] = j["prompt"]
    idx = [json.loads(l) for l in open(os.path.join(pack, "index.jsonl"), encoding="utf-8") if l.strip()]
    raw = open(os.path.join(pack, "seqs.bin"), "rb").read()
    feats = np.fromfile(os.path.join(pack, "feats.bin"), dtype=np.float16)
    off, out = 0, []
    for ix in idx:
        n, p = struct.unpack_from("ii", raw, off)
        off += 8
        seq = np.frombuffer(raw, np.int32, n, off).tolist()
        off += 4 * n
        f = feats[ix["feat_offset"]: ix["feat_offset"] + n * HIDDEN].reshape(n, HIDDEN)
        out.append((ix["id"], prompts[ix["id"]], seq, p, f))
    return out


@torch.no_grad()
def run_validate(model, tok, pack):
    """Template identity, feature agreement and top-1 agreement with the sequences Dara recorded. Returns True when all is well."""
    items = load_pack(pack)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ok = True
    seqs = [it[2] for it in items]
    feats = hidden_states(model, seqs, pad, chunk=4)
    dev = next(model.parameters()).device
    head = model.get_input_embeddings().weight          # tied head [vocab][hidden]
    cap = float(model.config.final_logit_softcapping or 0)
    cos_all, rel_all, top1_hit, top1_n, tmpl_bad = [], [], 0, 0, 0
    for (pid, text, seq, n_prompt, ref), f in zip(items, feats):
        if render_prompt(tok, text) != seq[:n_prompt]:
            tmpl_bad += 1
        a = torch.tensor(f, dtype=torch.float32)
        b = torch.tensor(ref, dtype=torch.float32)
        if not torch.isfinite(a).all():
            log(f"  id {pid}: NON-FINITE features (fp16 overflow?)")
            ok = False
            continue
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        cos_all.append(cos)
        rel_all.append((a - b).norm(dim=-1) / b.norm(dim=-1).clamp(min=1e-6))
        h = torch.tensor(f[n_prompt - 1: -1]).to(dev)    # positions that predict answer tokens
        lg = h @ head.T
        if cap > 0:
            lg = torch.tanh(lg.float() / cap) * cap
        pred = lg.argmax(-1).cpu()
        top1_hit += int((pred == torch.tensor(seq[n_prompt:])).sum())
        top1_n += len(pred)
    # generation: the exact path the data run uses (equal-length batch, greedy and sampled, NaN guard on). Dara's answer is the reference for greedy.
    agree = []
    for pid, text, seq, n_prompt, _ in items[:3]:
        ids = seq[:n_prompt]
        try:
            g = generate_batch(model, [ids] * 4, 24, False, pad)
            generate_batch(model, [ids] * 4, 24, True, pad)
        except Exception as e:  # noqa: BLE001
            log(f"  generation FAILED on id {pid}: {type(e).__name__}: {str(e)[:300]}")
            ok = False
            break
        ref = seq[n_prompt: n_prompt + 24]
        agree.append(sum(x == y for x, y in zip(g[0], ref)) / max(1, min(len(ref), len(g[0]))))
    if agree:
        log(f"generation: greedy first-24-token agreement with Dara's answers {[round(x, 2) for x in agree]} (informational; fp16 vs Q4-int8 diverge eventually)")
    cos = torch.cat(cos_all)
    rel = torch.cat(rel_all)
    log(f"template: {len(items) - tmpl_bad}/{len(items)} prompts tokenise identically to Dara")
    log(f"features vs Dara: cosine mean {cos.mean():.4f}, p1 {cos.quantile(0.01):.4f}, min {cos.min():.4f}; relative L2 mean {rel.mean():.4f}")
    log(f"top-1 vs Dara's own answers (teacher forced): {top1_hit}/{top1_n} = {top1_hit / top1_n:.4f}")
    if tmpl_bad:
        ok = False
    if cos.mean() < 0.97 or top1_hit / top1_n < 0.90:
        ok = False
    log("VALIDATION " + ("PASSED" if ok else "FAILED: do not generate data until this is understood (try dtype bf16, or WEIGHTS check)"))
    return ok


# --------------------------------------------------------------------------------------------------------------------------------------
def write_embd(gguf_path, out_path, hidden=HIDDEN):
    g = Gguf(gguf_path)
    with open(out_path, "wb") as f:
        for _, rows in g.row_chunks("token_embd.weight"):
            f.write(rows.numpy().astype(np.float16).tobytes())
    n = os.path.getsize(out_path) // (2 * hidden)
    log(f"wrote {out_path}: {n} rows x {hidden}")


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "embd", "prompts", "gen"):
        s = sub.add_parser(name)
        s.add_argument("--gguf", required=True)
        s.add_argument("--hf-dir")
        s.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
        s.add_argument("--device", default="cuda")
        if name == "validate":
            s.add_argument("--pack", required=True)
        if name == "embd":
            s.add_argument("--out", required=True)
        if name == "prompts":
            s.add_argument("--kit", required=True)
            s.add_argument("--out", required=True)
            s.add_argument("--samples", type=int, default=16, help="completions sampled per category per call (one batch, no padding)")
            s.add_argument("--repeats", type=int, default=1, help="calls per category")
            s.add_argument("--per", type=int, default=20, help="requests asked for per completion")
            s.add_argument("--shard", type=int, default=0)
            s.add_argument("--nshards", type=int, default=1)
            s.add_argument("--start-id", type=int, default=100000)
        if name == "gen":
            s.add_argument("--prompts", required=True)
            s.add_argument("--out-dir", required=True)
            s.add_argument("--shard", type=int, default=0)
            s.add_argument("--nshards", type=int, default=1)
            s.add_argument("--max-new", type=int, default=256)
            s.add_argument("--batch", type=int, default=48)
            s.add_argument("--sampled-frac", type=float, default=0.4)
            s.add_argument("--max-tokens", type=int, default=0, help="stop this shard after about this many tokens (0 = all prompts)")
            s.add_argument("--max-seconds", type=float, default=0, help="stop cleanly after this long (a Kaggle session ends at 12 h; unsaved temp data is lost)")
    a = ap.parse_args()
    if a.cmd == "embd":
        write_embd(a.gguf, a.out)
        return 0
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    model = build_model(a.gguf, a.hf_dir, a.device, dtype)
    tok = load_tokenizer(a.hf_dir)
    if a.cmd == "validate":
        return 0 if run_validate(model, tok, a.pack) else 1
    if a.cmd == "prompts":
        run_prompts(model, tok, a.kit, a.out, a.samples, a.repeats, a.per, a.shard, a.nshards, a.start_id + a.shard * 1_000_000)
    if a.cmd == "gen":
        prompts = [p for i, p in enumerate(read_jsonl(a.prompts)) if i % a.nshards == a.shard]
        run_gen(model, tok, prompts, a.out_dir, a.max_new, a.batch, a.sampled_frac, a.max_tokens, shard_name=f"gen {a.shard}", max_seconds=a.max_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
