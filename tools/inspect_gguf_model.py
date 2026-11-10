"""Summarize a GGUF model: architecture metadata, head tying, quantization coverage by tensor kind, tokenizer identity.
Usage: python -I tools/inspect_gguf_model.py <model.gguf> [<other.gguf> to compare tokenizers]"""
import collections
import hashlib
import sys

import numpy as np
from gguf import GGUFReader

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def field(r, k):
    f = r.fields.get(k)
    if f is None:
        return None
    return bytes(f.parts[f.data[0]]).decode() if f.types[0].name == "STRING" else f.parts[f.data[0]].tolist()[0]


def tok_digest(r):
    toks = r.fields["tokenizer.ggml.tokens"]
    mrg = r.fields["tokenizer.ggml.merges"]
    ty = r.fields["tokenizer.ggml.token_type"]
    h = hashlib.sha256()
    for i in toks.data:
        h.update(bytes(toks.parts[i]) + b"\0")
    for i in mrg.data:
        h.update(bytes(mrg.parts[i]) + b"\1")
    h.update(np.array([ty.parts[i][0] for i in ty.data], dtype=np.int32).tobytes())
    return h.hexdigest()[:16], len(toks.data), len(mrg.data), field(r, "tokenizer.ggml.pre")


def main():
    r = GGUFReader(sys.argv[1])
    arch = field(r, "general.architecture")
    print("architecture:", arch, "| name:", field(r, "general.name"), "| file_type:", field(r, "general.file_type"))
    for k in ["block_count", "embedding_length", "feed_forward_length", "attention.head_count", "attention.head_count_kv", "attention.key_length",
              "rope.freq_base", "rope.dimension_count", "ssm.conv_kernel", "ssm.state_size", "ssm.group_count", "ssm.time_step_rank", "ssm.inner_size",
              "full_attention_interval", "attention.layer_norm_rms_epsilon", "context_length"]:
        print(f"  {arch}.{k} = {field(r, arch + '.' + k)}")
    names = {t.name: t for t in r.tensors}
    print("output.weight present (untied head):", "output.weight" in names, "| token_embd:", names["token_embd.weight"].tensor_type.name, list(names["token_embd.weight"].shape))
    if "output.weight" in names:
        print("  output.weight:", names["output.weight"].tensor_type.name, list(names["output.weight"].shape))
    kinds = collections.defaultdict(collections.Counter)
    size = collections.Counter()
    for t in r.tensors:
        kind = t.name.split(".", 2)[2] if t.name.startswith("blk.") else t.name
        kinds[kind][t.tensor_type.name] += 1
        size[t.tensor_type.name] += int(t.n_bytes)
    print("tensor kinds -> quantization types (count):")
    for k in sorted(kinds):
        print(f"  {k:28s} {dict(kinds[k])}")
    tot = sum(size.values())
    print("bytes by type:", {k: f"{v / 2**20:.0f} MiB ({100 * v / tot:.1f}%)" for k, v in size.items()}, f"| total {tot / 2**30:.2f} GiB")
    print("tokenizer:", tok_digest(r))
    if len(sys.argv) > 2:
        print("other tokenizer:", tok_digest(GGUFReader(sys.argv[2])), "(identical to the first if the digests match)")


if __name__ == "__main__":
    main()
