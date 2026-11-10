"""Build an architecture manifest for one exact checkpoint.

Reads config.json and the safetensors *headers* only (HTTP range requests or local files),
so no weights are downloaded. Per-layer sizes are the checkpoint's stored bytes; the runtime
scales them by the chosen quantization.

Usage:
  python -I tools/make_manifest.py --repo Qwen/Qwen3.5-4B --out manifests/qwen3.5-4b.json
"""
import argparse
import json
import re
import struct
import sys
import urllib.request

HF = "https://huggingface.co/{repo}/resolve/{rev}/{path}"


def fetch(url, rng=None):
    req = urllib.request.Request(url)
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read(), r.headers


def commit_sha(repo, rev):
    data, _ = fetch(f"https://huggingface.co/api/models/{repo}/revision/{rev}")
    return json.loads(data)["sha"]


def shard_header(url):
    n = struct.unpack("<Q", fetch(url, (0, 7))[0])[0]
    if n > 100 << 20:
        raise ValueError("implausible safetensors header")
    return json.loads(fetch(url, (8, 8 + n - 1))[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--rev", default="main")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sha = commit_sha(a.repo, a.rev)  # pin the exact revision, not a moving branch
    url = lambda p: HF.format(repo=a.repo, rev=sha, path=p)
    cfg = json.loads(fetch(url("config.json"))[0])
    tc = cfg["text_config"]

    try:
        index = json.loads(fetch(url("model.safetensors.index.json"))[0])
        shards = sorted(set(index["weight_map"].values()))
    except Exception:
        shards = ["model.safetensors"]

    tensors = {}
    shapes = {}
    for s in shards:
        for k, v in shard_header(url(s)).items():
            if k != "__metadata__":
                tensors[k] = v["data_offsets"][1] - v["data_offsets"][0]
                shapes[k] = (v["shape"], v["dtype"])

    n = tc["num_hidden_layers"]
    layer_bytes = [0] * n
    other = {"embed_output": 0, "vision": 0, "mtp": 0, "unclassified": 0}
    lre = re.compile(r"\.language_model\.layers\.(\d+)\.|^model\.layers\.(\d+)\.")
    for name, size in tensors.items():
        m = lre.search(name)
        if m:
            layer_bytes[int(m.group(1) or m.group(2))] += size
        elif name.startswith("mtp.") or ".mtp." in name:
            other["mtp"] += size
        elif "visual" in name or "vision" in name:
            other["vision"] += size
        elif "embed_tokens" in name or "lm_head" in name or "language_model.norm" in name:
            other["embed_output"] += size
        else:
            other["unclassified"] += size

    types = tc["layer_types"]
    kv_heads, head_dim = tc["num_key_value_heads"], tc["head_dim"]
    # Recurrent state per DeltaNet layer: FP32 [value_heads, key_dim, value_dim] plus conv window.
    vh, kd, vd = tc["linear_num_value_heads"], tc["linear_key_head_dim"], tc["linear_value_head_dim"]
    conv_dim = 2 * tc["linear_num_key_heads"] * kd + vh * vd
    state_per_layer = vh * kd * vd * 4 + conv_dim * (tc["linear_conv_kernel_dim"] - 1) * 4
    # Real 2-D weight shapes [rows=out, cols=in] for the first layer of each kind, plus the tied
    # embedding / output head. Kernel tests take their dimensions from here.
    first = {"attention": types.index("full_attention"), "deltanet": types.index("linear_attention")}
    matrices = []
    for name, (shape, dtype) in sorted(shapes.items()):
        m = lre.search(name)
        if m:
            idx = int(m.group(1) or m.group(2))
            kind = next((k for k, i in first.items() if i == idx), None)
            if kind and len(shape) == 2:
                matrices.append({"name": f"{kind}.{name[m.end():]}", "rows": shape[0], "cols": shape[1], "dtype": dtype})
        elif len(shape) == 2 and ("embed_tokens" in name or "lm_head" in name) and "visual" not in name:
            matrices.append({"name": "output_head" if "lm_head" in name else "embed_tokens",
                             "rows": shape[0], "cols": shape[1], "dtype": dtype})
    manifest = {
        "name": a.repo.split("/")[-1],
        "source": {"repo": a.repo, "revision": sha},
        "family": "qwen35",
        "storage_dtype": tc.get("dtype", "bfloat16"),
        "native_context": tc["max_position_embeddings"],
        "layers": [
            {"kind": "attention" if t == "full_attention" else "deltanet", "stored_bytes": b}
            for t, b in zip(types, layer_bytes)
        ],
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "recurrent_state_bytes": state_per_layer * types.count("linear_attention"),
        "embed_output_bytes": other["embed_output"],
        "vision_bytes": other["vision"],
        "host_resident_bytes": 0,
        "mtp": {"config_layers": tc.get("mtp_num_hidden_layers", 0), "stored_bytes": other["mtp"]},
        "unclassified_bytes": other["unclassified"],
        "mrope_section": tc["rope_parameters"].get("mrope_section"),
        "image_token_id": cfg.get("image_token_id"),
        "matrices": matrices,
    }
    assert len(manifest["layers"]) == n and all(l["stored_bytes"] > 0 for l in manifest["layers"]), \
        "layer tensor name pattern did not match; inspect tensor names"
    with open(a.out, "w") as f:
        json.dump(manifest, f, indent=2)
    total = sum(l["stored_bytes"] for l in manifest["layers"]) + other["embed_output"] + other["vision"]
    print(f"{manifest['name']} @ {sha[:10]}: {n} layers, "
          f"kv_layers={types.count('full_attention')}, stored ~{total/2**30:.2f} GiB, "
          f"vision {other['vision']/2**20:.0f} MiB, mtp {other['mtp']/2**20:.0f} MiB, "
          f"unclassified {other['unclassified']} B")








if __name__ == "__main__":
    sys.exit(main())
