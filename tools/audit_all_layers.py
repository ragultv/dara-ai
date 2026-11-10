"""Audit the GGUF-vs-checkpoint tensor conventions across ALL 32 layers (and the final norm).

Expected conventions (established on layers 0 and 3 by tools/audit_vhead_order.py):
  DeltaNet layers : V-head axis de-interleaved (GGUF j <- checkpoint 2j / 2(j-16)+1) for attn_qkv V rows, attn_gate,
                    ssm_alpha, ssm_beta, ssm_a, ssm_dt.bias, ssm_conv1d V channels, ssm_out columns;
                    attn_qkv q/k rows and conv q/k channels identity; ssm_a = -exp(A_log); gated norm identical.
  Attention layers: attn_q ([query|gate] per head), attn_k, attn_v, attn_output identity.
  All layers      : input/post-attention norms (and q_norm/k_norm) have +1 folded in.
Each check verifies the EXPECTED mapping on a sample of heads and requires that the identity (or another decoy)
does NOT fit as well, so a passing check is discriminating. Layers run in parallel.

Usage: python -I tools/audit_all_layers.py  -> build/audit_all_layers.json + a summary
"""
import concurrent.futures as cf
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import audit_vhead_order as av  # noqa: E402

L = av.L
KINDS = [l["kind"] for l in json.load(open("manifests/qwen3.5-4b.json"))["layers"]]


def hf_head(j, n=32):
    return av.hf_head_expected(j, n)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-30))


def gg_all(tensors, data_off, name):
    """Whole tensor dequantized to [rows, cols] in one request."""
    t = tensors[name]
    ty, cols = t["type"], t["dims"][0]
    rows = int(np.prod(t["dims"][1:])) if len(t["dims"]) > 1 else 1
    be, bb = av.gp.BLOCK[ty]
    nbytes = rows * cols // be * bb  # the parsed header carries no byte size
    if ty == "F32":
        raw = av.gp.get(av.GG_URL, data_off + t["offset"], data_off + t["offset"] + nbytes - 1)
        return np.frombuffer(raw, dtype="<f4").astype(np.float64).reshape(rows, cols)
    rb = cols // be * bb
    raw = av.gp.get(av.GG_URL, data_off + t["offset"], data_off + t["offset"] + nbytes - 1)
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(rows, rb).copy()
    return av.dequantize(blk, av.QT[ty]).astype(np.float64).reshape(rows, cols)


def row_check(tensors, off, hf, gname, hname, base, nheads, head_rows, samples, expected, within=3):
    """Sampled per-head row check. GGUF head j should equal checkpoint head expected(j), not identity/decoys."""
    ok, worst_good, best_bad = True, 1.0, -1.0
    for j in samples:
        rows_g = [base + head_rows * j + within]
        e = expected(j)
        rows_e = [base + head_rows * e + within]
        G = av.gg_rows(tensors, off, gname, rows_g)[0]
        He = hf.tensor_rows(hname, rows_e)[0][0]
        c_good = corr(G, He)
        worst_good = min(worst_good, c_good)
        decoy = (j + 1) % nheads if e == j else j
        Hd = hf.tensor_rows(hname, [base + head_rows * decoy + within])[0][0]
        best_bad = max(best_bad, corr(G, Hd)) if best_bad > -1 else corr(G, Hd)
        if c_good < 0.97:
            ok = False
    return {"ok": ok and best_bad < 0.9, "match_corr_min": round(worst_good, 3), "decoy_corr_max": round(best_bad, 3)}


def audit_layer(i, tensors, off, hf):
    res = {}
    gp, hp = f"blk.{i}.", f"{L}{i}."
    ident = lambda j: j
    # norms: GGUF = HF + 1
    for g, h in (("attn_norm.weight", "input_layernorm.weight"), ("post_attention_norm.weight", "post_attention_layernorm.weight")):
        a = gg_all(tensors, off, gp + g)[0]
        b = hf.tensor_all(hp + h)
        res[g] = {"ok": bool(np.abs(a - (b + 1)).max() < 1e-6)}
    if KINDS[i] == "deltanet":
        la = hp + "linear_attn."
        qkv = la + "in_proj_qkv.weight"
        res["qkv_q_identity"] = row_check(tensors, off, hf, gp + "attn_qkv.weight", qkv, 0, 16, 128, [0, 5, 15], ident)
        res["qkv_k_identity"] = row_check(tensors, off, hf, gp + "attn_qkv.weight", qkv, 2048, 16, 128, [0, 5, 15], ident)
        res["qkv_v_deinterleaved"] = row_check(tensors, off, hf, gp + "attn_qkv.weight", qkv, 4096, 32, 128, [0, 1, 2, 15, 16, 31], hf_head)
        res["attn_gate_deinterleaved"] = row_check(tensors, off, hf, gp + "attn_gate.weight", la + "in_proj_z.weight", 0, 32, 128, [0, 1, 2, 15, 16, 31], hf_head)
        for g, h in (("ssm_beta.weight", "in_proj_b.weight"), ("ssm_alpha.weight", "in_proj_a.weight")):
            G = gg_all(tensors, off, gp + g)
            H = hf.tensor_all(la + h).reshape(32, -1)
            res[g] = {"ok": all(corr(G[j], H[hf_head(j)]) > 0.99 for j in range(32))}
        A = hf.tensor_all(la + "A_log")
        a = gg_all(tensors, off, gp + "ssm_a")[0]
        res["ssm_a"] = {"ok": bool(all(abs(a[j] - (-np.exp(A[hf_head(j)]))) < 1e-5 * (1 + abs(a[j])) for j in range(32)))}
        dt, dth = gg_all(tensors, off, gp + "ssm_dt.bias")[0], hf.tensor_all(la + "dt_bias")
        res["ssm_dt"] = {"ok": bool(all(abs(dt[j] - dth[hf_head(j)]) < 1e-3 * (1 + abs(dt[j])) for j in range(32)))}  # HF stores BF16
        conv = gg_all(tensors, off, gp + "ssm_conv1d.weight").reshape(8192, 4)
        convh = hf.tensor_all(la + "conv1d.weight").reshape(8192, 4)
        okq = np.abs(conv[:4096] - convh[:4096]).max() < 1e-6
        okv = all(np.abs(conv[4096 + 128 * j:4096 + 128 * (j + 1)] - convh[4096 + 128 * hf_head(j):4096 + 128 * (hf_head(j) + 1)]).max() < 1e-6 for j in range(32))
        res["conv1d"] = {"ok": bool(okq and okv)}
        gn = gg_all(tensors, off, gp + "ssm_norm.weight")[0]
        res["ssm_norm_identical"] = {"ok": bool(np.abs(gn - hf.tensor_all(la + "norm.weight")).max() < 1e-6)}
        Go = av.gg_rows(tensors, off, gp + "ssm_out.weight", [0])[0]
        Ho = hf.tensor_rows(la + "out_proj.weight", [0])[0][0]
        res["ssm_out_cols_deinterleaved"] = {"ok": all(corr(Go[128 * j:128 * (j + 1)], Ho[128 * hf_head(j):128 * (hf_head(j) + 1)]) > 0.97 for j in range(32))}
    else:
        sa = hp + "self_attn."
        # attn_q: per head 512 rows = [query(256) | gate(256)]; check both halves, identity expected
        r = {"ok": True, "min_corr": 1.0}
        for h in (0, 7, 15):
            for part in (0, 256):
                row = 512 * h + part + 3
                G = av.gg_rows(tensors, off, gp + "attn_q.weight", [row])[0]
                H = hf.tensor_rows(sa + "q_proj.weight", [row])[0][0]
                c = corr(G, H)
                r["min_corr"] = round(min(r["min_corr"], c), 3)
                if c < 0.97:
                    r["ok"] = False
        res["attn_q_identity[query|gate]"] = r
        res["attn_k_identity"] = row_check(tensors, off, hf, gp + "attn_k.weight", sa + "k_proj.weight", 0, 4, 256, [0, 1, 2, 3], ident)
        res["attn_v_identity"] = row_check(tensors, off, hf, gp + "attn_v.weight", sa + "v_proj.weight", 0, 4, 256, [0, 1, 2, 3], ident)
        Go = av.gg_rows(tensors, off, gp + "attn_output.weight", [0])[0]
        Ho = hf.tensor_rows(sa + "o_proj.weight", [0])[0][0]
        res["attn_output_cols_identity"] = {"ok": all(corr(Go[256 * j:256 * (j + 1)], Ho[256 * j:256 * (j + 1)]) > 0.97 for j in range(16))}
        for g, h in (("attn_q_norm.weight", "q_norm.weight"), ("attn_k_norm.weight", "k_norm.weight")):
            a = gg_all(tensors, off, gp + g)[0]
            b = hf.tensor_all(sa + h)
            res[g] = {"ok": bool(np.abs(a - (b + 1)).max() < 1e-6)}
    return res


def main():
    hf = av.Hf()
    tensors, off = av.load_gguf_header()
    results = {}
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(audit_layer, i, tensors, off, hf): i for i in range(32)}
        for f in cf.as_completed(futs):
            i = futs[f]
            try:
                results[i] = f.result()
            except Exception as e:  # report, don't hide: a failed fetch is not a pass
                results[i] = {"error": repr(e)}
            bad = [k for k, v in results[i].items() if k == "error" or not v.get("ok", False)]
            print(f"layer {i:2d} ({KINDS[i]:9s}): {'OK' if not bad else 'FAIL ' + ', '.join(bad)}", flush=True)
    # final norm
    a = gg_all(tensors, off, "output_norm.weight")[0]
    b = hf.tensor_all("model.language_model.norm.weight")
    results["final_norm"] = {"ok": bool(np.abs(a - (b + 1)).max() < 1e-6)}
    json.dump({str(k): v for k, v in results.items()}, open("build/audit_all_layers.json", "w"), indent=1)
    bad_layers = [k for k, v in results.items() if k != "final_norm" and any(kk == "error" or not vv.get("ok", False) for kk, vv in v.items())]
    print(f"\nfinal norm (+1 folded): {'OK' if results['final_norm']['ok'] else 'FAIL'}")
    print(f"{32 - len(bad_layers)}/32 layers match the expected conventions; failing: {sorted(bad_layers)}")
    return 1 if bad_layers or not results["final_norm"]["ok"] else 0


if __name__ == "__main__":
    sys.exit(main())
