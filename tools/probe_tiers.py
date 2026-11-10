"""Check what the four product tiers actually are on Hugging Face (existence, pinned revision, key config fields)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import gguf_probe as gp  # noqa: E402


def api(path):
    try:
        return json.loads(gp.get("https://huggingface.co/api/" + path, 0, 1 << 22))
    except Exception as e:  # noqa: BLE001
        return f"ERR {type(e).__name__}: {e}"


def main():
    for repo in ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B", "Qwen/Qwen3.5-27B"]:
        r = api(f"models/{repo}")
        print(f"{repo:22s} -> {r if isinstance(r, str) else r.get('sha')}")
        if isinstance(r, dict):
            sha = r["sha"]
            try:
                c = json.loads(gp.get(f"https://huggingface.co/{repo}/resolve/{sha}/config.json", 0, 1 << 20))
                t = c.get("text_config", c)
                keys = ["model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
                        "intermediate_size", "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
                        "linear_value_head_dim", "linear_conv_kernel_dim", "full_attention_interval", "vocab_size",
                        "max_position_embeddings", "tie_word_embeddings", "mtp_num_hidden_layers", "rms_norm_eps"]
                print("   ", {k: t.get(k) for k in keys if k in t})
                print("    rope:", t.get("rope_parameters"), "| attn_output_gate:", t.get("attn_output_gate"),
                      "| archs:", c.get("architectures"))
                lt = t.get("layer_types")
                if lt:
                    print("    layers:", len(lt), "pattern head:", lt[:8], "full-attn layers:", [i for i, x in enumerate(lt) if x == "full_attention"][:6], "...")
            except Exception as e:  # noqa: BLE001
                print("    config fetch failed:", e)
    for q in ["gemma-4", "gemma4", "gemma-3n"]:
        r = api(f"models?search={q}&limit=12")
        print(f"search '{q}':", [m["id"] for m in r] if isinstance(r, list) else r)


if __name__ == "__main__":
    main()
