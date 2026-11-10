"""Does the Nano tier (Gemma 4 E2B) exist on Hugging Face, and does the installed transformers have its reference code?"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import gguf_probe as gp  # noqa: E402


def main():
    for repo in ["google/gemma-4-E2B-it", "google/gemma-4-E2B", "google/gemma-4-E4B-it"]:
        try:
            r = json.loads(gp.get(f"https://huggingface.co/api/models/{repo}", 0, 1 << 22))
            sha = r["sha"]
            c = json.loads(gp.get(f"https://huggingface.co/{repo}/resolve/{sha}/config.json", 0, 1 << 20))
            t = c.get("text_config", c)
            print(f"{repo}: sha {sha[:10]} archs {c.get('architectures')} model_type {c.get('model_type')}")
            print("   text keys:", {k: t.get(k) for k in ["model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
                  "num_key_value_heads", "head_dim", "global_head_dim", "intermediate_size", "sliding_window", "vocab_size",
                  "hidden_size_per_layer_input", "vocab_size_per_layer_input", "num_kv_shared_layers", "final_logit_softcapping",
                  "max_position_embeddings", "tie_word_embeddings", "rms_norm_eps"] if k in t})
            lt = t.get("layer_types")
            if lt:
                print("   layer_types:", len(lt), "first 10:", lt[:10], "| full-attention at:", [i for i, x in enumerate(lt) if "full" in x or "global" in x][:8])
            print("   rope:", t.get("rope_parameters") or t.get("rope_scaling") or t.get("rope_theta"))
        except Exception as e:  # noqa: BLE001
            print(f"{repo}: not accessible ({type(e).__name__}: {str(e)[:100]})")
    sys.modules["torchao"] = None
    import transformers
    d = os.path.join(os.path.dirname(transformers.__file__), "models")
    print("transformers", transformers.__version__, "gemma-related model dirs:", sorted(x for x in os.listdir(d) if "gemma" in x))


if __name__ == "__main__":
    main()
