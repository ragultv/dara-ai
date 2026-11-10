# Full-model validation of Qwen3.5-4B. Dara supports GGUF only, so every Dara run uses the quantized model.
#   Dara (GGUF) vs Reference B (real HF modules, the SAME quantized weights dequantized by gguf-py) -> Dara's runtime error
#   Dara (GGUF) vs Reference A (real HF modules, the original BF16 checkpoint)                    -> quantization error
# Reference A/B are Python measuring tools (they need the HF config/tokenizer files and, for A, the BF16 checkpoint); Dara
# itself needs only the .gguf file.
# Usage: powershell -File tools/validate_4b.ps1 > build/validate_4b.txt
$ErrorActionPreference = "Continue"
$gen = ".\build\cuda\qwen_generate.exe"
$hf = "models/qwen3.5-4b"                                   # config.json + tokenizer + BF16 shards (reference tools only)
$gguf = "models/qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf"
function Cmp($dara, $ref, $t0, $title) {
    "`n##### $title"
    python -I -u tools/compare_logits.py --dara $dara --ref $ref --prompt-len $t0 --vocab 248320 --hidden 2560 --layers 32 2>&1
}
function Run-Gguf($k, $t0, $n) {
    "`n======== GGUF prompt p$k (prompt $t0 tokens, $n generated)"
    & $gen --gguf $gguf --tokens "build/prompts/p$k.tokens" --new $n --out "build/g_p$k" 2>&1 | Select-String "weights|prefill|restore|generated|memory"
    $feed = $t0 + $n - 1
    python -I -u tools/ref_forward.py --model $hf --gguf $gguf --tokens "build/g_p$k.tokens.txt" --feed $feed --layer-tokens $t0 --out "build/refB_p$k" 2>&1 | Select-Object -Last 1
    Cmp "build/g_p$k" "build/refB_p$k" $t0 "Dara(GGUF) vs Reference B, p$k"
}
Run-Gguf 0 5 8
Run-Gguf 1 6 12
Run-Gguf 2 11 12
Run-Gguf 3 25 12
# quantization error: the GGUF run's own tokens, scored by the BF16 reference
"`n======== quantization error: Dara(GGUF) tokens through Reference A (original BF16 weights)"
python -I -u tools/ref_forward.py --model $hf --tokens "build/g_p0.tokens.txt" --feed 12 --layer-tokens 5 --out "build/refA_on_g_p0" 2>&1 | Select-Object -Last 1
Cmp "build/g_p0" "build/refA_on_g_p0" 5 "Dara(GGUF) vs Reference A (differences here are QUANTIZATION error)"
"ALL DONE"
