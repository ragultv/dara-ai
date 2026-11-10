# Full-model validation of a Qwen3.5-family GGUF. Dara takes TEXT (its own tokenizer, from the GGUF) and runs the whole model;
# Reference B (real HF modules + the same GGUF weights dequantized by gguf-py) scores the very same token sequence.
# Usage: powershell -File tools/validate_gguf.ps1 -Tag 9b -Gguf models/qwen3.5-9b-gguf/Qwen3.5-9B-Q4_K_M.gguf -HfDir models/qwen3.5-9b -Hidden 4096 > build/validate_9b.txt
param([string]$Tag, [string]$Gguf, [string]$HfDir, [int]$Hidden, [int]$Layers = 32, [int]$Vocab = 248320)
$ErrorActionPreference = "Continue"
$gen = ".\build\cuda\qwen_generate.exe"
# prompts and generation lengths: (index into the fixed prompt set, tokens to generate)
$runs = @(@(0, 8), @(1, 12), @(2, 12), @(3, 12))
python -I -c "
import sys; sys.path.insert(0, 'tools'); import make_prompts, os
os.makedirs('build/prompts', exist_ok=True)
for k, t in enumerate(make_prompts.PROMPTS): open(f'build/prompts/t{k}.txt', 'w', encoding='utf-8', newline='').write(t)
"
foreach ($r in $runs) {
    $k = $r[0]; $n = $r[1]
    "`n======== $Tag prompt t$k ($n tokens generated)"
    & $gen --gguf $Gguf --prompt-file "build/prompts/t$k.txt" --new $n --out "build/${Tag}_t$k" 2>&1 | Select-String "prompt text|weights|prefill|restore|generated ids|memory|ERROR"
    $t0 = (Get-Content "build/${Tag}_t$k.tokens.txt").Split(" ").Count - $n
    $feed = $t0 + $n - 1
    python -I -u tools/ref_forward.py --model $HfDir --gguf $Gguf --tokens "build/${Tag}_t$k.tokens.txt" --feed $feed --layer-tokens $t0 --out "build/refB_${Tag}_t$k" 2>&1 | Select-Object -Last 1
    "##### Dara($Tag, GGUF) vs Reference B, prompt t$k (prompt $t0 tokens)"
    python -I -u tools/compare_logits.py --dara "build/${Tag}_t$k" --ref "build/refB_${Tag}_t$k" --prompt-len $t0 --vocab $Vocab --hidden $Hidden --layers $Layers 2>&1
}
"ALL DONE"
