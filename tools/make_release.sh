#!/usr/bin/env bash
# Builds a release candidate directory from the binaries currently in build/cuda, after re-running the regression checks.
# Usage: tools/make_release.sh <name> [--with-9b]      e.g. tools/make_release.sh rc2 --with-9b
# Never overwrites an existing release directory (each candidate stays reproducible). Any failed check aborts without creating it.
# Baselines are the immutable reference logits stored in release/rc1/results (validated against independent references).
set -u
NAME="${1:?usage: make_release.sh <name> [--with-9b]}"
WITH9B="${2:-}"
cd "$(dirname "$0")/.."
R="release/$NAME"
[ -e "$R" ] && { echo "refusing to overwrite $R"; exit 1; }
BASE=release/rc1/results
G4=models/qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf
G9=models/qwen3.5-9b-gguf/Qwen3.5-9B-Q4_K_M.gguf
GN=models/gemma4-e2b-gguf/gemma-4-E2B-it-Q4_K_M.gguf
TMP="build/release_check_$NAME"; mkdir -p "$TMP"
LOG="$TMP/summary.txt"; : > "$LOG"
fail() { echo "FAILED: $1" | tee -a "$LOG"; exit 1; }
ok() { echo "ok: $1" | tee -a "$LOG"; }

# 1. model logits against the frozen baselines
build/cuda/qwen_generate.exe --gguf $G4 --prompt "The capital of France is" --new 8 --out $TMP/q4 > $TMP/q4.txt 2>&1
cmp -s $BASE/res4b.logits.bin $TMP/q4.logits.bin && grep -q "restore, re-run 3 decode steps: logits bitwise identical" $TMP/q4.txt && ok "Qwen3.5-4B logits bitwise identical to rc1 baseline; checkpoint/restore bitwise identical" || fail "4B logits"
DARA_NO_GRAPH=1 build/cuda/qwen_generate.exe --gguf $G4 --prompt "The capital of France is" --new 8 --out $TMP/q4e > $TMP/q4e.txt 2>&1
cmp -s $BASE/res4b.logits.bin $TMP/q4e.logits.bin && ok "Qwen3.5-4B eager path (DARA_NO_GRAPH=1) bitwise identical to rc1 baseline" || fail "4B eager logits"
build/cuda/nano_generate.exe --gguf $GN --tokens build/nano_prompt.tokens.txt --new 12 --kv 1024 --no-layers --out $TMP/n > $TMP/n.txt 2>&1
cmp -s $BASE/nano_r-1.logits.bin $TMP/n.logits.bin && ok "Nano (Gemma 4 E2B) logits bitwise identical to rc1 baseline" || fail "Nano logits"
DARA_NO_GRAPH=1 build/cuda/nano_generate.exe --gguf $GN --tokens build/nano_prompt.tokens.txt --new 12 --kv 1024 --no-layers --out $TMP/ne > $TMP/ne.txt 2>&1
cmp -s $BASE/nano_r-1.logits.bin $TMP/ne.logits.bin && ok "Nano eager path (DARA_NO_GRAPH=1) bitwise identical to rc1 baseline" || fail "Nano eager logits"
if [ "$WITH9B" = "--with-9b" ]; then
  build/cuda/qwen_generate.exe --gguf $G9 --prompt-file build/prompts/t0.txt --new 8 --out $TMP/q9 > $TMP/q9.txt 2>&1
  cmp -s $BASE/9b_t0.logits.bin $TMP/q9.logits.bin && ok "Qwen3.5-9B (hybrid) logits bitwise identical to rc1 baseline" || fail "9B logits"
fi
# 2. graph / eager equivalence under relocation, restore, long decode, injected graph failures and allocation failures
build/cuda/nano_graph_test.exe $GN > $TMP/nano_graph.txt 2>&1 && tail -1 $TMP/nano_graph.txt | grep -q '^ok$' && ok "Nano graph tests: graphs == eager bitwise through re-capture, rewind, 560 replays; failure injection (capture/instantiate/launch/allocation) recovers cleanly" || fail "nano_graph_test"
build/cuda/qwen_graph_test.exe $G4 > $TMP/qwen_graph.txt 2>&1 && tail -1 $TMP/qwen_graph.txt | grep -q '^ok$' && ok "Qwen graph tests: graphs == eager bitwise through re-capture, DeltaNet-state restore, 560 replays; failure injection recovers cleanly" || fail "qwen_graph_test"
# 3. tokenizers and chat template vs Hugging Face
build/cuda/gemma_tokenizer_test.exe $GN build/golden/gemma_tokenizer_golden.json > $TMP/tok.txt 2>&1 && tail -1 $TMP/tok.txt | grep -q '^ok$' && ok "Gemma tokenizer identical to Hugging Face" || fail "gemma tokenizer"
build/cuda/gemma_chat_test.exe $GN build/golden/gemma_chat_golden.json > $TMP/chat.txt 2>&1 && tail -1 $TMP/chat.txt | grep -q '^ok$' && ok "Gemma chat template identical to Hugging Face" || fail "gemma chat template"
# 4. kernels and layers
build/cuda/gemv_multi_test.exe > $TMP/gemv_multi.txt 2>&1 && tail -1 $TMP/gemv_multi.txt | grep -q '^ok$' && ok "multi-token GEMV bitwise identical to single-vector kernels" || fail "gemv_multi_test"
for g in layer_4b_l0 layer_4b_l3 layer_9b_l0 layer_9b_l3; do
  build/cuda/layer_cuda_test.exe build/golden/$g.bin > $TMP/$g.txt 2>&1 && tail -1 $TMP/$g.txt | grep -q '^ok$' || fail "layer test $g"
done
ok "layer tests (4B and 9B, DeltaNet and attention layers)"

# 5. assemble the candidate
mkdir -p $R/bin $R/results
cp build/cuda/*.exe build/cuda/cudart64_13.dll $R/bin/
(cd $R/bin && sha256sum * > ../SHA256SUMS.txt)
cp $TMP/q4.logits.bin $R/results/res4b.logits.bin; cp $TMP/n.logits.bin $R/results/nano.logits.bin
[ "$WITH9B" = "--with-9b" ] && cp $TMP/q9.logits.bin $R/results/9b_t0.logits.bin
cp $TMP/*.txt $R/results/ 2>/dev/null
(cd $R/results && sha256sum *.bin > SHA256SUMS.txt)
SRC_HASH=$(find src tools tests docs CMakeLists.txt -type f 2>/dev/null | sort | xargs sha256sum | sha256sum | cut -d' ' -f1)
{
  echo "# Dara release candidate $NAME"
  echo
  echo "Created $(date -u +%Y-%m-%dT%H:%M:%SZ). Source tree is not under version control; identity = SHA-256 over every file in src, tools, tests, docs and CMakeLists.txt:"
  echo "source-tree sha256: $SRC_HASH"
  echo "GPU code targets: $(head -1 build/build_rc2.txt 2>/dev/null | sed 's/CUDA targets: //' | cut -c1-300)"
  echo "Toolchain: CUDA 13.1.1 (redist), MSVC 14.44, C++20."
  echo
  echo "## What was RUN"
  echo "GPU: $(grep -m1 '^GPU:' $TMP/n.txt)"
  echo "Only this architecture has been executed. Architectures compiled but NOT run on hardware: every other target listed above."
  echo
  echo "## Checks passed for this candidate (each compares against the immutable rc1 baselines or an independent oracle)"
  sed 's/^ok: /- /' "$LOG"
  echo
  echo "## Measured speed (not a controlled cross-runtime benchmark; this laptop GPU throttles, +-10%)"
  echo "CUDA-graph decode (default) and the eager fallback (DARA_NO_GRAPH=1) are BOTH supported execution paths; both are checked bitwise against the baseline above."
  echo "- Qwen 4B, graphs: $(grep -h 'warm decode' $TMP/q4.txt | sed 's/warm decode: //')"
  echo "- Qwen 4B, eager : $(grep -h 'warm decode' $TMP/q4e.txt | sed 's/warm decode: //')"
  echo "- Nano,     graphs: $(grep -h 'warm decode' $TMP/n.txt | sed 's/warm decode: //')"
  echo "- Nano,     eager : $(grep -h 'warm decode' $TMP/ne.txt | sed 's/warm decode: //')"
  [ "$WITH9B" = "--with-9b" ] && echo "- Qwen 9B (hybrid, streamed layers are never graph-captured): $(grep -h 'warm decode' $TMP/q9.txt | sed 's/warm decode: //')"
  echo "(Short runs: capture cost is included, so steady-state graph speed is somewhat higher than shown.)"
  echo
  echo "## What these checks do NOT establish"
  echo "- Failure recovery was tested with INJECTED faults (graph capture/instantiate/launch refusals and failed device allocations). That proves the recovery"
  echo "  code paths, not recovery from a GPU that is genuinely out of VRAM; real exhaustion on a nearly full card has not been exercised."
  echo "- Only the GPU named above was physically tested. Compilation for the other listed architectures is NOT hardware validation."
  echo "- Speed numbers are single-machine, short runs, with other programs possibly holding GPU memory; they are not comparisons with llama.cpp or any other runtime."
  echo
  echo "## Rule"
  echo "Later changes are measured against these results; a logits difference from the baseline is a blocker unless deliberately explained."
} > $R/RESULTS.md
echo "created $R"; cat $R/RESULTS.md
