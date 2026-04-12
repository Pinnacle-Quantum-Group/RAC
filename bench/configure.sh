#!/usr/bin/env bash
# configure.sh — auto-detect everything the three benchmarks need and
# fill in the YAML configs accordingly. Prints a status table showing
# what's ready and what still needs attention.
#
# No flags: run it, inspect the output, fix any ✗ rows the way the
# hints suggest. Safe to re-run — idempotent.
#
# Flags:
#   --quiet         don't print the status table (machine-readable JSON
#                   on stdout instead)
#   --update-yamls  rewrite the YAMLs with detected paths (DEFAULT: on)
#   --no-update     leave YAMLs untouched; just report status
#   --cache-dir DIR override HF cache location (default: ~/.cache/rac_bench)

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_YAML="${HERE}/configs/llama_cpp.yaml"
TINY_YAML="${HERE}/configs/tinygrad.yaml"
CACHE_DIR="${HF_HOME:-$HOME/.cache}/rac_bench"
[[ -n "${HF_HOME:-}" && "$HF_HOME" != "$HOME/.cache" ]] && CACHE_DIR="$HF_HOME/rac_bench"

UPDATE_YAMLS=1
QUIET=0
for arg in "$@"; do
  case "$arg" in
    --quiet)         QUIET=1 ;;
    --no-update)     UPDATE_YAMLS=0 ;;
    --update-yamls)  UPDATE_YAMLS=1 ;;
    --cache-dir=*)   CACHE_DIR="${arg#*=}" ;;
    -h|--help)
      awk '/^# /{sub(/^# ?/,""); print; next} {exit}' "$0"; exit 0 ;;
  esac
done

status=()
issues=()

row() {
    # row "<name>" "<glyph>" "<detail>"
    status+=("$(printf '  %-22s %s  %s' "$1" "$2" "$3")")
}
issue() { issues+=("$1"); }

# ── 1. RAC bench binary ────────────────────────────────────────────────
RAC_BIN=""
for cand in "${HERE}/../lib/build/bench_rac_transformer" \
            "${HERE}/../build/bench_rac_transformer" \
            "/tmp/rac_build/bench_rac_transformer" \
            "/tmp/rac_build_full/bench_rac_transformer"; do
  if [[ -x "$cand" ]]; then RAC_BIN="$(realpath "$cand")"; break; fi
done
if [[ -n "$RAC_BIN" ]]; then
  row "RAC bench binary"    "✓" "$RAC_BIN"
else
  row "RAC bench binary"    "✗" "not built"
  issue "build RAC bench:   cd lib/build && cmake .. && cmake --build . --target bench_rac_transformer"
fi

# ── 2. llama-bench ─────────────────────────────────────────────────────
LLAMA_BIN=""
for cand in \
    "$(command -v llama-bench 2>/dev/null || true)" \
    /tmp/llama.cpp/build/bin/llama-bench \
    "$HOME/llama.cpp/build/bin/llama-bench" \
    /usr/local/bin/llama-bench \
    /opt/homebrew/bin/llama-bench; do
  [[ -z "$cand" ]] && continue
  if [[ -x "$cand" ]]; then LLAMA_BIN="$(realpath "$cand" 2>/dev/null || echo "$cand")"; break; fi
done
if [[ -n "$LLAMA_BIN" ]]; then
  row "llama-bench"         "✓" "$LLAMA_BIN"
else
  row "llama-bench"         "✗" "not found"
  issue "install llama.cpp: ./bench/run_llama_cpp.sh --auto-build   (or apt install llama.cpp on Ubuntu 24.04+)"
fi

# ── 3. tinygrad ────────────────────────────────────────────────────────
TINYGRAD_STATUS=""
TINYGRAD_PY="$(command -v python3)"
# Check system python first
if python3 -c 'import tinygrad' 2>/dev/null; then
  TG_VER=$(python3 -c 'import tinygrad,sys; print(getattr(tinygrad,"__version__","?"))' 2>/dev/null)
  TINYGRAD_STATUS="system ($TG_VER)"
else
  # Check venv bootstrap location
  VENV_PY="${CACHE_DIR}/venv/bin/python3"
  if [[ -x "$VENV_PY" ]] && "$VENV_PY" -c 'import tinygrad' 2>/dev/null; then
    TG_VER=$("$VENV_PY" -c 'import tinygrad,sys; print(getattr(tinygrad,"__version__","?"))' 2>/dev/null)
    TINYGRAD_STATUS="venv ($TG_VER)"
    TINYGRAD_PY="$VENV_PY"
  fi
fi
if [[ -n "$TINYGRAD_STATUS" ]]; then
  row "tinygrad"            "✓" "$TINYGRAD_STATUS"
else
  row "tinygrad"            "✗" "not importable"
  issue "install tinygrad:  python3 bench/run_tinygrad.py --auto-install   (auto-venv on PEP 668)"
fi

# ── 4. huggingface_hub ─────────────────────────────────────────────────
HF_STATUS=""
if python3 -c 'import huggingface_hub' 2>/dev/null; then
  HF_STATUS="system"
else
  VENV_PY="${CACHE_DIR}/venv/bin/python3"
  if [[ -x "$VENV_PY" ]] && "$VENV_PY" -c 'import huggingface_hub' 2>/dev/null; then
    HF_STATUS="venv"
  fi
fi
if [[ -n "$HF_STATUS" ]]; then
  row "huggingface_hub"     "✓" "$HF_STATUS  (fast+resumable downloads)"
else
  row "huggingface_hub"     "⚠" "missing — urllib fallback (slower, no resume)"
  issue "install hf hub:    HF_BOOTSTRAP_VENV=1 python3 bench/fetch_model.py   (one-shot venv bootstrap)"
fi

# ── 5. TinyLlama safetensors cache ─────────────────────────────────────
SAFET="${CACHE_DIR}/TinyLlama--TinyLlama-1.1B-Chat-v1.0/main/model.safetensors"
SAFET_SIZE=0
if [[ -f "$SAFET" ]]; then
  SAFET_SIZE=$(stat -c%s "$SAFET" 2>/dev/null || stat -f%z "$SAFET" 2>/dev/null || echo 0)
fi
if [[ "$SAFET_SIZE" -gt 2000000000 ]]; then
  row "TinyLlama weights"   "✓" "$(printf '%.1f GB' "$(awk "BEGIN{print $SAFET_SIZE/1024/1024/1024}")") cached"
elif [[ "$SAFET_SIZE" -gt 0 ]]; then
  row "TinyLlama weights"   "⚠" "partial ($(printf '%.1f MB' "$(awk "BEGIN{print $SAFET_SIZE/1024/1024}")"))"
  issue "resume download:   python3 bench/fetch_model.py   (idempotent; will resume)"
else
  row "TinyLlama weights"   "✗" "not cached"
  issue "download weights:  python3 bench/fetch_model.py"
fi

# ── 6. TinyLlama GGUF (Q8_0) for llama.cpp ─────────────────────────────
GGUF="${CACHE_DIR}/TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
GGUF_SIZE=0
if [[ -f "$GGUF" ]]; then
  GGUF_SIZE=$(stat -c%s "$GGUF" 2>/dev/null || stat -f%z "$GGUF" 2>/dev/null || echo 0)
fi
if [[ "$GGUF_SIZE" -gt 500000000 ]]; then
  row "TinyLlama GGUF Q8_0" "✓" "$(printf '%.1f GB' "$(awk "BEGIN{print $GGUF_SIZE/1024/1024/1024}")") cached"
elif [[ "$GGUF_SIZE" -gt 0 ]]; then
  row "TinyLlama GGUF Q8_0" "⚠" "partial"
else
  row "TinyLlama GGUF Q8_0" "○" "not cached (fetched lazily by run_llama_cpp.sh)"
fi

# ── 7. Threads / hardware ──────────────────────────────────────────────
PHYS_CORES=$(grep -c "^processor" /proc/cpuinfo 2>/dev/null \
               || sysctl -n hw.physicalcpu 2>/dev/null \
               || nproc 2>/dev/null \
               || echo 16)
# Prefer physical over logical — matches the RAC bench thread-scaling finding
LOG_CORES=$(nproc 2>/dev/null || echo "$PHYS_CORES")
row "Threads detected"    "✓" "${PHYS_CORES} physical, ${LOG_CORES} logical (recommended: physical)"

# ── 8. Update YAMLs ────────────────────────────────────────────────────
yaml_set() {
  # yaml_set "key" "value" "file"
  local k="$1" v="$2" f="$3"
  if grep -qE "^${k}:" "$f" 2>/dev/null; then
    # sed -i works on GNU; BSD needs '' after -i. Use a tempfile for portability.
    awk -v k="$k" -v v="$v" '
      BEGIN { done = 0 }
      $0 ~ "^"k":" && !done { print k": "v; done=1; next }
      { print }
    ' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  fi
}

if [[ "$UPDATE_YAMLS" -eq 1 ]]; then
  if [[ -n "$LLAMA_BIN" ]]; then
    yaml_set "binary" "\"$LLAMA_BIN\"" "$LLAMA_YAML"
  fi
  yaml_set "threads" "$PHYS_CORES" "$LLAMA_YAML"
  row "llama_cpp.yaml"      "✓" "binary=${LLAMA_BIN:-auto}, threads=$PHYS_CORES"
  row "tinygrad.yaml"       "✓" "ready (no paths needed)"
fi

# ── 9. Emit ────────────────────────────────────────────────────────────
if [[ "$QUIET" -eq 1 ]]; then
  # JSON — for consumption by harness scripts
  printf '{"rac_bin":"%s","llama_bin":"%s","tinygrad":"%s","hf_hub":"%s","safet_bytes":%s,"gguf_bytes":%s,"threads":%s}\n' \
         "$RAC_BIN" "$LLAMA_BIN" "$TINYGRAD_STATUS" "$HF_STATUS" \
         "$SAFET_SIZE" "$GGUF_SIZE" "$PHYS_CORES"
else
  printf '\n  RAC bench configuration — Pinnacle Quantum Group\n'
  printf '  ────────────────────────────────────────────────\n'
  for line in "${status[@]}"; do echo "$line"; done
  printf '\n'
  if [[ ${#issues[@]} -gt 0 ]]; then
    printf '  To complete setup:\n'
    for hint in "${issues[@]}"; do printf '    • %s\n' "$hint"; done
    printf '\n'
    printf '  Or just run the whole pipeline and it will self-install:\n'
    printf '    ./bench/bench_harness.sh --auto-install --auto-build\n\n'
  else
    printf '  ✓ All components ready. Run the comparison:\n'
    printf '    ./bench/bench_harness.sh\n\n'
  fi
fi

# Exit 0 even if things are missing — this is a diagnostic, not a gate.
exit 0
