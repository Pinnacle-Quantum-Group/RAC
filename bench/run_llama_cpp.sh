#!/usr/bin/env bash
# run_llama_cpp.sh — invoke llama-bench against TinyLlama GGUF, emit
# numbers in the same machine-readable format as run_tinygrad.py and
# the RAC bench. Reads configs/llama_cpp.yaml for parameters.
#
# Self-bootstrapping: if llama-bench isn't on $PATH, prints clear
# install instructions. With --auto-build it clones + builds
# llama.cpp into /tmp/llama.cpp (requires cmake + a C++ compiler).
#
# Output is emitted on stdout as a single JSON line so
# bench_harness.sh can collate.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${HERE}/configs/llama_cpp.yaml"
AUTO_BUILD=0

for arg in "$@"; do
  case "$arg" in
    --auto-build) AUTO_BUILD=1 ;;
    --config=*)   CONFIG="${arg#*=}" ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//'
      exit 0 ;;
  esac
done

# Tiny YAML reader: extract "key: value" (no lists, no nesting).
yaml_get() {
  local key="$1" file="$2" default="${3:-}"
  local v
  v=$(grep -E "^${key}:[[:space:]]" "$file" 2>/dev/null \
        | sed -E "s/^${key}:[[:space:]]*//; s/[[:space:]]*#.*$//; s/^\"//; s/\"\$//") \
      || true
  if [[ -z "$v" ]]; then v="$default"; fi
  printf '%s\n' "$v"
}

BIN=$(yaml_get binary "$CONFIG" "")
GGUF_REPO=$(yaml_get gguf_repo "$CONFIG" "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
GGUF_FILE=$(yaml_get gguf_file "$CONFIG" "tinyllama-1.1b-chat-v1.0.Q8_0.gguf")
N_LAYERS=$(yaml_get n_layers_divisor "$CONFIG" "22")
PREFILL=$(yaml_get prefill_tokens "$CONFIG" "128")
DECODE=$(yaml_get decode_tokens "$CONFIG" "128")
THREADS=$(yaml_get threads "$CONFIG" "16")
REPEATS=$(yaml_get repeats "$CONFIG" "3")

# ── Locate llama-bench ───────────────────────────────────────────────────
if [[ -z "$BIN" ]]; then
  for cand in llama-bench \
              /tmp/llama.cpp/build/bin/llama-bench \
              "$HOME/llama.cpp/build/bin/llama-bench" \
              /usr/local/bin/llama-bench \
              /opt/homebrew/bin/llama-bench; do
    if command -v "$cand" >/dev/null 2>&1; then BIN="$cand"; break; fi
    if [[ -x "$cand" ]]; then BIN="$cand"; break; fi
  done
fi

if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  if [[ "$AUTO_BUILD" -eq 1 ]]; then
    echo "  llama-bench not found; auto-building into /tmp/llama.cpp ..." >&2
    if [[ ! -d /tmp/llama.cpp ]]; then
      git clone --depth 1 https://github.com/ggerganov/llama.cpp /tmp/llama.cpp >&2
    fi
    (cd /tmp/llama.cpp && cmake -S . -B build -DBUILD_SHARED_LIBS=OFF >&2 \
       && cmake --build build --target llama-bench -j >&2)
    BIN=/tmp/llama.cpp/build/bin/llama-bench
    [[ -x "$BIN" ]] || { echo "build failed" >&2; exit 2; }
  else
    cat >&2 <<EOF
  [ERROR] llama-bench not found.

  Install options (one-shot):

    # Ubuntu / Debian
    apt install -y cmake build-essential git && \\
    git clone --depth 1 https://github.com/ggerganov/llama.cpp /tmp/llama.cpp && \\
    cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build && \\
    cmake --build /tmp/llama.cpp/build --target llama-bench -j

    # Or rerun this script with --auto-build to do the above automatically.

  Once built, llama-bench lives at /tmp/llama.cpp/build/bin/llama-bench
  and this script will find it automatically.
EOF
    exit 3
  fi
fi

# ── Fetch the GGUF (idempotent — fetch_model.py caches) ──────────────────
echo "  fetching GGUF: ${GGUF_REPO}/${GGUF_FILE} (≈1.1 GB first time, cached after)..." >&2
GGUF_PATH=$(python3 "${HERE}/fetch_model.py" \
              --model "${GGUF_REPO}" --file "${GGUF_FILE}")
if [[ ! -s "$GGUF_PATH" ]]; then
  echo "  [ERROR] fetched path empty: $GGUF_PATH" >&2; exit 4
fi
GGUF_SZ=$(stat -c%s "$GGUF_PATH" 2>/dev/null || stat -f%z "$GGUF_PATH" 2>/dev/null || echo 0)
printf "  GGUF ready (%d MB) at %s\n" \
       "$((GGUF_SZ / 1024 / 1024))" "$GGUF_PATH" >&2

# ── Run llama-bench ──────────────────────────────────────────────────────
# llama-bench emits a CSV-like table on stdout + model-loading progress
# on stderr. We capture stdout into a tempfile and let stderr flow to
# the user's terminal so they see what's happening.
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

echo "  running llama-bench  [p=$PREFILL n=$DECODE t=$THREADS r=$REPEATS] ..." >&2
if ! "$BIN" -m "$GGUF_PATH" \
            -p "$PREFILL" -n "$DECODE" \
            -t "$THREADS" -r "$REPEATS" \
            -o csv > "$TMP"; then
  echo "  [ERROR] llama-bench exited non-zero" >&2
  exit 5
fi

# Parse the CSV: columns of interest are "t/s" (tokens/sec) for pp (prefill)
# and tg (decode). llama-bench 2024+ emits these under "test,t/s" headers.
# We defensively parse any CSV with these labels.
parse_metric() {
  local test_name="$1"
  python3 -c "
import csv, sys
best = None
for row in csv.DictReader(open('$TMP')):
    name = row.get('test','')
    tok  = row.get('t/s') or row.get('tokens_per_second')
    if not tok: continue
    if '$test_name' in name:
        try:
            v = float(tok)
            if best is None or v > best: best = v
        except: pass
print(best if best is not None else 0.0)
"
}

PP_TPS=$(parse_metric "pp${PREFILL}")
TG_TPS=$(parse_metric "tg${DECODE}")

# Per-layer derived values
if [[ "$N_LAYERS" -gt 0 ]]; then
  PP_TPS_LAYER=$(awk "BEGIN{printf \"%.2f\", $PP_TPS * $N_LAYERS}")
  TG_TPS_LAYER=$(awk "BEGIN{printf \"%.2f\", $TG_TPS * $N_LAYERS}")
  TG_MS_LAYER=$(awk "BEGIN{printf \"%.4f\", 1000.0 / ($TG_TPS * $N_LAYERS)}")
else
  PP_TPS_LAYER="$PP_TPS"; TG_TPS_LAYER="$TG_TPS"
  TG_MS_LAYER=$(awk "BEGIN{printf \"%.4f\", 1000.0 / $TG_TPS}")
fi

cat <<JSON
{"framework": "llama.cpp",
 "model": "$GGUF_REPO/$GGUF_FILE",
 "quant": "Q8_0",
 "threads": $THREADS,
 "prefill_T": $PREFILL,
 "decode_N": $DECODE,
 "prefill_tok_s_model": $PP_TPS,
 "decode_tok_s_model":  $TG_TPS,
 "prefill_tok_s_per_layer": $PP_TPS_LAYER,
 "decode_tok_s_per_layer":  $TG_TPS_LAYER,
 "decode_ms_per_layer":     $TG_MS_LAYER}
JSON
