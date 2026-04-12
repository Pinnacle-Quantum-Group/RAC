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
PRECISION=$(yaml_get precision "$CONFIG" "f32")
# Precision can also be overridden by the env var RAC_LLAMA_QUANT.
PRECISION="${RAC_LLAMA_QUANT:-$PRECISION}"
GGUF_FILE_Q8_0=$(yaml_get gguf_file_q8_0 "$CONFIG" "tinyllama-1.1b-chat-v1.0.Q8_0.gguf")
GGUF_FILE_F32=$(yaml_get gguf_file_f32  "$CONFIG" "tinyllama-1.1b-chat-v1.0.F32.gguf")
GGUF_FILE=$(yaml_get gguf_file "$CONFIG" "tinyllama-1.1b-chat-v1.0.Q8_0.gguf")
# Pick filename by precision; fall back to generic gguf_file if both empty.
case "$PRECISION" in
  f32|F32|fp32)  GGUF_FILE="$GGUF_FILE_F32"; QUANT="F32" ;;
  q8_0|Q8_0)     GGUF_FILE="$GGUF_FILE_Q8_0"; QUANT="Q8_0" ;;
  *)             QUANT="$PRECISION" ;;  # pass-through for user-defined quants
esac
N_LAYERS=$(yaml_get n_layers_divisor "$CONFIG" "22")
PREFILL=$(yaml_get prefill_tokens "$CONFIG" "128")
DECODE=$(yaml_get decode_tokens "$CONFIG" "128")
THREADS_RAW=$(yaml_get threads "$CONFIG" "auto")
case "$THREADS_RAW" in
  auto|AUTO) THREADS="$(nproc 2>/dev/null || echo 16)" ;;
  *)         THREADS="$THREADS_RAW" ;;
esac
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

# ── Fetch / produce the GGUF ─────────────────────────────────────────────
# Q8_0 is a simple HuggingFace download from TheBloke.
# F32 is not always published as GGUF — in that case we convert the
# safetensors we already cached for the RAC bench using llama.cpp's
# convert_hf_to_gguf.py. One-time ~8 GB conversion, cached afterwards.
get_f32_gguf() {
  # Try the HF download first (fast path).
  local got
  got=$(python3 "${HERE}/fetch_model.py" \
          --model "${GGUF_REPO}" --file "${GGUF_FILE}" 2>/dev/null || true)
  if [[ -s "$got" ]]; then printf '%s' "$got"; return 0; fi

  # No pre-built F32 GGUF — convert from safetensors ourselves.
  local src_dir cache_out convert_py
  src_dir=$(python3 "${HERE}/fetch_model.py" \
              --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 2>&1 | tail -1)
  cache_out="$(dirname "$src_dir")/tinyllama-f32.gguf"
  if [[ -s "$cache_out" ]]; then printf '%s' "$cache_out"; return 0; fi

  # Find convert_hf_to_gguf.py inside the auto-built llama.cpp clone.
  for cand in /tmp/llama.cpp/convert_hf_to_gguf.py \
              "$HOME/llama.cpp/convert_hf_to_gguf.py" \
              /tmp/llama.cpp/convert-hf-to-gguf.py; do
    [[ -f "$cand" ]] && convert_py="$cand" && break
  done
  if [[ -z "${convert_py:-}" ]]; then
    echo "  [ERROR] F32 GGUF not on HF hub and convert_hf_to_gguf.py not found." >&2
    echo "         Run with --auto-build so llama.cpp is cloned, or set" >&2
    echo "         precision: q8_0 in configs/llama_cpp.yaml for a quick fallback." >&2
    return 1
  fi

  echo "  [convert] producing F32 GGUF from safetensors ($src_dir) ..." >&2
  echo "            (one-time, ~8 GB, cached at $cache_out)" >&2
  if ! python3 "$convert_py" "$src_dir" \
         --outtype f32 --outfile "$cache_out" >&2; then
    echo "  [ERROR] convert_hf_to_gguf.py failed" >&2
    return 1
  fi
  printf '%s' "$cache_out"
}

echo "  fetching GGUF: ${GGUF_REPO}/${GGUF_FILE} (precision=$PRECISION) ..." >&2
if [[ "$PRECISION" == "f32" || "$PRECISION" == "F32" || "$PRECISION" == "fp32" ]]; then
  GGUF_PATH=$(get_f32_gguf) || exit 4
else
  GGUF_PATH=$(python3 "${HERE}/fetch_model.py" \
                --model "${GGUF_REPO}" --file "${GGUF_FILE}")
fi
if [[ ! -s "$GGUF_PATH" ]]; then
  echo "  [ERROR] fetched path empty: $GGUF_PATH" >&2; exit 4
fi
GGUF_SZ=$(stat -c%s "$GGUF_PATH" 2>/dev/null || stat -f%z "$GGUF_PATH" 2>/dev/null || echo 0)
printf "  GGUF ready (%d MB, quant=%s) at %s\n" \
       "$((GGUF_SZ / 1024 / 1024))" "$QUANT" "$GGUF_PATH" >&2

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

# Parse the CSV. llama-bench schemas seen in the wild:
#   pre-2024:  columns include "test" + "t/s"
#   2024+:     columns "test" + "avg_ts" or "tokens_per_second"
#   recent:    separate "n_prompt" / "n_gen" columns, "test" may be empty
# We probe each option and fall back through them.
parse_metric() {
  local test_name="$1"
  local prefill="$2"
  local decode="$3"
  python3 - "$TMP" "$test_name" "$prefill" "$decode" <<'PY'
import csv, sys
path, test_name, prefill, decode = sys.argv[1:]
prefill, decode = int(prefill), int(decode)
best = None

def try_parse(row, keys):
    for k in keys:
        v = row.get(k)
        if v in (None, "", "—"): continue
        try: return float(v)
        except ValueError: continue
    return None

tps_keys = ("avg_ts", "t/s", "tokens_per_second", "avg_tps")

with open(path) as f:
    rows = list(csv.DictReader(f))

for row in rows:
    # Classic form: test="pp128" or "tg128"
    name = row.get("test", "") or ""
    if test_name and test_name in name:
        v = try_parse(row, tps_keys)
        if v is not None and (best is None or v > best): best = v
        continue
    # New form: n_prompt / n_gen columns.
    try:
        np_ = int(row.get("n_prompt") or 0)
        ng_ = int(row.get("n_gen") or 0)
    except ValueError:
        np_ = ng_ = 0
    if test_name.startswith("pp") and np_ == prefill and ng_ == 0:
        v = try_parse(row, tps_keys)
        if v is not None and (best is None or v > best): best = v
    elif test_name.startswith("tg") and ng_ == decode and np_ == 0:
        v = try_parse(row, tps_keys)
        if v is not None and (best is None or v > best): best = v

print(best if best is not None else 0.0)
PY
}

PP_TPS=$(parse_metric "pp${PREFILL}" "$PREFILL" "$DECODE")
TG_TPS=$(parse_metric "tg${DECODE}"  "$PREFILL" "$DECODE")

# Diagnostic: if both metrics came back zero the schema doesn't match.
# Print a snippet of the raw CSV so users can see what llama-bench gave us.
if awk "BEGIN{exit !($PP_TPS==0 && $TG_TPS==0)}"; then
  echo "  [WARN] llama-bench CSV parse returned 0 for both metrics." >&2
  echo "  [DEBUG] raw llama-bench output follows (first 20 lines):" >&2
  head -20 "$TMP" | sed 's/^/    /' >&2
  echo "  [HINT] if your llama-bench emits different columns, share the"      >&2
  echo "         output above so the parser can be extended." >&2
fi

# Derive a genuine per-layer latency from the measured full-model throughput.
# Assumption: layer cost distributes roughly evenly across the 22 layers (fine
# for transformers modulo the final norm + lm_head overhead). Everything else
# we report is the *measured* full-model throughput — no multiplication fiction.
PREFILL_MS_MODEL=$(awk "BEGIN{printf \"%.3f\", ($PP_TPS > 0) ? 1000.0 / $PP_TPS : 0}")
DECODE_MS_MODEL=$(awk "BEGIN{printf \"%.3f\", ($TG_TPS > 0) ? 1000.0 / $TG_TPS : 0}")
if [[ "$N_LAYERS" -gt 0 ]] && awk "BEGIN{exit !($PP_TPS>0 && $TG_TPS>0)}"; then
  PREFILL_MS_PER_LAYER=$(awk "BEGIN{printf \"%.4f\", 1000.0 / $PP_TPS / $N_LAYERS}")
  DECODE_MS_PER_LAYER=$(awk "BEGIN{printf \"%.4f\", 1000.0 / $TG_TPS / $N_LAYERS}")
else
  PREFILL_MS_PER_LAYER=0; DECODE_MS_PER_LAYER=0
fi

cat <<JSON
{"framework": "llama.cpp",
 "model": "$GGUF_REPO/$GGUF_FILE",
 "quant": "$QUANT",
 "threads": $THREADS,
 "prefill_T": $PREFILL,
 "decode_N": $DECODE,
 "n_layers": $N_LAYERS,
 "scope": "full_model",
 "prefill_tok_s_model":  $PP_TPS,
 "decode_tok_s_model":   $TG_TPS,
 "prefill_ms_per_token": $PREFILL_MS_MODEL,
 "decode_ms_per_token":  $DECODE_MS_MODEL,
 "prefill_ms_per_layer_derived": $PREFILL_MS_PER_LAYER,
 "decode_ms_per_layer_derived":  $DECODE_MS_PER_LAYER}
JSON
