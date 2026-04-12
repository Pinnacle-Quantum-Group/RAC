#!/usr/bin/env bash
# bench_harness.sh — run RAC, llama.cpp, tinygrad back-to-back against
# TinyLlama-1.1B-Chat-v1.0 and print a side-by-side table.
#
# Flags:
#   --auto-install     pip install tinygrad if missing
#   --auto-build       git clone + build llama.cpp if missing
#   --layer N          which TinyLlama decoder layer to load (default 0)
#   --skip-rac         skip RAC run
#   --skip-llama       skip llama.cpp run
#   --skip-tinygrad    skip tinygrad run
#   --bin-dir DIR      where bench_rac_transformer lives (default: auto-detect)
#
# Output goes to stdout as both (a) individual JSON lines per framework
# and (b) a final Markdown comparison table.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR=""
AUTO_INSTALL=0
AUTO_BUILD=0
LAYER=0
SKIP_RAC=0
SKIP_LLAMA=0
SKIP_TINY=0

for arg in "$@"; do
  case "$arg" in
    --auto-install) AUTO_INSTALL=1 ;;
    --auto-build)   AUTO_BUILD=1 ;;
    --layer=*)      LAYER="${arg#*=}" ;;
    --skip-rac)     SKIP_RAC=1 ;;
    --skip-llama)   SKIP_LLAMA=1 ;;
    --skip-tinygrad) SKIP_TINY=1 ;;
    --bin-dir=*)    BIN_DIR="${arg#*=}" ;;
    -h|--help)
      # Print only the leading comment block (contiguous lines starting '#')
      awk 'NR==1 || /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
      exit 0 ;;
  esac
done

# Auto-detect the RAC bench binary (built via cmake in lib/build)
if [[ -z "$BIN_DIR" ]]; then
  for cand in "${HERE}/../lib/build" "${HERE}/../build" "/tmp/rac_build"; do
    if [[ -x "$cand/bench_rac_transformer" ]]; then
      BIN_DIR="$cand"
      break
    fi
  done
fi
RAC_BIN="${BIN_DIR}/bench_rac_transformer"

# ── Fetch weights once (both RAC and tinygrad pull the same files) ─────
MODEL_ID="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
echo "  fetching $MODEL_ID (idempotent, cached)..." >&2
MODEL_DIR=$(python3 "${HERE}/fetch_model.py" --model "$MODEL_ID") || {
  echo "fetch failed" >&2; exit 2
}
SAFET="$MODEL_DIR/model.safetensors"

RAC_JSON=""
LLAMA_JSON=""
TINY_JSON=""

run_rac() {
  [[ "$SKIP_RAC" -eq 1 ]] && return 0
  if [[ ! -x "$RAC_BIN" ]]; then
    echo "  [WARN] RAC bench binary not found (set --bin-dir=DIR or build first)" >&2
    return 1
  fi
  # The C bench prints a human-readable block; parse the three lines we need.
  OUT=$("$RAC_BIN" --safetensors "$SAFET" --layer "$LAYER" \
                   --prefill-iters 20 --decode-iters 100 2>/dev/null)
  echo "$OUT" >&2
  local pre_ms pre_tps pre_gf dec_ms dec_tps dec_gf
  pre_ms=$(echo "$OUT" | awk '/prefill T=/{print $(NF-5)}' | head -1)
  pre_tps=$(echo "$OUT" | awk '/prefill T=/{print $(NF-3)}' | head -1)
  pre_gf=$(echo "$OUT"  | awk '/prefill T=/{print $(NF-1)}' | head -1)
  dec_ms=$(echo "$OUT"  | awk '/decode  T=1:/{print $(NF-5)}' | head -1)
  dec_tps=$(echo "$OUT" | awk '/decode  T=1:/{print $(NF-3)}' | head -1)
  dec_gf=$(echo "$OUT"  | awk '/decode  T=1:/{print $(NF-1)}' | head -1)
  RAC_JSON=$(cat <<JSON
{"framework": "RAC",
 "model": "$MODEL_ID",
 "layer": $LAYER,
 "prefill_ms_per_layer":    ${pre_ms:-0},
 "prefill_tok_s_per_layer": ${pre_tps:-0},
 "prefill_gflops_per_layer":${pre_gf:-0},
 "decode_ms_per_layer":     ${dec_ms:-0},
 "decode_tok_s_per_layer":  ${dec_tps:-0},
 "decode_gflops_per_layer": ${dec_gf:-0}}
JSON
)
}

run_llama() {
  [[ "$SKIP_LLAMA" -eq 1 ]] && return 0
  ARGS=()
  [[ "$AUTO_BUILD" -eq 1 ]] && ARGS+=(--auto-build)
  LLAMA_JSON=$(bash "${HERE}/run_llama_cpp.sh" "${ARGS[@]}" 2>/dev/null) || {
    echo "  [WARN] llama.cpp run failed; skipping" >&2
    bash "${HERE}/run_llama_cpp.sh" "${ARGS[@]}" >&2 || true
    return 1
  }
}

run_tiny() {
  [[ "$SKIP_TINY" -eq 1 ]] && return 0
  ARGS=(--layer "$LAYER")
  [[ "$AUTO_INSTALL" -eq 1 ]] && ARGS+=(--auto-install)
  TINY_JSON=$(python3 "${HERE}/run_tinygrad.py" "${ARGS[@]}" 2>/dev/null) || {
    echo "  [WARN] tinygrad run failed; skipping" >&2
    python3 "${HERE}/run_tinygrad.py" "${ARGS[@]}" >&2 || true
    return 1
  }
}

echo "  ── RAC ──────────────────────────────────" >&2; run_rac   || true
echo "  ── llama.cpp ────────────────────────────" >&2; run_llama || true
echo "  ── tinygrad ─────────────────────────────" >&2; run_tiny  || true

# ── Emit machine-readable lines first ──────────────────────────────────
echo ""
echo "# Individual framework results (JSON)"
[[ -n "$RAC_JSON"   ]] && echo "$RAC_JSON"   | tr -d '\n' | sed 's/}$/}\n/'
[[ -n "$LLAMA_JSON" ]] && echo "$LLAMA_JSON" | tr -d '\n' | sed 's/}$/}\n/'
[[ -n "$TINY_JSON"  ]] && echo "$TINY_JSON"  | tr -d '\n' | sed 's/}$/}\n/'

# ── Collate into a Markdown table ──────────────────────────────────────
python3 - "$RAC_JSON" "$LLAMA_JSON" "$TINY_JSON" <<'PY'
import json, sys

rows = []
for raw in sys.argv[1:]:
    if not raw.strip(): continue
    try: rows.append(json.loads(raw))
    except json.JSONDecodeError: continue

if not rows:
    print("\n(no results)")
    sys.exit(0)

def num(x, default="—"):
    try: return f"{float(x):.2f}"
    except: return default

print("\n## TinyLlama-1.1B-Chat-v1.0 — single-layer inference\n")
print("| framework | prefill tok/s/layer | prefill ms/layer | decode tok/s/layer | decode ms/layer |")
print("|---|---|---|---|---|")
for r in rows:
    print("| {fw} | {pp_t} | {pp_ms} | {d_t} | {d_ms} |".format(
        fw    = r.get("framework","?"),
        pp_t  = num(r.get("prefill_tok_s_per_layer")),
        pp_ms = num(r.get("prefill_ms_per_layer")),
        d_t   = num(r.get("decode_tok_s_per_layer")),
        d_ms  = num(r.get("decode_ms_per_layer")),
    ))

# Ratio row vs RAC (if present)
rac = next((r for r in rows if r.get("framework") == "RAC"), None)
if rac is not None and len(rows) > 1:
    print("\n**Ratios vs RAC (higher = framework is faster than RAC):**\n")
    print("| framework | prefill× | decode× |")
    print("|---|---|---|")
    for r in rows:
        if r.get("framework") == "RAC": continue
        try:
            pp = float(r.get("prefill_tok_s_per_layer", 0)) / float(rac.get("prefill_tok_s_per_layer", 1) or 1)
            dd = float(r.get("decode_tok_s_per_layer", 0))  / float(rac.get("decode_tok_s_per_layer", 1) or 1)
            print(f"| {r['framework']} | {pp:.2f}× | {dd:.2f}× |")
        except: pass
PY
