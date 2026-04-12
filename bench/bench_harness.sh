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

# Run configure.sh up front: auto-detect tools + update YAMLs with the
# resolved llama-bench path / thread count. Silent JSON output.
if [[ -x "${HERE}/configure.sh" ]]; then
  "${HERE}/configure.sh" --quiet >/dev/null 2>&1 || true
fi

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

_build_rac_bench() {
  # Build bench_rac_transformer + librac_avx2 via the top-level CMake
  # project using explicit -S/-B so a stray in-source cmake cache
  # (eg. from someone running `cmake .` inside lib/) can't redirect
  # the generator. Uses lib/build as the out-of-tree build dir.
  # Idempotent.
  local src_dir build_dir
  src_dir="$(cd "${HERE}/../lib" && pwd)"
  build_dir="${src_dir}/build"

  # Clean up stale in-source cache if one exists. Without this, cmake
  # writes to src_dir instead of build_dir, and the subsequent
  # `cmake --build` step fails with "Error: could not load cache".
  if [[ -f "$src_dir/CMakeCache.txt" ]]; then
    echo "  [auto-build] removing stale in-source cmake cache at $src_dir/CMakeCache.txt" >&2
    rm -f "$src_dir/CMakeCache.txt"
    rm -rf "$src_dir/CMakeFiles"
  fi

  mkdir -p "$build_dir"
  echo "  [auto-build] configuring RAC bench (src=$src_dir build=$build_dir)" >&2

  if ! cmake -S "$src_dir" -B "$build_dir" \
         -DCMAKE_BUILD_TYPE=Release -DRAC_ENGINE=OFF \
         2>&1 | sed 's/^/    /' >&2
  then
    echo "  [ERR] cmake configure failed" >&2
    return 1
  fi

  if ! cmake --build "$build_dir" --target bench_rac_transformer \
         -j"$(nproc 2>/dev/null || echo 4)" \
         2>&1 | sed 's/^/    /' >&2
  then
    echo "  [ERR] build failed" >&2
    return 1
  fi

  if [[ -x "$build_dir/bench_rac_transformer" ]]; then
    BIN_DIR="$build_dir"
    RAC_BIN="$build_dir/bench_rac_transformer"
    echo "  [auto-build] bench_rac_transformer ready at $RAC_BIN" >&2
    return 0
  fi
  echo "  [ERR] expected binary not produced" >&2
  return 1
}

# If the bench binary is missing and --auto-build was requested, build it.
if [[ ! -x "$RAC_BIN" && "$AUTO_BUILD" -eq 1 ]]; then
  _build_rac_bench || true
fi

# ── PEP 668-safe Python deps ───────────────────────────────────────────
# Debian / Ubuntu 23.04+ ship externally-managed Python. `pip install` on
# the system interpreter errors out. When --auto-install is set we
# provision a private venv under ~/.cache/rac_bench/venv and front-load
# its bin on PATH so everything downstream (fetch_model.py, tinygrad,
# huggingface_hub) uses it automatically.
BENCH_VENV="${HOME}/.cache/rac_bench/venv"

_ensure_venv() {
  if [[ -d "$BENCH_VENV" && -x "$BENCH_VENV/bin/python3" ]]; then
    return 0
  fi
  echo "  [auto-install] creating bench venv at $BENCH_VENV" >&2
  python3 -m venv "$BENCH_VENV" 2>&1 | sed 's/^/    /' >&2 || {
    echo "  [WARN] python3 -m venv failed; is python3-venv installed? (apt install python3-venv)" >&2
    return 1
  }
  "$BENCH_VENV/bin/python3" -m pip install --quiet --upgrade pip setuptools wheel \
    2>&1 | sed 's/^/    /' >&2 || true
}

_bench_pip_install() {
  # $1..: package names. Tries system pip first; on PEP 668 failure falls
  # back to the bench venv. Idempotent — skips if import succeeds in the
  # venv Python (not system Python, because the venv is what the bench
  # ends up using via PATH override below).
  local pkg check_py
  # Prefer venv python for the module check so we know the venv has it
  # (what matters downstream). Fall back to system python3 if no venv yet.
  check_py=$([[ -x "$BENCH_VENV/bin/python3" ]] && echo "$BENCH_VENV/bin/python3" || echo python3)
  for pkg in "$@"; do
    local mod="$pkg"
    case "$pkg" in
      huggingface_hub|huggingface-hub) mod=huggingface_hub ;;
      *) mod="${pkg//-/_}" ;;
    esac
    if "$check_py" -c "import $mod" 2>/dev/null; then
      echo "  [auto-install] $pkg already present in $check_py" >&2
      continue
    fi
    # System attempt (will fail silently on PEP 668)
    if pip install --quiet "$pkg" 2>/dev/null; then
      echo "  [auto-install] pip installed $pkg (system)" >&2
      check_py=python3
      continue
    fi
    # Venv fallback. Show pip output so heavy installs (torch ~800 MB)
    # report progress + any failure. Previously --quiet hid torch's
    # compile/download errors completely.
    _ensure_venv || return 1
    echo "  [auto-install] installing $pkg into $BENCH_VENV ..." >&2
    if "$BENCH_VENV/bin/pip" install "$pkg" 2>&1 | sed 's/^/    /' >&2; then
      if "$BENCH_VENV/bin/python3" -c "import $mod" 2>/dev/null; then
        echo "  [auto-install] $pkg installed and importable" >&2
        check_py="$BENCH_VENV/bin/python3"
      else
        echo "  [WARN] $pkg reported install OK but 'import $mod' still fails in the venv" >&2
      fi
    else
      echo "  [WARN] failed to install $pkg — continuing without it" >&2
    fi
  done
}

if [[ "$AUTO_INSTALL" -eq 1 ]]; then
  _bench_pip_install huggingface_hub
  # For the F32-GGUF conversion path in run_llama_cpp.sh we need the
  # full convert_hf_to_gguf.py dependency set: transformers (for
  # AutoConfig), safetensors, sentencepiece, numpy, torch. torch is
  # heavy but convert_hf_to_gguf imports it unconditionally for
  # tensor ops during conversion. On the bench venv this is a one-
  # time pip cost.
  _bench_pip_install transformers safetensors sentencepiece numpy torch
  # If the venv exists, front-load it so subsequent python3 calls pick it up.
  if [[ -x "$BENCH_VENV/bin/python3" ]]; then
    export PATH="$BENCH_VENV/bin:$PATH"
    export VIRTUAL_ENV="$BENCH_VENV"
  fi
fi

# ── Fetch weights once (both RAC and tinygrad pull the same files) ─────
# If --auto-install was requested, let fetch_model.py + run_tinygrad.py
# bootstrap a venv under $HF_HOME/rac_bench/venv (bypasses PEP 668 on
# Debian/Ubuntu 3.12+ system Python).
if [[ "$AUTO_INSTALL" -eq 1 ]]; then
  export HF_BOOTSTRAP_VENV=1
fi
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
    echo "  [WARN] RAC bench binary not found. Re-run with --auto-build, or:" >&2
    echo "         cd lib && mkdir -p build && cd build &&" >&2
    echo "         cmake .. -DCMAKE_BUILD_TYPE=Release -DRAC_ENGINE=OFF &&" >&2
    echo "         cmake --build . --target bench_rac_transformer -j\$(nproc)" >&2
    return 1
  fi
  if command -v ldd >/dev/null 2>&1; then
    if ! ldd "$RAC_BIN" 2>/dev/null | grep -q libgomp; then
      echo "  [WARN] $RAC_BIN is not linked against libgomp — OpenMP pragmas are no-ops." >&2
    fi
  fi
  # Default OMP_NUM_THREADS to PHYSICAL core count, not logical. This
  # bench's decode path is a pure GEMV (memory-bandwidth bound); SMT
  # siblings on the same physical core share L1/L2 and just thrash each
  # other when both are running DRAM-streaming workloads. Using physical
  # cores gave ~2x decode speedup on the 5950X (16 phys / 32 logical).
  #
  # Prefill is compute-bound and would benefit from all 32 logical cores,
  # but the difference is smaller than the decode loss. One setting that
  # works reasonably for both is "physical". User can override via
  # OMP_NUM_THREADS=X in the environment.
  if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
    local _phys
    _phys=$(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket:/ {c=$2} /^Socket\(s\):/ {s=$2} END {if (c && s) print c*s}')
    if [[ -n "$_phys" && "$_phys" -gt 0 ]]; then
      OMP_NUM_THREADS="$_phys"
    else
      OMP_NUM_THREADS=$(nproc 2>/dev/null || echo 1)
    fi
  fi
  export OMP_NUM_THREADS
  local quant_flag="" quant_name="F32"
  if [[ "${RAC_BENCH_QUANT:-}" == "q8_0" || "${RAC_BENCH_QUANT:-}" == "Q8_0" ]]; then
    quant_flag="--q8_0"
    quant_name="Q8_0"
  fi
  echo "  [info] OMP_NUM_THREADS=$OMP_NUM_THREADS  mode=full-model  quant=$quant_name" >&2

  # Run the FULL-MODEL bench (all 22 layers + KV cache). This gives numbers
  # directly comparable to llama-bench's full-model tok/s — no n_layers
  # multiplication anywhere. --layer is ignored in full-model mode.
  OUT=$("$RAC_BIN" --safetensors "$SAFET" --full-model $quant_flag \
                   --prefill-iters 5 --decode-iters 50 2>/dev/null)
  echo "$OUT" >&2

  # Parse the full-model output format:
  #   prefill T=128:   X.XX ms/token   Y.YY tok/s   Z.Z GFLOPS
  #   decode  T=1:     X.XX ms/token   Y.YY tok/s   Z.Z GFLOPS
  local pre_ms pre_tps pre_gf dec_ms dec_tps dec_gf
  pre_ms=$(echo "$OUT"  | awk '/prefill T=/{print $(NF-5)}' | head -1)
  pre_tps=$(echo "$OUT" | awk '/prefill T=/{print $(NF-3)}' | head -1)
  pre_gf=$(echo "$OUT"  | awk '/prefill T=/{print $(NF-1)}' | head -1)
  dec_ms=$(echo "$OUT"  | awk '/decode  T=1:/{print $(NF-5)}' | head -1)
  dec_tps=$(echo "$OUT" | awk '/decode  T=1:/{print $(NF-3)}' | head -1)
  dec_gf=$(echo "$OUT"  | awk '/decode  T=1:/{print $(NF-1)}' | head -1)
  # "quant" here describes the DECODE path. Prefill is always F32 — Q8_0
  # only substitutes for the linear-layer GEMV during decode, where the
  # memory bandwidth advantage matters most.
  local quant_label="F32"
  [[ "$quant_flag" == "--q8_0" ]] && quant_label="F32 (prefill) / Q8_0 (decode)"
  RAC_JSON=$(cat <<JSON
{"framework": "RAC",
 "model": "$MODEL_ID",
 "quant": "$quant_label",
 "scope": "full_model",
 "n_layers": 22,
 "threads": $OMP_NUM_THREADS,
 "prefill_ms_per_token": ${pre_ms:-0},
 "prefill_tok_s_model":  ${pre_tps:-0},
 "prefill_gflops":       ${pre_gf:-0},
 "decode_ms_per_token":  ${dec_ms:-0},
 "decode_tok_s_model":   ${dec_tps:-0},
 "decode_gflops":        ${dec_gf:-0}}
JSON
)
}

run_llama() {
  [[ "$SKIP_LLAMA" -eq 1 ]] && return 0
  ARGS=()
  [[ "$AUTO_BUILD" -eq 1 ]] && ARGS+=(--auto-build)
  # Capture stdout (JSON) into a tempfile; let stderr flow to the user
  # so they see GGUF download progress + llama-bench model-load output.
  local out; out=$(mktemp)
  if bash "${HERE}/run_llama_cpp.sh" "${ARGS[@]}" > "$out"; then
    LLAMA_JSON=$(cat "$out")
  else
    echo "  [WARN] llama.cpp run failed; see stderr above" >&2
    LLAMA_JSON=""
  fi
  rm -f "$out"
}

run_tiny() {
  [[ "$SKIP_TINY" -eq 1 ]] && return 0
  ARGS=(--layer "$LAYER")
  [[ "$AUTO_INSTALL" -eq 1 ]] && ARGS+=(--auto-install)
  local out; out=$(mktemp)
  if python3 "${HERE}/run_tinygrad.py" "${ARGS[@]}" > "$out"; then
    TINY_JSON=$(cat "$out")
  else
    echo "  [WARN] tinygrad run failed; see stderr above" >&2
    TINY_JSON=""
  fi
  rm -f "$out"
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

def pp_tps(r):
    """Full-model prefill tok/s, preferring measured over derived."""
    return (r.get("prefill_tok_s_model")
            or r.get("prefill_tok_s_per_layer")  # old RAC schema
            or 0)

def tg_tps(r):
    return (r.get("decode_tok_s_model")
            or r.get("decode_tok_s_per_layer")
            or 0)

print("\n## TinyLlama-1.1B-Chat-v1.0 — full-model inference\n")
print("| framework | quant | threads | prefill tok/s | prefill ms/tok | decode tok/s | decode ms/tok |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    pp_t = pp_tps(r)
    tg_t = tg_tps(r)
    pp_ms = (1000.0 / pp_t) if pp_t else 0
    tg_ms = (1000.0 / tg_t) if tg_t else 0
    print("| {fw} | {q} | {t} | {pp_t} | {pp_ms} | {d_t} | {d_ms} |".format(
        fw    = r.get("framework","?"),
        q     = r.get("quant", "?"),
        t     = r.get("threads", "—"),
        pp_t  = num(pp_t),
        pp_ms = num(pp_ms),
        d_t   = num(tg_t),
        d_ms  = num(tg_ms),
    ))

# Ratio row vs RAC (if present)
rac = next((r for r in rows if r.get("framework") == "RAC"), None)
if rac is not None and len(rows) > 1:
    rac_pp = float(pp_tps(rac) or 1)
    rac_tg = float(tg_tps(rac) or 1)
    print("\n**Ratios vs RAC (higher = framework is faster than RAC at the same workload):**\n")
    print("| framework | prefill× | decode× | quant |")
    print("|---|---|---|---|")
    for r in rows:
        if r.get("framework") == "RAC": continue
        try:
            pp = float(pp_tps(r) or 0) / rac_pp
            dd = float(tg_tps(r) or 0) / rac_tg
            print(f"| {r['framework']} | {pp:.2f}× | {dd:.2f}× | {r.get('quant','?')} |")
        except Exception: pass

# Footnote: make the workload boundaries explicit so nobody re-invents the
# previous × n_layers lie.
print("\n_**Notes on comparability:**_")
print("- All numbers are **full-model** throughput (one token end-to-end through "
      "the whole 22-layer TinyLlama stack), not per-layer estimates.")
print("- RAC runs F32. llama.cpp's default Q8_0 uses 4x less memory bandwidth — "
      "for a true apples-to-apples compare, feed llama-bench an F32 GGUF and "
      "match OMP_NUM_THREADS.")
print("- Decode numbers assume a warm KV cache (RAC reseeds the cache with a "
      "T=128 prefill before timing decode; llama-bench does the same).")
PY
