#!/usr/bin/env bash
# coverage.sh — code coverage harness for RAC (C + Rust + PyTorch)
# Pinnacle Quantum Group — April 2026
#
# Runs every test layer (BVT + DVT + E2E + perf where applicable) and
# emits coverage reports.
#
# Outputs:
#   build-cov/c/         gcov + lcov HTML for the C library
#   build-cov/rust/      llvm-cov HTML + lcov.info for the Rust crate
#   build-cov/python/    coverage.py HTML for rac_torch
#
# Requirements:
#   C:      gcc, lcov, genhtml
#   Rust:   cargo-llvm-cov (install with: cargo install cargo-llvm-cov)
#   Python: coverage (pip install coverage)
#
# Usage:
#   ./lib/coverage.sh           # all languages
#   ./lib/coverage.sh c         # only C
#   ./lib/coverage.sh rust
#   ./lib/coverage.sh python

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUT="$REPO/build-cov"
mkdir -p "$OUT"

TARGET="${1:-all}"

run_c() {
  echo "── C coverage ──────────────────────────────────────────────"
  local D="$OUT/c"
  rm -rf "$D" && mkdir -p "$D"
  pushd "$D" >/dev/null

  # Coverage-instrument rac_cpu.c (the TU we track coverage for).
  local CFLAGS="-O0 -g --coverage -fprofile-arcs -ftest-coverage -I${REPO}/lib/c -fopenmp"
  gcc $CFLAGS -c "${REPO}/lib/c/rac_cpu.c" -o rac_cpu.o

  # Also compile the HAL / AVX2 support objects (not instrumented — they
  # just provide linkage for BVT/DVT/E2E).
  local HAL_CFLAGS="-O2 -g -mavx2 -mfma -fopenmp -I${REPO}/lib/c"
  gcc $HAL_CFLAGS -c "${REPO}/lib/c/rac_avx2.c"        -o rac_avx2.o
  gcc $HAL_CFLAGS -c "${REPO}/lib/c/rac_hal.c"         -o rac_hal.o
  gcc $HAL_CFLAGS -c "${REPO}/lib/c/rac_zen3_kern.S"   -o rac_zen3_kern.o
  gcc $HAL_CFLAGS -c "${REPO}/lib/c/rac_avx512_kern.S" -o rac_avx512_kern.o

  local LINK_LIBS="rac_cpu.o rac_avx2.o rac_hal.o rac_zen3_kern.o rac_avx512_kern.o -lm"

  # Build BVT/DVT with coverage enabled on rac_cpu.o.
  for t in test_rac_lib_bvt test_rac_lib_dvt; do
    gcc $CFLAGS "${REPO}/lib/c/${t}.c" $LINK_LIBS -o "$t"
    ./"$t" > "${t}.log" 2>&1 || echo "  $t reported failures (see ${t}.log)"
  done

  # Coverage harness that targets the new transformer primitives.
  gcc $CFLAGS "${REPO}/lib/Testing/cov_transformer.c" rac_cpu.o -lm -o cov_transformer
  ./cov_transformer > cov_transformer.log

  # Generate line-coverage report.
  if command -v lcov >/dev/null 2>&1; then
    lcov --capture --directory . --output-file cov.info --no-external 2>/dev/null || true
    lcov --remove cov.info '/usr/*' --output-file cov.info 2>/dev/null || true
    if command -v genhtml >/dev/null 2>&1; then
      genhtml cov.info --output-directory html --quiet 2>/dev/null || true
      echo "  C coverage HTML:   $D/html/index.html"
    fi
  fi
  echo "  C coverage summary:"
  gcov -n -o . rac_cpu.o 2>&1 | grep -E "File.*rac_cpu\.c|Lines executed" | head -2 | sed 's/^/    /'
  popd >/dev/null
}

run_rust() {
  echo "── Rust coverage ───────────────────────────────────────────"
  local D="$OUT/rust"
  rm -rf "$D" && mkdir -p "$D"
  pushd "$REPO/lib/rust" >/dev/null
  if command -v cargo-llvm-cov >/dev/null 2>&1; then
    cargo llvm-cov --html --output-dir "$D" 2>&1 | tail -20
    cargo llvm-cov --lcov --output-path "$D/lcov.info" >/dev/null 2>&1 || true
    echo "  Rust coverage report: $D/html/index.html"
  else
    echo "  cargo-llvm-cov not installed; falling back to 'cargo test'"
    cargo test --lib 2>&1 | tail -10
  fi
  popd >/dev/null
}

run_python() {
  echo "── Python coverage ─────────────────────────────────────────"
  local D="$OUT/python"
  rm -rf "$D" && mkdir -p "$D"
  pushd "$REPO/rac_torch" >/dev/null
  if python3 -c "import coverage" 2>/dev/null; then
    python3 -m coverage erase
    for t in test_rac_bvt.py test_rac_dvt.py test_rac_e2e.py; do
      python3 -m coverage run -a --source=rac_torch "$t" || true
    done
    python3 -m coverage report --include='rac_torch.py'
    python3 -m coverage html -d "$D"
    echo "  Python coverage report: $D/index.html"
  else
    echo "  coverage module not installed; running plain tests"
    python3 test_rac_bvt.py || true
  fi
  popd >/dev/null
}

case "$TARGET" in
  all)    run_c; run_rust; run_python ;;
  c)      run_c ;;
  rust)   run_rust ;;
  python) run_python ;;
  *)      echo "Usage: $0 [all|c|rust|python]"; exit 2 ;;
esac

echo ""
echo "Coverage run complete — see $OUT/"
