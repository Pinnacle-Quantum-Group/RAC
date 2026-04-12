## RAC transformer inference — three-way bench

Side-by-side single-layer transformer inference: **RAC** vs **llama.cpp**
vs **tinygrad**, same shape, same weights, same hardware, same scenarios.

### What's here

| file | purpose |
|---|---|
| `bench_rac_transformer.c` | RAC decoder-layer bench (synthetic OR real HF weights) |
| `safetensors_reader.{h,c}` | minimal F32/F16/BF16 safetensors loader |
| `test_safetensors.c` | 18 BVTs for the reader, synthesizes a tiny file in /tmp |
| `fetch_model.py` | HF downloader + cache manager; prints resolved path |
| `configs/llama_cpp.yaml` | llama-bench comparison config (fill in) |
| `configs/tinygrad.yaml` | tinygrad comparison config (fill in) |

### Quick-start: synthetic weights (no network)

```bash
cd build/
cmake --build . --target bench_rac_transformer
./bench_rac_transformer                         # tiny shape (d=512, 8 heads)
./bench_rac_transformer --config tinyllama      # full TinyLlama shape, random init
```

### Real-weights run (HF download + auto-convert dtype)

```bash
# 1. Fetch the safetensors + config to ~/.cache/rac_bench/
MODEL_DIR=$(python3 bench/fetch_model.py \
                --model TinyLlama/TinyLlama-1.1B-Chat-v1.0)

# 2. Feed one layer's weights (default: layer 0) into the RAC bench.
#    --safetensors implies --config tinyllama (d=2048, 32 heads, 4 kv-heads)
./bench_rac_transformer --safetensors "$MODEL_DIR/model.safetensors"
./bench_rac_transformer --safetensors "$MODEL_DIR/model.safetensors" --layer 5
```

`fetch_model.py` uses `huggingface_hub` if installed (fast, resumable,
parallel), else falls back to `urllib` (dep-free, slower). Cache dir
is `~/.cache/rac_bench/` or `$HF_HOME/rac_bench/` if set. Gated repos:
`HF_TOKEN=...` or `--token ...`.

**PEP 668 / Debian 3.12+:** if `pip install huggingface_hub` fails
with `externally-managed-environment`, set `HF_BOOTSTRAP_VENV=1` and
re-run. `fetch_model.py` will auto-create a venv under
`~/.cache/rac_bench/venv` and re-exec itself under that interpreter.
`bench_harness.sh --auto-install` sets this flag automatically.

**Per-file fetch** (e.g. GGUF for llama.cpp):

```bash
GGUF=$(python3 bench/fetch_model.py \
          --model TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
          --file tinyllama-1.1b-chat-v1.0.Q8_0.gguf)
echo "$GGUF"   # absolute cached path
```

### Model shapes

| preset | d_model | heads | kv_heads (GQA) | d_head | d_ff |
|---|---|---|---|---|---|
| `--config tiny` | 512 | 8 | 8 | 64 | 1536 |
| `--config tinyllama` | 2048 | 32 | **4** | 64 | 5632 |

TinyLlama uses 4 KV heads (GQA): 32 query heads share 4 key/value heads.
`bench_rac_transformer.c` handles the repeat mapping inline.

### Scenarios

| scenario | T | bottleneck | measures |
|---|---|---|---|
| **prefill** | 128 (override `--prefill N`) | compute-bound GEMM | peak GFLOPS |
| **decode** | 1 | memory-bound GEMV | token latency |

### Running all three frameworks (one command)

```bash
# On your box, from the repo root:
./bench/bench_harness.sh --auto-install --auto-build --layer 0
```

That command:
1. Fetches TinyLlama weights into `~/.cache/rac_bench/` (via `fetch_model.py`)
2. Runs the RAC bench with real layer-0 weights
3. `--auto-build` clones + builds `llama.cpp` into `/tmp/llama.cpp` if missing,
   then runs `llama-bench` on the Q8_0 GGUF
4. `--auto-install` pip-installs `tinygrad` if missing, then runs a minimal
   decoder-layer forward pass with real weights
5. Prints individual JSON lines per framework, followed by a Markdown
   side-by-side table with ratios vs RAC

**Flags:** `--skip-rac`, `--skip-llama`, `--skip-tinygrad`, `--layer N`,
`--bin-dir DIR` (override auto-detection of `bench_rac_transformer`).

### Running each framework individually

**RAC only** — see the "Real-weights run" section above.

**llama.cpp only:**
```bash
# Prerequisite: llama-bench on $PATH, or pass --auto-build to clone/build into /tmp.
./bench/run_llama_cpp.sh                 # uses configs/llama_cpp.yaml
./bench/run_llama_cpp.sh --auto-build    # builds llama.cpp first if missing
```
Output: one JSON line on stdout with prefill/decode tok/s (model-total
and per-layer via the `n_layers_divisor`).

**tinygrad only:**
```bash
python3 bench/run_tinygrad.py                  # uses configs/tinygrad.yaml
python3 bench/run_tinygrad.py --auto-install   # pip install tinygrad if missing
python3 bench/run_tinygrad.py --layer 5        # different decoder layer
```
Output: one JSON line with prefill/decode ms/layer and tok/s/layer.
Loads real TinyLlama layer weights via tinygrad's `safe_load`.

### Fair-comparison notes

- **Tensor identity.** With `--safetensors` pointing at the same file all
  three frameworks consume, the only variable is the execution engine.
- **Precision.** RAC is FP32 internally (BF16 / F16 HF weights are
  upcast on load). For apples-to-apples use F32 in llama.cpp
  (`--type f32` / `-t f32` as supported) and `dtype: float32` in
  tinygrad. Report Q8_0 / F16 separately as informational tiers.
- **Threading.** Pin each to the same physical core count. RAC: OpenMP,
  set `OMP_NUM_THREADS`. llama.cpp: `-t N`. tinygrad: `OMP_NUM_THREADS`
  plus `TINYGRAD_CLANG_PARALLEL=1`.
- **Warmup.** Each framework runs one untimed pass before timing.
- **Layer scaling.** For whole-model framework comparisons, divide by
  `n_layers_divisor` (22 for TinyLlama-1.1B) to compare per-layer.

### Rough expected numbers on Ryzen 9 5950X (FP32)

| framework | prefill (tok/s/layer) | decode (ms/layer) |
|---|---|---|
| RAC (CORDIC) | 400–600 | 0.3–0.8 |
| llama.cpp (F32, cBLAS) | 1500–3000 | 0.1–0.3 |
| tinygrad (CLANG/F32) | 200–800 | 0.5–2.0 |

RAC's competitive advantage isn't FP32-on-x86-silicon (where tensor-
oriented kernels beat CORDIC). The point of this bench is to establish
where RAC sits on commodity silicon so the FPGA/ASIC numbers have a
ground-truth baseline.
