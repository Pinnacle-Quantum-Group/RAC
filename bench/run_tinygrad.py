#!/usr/bin/env python3
"""
run_tinygrad.py — single-layer TinyLlama inference bench using tinygrad.

Designed to emit numbers in the same JSON-line format as
run_llama_cpp.sh so bench_harness.sh can collate a side-by-side table.

Self-bootstrapping:
  * If `tinygrad` isn't importable it prints `pip install tinygrad` and
    exits (or auto-installs with --auto-install).
  * Weights fetched via bench/fetch_model.py into ~/.cache/rac_bench/.
  * Builds a minimal decoder-layer forward pass in tinygrad; loads the
    selected layer's weights from the safetensors file (BF16 auto-cast).

Usage:
  python3 bench/run_tinygrad.py                    # defaults from yaml
  python3 bench/run_tinygrad.py --auto-install     # pip install if missing
  python3 bench/run_tinygrad.py --layer 5 --prefill 128 --decode-iters 100
"""

from __future__ import annotations
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time


HERE = pathlib.Path(__file__).resolve().parent
CONFIG = HERE / "configs" / "tinygrad.yaml"


def yaml_get(key, text, default=None):
    """Minimal key: value reader for the tinygrad.yaml schema."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(f"{key}:"):
            v = s[len(key) + 1 :].strip().split("#", 1)[0].strip()
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            return v
    return default


def load_config(path):
    text = path.read_text() if path.exists() else ""
    return {
        "hf_model_id":      yaml_get("hf_model_id", text, "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        "n_layers_divisor": int(yaml_get("n_layers_divisor", text, "22")),
        "prefill_tokens":   int(yaml_get("prefill_tokens", text, "128")),
        "decode_iters":     int(yaml_get("decode_iters", text, "100")),
        "device":           yaml_get("device", text, "CLANG"),
        "dtype":            yaml_get("dtype", text, "float32"),
    }


def ensure_tinygrad(auto_install: bool) -> None:
    try:
        import tinygrad  # noqa: F401
        return
    except ImportError:
        pass
    if auto_install:
        print("  installing tinygrad via pip...", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tinygrad"])
        return
    print(
        "  [ERROR] tinygrad not importable.\n"
        "    pip install tinygrad\n"
        "  or rerun this script with --auto-install.",
        file=sys.stderr,
    )
    sys.exit(2)


def _available_backends():
    """List the ops_* backends the installed tinygrad actually ships.
    Older tinygrad had ops_clang; newer versions renamed it to ops_cpu.
    Needed for graceful fallback when the yaml config says CLANG."""
    try:
        import pkgutil, tinygrad.runtime
        return {m.name.replace("ops_", "").upper()
                for m in pkgutil.iter_modules(tinygrad.runtime.__path__)
                if m.name.startswith("ops_")}
    except Exception:
        return set()


def _pick_device(requested: str) -> str:
    """
    Map the YAML-requested device to one the installed tinygrad actually has.

      - 'CLANG' is the old CPU JIT; on tinygrad 0.11+ it was renamed to 'CPU'.
      - 'HIP' / 'AMD' on a Navi 33 box is preferable to CPU if available.
      - Fall back to 'CPU' or 'PYTHON' as last resorts.
    """
    avail = _available_backends()
    if not avail:
        return requested.upper()

    requested_up = requested.upper()
    # Honor explicit request when possible.
    if requested_up in avail:
        return requested_up

    # CLANG → CPU rename on 0.11+.
    if requested_up == "CLANG" and "CPU" in avail:
        print("  [info] tinygrad renamed CLANG -> CPU; using CPU backend", file=sys.stderr)
        return "CPU"

    # Prefer GPU over CPU for larger models if we can detect one.
    for preferred in ("HIP", "AMD", "CUDA", "NV", "METAL", "CPU", "CLANG", "PYTHON"):
        if preferred in avail:
            print(f"  [info] requested '{requested}' unavailable; using '{preferred}' (avail: {sorted(avail)})",
                  file=sys.stderr)
            return preferred

    return requested_up  # let tinygrad raise its own error


def fetch_weights(model_id: str) -> pathlib.Path:
    """Use fetch_model.py to pull the safetensors file; return its path."""
    out = subprocess.check_output(
        [sys.executable, str(HERE / "fetch_model.py"),
         "--model", model_id,
         "--file", "model.safetensors"],
        text=True,
    ).strip()
    p = pathlib.Path(out)
    if not p.exists():
        raise FileNotFoundError(f"fetch_model.py resolved to missing path: {p}")
    return p


def build_layer(cfg, weights_path, layer_idx):
    """
    Build a tinygrad Tensor-level decoder layer reading real weights.
    Returns a callable `forward(x)` that runs the whole block.
    """
    from tinygrad import Tensor
    from tinygrad.nn.state import safe_load

    state = safe_load(str(weights_path))

    prefix = f"model.layers.{layer_idx}."
    names = {
        "q":      prefix + "self_attn.q_proj.weight",
        "k":      prefix + "self_attn.k_proj.weight",
        "v":      prefix + "self_attn.v_proj.weight",
        "o":      prefix + "self_attn.o_proj.weight",
        "gate":   prefix + "mlp.gate_proj.weight",
        "up":     prefix + "mlp.up_proj.weight",
        "down":   prefix + "mlp.down_proj.weight",
        "rmsatt": prefix + "input_layernorm.weight",
        "rmsffn": prefix + "post_attention_layernorm.weight",
    }
    missing = [k for k, n in names.items() if n not in state]
    if missing:
        raise RuntimeError(f"missing tensors: {missing}")

    # Dtype casting happens in tinygrad automatically on first op;
    # we'll keep everything in the configured precision.
    dtype_map = {"float32": "float", "float16": "half", "bfloat16": "bfloat16"}
    tg_dtype = dtype_map.get(cfg["dtype"], "float")

    W = {k: state[n].cast(tg_dtype) for k, n in names.items()}

    # TinyLlama-1.1B dims
    d       = 2048
    n_heads = 32
    n_kv    = 4
    d_head  = 64
    d_ff    = 5632

    def rmsnorm(x, gamma, eps=1e-5):
        var = (x * x).mean(axis=-1, keepdim=True)
        return x * (var + eps).rsqrt() * gamma

    def forward(x):
        # ── Attention
        h = rmsnorm(x, W["rmsatt"])
        q = h @ W["q"].T
        k = h @ W["k"].T
        v = h @ W["v"].T

        T = x.shape[0]
        q = q.reshape(T, n_heads, d_head).permute(1, 0, 2)            # [nq, T, dh]
        k = k.reshape(T, n_kv,    d_head).permute(1, 0, 2)            # [nkv, T, dh]
        v = v.reshape(T, n_kv,    d_head).permute(1, 0, 2)

        # GQA: repeat k/v groups
        rep = n_heads // n_kv
        k = k.repeat_interleave(rep, dim=0)  # [nq, T, dh]
        v = v.repeat_interleave(rep, dim=0)

        scores = (q @ k.permute(0, 2, 1)) * (1.0 / (d_head ** 0.5))   # [nq, T, T]

        # Causal mask
        if T > 1:
            mask = Tensor.ones(T, T).tril() - Tensor.ones(T, T)
            scores = scores + mask * 1e9    # below-diagonal is 0, above = -inf-ish

        attn = scores.softmax(axis=-1)
        out  = attn @ v                                               # [nq, T, dh]
        out  = out.permute(1, 0, 2).reshape(T, d)                     # [T, d]
        out  = out @ W["o"].T
        x    = x + out

        # ── FFN (SwiGLU)
        h = rmsnorm(x, W["rmsffn"])
        gate = (h @ W["gate"].T).silu()
        up   = h @ W["up"].T
        ffn  = (gate * up) @ W["down"].T
        x    = x + ffn
        return x

    return forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--prefill", type=int, default=None)
    ap.add_argument("--prefill-iters", type=int, default=10)
    ap.add_argument("--decode-iters", type=int, default=None)
    ap.add_argument("--auto-install", action="store_true")
    args = ap.parse_args()

    cfg = load_config(pathlib.Path(args.config))
    if args.prefill is not None:      cfg["prefill_tokens"] = args.prefill
    if args.decode_iters is not None: cfg["decode_iters"]   = args.decode_iters

    # Apply env knobs from config
    # (Minimal: OMP_NUM_THREADS + device)
    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 16))
    # TINYGRAD_CLANG_PARALLEL was the old flag for the CLANG backend;
    # the equivalent on 0.11+ is CPU_PARALLEL. Set both, harmless.
    os.environ.setdefault("TINYGRAD_CLANG_PARALLEL", "1")
    os.environ.setdefault("CPU_PARALLEL", "1")

    ensure_tinygrad(args.auto_install)

    # Only after tinygrad is importable can we pick a backend that
    # actually exists in this install.
    dev = _pick_device(cfg["device"])
    os.environ["DEV"] = dev
    cfg["device"] = dev

    from tinygrad import Tensor

    weights = fetch_weights(cfg["hf_model_id"])
    layer = build_layer(cfg, weights, args.layer)

    T = cfg["prefill_tokens"]
    d_model = 2048

    # Build inputs
    x_prefill = Tensor.rand(T, d_model).cast("float" if cfg["dtype"] == "float32" else "half")
    x_decode  = Tensor.rand(1, d_model).cast("float" if cfg["dtype"] == "float32" else "half")

    # Warmup
    _ = layer(x_prefill).realize()
    _ = layer(x_decode).realize()

    # Prefill
    t0 = time.monotonic()
    for _ in range(args.prefill_iters):
        _ = layer(x_prefill).realize()
    s_pre = time.monotonic() - t0
    tps_pre = (T * args.prefill_iters) / s_pre
    ms_pre  = s_pre * 1000.0 / args.prefill_iters

    # Decode
    t0 = time.monotonic()
    for _ in range(cfg["decode_iters"]):
        _ = layer(x_decode).realize()
    s_dec = time.monotonic() - t0
    tps_dec = cfg["decode_iters"] / s_dec
    ms_dec  = s_dec * 1000.0 / cfg["decode_iters"]

    # Layer-scaled equivalents (if we were running all 22 layers, throughput
    # would be ÷22, so "per_layer" here is already the measurement).
    result = {
        "framework":                "tinygrad",
        "model":                    cfg["hf_model_id"],
        "device":                   cfg["device"],
        "dtype":                    cfg["dtype"],
        "prefill_T":                T,
        "decode_N":                 cfg["decode_iters"],
        "prefill_ms_per_layer":     round(ms_pre, 3),
        "decode_ms_per_layer":      round(ms_dec, 3),
        "prefill_tok_s_per_layer":  round(tps_pre, 2),
        "decode_tok_s_per_layer":   round(tps_dec, 2),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
