#!/usr/bin/env python3
"""
bench/fetch_model.py — Hugging Face model fetcher for the RAC bench suite.

Downloads model files (safetensors, GGUF, config, tokenizer) into a local
cache keyed by (repo_id, revision, filename) and prints the resolved
path. Harness scripts source from it via command substitution, e.g.:

    MODEL_DIR=$(python3 bench/fetch_model.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0)
    GGUF_PATH=$(python3 bench/fetch_model.py --model TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \\
                                             --file tinyllama-1.1b-chat-v1.0.Q8_0.gguf)

Strategy:
  1. If `huggingface_hub` is installed, use snapshot_download / hf_hub_download.
     (pip install huggingface_hub)
  2. Otherwise fall back to plain urllib against the HF resolve endpoint.
     Works without any extra deps but is slower (no resume, no parallel).

Cache layout (under ~/.cache/rac_bench/ or $HF_HOME/rac_bench/):
    <repo_id_slug>/<revision>/<filename>

Env vars honoured:
    HF_HOME    — overrides cache root
    HF_TOKEN   — passed as Bearer token for gated repos
    HF_HUB_ENABLE_HF_TRANSFER — if set and hf_transfer installed, used for speed
"""

from __future__ import annotations
import argparse
import json
import os
import pathlib
import shutil
import sys
import urllib.request
import urllib.error


DEFAULT_MODEL    = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_REVISION = "main"

# Files we consider "essential" when the user asks for a whole-repo snapshot.
# Everything else is left for on-demand --file fetches.
DEFAULT_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",         # some repos ship this
    "model.safetensors",
    "pytorch_model.bin",       # some repos only have pt
]


def cache_root() -> pathlib.Path:
    """Resolve the cache root, honouring $HF_HOME for cohabitation."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return pathlib.Path(hf_home).expanduser() / "rac_bench"
    return pathlib.Path.home() / ".cache" / "rac_bench"


def slug(repo_id: str) -> str:
    """Filesystem-safe slug for a HF repo id (org/name → org--name)."""
    return repo_id.replace("/", "--")


def target_dir(repo_id: str, revision: str) -> pathlib.Path:
    d = cache_root() / slug(repo_id) / revision
    d.mkdir(parents=True, exist_ok=True)
    return d


def have_hub() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        return False


def maybe_bootstrap_venv() -> None:
    """If HF_BOOTSTRAP_VENV=1 and huggingface_hub is missing, build a
    venv under cache_root()/venv with pip + huggingface_hub and re-exec
    ourselves from that interpreter. Lets Debian/Ubuntu users opt into
    a non-PEP-668 pip install without polluting system Python."""
    if os.environ.get("HF_BOOTSTRAP_VENV") != "1":
        return
    if have_hub():
        return
    venv_dir = cache_root() / "venv"
    if not venv_dir.exists():
        import subprocess
        sys.stderr.write(f"  bootstrapping venv at {venv_dir} ...\n")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        subprocess.check_call([
            str(venv_dir / "bin" / "python3"), "-m", "pip", "install",
            "-q", "--upgrade", "pip", "huggingface_hub",
        ])
    os.execv(str(venv_dir / "bin" / "python3"),
             [str(venv_dir / "bin" / "python3"), __file__] + sys.argv[1:])


# ── huggingface_hub path ────────────────────────────────────────────────

def fetch_hub(repo_id: str, revision: str, filenames: list[str] | None,
              token: str | None) -> pathlib.Path:
    from huggingface_hub import hf_hub_download, snapshot_download
    dest = target_dir(repo_id, revision)

    if filenames is None:
        # snapshot_download pulls the whole revision; we then symlink into
        # our cache_root layout for consistency with the urllib path.
        snap = snapshot_download(repo_id=repo_id, revision=revision,
                                 token=token, local_dir=str(dest),
                                 local_dir_use_symlinks=False,
                                 allow_patterns=DEFAULT_FILES)
        return pathlib.Path(snap)

    # Per-file fetch: download each to the cache root and hardlink/copy
    # into our layout.
    for fn in filenames:
        target = dest / fn
        if target.exists():
            continue
        got = hf_hub_download(repo_id=repo_id, filename=fn,
                              revision=revision, token=token,
                              local_dir=str(dest),
                              local_dir_use_symlinks=False)
        got_p = pathlib.Path(got)
        if got_p.resolve() != target.resolve():
            # Older huggingface_hub versions put the file under a nested
            # snapshots/ tree; copy it up to the requested filename.
            shutil.copy2(got_p, target)
    return dest


# ── urllib fallback ─────────────────────────────────────────────────────

_NEGATIVE_CACHE_NAME = ".404_cache"

def _load_404_cache(dest_dir: pathlib.Path) -> set[str]:
    f = dest_dir / _NEGATIVE_CACHE_NAME
    if not f.exists(): return set()
    try:
        return set(line.strip() for line in f.read_text().splitlines() if line.strip())
    except Exception:
        return set()

def _mark_404(dest_dir: pathlib.Path, filename: str) -> None:
    f = dest_dir / _NEGATIVE_CACHE_NAME
    try:
        with open(f, "a") as fh:
            fh.write(filename + "\n")
    except Exception:
        pass


def _download_one(url: str, target: pathlib.Path, token: str | None) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    # Per-directory 404 cache: if we already asked HF for this filename
    # and got 404, don't ask again on every run.
    negative = _load_404_cache(target.parent)
    if target.name in negative:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    sys.stderr.write(f"  download {url} -> {target}\n")
    try:
        with urllib.request.urlopen(req) as resp:
            with open(target.with_suffix(target.suffix + ".part"), "wb") as f:
                shutil.copyfileobj(resp, f, length=1 << 20)
        target.with_suffix(target.suffix + ".part").replace(target)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            sys.stderr.write(f"  [skip] 404: {url}\n")
            _mark_404(target.parent, target.name)
            return
        raise


def fetch_urllib(repo_id: str, revision: str,
                 filenames: list[str] | None,
                 token: str | None) -> pathlib.Path:
    dest = target_dir(repo_id, revision)
    if filenames is None:
        # Optimization: if model.safetensors is already cached, drop
        # pytorch_model.bin from the default list (avoids the noisy
        # [skip] 404 for safetensors-only repos like TinyLlama).
        safet = dest / "model.safetensors"
        ptbin = dest / "pytorch_model.bin"
        files = list(DEFAULT_FILES)
        if safet.exists() and safet.stat().st_size > 0 and not ptbin.exists():
            files = [f for f in files if f != "pytorch_model.bin"]
        filenames = files
    base = f"https://huggingface.co/{repo_id}/resolve/{revision}"
    for fn in filenames:
        url = f"{base}/{fn}"
        _download_one(url, dest / fn, token)
    return dest


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch HF model files into a local cache for RAC benches.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"HF repo id (default: {DEFAULT_MODEL})")
    ap.add_argument("--revision", default=DEFAULT_REVISION,
                    help=f"branch / tag / commit (default: {DEFAULT_REVISION})")
    ap.add_argument("--file", action="append",
                    help="specific file(s) to fetch; can be repeated. "
                         "Omit to fetch a default set (config, tokenizer, "
                         "safetensors). Pass a GGUF filename here to fetch "
                         "just that quant for llama.cpp.")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF access token (or set $HF_TOKEN).")
    ap.add_argument("--print-file",
                    help="after fetching, print the absolute path to this "
                         "specific file (useful for shell command sub).")
    args = ap.parse_args()
    maybe_bootstrap_venv()

    try:
        if have_hub():
            path = fetch_hub(args.model, args.revision, args.file, args.token)
        else:
            sys.stderr.write(
                "  note: huggingface_hub not installed; falling back to urllib\n"
                "         (`pip install huggingface_hub` for resume + parallel)\n")
            path = fetch_urllib(args.model, args.revision, args.file, args.token)
    except Exception as e:
        sys.stderr.write(f"fetch failed: {e}\n")
        return 1

    if args.print_file:
        target = path / args.print_file
        if not target.exists():
            sys.stderr.write(f"requested --print-file not in cache: {target}\n")
            return 2
        print(str(target.resolve()))
    elif args.file and len(args.file) == 1:
        print(str((path / args.file[0]).resolve()))
    else:
        print(str(path.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
