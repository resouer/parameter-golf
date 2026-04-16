#!/usr/bin/env python3
"""Round-23 FLA feasibility wrapper.

The actual GDN training logic lives in `train_gdn_7k.py`, imported from the
fetched #1370 scaffold. `evaluate.py` still calls `torchrun train_gpt.py`, so
this wrapper preserves the normal repo entrypoint while we validate the FLA
lane on Heimdall.
"""

import importlib
import fcntl
import os
import shutil
import sys
import traceback
from pathlib import Path
import subprocess

# These explicit defaults serve two purposes:
# 1. They make the remote evaluate.py vocabulary detector pick the SP8192 path.
# 2. They keep the wrapper/runtime aligned with the intended W43 score probe.
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", 8192))
DATA_PATH = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
ARCH_MODE = os.environ.get("ARCH_MODE", "K")
os.environ.setdefault("VOCAB_SIZE", str(VOCAB_SIZE))
os.environ.setdefault("DATA_PATH", DATA_PATH)
os.environ.setdefault("TOKENIZER_PATH", TOKENIZER_PATH)
os.environ.setdefault("ARCH_MODE", ARCH_MODE)
if ARCH_MODE == "D":
    os.environ.setdefault("XSA_EVAL", "1")


_W40_VENDOR_DIR = Path(__file__).resolve().parent / ".w40_vendor"
_W40_VENDOR_PKGS = [
    "triton==3.2.0",
    "flash-linear-attention==0.4.2",
    "fla-core==0.4.2",
    "transformers==5.5.4",
    "tokenizers==0.22.2",
    "safetensors==0.7.0",
]


def _ensure_w40_vendor_on_path() -> None:
    p = str(_W40_VENDOR_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _ensure_fla_vendor_installed() -> None:
    _ensure_w40_vendor_on_path()
    try:
        from fla.layers.gated_deltanet import GatedDeltaNet  # noqa: F401
        print("w40_wrapper: local vendor layered import already works", flush=True)
        return
    except Exception:
        pass

    lock_path = _W40_VENDOR_DIR.parent / ".w40_vendor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        _ensure_w40_vendor_on_path()
        try:
            from fla.layers.gated_deltanet import GatedDeltaNet  # noqa: F401
            print("w40_wrapper: local vendor layered import became available after lock wait", flush=True)
            return
        except Exception:
            pass
        print(f"w40_wrapper: installing vendored FLA deps into {_W40_VENDOR_DIR}", flush=True)
        shutil.rmtree(_W40_VENDOR_DIR, ignore_errors=True)
        _W40_VENDOR_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-deps",
                "--target",
                str(_W40_VENDOR_DIR),
                *_W40_VENDOR_PKGS,
            ],
            stdout=sys.stderr,
            stderr=sys.stderr,
        )
        importlib.invalidate_caches()
        _ensure_w40_vendor_on_path()
        from fla.layers.gated_deltanet import GatedDeltaNet  # noqa: F401
        print("w40_wrapper: vendored FLA layered import succeeded", flush=True)


def main():
    _ensure_fla_vendor_installed()
    print("w40_wrapper: importing train_gdn_7k", flush=True)
    try:
        from train_gdn_7k import main as train_main
    except Exception:
        traceback.print_exc()
        raise
    print("w40_wrapper: import ok, entering train_main", flush=True)
    train_main()


if __name__ == "__main__":
    main()
