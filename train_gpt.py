#!/usr/bin/env python3
"""Round-23 FLA feasibility wrapper.

The actual GDN training logic lives in `train_gdn_7k.py`, imported from the
fetched #1370 scaffold. `evaluate.py` still calls `torchrun train_gpt.py`, so
this wrapper preserves the normal repo entrypoint while we validate the FLA
lane on Heimdall.
"""

import traceback


def main():
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
