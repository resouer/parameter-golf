#!/usr/bin/env python3
"""Round-23 FLA feasibility wrapper.

The actual GDN training logic lives in `train_gdn_7k.py`, imported from the
fetched #1370 scaffold. `evaluate.py` still calls `torchrun train_gpt.py`, so
this wrapper preserves the normal repo entrypoint while we validate the FLA
lane on Heimdall.
"""

from train_gdn_7k import main


if __name__ == "__main__":
    main()
