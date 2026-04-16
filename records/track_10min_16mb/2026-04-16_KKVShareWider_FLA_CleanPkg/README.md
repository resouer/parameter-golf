# Record: K_KVShare_Wider full-recipe FLA (cleanpkg)

**val_bpb: 1.0409** (3-seed mean, std 0.0011) | **3.1648 nats** | **<= 15.97 MB conservative packaged upper bound** | 8xH100 SXM, 600s | No TTT

This record takes the FLA / GatedDeltaNet family from the round-23 exploration lane and turns it into a candidate on a single canonical script revision. The nearest prior family reference is PR #1370, but the strongest point here is not the early trimmed scaffold. The winning configuration is `K_KVShare_Wider` on the fuller upstream-style recipe.

The main hardening step in this folder is that `train_gpt.py` no longer downloads dependencies at runtime. Required Python packages are declared in `requirements.txt` and must be installed before evaluation. The candidate script itself fails fast if the FLA stack is missing.

## Results (8xH100 80GB SXM, 600s, no TTT)

| Seed | Steps | Post-EMA BPB | **Quantized BPB** | val_loss (nats) | Artifact |
|------|------:|-------------:|------------------:|----------------:|---------:|
| 1337 | 1652 | 1.020660 | **1.03967403** | 3.16104735 | 15,762,406 |
| 42 | 1652 | 1.022042 | **1.04153708** | 3.16671180 | 15,870,797 |
| 2025 | 1583 | 1.023994 | **1.04148177** | 3.16654364 | 15,648,800 |
| **Mean** | **1629** | **1.022232** | **1.04089763** | **3.16476760** | **15,760,668** |

## Technique Summary

Core mechanism:
- FLA / GatedDeltaNet family (`K_KVShare_Wider`)
- KV sharing used to buy width rather than depth
- fuller upstream-style recipe instead of the earlier trimmed feasibility scaffold
- EMA + SWA + late QAT + int6 artifact path

What this candidate does **not** use:
- no TTT
- no SLOT
- no n-gram overlay
- no SWA/XSA scoring path for the final metric (`K_KVShare_Wider` has no SWA layers)

Final scored line in all three logs is the exact no-XSA roundtrip line:
- `final_int6_roundtrip_exact`

## Compliance Notes

- Validation data is not used for training. The script trains from `train_files` and scores separately on `val_files`.
- The candidate does not perform eval-time adaptation.
- `train_gpt.py` does not perform network downloads during evaluation.
- Dependencies are preinstalled via `requirements.txt` before running the script.
- The raw compressed model artifact stays under 16,000,000 bytes on every seed.

Size accounting used for this candidate:
- max artifact bytes across canonical seeds: `15,870,797`
- code files in this folder: `train_gpt.py`, `train_gdn_7k.py`, `architectures.py`, `configs.py`
- a conservative draft packaged-folder audit (including README / submission.json / requirements / logs) remains under `16,000,000` bytes.

## Reproducibility

Install dependencies before evaluation:

```bash
pip install -r requirements.txt
```

Prepare the SP8192 cached dataset/tokenizer as usual, then run one seed with:

```bash
SEED=$SEED ARCH_MODE=K MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=0 EVAL_COMPILE_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

`EVAL_COMPILE_ENABLED=0` is an operational stability setting used to avoid final-tail DDP/NCCL flakiness. It does not change the model family or the quantized scoring path.
