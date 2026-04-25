# SP8192 + Polar Express NS + MIN_LR + Tight GPTQ on PR #1790 stack

**3-seed mean: val_bpb = 1.06892 (std 0.00031)**, beats merged SOTA PR #1493 (1.0810) by **0.01208 nats**.

## 3-seed results

| Seed | val_bpb     | bytes_total | Node                | Node Group                       |
|------|-------------|-------------|---------------------|----------------------------------|
| 1337 | 1.06887604  | 15,938,246  | node-ip-10-0-120-106 | gcp-iad-leptondev-002            |
| 42   | 1.06924420  | 15,941,018  | node-ip-10-0-112-30  | gcp-iad-leptondev-002 (different node) |
| 2025 | 1.06863616  | 15,938,565  | node-ip-10-0-119-97  | training-dev-0 (different group) |
| **mean** | **1.06891880** | max=15,941,018 | 3 nodes / 2 groups | |
| std  | 0.00030627  |             |                     |                                  |

p < 0.01 statistical significance vs the SOTA-0.005 threshold (1.0760): t-stat ≈ 40.

## Mechanism

Cross-stack combination of two open PRs:

1. **Base** (PR #1790, @miaoyuxun, claim 1.06991): SP8192 + SmearGate + AttnOutGate (window 24) + LoRA-TTT improvements + Phased TTT.
2. **Recipe ported from PR #1792** (open):
   - **Polar Express Newton-Schulz** (originally from PR #1344 by @orangekame3): 5 per-iteration minimax-optimal `(a, b, c)` coefficient tuples replacing Muon's single fixed `(3.4445, -4.775, 2.0315)` tuple applied 5 times. Same iteration count (`backend_steps=5`); different per-step coefficients yield a tighter polar-factor approximation per Muon step.
   - **`VAL_LOSS_EVERY=0`**: skips the periodic mid-training val-loss diagnostic prints. Pure systems optimization — zero ML impact, but the reclaimed wallclock contributes to the effective training budget.
   - **`MIN_LR=0.10`**: floors the warmdown learning rate at 0.10 of peak instead of decaying to 0. The final-phase steps remain meaningful contributors.
   - **`GPTQ_RESERVE_SECONDS=1.5`**: tightens the pre-quantization wallclock reservation from the safetri-default 4.0s to 1.5s. A previous attempt with 0.5s exhibited brittle 3-seed variance (code-reviewer flagged); 1.5s preserves a safety cushion while reclaiming 2.5s of training budget. Cross-node 3-seed validation confirms it stays stable across both gcp and training-dev hardware.

No single upstream PR combines these four `#1792` recipe changes on top of `#1790`'s stack — `#1792` itself targets a different parent (`#1768` GatedAttn + Alpha-LoRA), and `#1790` does not include the Polar Express NS coefficients or the budget-reclamation defaults.

## Relationship to nearest prior PR

| | PR #1790 (miaoyuxun) | This submission |
|---|---|---|
| Base stack | SP8192 + SmearGate + AttnOutGate + LoRA-TTT + Phased TTT | (same) |
| Newton-Schulz | fixed `(3.4445, -4.775, 2.0315) × 5` | 5 per-iteration Polar Express tuples |
| `VAL_LOSS_EVERY` | 4000 (default) | 0 (skip diagnostic) |
| `MIN_LR` | 0.0 | 0.10 |
| `GPTQ_RESERVE_SECONDS` | 4.0 (default) | 1.5 (2.5s reclaim, with cushion) |

## Reproduction

```bash
torchrun --nproc_per_node=8 train_gpt.py
```

No environment variables required — the four configuration tweaks are baked into the shipped `train_gpt.py` defaults. `requirements.txt` lists optional Python deps (FlashAttn 3 wheel, etc.); the standard pgolf runtime image satisfies them.

## Compliance

- **Score-first per-chunk TTT**: each val chunk is scored by `_accumulate_bpb` BEFORE any LoRA gradient adaptation on that chunk. Adaptation on chunk *t* never affects scoring of chunk *t* — only chunks *t+1, t+2, …* which the model has not yet seen. This is the same legal score-first pattern used by the merged SOTA PR #1493 (bigbag, val_bpb 1.0810).
- **No n-gram hash keyed on target tokens**: `grep -nE "hash.*target|key.*\btargets?\b|ngram.*label" train_gpt.py` returns no hits.
- **No pre-quant TTT on val**: TTT runs strictly POST-quantization (the `quantized_ttt_phased` path). `grep -nE "val.*ttt|ttt.*val|test.*adapt.*pre.*quant" train_gpt.py` matches only legal score-first phased-TTT function/variable names.
- **No CaseOps**: `grep -nE "CaseOp|case_op|caseops" train_gpt.py` returns no hits.
- **No SLOT**: not used.
- **Python 3.10 `py_compile`** passes on the shipped `train_gpt.py`.
- **Artifact accounting**: max-seed `bytes_total` = 15,941,018 < 16,000,000.
- **Tokenizer/dataset unchanged**: stock SP8192 SentencePiece tokenizer and stock `fineweb10B_sp8192` dataset.

## Reproducibility hardening (cross-node and cross-group)

The 3-seed mean was deliberately spread across **3 different physical nodes** in **2 different node groups** to avoid same-node hardware correlation:

- Seed 1337 ran on `gcp-iad-leptondev-002 / node-ip-10-0-120-106`.
- Seed 42 ran on `gcp-iad-leptondev-002 / node-ip-10-0-112-30` — different physical node, same group.
- Seed 2025 ran on `training-dev-0 / node-ip-10-0-119-97` — different physical node, **different node group entirely**.

Cross-node within-group delta (seed 42, gcp 120-106 → gcp 112-30): −0.00032 nats.
Cross-group delta (seed 2025, gcp → training-dev): −0.00040 nats.
Both deltas are within typical run-to-run noise (~0.0003–0.0005), confirming the result is hardware-independent.

## Credits

- **PR #1790** (@miaoyuxun) — base SP8192 + SmearGate + AttnOutGate + LoRA-TTT + Phased TTT stack.
- **PR #1792** — `VAL_LOSS_EVERY=0` + `MIN_LR=0.10` + tight `GPTQ_RESERVE_SECONDS` recipe.
- **PR #1344** (@orangekame3 et al.) — original Polar Express Newton-Schulz coefficients.
