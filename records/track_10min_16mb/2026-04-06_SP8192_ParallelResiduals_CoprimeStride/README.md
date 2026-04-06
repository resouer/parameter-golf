# Record: SP8192 + Parallel Residuals + Coprime-Stride Loader

**val_bpb = 1.08459** (3-seed mean, std 0.00069) | 15.99 MB | 8xH100 SXM | ~115s eval

## Results (3-seed)

| Seed | BPB | val_loss (nats) | Artifact |
|------|-----|-----------------|----------|
| 1337 | **1.08414** | 2.80045 | 15,985,531 |
| 42 | **1.08424** | 2.80070 | 15,989,295 |
| 2025 | **1.08538** | 2.80365 | 15,986,932 |
| **Mean** | **1.08459** | **2.80160** | |

Merged SOTA (PR #1019, 3-seed mean): **2.88218 nats** (1.1147 BPB). This run: **2.80160 nats**. Delta: **-0.0806 nats**. Clears the 0.005-nat threshold.

## Changes from Base (PR #1394)

### 1. Parallel Residuals (from layer 7)
Layers 7-10 execute attention and MLP in parallel (PaLM-style) instead of sequential. The normalized input feeds both branches simultaneously, with learned per-channel scales (`attn_scale`, `mlp_scale`) controlling the contribution of each. Zero additional parameters beyond the existing scale vectors. Nearest PR: #1334 (parallel residuals on SP4096). Different: applied to SP8192 stack with depth recurrence, where the parallel execution interacts with the looped layers 4-5 differently than on SP4096.

### 2. Coprime-Stride Data Loader
Replaces standard sequential shard traversal with coprime-stride ordering. For each shard, a stride coprime to the number of sequences is selected, ensuring all sequences are visited exactly once in a pseudo-random order without repetition. This provides better data diversity within each epoch without additional compute cost. Not present in any SP8192 submission.

### Architecture
- SP8192 vocabulary (8192 BPE tokens via SentencePiece)
- 11 transformer layers, dim 512, MLP 4x, 8 heads / 4 KV heads (GQA)
- Depth recurrence: layers 4-5 looped 2x (effective 13 layers)
- XSA-all (exclusive self-attention on all 11 layers)
- Skip gates, RMSNorm, LeakyReLU(0.5)^2 activation
- MuonEq-R optimizer (row-normalized Newton-Schulz)
- GPTQ int6 weights + int8 embeddings + brotli compression
- SDClip (std-dev based quantization clipping)
- EMA (decay 0.997)

### Compression
- Code: lzma+base85 self-extracting (43KB -> 15.8KB)
- Model: GPTQ int6 + brotli-11 (~15.97MB)
- Total artifact: ~15.99MB (under 16MB limit)

## Compliance
- All techniques are training-side architecture changes. No eval-time adaptation.
- No SLOT, no TTT, no n-gram caches.
- Eval uses `torch.inference_mode()` for scoring. Model weights frozen at eval time.
- GPTQ calibration uses AR self-generated training data (not validation data).
- Sliding window evaluation with stride 64, standard BPB calculation.

## Reproduction

```bash
pip install brotli
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

No env vars needed. Code defaults are the submission config. SP8192 data downloads automatically from `kevclark/parameter-golf` on first run.

## Credits
Base: PR #1394 (@clarkkev) — SP8192 + Depth Recurrence + MuonEq-R + SDClip + GPTQ int6.
Parallel residuals pattern: PR #1334 (@aryanbhosale) — first demonstrated on SP4096.
