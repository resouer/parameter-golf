# Record: SP8192 + Parallel Residuals + Coprime-Stride + Legal Score-First TTT

**val_bpb = 1.08286** (3-seed mean, std 0.00070) | 15.99 MB | 8xH100 SXM | ~1750s total (588s train + ~430s TTT eval)

## Results (3-seed)

| Seed | Pre-TTT BPP | Post-TTT BPP | TTT Gain | val_loss (nats) | Artifact |
|------|------------|-------------|----------|-----------------|----------|
| 1337 | ~1.084 | **1.08255** | -0.0015 | 2.79633 | 15,988,547 |
| 42 | ~1.084 | **1.08237** | -0.0016 | 2.79588 | 15,990,325 |
| 2025 | ~1.084 | **1.08366** | -0.0007 | 2.79921 | 15,989,566 |
| **Mean** | **~1.084** | **1.08286** | **-0.0013** | **2.79714** | |

Merged SOTA (PR #1019): **2.88218 nats** (1.1147 BPP). This run: **2.79714 nats**. Delta: **-0.0850 nats**. Clears the 0.005-nat threshold.

## Changes from Base (PR #1394)

### 1. Parallel Residuals (from layer 7)
Layers 7-10 execute attention and MLP in parallel (PaLM-style) instead of sequential. Zero additional parameters. Measured delta: **-0.0016 BPP** (R10).

### 2. Coprime-Stride Data Loader
Coprime-stride shard traversal for better data diversity. Each shard is traversed with a stride coprime to the number of sequences, ensuring all sequences visited exactly once in pseudo-random order. Measured delta: **-0.0016 BPP** (R10).

### 3. Legal Score-First TTT (eval-time)
Score-first test-time training on the quantized model. Each sliding-window chunk is scored under `torch.inference_mode()` BEFORE any gradient update. Training on a chunk only happens AFTER scoring. Last chunk is score-only. Config: SGD with momentum 0.9, LR=0.005, 3 epochs per chunk, 32768 tokens per chunk. Measured delta: **-0.0015 BPP** (R12).

Pattern follows PR #549 precedent:
```python
for chunk in chunks:
    # Phase 1: SCORE (no grad)
    with torch.inference_mode():
        nll = model(batch); loss_sum += nll.sum()
    # Phase 2: TRAIN (only on scored chunk)
    if not last_chunk:
        for epoch in range(3):
            loss = model(x, y); loss.backward(); optimizer.step()
```

## Architecture
- SP8192 (8192 BPE tokens via SentencePiece)
- 11 layers, dim 512, MLP 4x, 8 heads / 4 KV heads (GQA)
- Depth recurrence: layers 4-5 looped 2x (effective 13 layers)
- XSA-all, skip gates, RMSNorm, LeakyReLU(0.5)^2
- MuonEq-R optimizer, EMA (0.997)
- GPTQ int6 weights + int8 embeddings + brotli + SDClip

## Compliance
- All training-side techniques are architecture changes. LEGAL.
- TTT is score-first: strict score-before-update ordering per PR #549.
- `torch.inference_mode()` during scoring prevents gradient accumulation.
- No SLOT, no pre-quant TTT, no n-gram caches, no eval-time logit bias.
- GPTQ calibration uses AR self-generated training data (not validation).

## Reproduction
```bash
pip install brotli
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
No env vars needed. SP8192 data downloads automatically.

## Credits
Base: PR #1394 (@clarkkev). Parallel residuals: PR #1334 (@aryanbhosale). TTT pattern: PR #549 (@abaybektursun), PR #1413 (@dexhunter).
