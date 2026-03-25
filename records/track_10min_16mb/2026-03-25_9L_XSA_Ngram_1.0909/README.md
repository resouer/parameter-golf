# Record: 9L XSA-all + LeakyReLU(0.5)^2 + Online 5-gram Eval Cache

**val_bpb: ~1.090** (3-seed mean, pending seed 7) | **~14.7 MB** | 8xH100 SXM

## Results (8xH100 SXM)

| Seed | Pre-ngram BPB | Post-ngram BPB | Artifact |
|------|---------------|----------------|----------|
| 1337 | 1.1700 | **1.0898** | 14.68 MB |
| 42 | ~1.170 | **1.0909** | 14.69 MB |
| 7 | pending | pending | pending |
| **Mean** | **~1.170** | **~1.090** | |

## Key Techniques

### Training (9L/512d, ~17.6M params)
- 9 transformer layers, 512d, 8H/4KV (GQA), MLP 2x
- **XSA on all 9 layers** (Exclusive Self-Attention)
- **LeakyReLU(0.5)^2** activation (eliminates dead neurons)
- SmearGate temporal gating
- BigramHash(4096 buckets, dim=128)
- OrthoInit, LN Scale, Partial RoPE (25%)
- Muon optimizer (lr=0.02, momentum 0.92->0.99, WD=0.04)
- seq2048, batch 786K tokens, warmdown 3500

### Quantization
- Standard per-row int8 (NOT int6 — int8 gives near-zero degradation)
- No GPTQ (standard percentile clipping is sufficient at int8)
- zstd-22 compression
- 14.7MB artifact (well under 16MB)

### Eval-time Innovation: Online 5-gram Cache
- Hashed 5-gram frequency table (4M buckets) accumulated from scored tokens
- Fixed-weight linear mixing: `mixed = 0.8 * p_model + 0.2 * p_ngram`
- Strictly causal: cache updated AFTER each segment is scored
- No target-aware gating (legal per competition rules)
- **-0.079 BPB improvement** (1.170 → 1.091) for zero artifact cost
- Eval time: ~132 seconds (well within 600s budget)

Inspired by and credited to @deanbrr (PR #659) and @newjordan (PR #674) for the n-gram eval cache concept.

## Architecture Details

The key insight: at int8 quantization, a 9-layer model has near-zero quantization
degradation (~0.001 BPB) and easily fits in 16MB (14.7MB). The n-gram eval cache
then provides a massive -0.079 BPB improvement at eval time, complementing the
neural model's predictions on repetitive patterns in web text.

## Reproduce

```bash
SEED=1337 NUM_LAYERS=9 MLP_MULT=2 QUANT_BITS=8 GPTQ_ENABLED=0 PRUNE_PCT=0 NGRAM_ENABLED=1 \
  torchrun --nproc_per_node=8 train_gpt.py
```
