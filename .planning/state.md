Current round: Round 142

## [2026-03-30 09:40] Round 116

- Branch: `autoresearch/exp-familyb-strict-causal-backoff`
- Base host: `/Users/zlei/code/parameter-golf-1060-loader-phase2 @ 0d6643923382fd124b41cd37d70e44855925b9bc`
- Deliverable: strict-legal `Family B` implementation packet
- Scope: `train_gpt.py` only
- Launch state: no launch yet

## [2026-03-30 11:15] Round 124

- Current round: Round 124
- Branch: `autoresearch/exp-familyb-batchwise-mixer-speedup`
- Base head: `6d74808adc021168a1115162eb2e5cccffdffe57`
- Deliverable: strict-legal `Family B` follow-up packet after `#116 = 0.39421265`
- Scope:
  - keep the same strict-legal Family B env surface as `#116`
  - patch `train_gpt.py`
  - no launch in this round
- Intended delta:
  - batchwise contiguous score-first mixer pass
  - reuse per-order hash/key computations for both mixing and update

## [2026-03-30 12:20] Round 126

- Current round: Round 126
- Branch: `autoresearch/exp-familyb-safe-batch-cache`
- Base head: `88e9ebbd70c87264b66055f7bf3bb30117e9b789`
- Deliverable: strict-legal `Family B` safe repair packet after the `#124` non-contiguous-target failure
- Scope:
  - preserve `#116` semantics exactly
  - patch `train_gpt.py`
  - no launch in this round
- Intended delta:
  - remove the broken one-pass contiguous-union scorer from `#124`
  - precompute the contiguous batch-update key range once
  - score each window by slicing that batch cache, then do the single post-batch update exactly like `#116`

## [2026-03-30 13:40] Round 129

- Current round: Round 129
- Branch: `autoresearch/exp-familyb-count-prefetch`
- Base head: `e8d6d047977a9a976a29153e38e7dc0827ad8952`
- Deliverable: strict-legal `Family B` bounded follow-up packet after clean `#126` closeout
- Scope:
  - keep `#126` as the base
  - patch `train_gpt.py`
  - no launch in this round
- Intended delta:
  - prefetch per-order table counts once for the contiguous batch cache
  - slice those cached counts per window during score-first mixing
  - keep the same post-batch update path as `#126`

## [2026-03-30 14:25] Round 134

- Current round: Round 134
- Branch: `autoresearch/exp-familyb-safe-batch-cache`
- Base head: `e8d6d047977a9a976a29153e38e7dc0827ad8952`
- Deliverable: single-node Family B parity confirmation lane on the clean `#126` keep-base
- Scope:
  - no new model-path change
  - minimal re-gate
  - launch under the official RunPod parity launcher mode
- Intended delta:
  - switch env/image/bootstrap surface to the official RunPod parity mode
  - keep the Family B method surface exactly on `#126`

## [2026-03-30 14:35] Round 136

- Current round: Round 136
- Branch: `autoresearch/exp-familyb-safe-batch-cache`
- Base head: `e8d6d047977a9a976a29153e38e7dc0827ad8952`
- Deliverable: rerun clean `#126` under the repaired parity launcher path
- Scope:
  - no new model-path change
  - minimal re-gate
  - relaunch under the official RunPod parity mode after the checkout-dir repair
- Intended delta:
  - keep the same official parity image/label surface
  - use repaired checkout dir `/workspace/parameter-golf-parity`

## [2026-03-30 15:04] Round 142

- Current round: Round 142
- Branch: `autoresearch/exp-familyb-safe-batch-cache`
- Base head: `e8d6d047977a9a976a29153e38e7dc0827ad8952`
- Deliverable: corrected full Family B parity rerun on the same clean `#126` keep-base
- Scope:
  - no new model-path change
  - minimal re-gate
  - relaunch under the same repaired official RunPod parity mode
  - explicitly restore the full Family B env surface
- Intended delta:
  - keep the same official parity image/label surface
  - keep the same repaired checkout dir `/workspace/parameter-golf-parity`
  - restore:
    - `USE_NGRAM_MIXER=1`
    - `NGRAM_ORDER=10`
    - `NGRAM_MIN_ORDER=2`
    - `NGRAM_BUCKETS=4194304`
    - `NGRAM_MIN_COUNT=1`
    - `ALPHA_BASE=0.20`
    - `ALPHA_RANGE=0.55`
    - `ALPHA_CENTER=3.0`
    - `NGRAM_BATCH_SEQS=128`
    - `TTT_ENABLED=0`
    - `TRIGRAM=0`

## [2026-03-30 18:44] Round 151

- Current round: Round 151
- Branch: `autoresearch/exp-familyb-count-prefetch`
- Base head: `e8d6d047977a9a976a29153e38e7dc0827ad8952`
- Deliverable: safer standalone-cache/state Family B candidate after retiring synchronized reuse as a base-preserving repair path
- Scope:
  - keep the same semantic base as `#142`
  - keep the standalone causal-backoff evaluation control flow
  - patch `train_gpt.py`
  - no launch in this round
- Intended delta:
  - prefetch Family B count/state lookups once for the fixed batch key cache
  - reuse those prefetched counts per window inside the standalone scorer
  - avoid distributed shared-pass / shared-state tricks entirely

## [2026-03-30 18:48] Round 152

- Current round: Round 152
- Branch: `autoresearch/exp-familyb-count-prefetch`
- Base head: `4dc08a7c53a3614bc1edc9b43ff9819eae1e7c10`
- Deliverable: launch-authorized safer standalone count/state prefetch candidate on the corrected parity/full-Family-B surface
- Scope:
  - keep the same semantic base as `#142`
  - keep the same accepted standalone count/state prefetch code surface
  - no new code changes in this round
- Intended delta:
  - minimal re-gate only
  - launch the accepted standalone-cache/state candidate on the single allowed AWS node
  - report `job / job id / node group / direct state`

## [2026-03-30 19:53] Round 153

- Current round: Round 153
- Branch: `autoresearch/exp-familyb-count-prefetch`
- Base head: `4dc08a7c53a3614bc1edc9b43ff9819eae1e7c10`
- Deliverable: confirmation rerun of the accepted standalone count/state prefetch candidate to decide whether the residual semantic delta vs `#142` is noise or persistent
- Scope:
  - keep the same semantic base as `#142`
  - keep the same accepted standalone count/state prefetch code surface
  - keep the same corrected parity/full-Family-B launch surface
  - no new code changes in this round
- Intended delta:
  - minimal re-gate only
  - relaunch the accepted standalone-cache/state candidate on the single allowed AWS node
  - report `job / job id / node group / direct state`

## [2026-03-30 20:54] Round 154

- Current round: Round 154
- Branch: `autoresearch/exp-familyb-sparse-update`
- Base head: `4dc08a7c53a3614bc1edc9b43ff9819eae1e7c10`
- Deliverable: bounded sparse-update follow-up on top of the confirmed standalone count/state prefetch line
- Scope:
  - keep the same semantic base as `#142`
  - keep the same standalone-prefetch control flow confirmed by `#152/#153`
  - patch `train_gpt.py`
  - no launch in this round
- Intended delta:
  - replace dense full-bucket `np.bincount(..., minlength=...)` post-batch updates
  - use sparse unique-key accumulation only for the keys touched by the current batch
  - leave Family B scoring semantics and update order unchanged

## [2026-03-31 06:37] Round 171

- Current round: Round 171
- Branch: `autoresearch/exp-familyb-vectorized-order-cache`
- Base head: `c6e77c4d6e449273a3c8c9ff2510e0ccfb6bfeea`
- Deliverable: bounded keep-line follow-up that vectorizes Family B order-cache key construction
- Scope:
  - keep confirmed sparse-update keep-line `c6e77c4`
  - keep semantic/control anchor at `#142`
  - patch `train_gpt.py`
  - no launch in this round
- Intended delta:
  - precompute per-order offset and prime vectors once in the mixer
  - compute the same `ctx_hash` / `full_key` values with NumPy bulk gather/XOR instead of the inner Python token loop
  - leave Family B scoring semantics and update order unchanged
