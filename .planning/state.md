Current round: Round 146

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

## [2026-03-30 13:37] Round 129

- Current round: Round 129
- Branch: `autoresearch/exp-familyb-reuse-sliding-pass`
- Base head: `e8d6d047977a9a976a29153e38e7dc0827ad8952`
- Deliverable: strict-legal `Family B` bounded follow-up packet after the clean `#126` closeout
- Scope:
  - keep the same strict-legal `Family B` scoring semantics as `#116/#126`
  - patch `train_gpt.py`
  - no launch in this round
- Intended delta:
  - reuse the already-required sliding exact pass when `FAMILYB_REUSE_SLIDING_PASS=1`
  - feed the strict-legal causal backoff scorer from that shared pass instead of launching a second full model-forward eval
  - keep the old standalone `eval_val_causal_backoff_mixer(...)` path available as the fallback

## [2026-03-30 15:58] Round 143

- Current round: Round 143
- Branch: `autoresearch/exp-familyb-reuse-sliding-pass-parity`
- Base head: `e8d6d047977a9a976a29153e38e7dc0827ad8952`
- Deliverable: bounded strict-legal `Family B` follow-up packet after the parity-confirmed `#142` closeout
- Scope:
  - keep the same strict-legal `Family B` scoring semantics as `#142`
  - patch `train_gpt.py`
  - no launch in this round
- Intended delta:
  - enable the already-prepared sliding-pass reuse path as the chosen next bounded packet
  - reuse the already-required sliding exact pass when `FAMILYB_REUSE_SLIDING_PASS=1`
  - keep the old standalone `eval_val_causal_backoff_mixer(...)` path available as the fallback

## [2026-03-30 17:24] Round 146

- Current round: Round 146
- Branch: `autoresearch/exp-familyb-reuse-sliding-sync`
- Base head: `907640e977ee41768db004f4e39171e605c7e4ee`
- Deliverable: strict-legal `Family B` safe repair packet after the `#143` rank-sharded reuse semantic failure
- Scope:
  - keep the same parity-confirmed `#142` Family B scoring semantics
  - patch `train_gpt.py`
  - no launch in this round
- Intended delta:
  - replace rank-local reuse with synchronized global-batch payload gathering
  - let one authoritative mixer state score gathered Family B windows in global order
  - keep the standalone `eval_val_causal_backoff_mixer(...)` path available as the fallback
