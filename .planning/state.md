Current round: Round 116

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
