## [2026-03-30 09:40] Round 116

### Research Findings
- Clean host stays `#64` / `0d6643923382fd124b41cd37d70e44855925b9bc`.
- Old `pgolf` gives the causal backoff/mixer seed, but only after stripping contaminated features.
- The first Family B packet should be single-file and strict-legal by construction.

### Decision
- Build `Family B` as a minimal strict-legal causal backoff/mixer port on top of the clean #64 host.
- Keep the neural exact evaluator intact and add a parallel Family B exact read.
- Keep the first packet no-launch and pre-launch-ready only.

### Codex Review
- Scope stays `train_gpt.py` only.
- Required excludes: no TTT, no two-pass rescoring, no PAQ adaptive logic, no order-adaptive logic, no hindsight surfaces, no extra side files.
- Success condition for this round is compile-clean plus launcher dry-run clean with `Round 116` planning validation.

## [2026-03-30 11:15] Round 124

### Research Findings
- `#116` on the strict-legal Family B surface already landed `final_int6_causal_backoff_exact val_bpb:0.39421265` under the byte cap.
- The same run showed the current exact-tail implementation is extremely slow:
  - `final_int6_causal_backoff eval_time:1710133ms`
  - `TIMING:final_eval=1851.3s`
- The current code recomputes the same context/full hashes twice on each contiguous scored batch union:
  - once during score-first mixing
  - again during the post-score update

### Decision
- Keep the exact same strict-legal Family B env surface as `#116`.
- Apply a narrow implementation-side speedup only:
  - score each contiguous batch union once
  - cache/reuse per-order context/full-key computations for both mixing and update
- Do not change family, legality surface, or broader host choice in this round.

### Codex Review
- File scope remains `train_gpt.py` plus the mandatory `Round 124` planning-file updates.
- Success condition is:
  - `python3 -m py_compile train_gpt.py`
  - clean `run_lepton.py --dry-run --round 124 ...`
  - packet writeup must explain expected value vs `#116` as a semantics-preserving exact-tail runtime reduction, not a new scoring family.

## [2026-03-30 12:20] Round 126

### Research Findings
- The recovered `#124` failure is a code-level invariant break, not a Family B failure:
  - `Non-contiguous scored targets in causal backoff batch: targets=6208 probs=8192`
- The unsafe assumption was that one batch’s scored tokens could always be collapsed into a single deduplicated contiguous target list.
- The exact-tail speedup still matters, but the next packet must re-establish semantic safety first.

### Decision
- Replace the broken `#124` one-pass contiguous scorer with a safe cache-sharing path:
  - precompute the contiguous batch-update key range once
  - score each window independently by slicing that batch cache
  - keep the single post-batch update exactly like `#116`
- Keep the exact same strict-legal Family B env surface as `#116/#124`.
- Keep the packet no-launch and pre-launch-ready only.

### Codex Review
- File scope remains `train_gpt.py` plus the mandatory `Round 126` planning-file updates.
- Success condition is:
  - `python3 -m py_compile train_gpt.py`
  - clean `run_lepton.py --dry-run --round 126 ...`
  - packet writeup must explain why this preserves `#116` semantics while replacing the broken `#124` optimization path.

## [2026-03-30 13:37] Round 129

### Research Findings
- `#126` closed cleanly and repaired `#124`'s exact-tail crash without changing the winning strict-legal Family B surface:
  - `final_int6_causal_backoff_exact val_bpb:0.39442306`
  - `TIMING:final_eval=1813.1s`
- The remaining obvious waste is that strict-legal Family B still launches a second full sliding-style eval pass after the already-required `final_int6_sliding_window` pass.
- The shared-pass opportunity is narrow and local:
  - sliding exact and Family B exact use the same window order, same logits source, and the same score-first window semantics
  - the avoidable duplication is the second model-forward sweep, not the mixer formula itself

### Decision
- Keep `#126` as the semantic base and add one bounded opt-in follow-up:
  - `FAMILYB_REUSE_SLIDING_PASS=1`
- Reuse the main sliding exact pass to supply target probabilities and entropy to the same strict-legal Family B scorer.
- Preserve the old standalone causal-backoff path as the fallback when the new env flag is off.

### Codex Review
- File scope remains `train_gpt.py` plus the mandatory `Round 129` planning-file updates.
- Success condition is:
  - `python3 -m py_compile train_gpt.py`
  - clean `run_lepton.py --dry-run --round 129 --env USE_NGRAM_MIXER=1 --env FAMILYB_REUSE_SLIDING_PASS=1 ...`
  - packet writeup must state the explicit boundary:
    - same strict-legal scoring semantics intended as `#116/#126`
    - shared-pass timing/logging is a bounded implementation change, not a new scoring family

## [2026-03-30 15:58] Round 143

### Research Findings
- `#142` has now closed as the corrected full-env parity rerun on the official-targeted RunPod image with the full Family B env surface visibly active.
- The parity-confirmed keep-base is therefore the same clean semantic surface as `#126`, but now with stronger env evidence:
  - `TIMING:final_eval=1790.8s`
  - `final_int6_causal_backoff_exact val_bpb:0.39344583`
  - `final_int6_causal_backoff eval_time:1656267ms`
- The training-side regime already improved materially under parity (`~87.9ms/step` through wallclock cap), so the dominant remaining exact-tail waste is even more concentrated in the standalone Family B causal-backoff eval pass itself.

### Decision
- Choose the bounded `FAMILYB_REUSE_SLIDING_PASS=1` follow-up as the single next packet on top of the parity-confirmed Family B keep-base.
- Keep the exact same strict-legal Family B env surface as `#142`; only add the reuse flag as the bounded implementation delta.
- Preserve the standalone `eval_val_causal_backoff_mixer(...)` path as the fallback when the reuse flag is off.

### Codex Review
- Scope remains `train_gpt.py` plus the mandatory `Round 143` planning-file updates.
- Success condition is:
  - `python3 -m py_compile train_gpt.py`
  - clean `run_lepton.py --dry-run --round 143 --env USE_NGRAM_MIXER=1 --env FAMILYB_REUSE_SLIDING_PASS=1 ...`
  - packet writeup must state the new governing read:
    - keep-base is parity-confirmed `#142`
    - bounded EV is to attack the remaining standalone Family B closeout pass
    - this is still a no-launch packet, not a new family/search pivot

## [2026-03-30 17:24] Round 146

### Research Findings
- `#143` proved the reuse-pass idea can deliver a real runtime win, but its first implementation was not behavior-preserving:
  - `#142`: `final_int6_causal_backoff_exact val_bpb:0.39344583`, `TIMING:final_eval=1790.8s`
  - `#143`: `final_int6_causal_backoff_exact val_bpb:0.82366213`, `TIMING:final_eval=284.6s`
- The exact culprit is not parity drift or a changed Family B formula.
- The culprit is that `#143` piggybacked Family B scoring onto rank-sharded sliding eval, so each rank updated a local mixer on only its own `my_windows` slice.

### Decision
- Keep `#142` as the semantic base and keep the bounded reuse-pass objective.
- Replace the unsafe rank-local reuse path with a synchronized global-batch reuse path:
  - sliding exact may stay rank-sharded for the neural metric
  - but Family B window payloads must be gathered back into one global order before any mixer update happens
  - a single authoritative mixer state should score those gathered windows
- Keep the old standalone causal-backoff path as the fallback when the reuse flag is off.

### Codex Review
- Scope remains `train_gpt.py` plus the mandatory `Round 146` planning-file updates.
- Success condition is:
  - `python3 -m py_compile train_gpt.py`
  - clean `run_lepton.py --dry-run --round 146 --env USE_NGRAM_MIXER=1 --env FAMILYB_REUSE_SLIDING_PASS=1 ...`
  - packet writeup must state the safe repair boundary explicitly:
    - no rank-sharded Family B mixer state
    - gathered global order before any reuse-path mixer update
    - no launch in this round
