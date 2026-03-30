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
