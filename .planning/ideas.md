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
