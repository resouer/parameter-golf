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
