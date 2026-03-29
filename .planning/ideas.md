## [2026-03-28 18:15] Round 16

### Research Findings
- The exact byte ledger for the repaired tied-embed lane is: unpruned model `16,070,564`, code `124,051`, unpruned total `16,194,615`, required giveback `194,615`.
- The only approved giveback source is `selective_prune_low_error_pm1` inside the existing mixed-export exporter.
- The launch policy is fixed to `SEED=1337`, `TTT_ENABLED=0`, `EXPORT_FP16_EMBED=1`, `EXPORT_FP16_LATE_K_LAYERS=0`, `TARGET_TOTAL_BYTES=16000000`.

### Decision
- Run one smoke on `autoresearch/exp-fp16-embed-export-size-neutral` to verify that exact package budgeting repairs the standalone tied-embed lane.
- Route the job explicitly if the default scheduler does not assign a node group.
- Treat any final package over `16,000,000` bytes or any non-beneficial BPP result as a direct NO-GO for this repaired lane.

### Codex Review
- Local code verification passed: `python3 -m py_compile train_gpt.py`.
- The branch is pushed and ready for remote checkout at commit `677944d`.
- The implementation boundary is still single-mechanism: mixed export plus exact size budgeting only.
