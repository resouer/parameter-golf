## [2026-03-28 18:15] Round 16

### Research Findings
- `#20` proved the tied-embed mixed-export path is real, but the saved run finished at `Serialized model int6+lzma=16,070,564` and `Total submission size int6+lzma=16,192,977`, so the lane was over the real cap by `192,977` bytes.
- The root cause was exporter budgeting, not mechanism failure: the branch was pruning against legacy `TARGET_MB=15.9` in binary MiB (`16,672,358` bytes), so it logged `selective_prune: already fits` even though the real decimal package cap is `16,000,000`.
- Branch `autoresearch/exp-fp16-embed-export-size-neutral` at commit `677944d` converts that budget to exact total-byte accounting and logs `mixed_export_size_budget ... giveback_needed_bytes=...` before final export.

### Paradigm Assumptions
- Keep the lane inside a single mechanism family: tied-embedding mixed export plus exact byte giveback.
- Do not add `prune-before-quant`, late-key fp16, or any TTT path.
- Preserve the standalone export policy from `#20`: `TTT_ENABLED=0`, `EXPORT_FP16_EMBED=1`, `EXPORT_FP16_LATE_K_LAYERS=0`.

### Frontier Snapshot
- Current internal gate for this lane is one `seed=1337` smoke only, with a hard package limit of `16,000,000` total bytes.
- `#21 prune-before-quant` is already assigned as a separate lane and must not be folded into `#16`.
- `#17` is the separate late-key-only mixed-export lane; `#16` should stay tied-embed-only.

### Comparable Methods
- `#20` is the direct predecessor: same tied-embed mixed-export policy, but with a bogus size target.
- The repaired branch keeps the same export tensor set and only changes how byte giveback is enforced.
- The prior TTT family is not a valid comparator for this lane because it changes the mechanism category and has already been gated off.

### Novelty-Relevant Findings
- The repair is not a new model family; it is a validity repair for an already-proven mixed-export mechanism.
- The explicit giveback source is existing `selective_prune_low_error_pm1`, which was already on-path in `#20`.
- The saved `#20` run exposed `4,136,264` eligible `±1` candidates, so the giveback source is concrete rather than speculative.

### Compliance & Risk Status
- Compliance is clean if the final run logs `Total submission size <= 16,000,000` with the approved export policy.
- The main technical risk is quality: tied-embed mixed export may still fail the supervisor’s “clean benefit” bar even if size is repaired.
- A second risk is launcher-side scheduling; the first launch attempt failed with `no available node groups found`, so explicit node-group routing may be required.

### Known Failures
- The original `#20` run is invalid as a standalone candidate because it was over cap.
- The first `#16` launch attempt failed before submission because no node group was allocated under the default scheduler path.
- The second `#16` launch attempt was blocked by missing Round 16 planning files, not by repo code or experiment policy.
