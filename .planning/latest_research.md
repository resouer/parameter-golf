## [2026-03-30 09:40] Round 116

### Research Findings
- `#111` already established that active `parameter-golf*` worktrees have no clean strict-legal `Family B` seed ready to launch.
- The cleanest execution host remains `#64` at `/Users/zlei/code/parameter-golf-1060-loader-phase2 @ 0d6643923382fd124b41cd37d70e44855925b9bc`.
- Old `/Users/zlei/code/pgolf/train_gpt.py` contains the nearest algorithmic seed for causal backoff/mixer, but it is contaminated by excluded surfaces such as score-first TTT, two-pass rescoring, PAQ adaptive mixing, and order-adaptive extras.

### Paradigm Assumptions
- Family B must stay on the strict-legal surface: score-first, backward-looking only, full-vocab normalized mixture, and no validation-time training.
- The first implementation packet should stay as narrow as possible: `train_gpt.py` only on the clean #64 host.
- The current queue decision should preserve Family B as the clean backup while `#115` decides whether A1 gets a narrow runtime-fix follow-up.

### Frontier Snapshot
- Official merged SOTA remains `#549 = 1.1194`.
- Strict-legal `Family B` public frontier remains materially stronger on raw score, with `#1094 = 0.3958`.
- `#110` proved the current A1 packet can land a near-`#549` exact result, but not within a healthy runtime envelope.

### Comparable Methods
- `#1094` is the relevant strict-legal causal backoff/mixer reference surface.
- Old local `pgolf` n-gram code is usable only as a stripped seed, not as a direct execution base.
- `#110` is a quality-control packet on A1, not a Family B implementation surface.

### Novelty-Relevant Findings
- The actionable local novelty here is not the family itself, but the clean port boundary: minimal strict-legal mixer on the clean #64 host without dragging in old eval-time contamination.
- A narrow clean port keeps the next-machine option alive without broad family drift.
- Keeping the neural exact path intact gives a direct local control against the mixed exact read.

### Compliance & Risk Status
- `#1082` remains open, so Family B must stay on the strictest legal interpretation.
- First packet must exclude TTT, two-pass, hindsight rescoring, normalization-bug surfaces, and extra side systems such as `swarm_agents.py` / `kg_data.py`.
- Main technical risk is runtime overhead from the mixer eval path; the packet should remain pre-launch until compile and dry-run are clean.

### Known Failures
- Old `pgolf` mixer surfaces are not directly reusable because they mix in excluded features.
- Family B is not already launch-ready from repo state alone.
- `#110` already showed that near-frontier quality without runtime health is insufficient as a next-machine default.

## [2026-03-30 11:15] Round 124

### Research Findings
- `#116` proved the strict-legal `Family B` surface is the current single-node main line:
  - `final_int6_causal_backoff_exact val_bpb:0.39421265`
  - `Total submission size int6+lzma: 15881542 bytes`
- The same run also exposed the dominant practical cost on this surface:
  - `final_int6_causal_backoff eval_time:1710133ms`
  - `TIMING:final_eval=1851.3s`
- The largest obvious waste in the current implementation is inside `eval_val_causal_backoff_mixer(...)`:
  - per-window score-first mixing loops over the same contiguous scored union range
  - `StrictCausalBackoffMixer` then recomputes the same context hashes again in `update(...)`

### Paradigm Assumptions
- `#124` should stay on the exact same strict-legal `Family B` surface:
  - score-first
  - backward-looking only
  - no TTT
  - no two-pass / hindsight / normalization-bug drift
- Under the new single-node budget, the highest-EV next move is runtime improvement on the already-winning Family B exact path, not reopening the broader `#1019` host line.
- The follow-up should preserve the mixer formula and env surface unless a knob change is strictly required.

### Frontier Snapshot
- `#116` is now the strongest local strict-legal read and beats the recent public `Family B` marker `#1094 = 0.3958`.
- `#120/#123` further reinforced that the broader `#1019` host family is runtime-negative under the current envelope.
- Current single-node strategy therefore collapses to `Family B` first.

### Comparable Methods
- `#116` is the direct control and must remain the comparison baseline.
- The acceptable follow-up should look like a semantics-preserving speedup or another very narrow legal delta on top of the same clean host.
- Broad family pivots or reopening A1 host/runtime tuning are out of scope for this round.

### Novelty-Relevant Findings
- The main actionable local novelty is now implementation-side:
  - score the contiguous batch union once
  - reuse the same per-order context/full-key computations for both mixing and post-score updates
- This keeps the legal surface unchanged while targeting the heaviest visible exact-tail cost.

### Compliance & Risk Status
- Compliance boundary is unchanged from `#116` and must stay strict-legal.
- Main risk is semantic drift from the accepted `#116` mixer; the follow-up should preserve score-then-update exactly.
- Because local Python lacks `numpy`, deep semantic A/B execution is not available in this workspace; validation will therefore rely on code inspection, `py_compile`, and clean launcher dry-run.

### Known Failures
- `#116` already proved that raw training-body runtime on this host family is not healthy.
- The `#116` exact-tail implementation is extraordinarily slow even though the final score is excellent.
- Reopening broader `#1019` host-runtime-negative lines would violate the new single-node mainline decision.

## [2026-03-30 12:20] Round 126

### Research Findings
- `#124` did not fail on infra or generic closeout; it failed inside the new batchwise optimization with:
  - `RuntimeError: Non-contiguous scored targets in causal backoff batch: targets=6208 probs=8192`
- The thrown guard proved the `#124` one-pass contiguous-union scorer had introduced a hidden correctness invariant that is false on the real Family B exact path.
- The missing `final_int6_causal_backoff_exact` / `TIMING:final_eval` markers are therefore genuinely absent evidence from `#124`, not lost logs.

### Paradigm Assumptions
- The repair must preserve `#116` semantics exactly:
  - score each window against the same pre-update mixer tables
  - then perform one post-batch update on the contiguous batch range
- The replacement should stay on the exact same strict-legal Family B surface as `#116/#124`.
- The repair should target the same exact-tail waste, but it must not rely on a deduplicated union of scored targets.

### Frontier Snapshot
- `#116 = 0.39421265` remains the governing Family B success packet.
- `#124` is a code-path failure of the attempted optimization, not a Family B quality failure.
- Under the single-node strategy, the next acceptable Family B packet must first re-establish semantic correctness.

### Comparable Methods
- `#116` is the semantic control: per-window score-first mixing plus one post-batch contiguous update.
- `#124` is the negative control: one-pass contiguous union with a hidden non-contiguity bug.
- The correct safe-repair target is between them:
  - keep `#116` semantics
  - keep the `#124` idea of reusing batch-range hashes
  - remove the broken contiguous-union assumption

### Novelty-Relevant Findings
- The safe optimization is to precompute the contiguous batch-update key range once and slice that cache for each per-window score.
- This still eliminates the duplicated hash rebuild between scoring and update, but it does not collapse overlapping scored windows into one deduplicated target list.
- That makes it a narrow implementation repair, not a family or legality change.

### Compliance & Risk Status
- Compliance boundary is unchanged: strict-legal Family B only.
- Main technical risk is still semantic drift from `#116`; the safe repair is acceptable only if code inspection, compile, and dry-run all point back to `#116` semantics.
- A deeper semantic A/B harness is still unavailable locally because this shell lacks `numpy`.

### Known Failures
- `#124` proved that a single contiguous-union scorer is unsafe on the real exact path.
- Missing Family B closeout markers from `#124` cannot be treated as hidden positive evidence.
- Any future speedup that again assumes scored-target contiguity across a batch is suspect until explicitly justified.

## [2026-03-30 13:40] Round 129

### Research Findings
- `#126` cleanly closed the strict-legal Family B path that `#124` broke:
  - `TIMING:final_eval=1813.1s`
  - `final_int6_causal_backoff_exact val_bpb:0.39442306`
- Relative to `#116`, `#126` already gave a small exact-tail runtime win without reopening the `#124` correctness failure mode:
  - `#116`: `final_int6_causal_backoff eval_time:1710133ms`, `TIMING:final_eval=1851.3s`
  - `#126`: `final_int6_causal_backoff eval_time:1671402ms`, `TIMING:final_eval=1813.1s`
- The remaining visible waste in `#126` is that each window still re-reads the same per-order table counts from `ctx_tables` / `full_tables` even though those tables are fixed for the whole pre-update batch.

### Paradigm Assumptions
- Keep-base should be `#126`, not `#116`:
  - it already preserves the repaired strict-legal closeout path
  - it already improved exact-tail timing slightly
  - it does not rely on the broken contiguous-union scorer from `#124`
- The next move should remain a bounded implementation-side speedup only:
  - no family drift
  - no legality drift
  - no change to score-then-update semantics

### Frontier Snapshot
- `#116` and `#126` both validate Family B as the live strict-legal champion family.
- `#126` is the more relevant base because it proved the repaired closeout path can finish cleanly.
- Under the single-node budget, the best-EV next move is still exact-tail runtime reduction on this same Family B surface.

### Comparable Methods
- `#116` is the semantic and quality baseline.
- `#126` is the repaired keep-base.
- The acceptable follow-up should only optimize the `#126` exact-tail implementation path, not alter training, model family, or the mixer formula.

### Novelty-Relevant Findings
- The safe bounded follow-up is to prefetch per-order table counts once for the contiguous batch cache, then slice those cached counts per window during score-first mixing.
- This removes repeated `ctx_tables/full_tables` indexing across windows while preserving:
  - the same batch key cache
  - the same per-window score order
  - the same single post-batch update
- Unlike `#124`, this does not collapse window scores into a single union probability array, so it does not reintroduce the non-contiguous-target failure mode.

### Compliance & Risk Status
- Compliance boundary remains strict-legal Family B only.
- Main risk is subtle semantic drift from `#126` if the count-prefetch cache is misaligned with the per-window slices.
- Local Python still lacks `numpy`, so validation remains limited to code inspection, `py_compile`, and clean launcher dry-run rather than a semantic A/B harness.

### Known Failures
- `#124` remains the explicit negative control for unsafe contiguous-union scoring.
- The broader pure-neural `#110/#120/#123` line remains runtime-negative and is out of scope for this round.
- Any follow-up that changes scored-window boundaries or mixer update order would violate the bounded speedup requirement for this round.
