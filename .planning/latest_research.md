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

## [2026-03-30 13:37] Round 129

### Research Findings
- `#126` has now established the clean post-repair Family B baseline:
  - `final_int6_causal_backoff_exact val_bpb:0.39442306`
  - `TIMING:final_eval=1813.1s`
- That means the next highest-EV move is no longer semantic repair, but exact-tail speed on the same winning strict-legal surface.
- The heaviest remaining waste is architectural duplication:
  - `final_int6_sliding_window` and `final_int6_causal_backoff` both run over the same windows
  - today the Family B closeout still pays for a second full model-forward sweep after the sliding exact pass

### Paradigm Assumptions
- Keep the strict-legal Family B surface unchanged:
  - score-first
  - backward-looking only
  - full-vocab normalized mix
  - no TTT
  - no two-pass / hindsight / normalization-bug drift
- Treat `#126` as the semantic control.
- Restrict this round to a bounded implementation-side change that can be gated off with one env flag.

### Frontier Snapshot
- `#116 = 0.39421265` remains the strongest prior Family B evidence.
- `#126 = 0.39442306` confirms the safe-repair line is still competitive and, critically, cleanly reaches causal-backoff closeout.
- The broader `#1019` host family remains runtime-negative and does not re-enter the single-node main budget from this round.

### Comparable Methods
- `#116` is the clean semantic control.
- `#124` is the negative control for unsafe exact-tail acceleration.
- `#126` is the current safe-repair base that already removed the broken contiguous-union assumption.
- The bounded follow-up here is to share the sliding pass with Family B scoring instead of scheduling a second standalone causal-backoff eval pass.

### Novelty-Relevant Findings
- The new packet does not invent a new Family B formula; its bounded novelty is implementation-side:
  - reuse the already-required sliding exact pass
  - feed the existing strict-legal causal-backoff scorer from those same logits / probabilities / entropies
  - preserve the old standalone path as a fallback behind `FAMILYB_REUSE_SLIDING_PASS=1`
- This is narrower than `#124` because it avoids a new scorer/update invariant and instead reuses the already accepted `#126` batch scorer.

### Compliance & Risk Status
- Compliance boundary is unchanged and still strict-legal.
- Main technical risk is not legality drift but semantic drift from `#116/#126` due to the new shared-pass orchestration.
- The packet should therefore stay explicitly no-launch until compile and launcher dry-run are clean.
- Local validation remains bounded: this shell still does not have a deeper semantic A/B harness for full proof of identity.

### Known Failures
- `#124` showed that exact-tail speedups can easily introduce hidden invariants that only fail at closeout time.
- Reusing the sliding pass changes evaluation orchestration and logging shape, so timing lines must be interpreted as shared-pass cost when the new flag is enabled.
- This round does not fix the broader pure-neural runtime envelope; it is only a Family B bounded follow-up.

## [2026-03-30 15:58] Round 143

### Research Findings
- `#142` completed as the corrected full-env parity rerun and reached the full Family B causal-backoff closeout surface:
  - `TIMING:final_eval=1790.8s`
  - `final_int6_causal_backoff_exact val_bpb:0.39344583`
  - `final_int6_causal_backoff eval_time:1656267ms`
- That means the governing keep-base is no longer just `#126` by ordinary Lepton evidence; it is now the same Family B surface with parity-targeted evidence that the env/image path can also close cleanly.
- Under this stronger base, the dominant remaining exact-tail cost is still the standalone Family B causal-backoff pass itself rather than training-body runtime.

### Paradigm Assumptions
- Keep the strict-legal Family B scoring semantics unchanged from `#142`.
- Keep the repaired parity launcher/env findings as evidence only; this round is still a code packet, not a launch.
- Restrict the implementation delta to reusing the already-required sliding exact pass when explicitly enabled by `FAMILYB_REUSE_SLIDING_PASS=1`.

### Frontier Snapshot
- `#142` is now the parity-confirmed keep-base for strict-legal Family B.
- The queue is empty again under the single-node budget after the clean parity closeout.
- The highest-EV next move is to reduce the remaining exact-tail Family B eval cost without reopening env-parity or legality questions.

### Comparable Methods
- `#126` remains the repaired semantic control.
- `#142` is the parity-confirmed control and therefore the correct governing base for the next packet.
- The chosen bounded follow-up is the already-prepared reuse-sliding-pass path, because it attacks the largest remaining exact-tail duplicate work directly.

### Novelty-Relevant Findings
- The bounded novelty remains orchestration-only:
  - reuse the sliding exact pass that already computes the needed logits/probabilities/entropies
  - feed those same windows into the unchanged strict-legal Family B scorer
  - keep the standalone causal-backoff eval as the fallback when the reuse flag is off
- Relative to `#142`, this is the narrowest next move that targets the still-dominant `final_int6_causal_backoff` time without changing the Family B formula.

### Compliance & Risk Status
- Compliance boundary remains strict-legal Family B only.
- Main risk is semantic drift from `#142` due to shared-pass orchestration or timing/log interpretation.
- Local validation remains bounded to code inspection, `py_compile`, and launcher dry-run; this shell still lacks a deeper semantic A/B harness.

### Known Failures
- `#124` remains the negative control for unsafe exact-tail acceleration.
- `#136` remains the negative control for parity launch-spec omission; it is no longer the governing explanation after `#142`.
- Shared-pass timing must still be interpreted as shared exact-tail cost when `FAMILYB_REUSE_SLIDING_PASS=1` is enabled.
