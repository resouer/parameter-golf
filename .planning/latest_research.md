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
