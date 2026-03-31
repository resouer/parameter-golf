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

## [2026-03-30 14:25] Round 134

### Research Findings
- The parity audit is now fully concrete:
  - official validation template image = `runpod/parameter-golf@sha256:394ad73416e110e306004171d712e5835c94e3a09032bed59fcd3c1a959a4f85`
  - current ordinary Lepton image = `pytorch/pytorch@sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6`
- `#126` is the cleanest Family B confirmation surface because it already closed the causal-backoff exact path cleanly:
  - `final_int6_causal_backoff_exact val_bpb:0.39442306`
  - `TIMING:final_eval=1813.1s`
- `#129` is not the keep-base for parity confirmation yet because it had not produced decisive Family B causal-backoff closeout markers when this round was chosen.

### Paradigm Assumptions
- This round is a bounded parity-confirmation lane, not a new family or search pivot.
- The model/code surface should stay exactly on accepted `#126`.
- The launcher should switch only the env/image/bootstrap surface to the official RunPod parity mode and make the label explicit.

### Frontier Snapshot
- `#116/#126` keep Family B as the governing strict-legal champion family.
- The live queue is empty under the single-node budget at this decision point.
- The highest-EV next question is no longer family choice; it is whether the clean `#126` evidence survives under the official-targeted env surface.

### Comparable Methods
- `#126` is the direct semantic control for this round.
- Ordinary Lepton launches remain the baseline unverified env path and must stay labeled `env-parity-unverified`.
- The new parity-mode launcher path is the only acceptable env change in this round.

### Novelty-Relevant Findings
- The new information gain here is environmental, not algorithmic:
  - use the official RunPod image target
  - preserve the clean `#126` Family B base
  - make parity labeling explicit in the launcher/job env
- This gives a bounded confirmation run without reopening model-family search.

### Compliance & Risk Status
- Compliance boundary remains strict-legal Family B only.
- Main risk is env-stack mismatch rather than method drift.
- Until parity mode is actually used, the correct evidence label remains `env-parity-unverified`.

### Known Failures
- The generic Lepton image is not parity-equivalent to the official RunPod image.
- `#124` remains the explicit negative control for unsafe Family B exact-tail optimization.
- Using `#129` as the parity keep-base right now would mix env confirmation with unresolved optimization-side uncertainty.

## [2026-03-30 14:35] Round 136

### Research Findings
- `#134` did launch under parity mode, but it failed before live evaluation with a launcher-path collision:
  - `fatal: destination path '/workspace/parameter-golf' already exists and is not an empty directory.`
- `#135` repaired that bounded parity-bootstrap issue by moving parity-mode checkout to `/workspace/parameter-golf-parity`.
- The repair leaves the intended env surface unchanged:
  - same official RunPod digest image
  - same `runpod-template` mode
  - same `runpod-template-targeted` label

### Paradigm Assumptions
- This round remains a bounded parity-confirmation rerun, not a new family or method pivot.
- The model/code surface must stay exactly on clean keep-base `#126`.
- The only meaningful changed variable from `#134` to `#136` is the repaired launcher checkout path.

### Frontier Snapshot
- `#126` remains the best clean Family B control for parity confirmation.
- The node is idle again after the bounded launcher failure on `#134`.
- The highest-EV next question is whether the repaired parity launcher now allows the clean `#126` surface to enter real evaluation.

### Comparable Methods
- `#134` is now the negative control for the old parity-bootstrap path.
- `#135` is the launcher-only repair lane that fixed the checkout-dir collision.
- `#136` should therefore be read as `#126` rerun under repaired parity mode, not as a new Family B experiment.

### Novelty-Relevant Findings
- The new information gain is narrow and infrastructural:
  - keep the same Family B method surface
  - reuse the same official parity image/label
  - verify that the repaired checkout path eliminates the `/workspace/parameter-golf` collision
- That keeps the lane attributable to the launcher fix rather than to any method drift.

### Compliance & Risk Status
- Compliance boundary remains strict-legal Family B only.
- Main risk is still env/bootstrap behavior on the official image surface, not model semantics.
- Because the launcher repair is bounded, the rerun should be read as a parity-bootstrap confirmation attempt rather than a new method result.

### Known Failures
- `#134` cannot be read as a parity-quality verdict because it died before evaluation body.
- The old parity checkout target `/workspace/parameter-golf` is now a known-bad path on the official image surface.
- Reopening `#129` or changing Family B method surfaces would confound the launcher repair confirmation.

## [2026-03-30 15:04] Round 142

### Research Findings
- `#136` established that the repaired parity launcher path is now operational:
  - same official RunPod digest image
  - same repaired checkout dir `/workspace/parameter-golf-parity`
  - real training body and exact-tail closeout both ran to control-plane completion
- `#141` recovered the missing-boundary root cause:
  - `#136` launched with `job[0] env=(none)`
  - `USE_NGRAM_MIXER` defaults to `0`
  - `final_int6_causal_backoff(_exact)` only runs when the Family B env surface is explicitly enabled
- So the next highest-EV question is no longer parity launcher health; it is whether the full Family B env surface survives cleanly on the same repaired parity path.

### Paradigm Assumptions
- This round remains a bounded parity-confirmation rerun, not a new family or method pivot.
- The model/code surface must stay exactly on clean keep-base `#126`.
- The only changed variable from `#136` to this round is the restored Family B env surface.

### Frontier Snapshot
- `#126` remains the cleanest accepted Family B keep-base.
- `#136` gives real parity-side runtime evidence, but it is incomplete as a Family B verdict because causal-backoff was not activated.
- The node is free again, so the correct next move is an immediate corrected parity rerun rather than a search pivot.

### Comparable Methods
- `#126` is the known-good Family B closeout control under the ordinary Lepton image.
- `#136` is the negative control for a parity rerun that dropped the env surface.
- This round is the corrected parity counterpart: same code, same image path, same launcher repair, but with the full Family B env surface restored.

### Novelty-Relevant Findings
- The new information gain is narrow and attributable:
  - same keep-base code
  - same parity image/bootstrap surface
  - explicit restoration of the Family B env activation surface
- That keeps the round interpretable as a missing-closeout-surface correction, not a new algorithmic result.

### Compliance & Risk Status
- Compliance boundary remains strict-legal Family B only.
- Main risk is no longer parity bootstrap; it is accidental env drift or an incomplete env restore.
- The launch record must therefore explicitly show the full Family B env surface, or the run cannot be read as a valid parity verdict.

### Known Failures
- `#134` remains the negative control for the old parity checkout-dir collision.
- `#136` remains the negative control for a parity rerun that omitted the Family B env surface.
- Any relaunch that changes the keep-base code or the parity image/bootstrap at the same time would confound the correction.

## [2026-03-30 18:44] Round 151

### Research Findings
- `#147` and `#150` have now split the synchronized reuse family from the semantic control cleanly:
  - runtime-side win reproduced (`506.9s`, `502.8s`)
  - semantic delta vs `#142` also reproduced (`0.39390852`, `0.39386201` vs `0.39344583`)
- That means synchronized reuse is no longer a noise question; it is a persistent speed/semantics tradeoff relative to `#142`.
- The accepted `#149` ranking explicitly puts the safer standalone-cache/state optimization next once that tradeoff becomes persistent.

### Paradigm Assumptions
- Keep `#142 / e8d6d04` as the semantic comparison anchor.
- Stop treating synchronized reuse as a base-preserving repair family.
- Switch next to the standalone cache/state idea that keeps the standalone causal-backoff control flow intact.

### Frontier Snapshot
- `#142` remains the semantic control.
- `#147/#150` now jointly characterize the synchronized reuse family:
  - reproducible runtime win
  - reproducible small semantic drift
- The next bounded move is therefore not another distributed/reuse repair, but the safer standalone cache/state branch already prototyped in `#129`.

### Comparable Methods
- `#142`: control, fully trusted semantics, slow exact tail.
- `#147/#150`: synchronized reuse, fast exact tail, persistent small semantic drift.
- `#151`: standalone count/state prefetch on the same semantic base, intended to attack exact-tail cost without changing evaluation control flow.

### Novelty-Relevant Findings
- The revived candidate does not change the Family B formula or base.
- It only prefetches table-count/state lookups once for the fixed batch cache and then slices those prefetched counts per window.
- That makes it strictly safer than shared-pass reuse from a semantic-control standpoint.

### Compliance & Risk Status
- Compliance boundary remains strict-legal Family B only.
- Main risk is now implementation payoff: this candidate may deliver less speedup than synchronized reuse.
- But it is ranked next because it removes the distributed/state-sharing failure mode from the optimization family entirely.

### Known Failures
- Do not open another confirmation rerun on synchronized reuse.
- Do not reintroduce rank-sharded or shared-pass mixer-state tricks under the label of “safer cache/state.”
- If this standalone candidate still fails to preserve `#142`, the next move should follow the `#149` switch ladder rather than more local patching on the same idea.

## [2026-03-30 18:48] Round 152

### Research Findings
- `#151` has already been accepted as the next safer switch candidate.
- There is no new method delta in this round; the information gain is whether the accepted standalone count/state prefetch candidate can run cleanly on the same corrected parity/full-Family-B surface that `#147/#150` used.
- That keeps the next live result attributable to:
  - same semantic base `#142`
  - same parity launcher/image/env surface
  - different optimization family: standalone count/state prefetch instead of synchronized reuse

### Paradigm Assumptions
- Keep `#142 / e8d6d04` as the semantic comparison anchor.
- Keep the accepted `4dc08a7` code surface unchanged.
- Read this as a family switch inside the same semantic-base comparison frame, not as a new formula/base search.

### Frontier Snapshot
- `#142` remains the control.
- `#147/#150` have retired synchronized reuse as a base-preserving repair family.
- `#151/#152` now advance the next ranked safer candidate from the accepted switch ladder.

### Comparable Methods
- `#142`: control, fully trusted semantics, slow exact tail.
- `#147/#150`: synchronized reuse, fast exact tail, persistent small semantic drift.
- `#152`: standalone count/state prefetch on the same semantic base, intended to chase runtime without inherited shared-state failure modes.

### Novelty-Relevant Findings
- The live question is now cleanly isolated:
  - does standalone count/state prefetch buy runtime on the parity/full-Family-B surface
  - without reproducing the persistent semantic drift seen in synchronized reuse

### Compliance & Risk Status
- Compliance boundary remains strict-legal Family B only.
- Main risk is implementation payoff and possible smaller speed gain than synchronized reuse.
- But the live experiment is substantially safer semantically because it stays inside the standalone scorer path.

### Known Failures
- Do not keep treating synchronized reuse as an active repair family.
- Do not change semantic base or parity surface in the same round as this family switch.
- If this safer candidate still fails to preserve `#142`, escalate to the next `#149` switch-base ladder rather than mixing families again.

## [2026-03-30 19:53] Round 153

### Research Findings
- `#152` did reach full strict-legal Family B closeout on the safer standalone count/state prefetch path.
- But versus semantic base `#142`, it still showed a small residual semantic delta and no clear tail-time win:
  - `#142`: `final_int6_causal_backoff_exact val_bpb:0.39344583`, `TIMING:final_eval=1790.8s`
  - `#152`: `final_int6_causal_backoff_exact val_bpb:0.39363898`, `TIMING:final_eval=1804.7s`
- The next information gain is therefore not a new method change.
- It is whether that residual delta is just run-to-run noise or a persistent property of the standalone-prefetch candidate.

### Paradigm Assumptions
- Keep `#142 / e8d6d04` as the semantic comparison anchor.
- Keep the accepted `4dc08a7` code surface unchanged.
- Keep the same corrected parity/full-Family-B launch surface unchanged.
- Read this round as a confirmation rerun, not a new optimization search.

### Frontier Snapshot
- `#142` remains the semantic control.
- `#147/#150` confirmed synchronized reuse is a persistent speed/semantics tradeoff and is no longer the active repair family.
- `#152/#153` now test whether the safer standalone-prefetch family has a persistent residual delta of its own.

### Comparable Methods
- `#142`: control, fully trusted semantics, slow exact tail.
- `#147/#150`: synchronized reuse, fast exact tail, persistent small semantic drift.
- `#152`: standalone count/state prefetch, full closeout, small residual delta vs `#142`.
- `#153`: confirmation rerun on the same standalone-prefetch candidate.

### Novelty-Relevant Findings
- No new novelty is introduced in this round.
- The information gain is purely statistical/diagnostic:
  - does `4dc08a7` reproduce its small delta vs `#142`
  - or does it collapse back toward the `#142` value on rerun

### Compliance & Risk Status
- Compliance boundary remains strict-legal Family B only.
- Main risk is wasted cycle if the residual delta is already effectively established.
- But this is still the cheapest clean way to decide noise vs persistent drift before abandoning or escalating the family.

### Known Failures
- Do not change code, semantic base, or parity surface in this round.
- Do not reopen synchronized reuse for another repair attempt here.
- If `#153` reproduces the same residual delta vs `#142`, treat that as persistence and move back to the next `#149` switch ladder.

## [2026-03-30 20:54] Round 154

### Research Findings
- `#152/#153` jointly establish standalone count/state prefetch as the current safe working line:
  - full strict-legal Family B closeout lands cleanly
  - residual semantic delta vs `#142` now reads like noise, not persistent drift
- But the exact tail still spends most of its time in the standalone causal-backoff closeout:
  - `#153 final_int6_causal_backoff eval_time = 1655741ms`
  - this remains far larger than the sliding/roundtrip exact subpasses
- On the confirmed `4dc08a7` safe line, the post-batch update still allocates dense full-bucket `np.bincount(..., minlength=4,194,304)` vectors for every order.

### Paradigm Assumptions
- Keep `4dc08a7` as the current safe working line after `#152/#153`.
- Keep the Family B formula, scoring order, and standalone control flow unchanged.
- Only target the post-batch table-update implementation detail inside the standalone scorer.

### Frontier Snapshot
- `#142` remains the semantic control.
- `#152/#153` have now promoted standalone-prefetch from “candidate” to current safe working line.
- The next bounded move should attack exact-tail cost inside that same safe line, not reopen shared-state or base-switch questions.

### Comparable Methods
- `#142`: clean semantic control, slow exact tail.
- `#152/#153`: safe standalone-prefetch line, semantics consistent with `#142`, but still heavy exact-tail cost.
- `#154`: sparse post-batch table updates inside the same standalone scorer path.

### Novelty-Relevant Findings
- This follow-up does not change what gets counted or when it gets counted.
- It only changes how the batch update is accumulated:
  - from dense full-bucket `np.bincount(..., minlength=...)`
  - to sparse unique-key accumulation on only the keys touched by the batch

### Compliance & Risk Status
- Compliance boundary remains strict-legal Family B only.
- Main risk is implementation correctness in the accumulator path, not semantic policy drift.
- Risk is bounded because the change stays entirely inside the already-confirmed standalone scorer/update path.

### Known Failures
- Do not reopen synchronized/shared-pass reuse.
- Do not change semantic base, parity surface, or Family B formula in this round.
- If this sparse-update follow-up still cannot improve exact-tail behavior, escalate back to the next `#149` ladder rather than layering unbounded local tweaks.

## [2026-03-31 02:47] Round 163

### Research Findings
- `#159/#160` showed that valid-count-cache was a clean bounded follow-up, but not a clear promotion over sparse-update keep-line `c6e77c4`.
- `#161/#162` showed the same for alpha-cache: clean closeout, tiny timing movement, but no semantic/submission promotion over `#156`.
- Keep-line therefore remains sparse-update `c6e77c4d6e449273a3c8c9ff2510e0ccfb6bfeea`.
- Inside the confirmed keep-line scorer, we still rebuild per-order candidate positions from full-length boolean arrays on every score pass:
  - `valid & ~mixed_mask`
  - `np.nonzero(...)`
- That candidate selection work is repeated in both `mix_target_probs_cached(...)` and `mix_target_probs_count_cached(...)`, even though the base valid positions are already fixed by the cached target range.

### Paradigm Assumptions
- Keep semantic anchor at `#142`.
- Keep sparse-update `c6e77c4` as the confirmed keep-line.
- Keep Family B formula, scoring order, window semantics, and single post-batch update semantics unchanged.
- Only optimize how per-order valid candidate indices are reused inside the existing cached scorer path.

### Frontier Snapshot
- `#156` remains the keep-line result to beat:
  - submission `1.11238486`
  - exact `0.39332185`
  - `TIMING:final_eval=1178.3s`
- `#159/#160` and `#162` are clean but non-promoting bounded follow-ups.
- The next packet should therefore stay bounded and behavior-preserving on top of `c6e77c4`, not reopen base-promotion or parity-surface questions.

### Comparable Methods
- `#156`: sparse-update keep-line, current safe working line.
- `#159/#160`: valid-count-cache follow-up, mixed confirmation pair, not promoted.
- `#162`: alpha-cache follow-up, clean but non-promoting.
- `#163`: valid-index-cache follow-up, reusing per-order valid positions instead of recomputing them from full boolean masks every pass.

### Novelty-Relevant Findings
- This follow-up does not change:
  - which positions are valid for each order
  - how probabilities are mixed
  - when post-batch updates happen
- It only caches and reuses the already-determined valid indices for each order so the scorer stops rebuilding the same candidate set from full-length boolean masks each pass.

### Compliance & Risk Status
- Compliance boundary remains strict-legal Family B only.
- Main risk is index-slicing correctness when cutting a fixed batch cache down to per-window caches.
- Risk is bounded because the change stays inside the existing cached scorer path and leaves formula/control flow intact.

### Known Failures
- Do not promote `508892d` as keep-line.
- Do not reintroduce valid-count-cache or alpha-cache assumptions into the base packet.
- Do not change Family B formula, order traversal, or update timing in this round.
