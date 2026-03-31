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

## [2026-03-30 13:40] Round 129

### Research Findings
- `#126` is the correct keep-base because it already repaired `#124` and closed the Family B causal-backoff path cleanly.
- It also delivered a small exact-tail timing win over `#116`, so the next move should stack on `#126` rather than reset to `#116`.
- The remaining bounded optimization target is repeated per-window table lookup work inside the exact-tail mixer.

### Decision
- Keep base = `#126` / `e8d6d047977a9a976a29153e38e7dc0827ad8952`.
- Apply one bounded follow-up only:
  - prefetch per-order `ctx_tables/full_tables` counts once for the contiguous batch cache
  - slice those cached counts per window during score-first mixing
  - keep the same post-batch update path as `#126`
- Do not change family, legality surface, scored-window boundaries, or update order in this round.

### Codex Review
- File scope remains `train_gpt.py` plus the mandatory `Round 129` planning-file updates.
- Success condition is:
  - `python3 -m py_compile train_gpt.py`
  - clean `run_lepton.py --dry-run --round 129 ...`
  - packet writeup must explain why `#126` is the keep-base and why the new count-prefetch path is safer than `#124` while still targeting exact-tail speed.

## [2026-03-30 14:25] Round 134

### Research Findings
- The parity audit now makes the env gap concrete: official validation targets the RunPod image `runpod/parameter-golf@sha256:394ad73416e110e306004171d712e5835c94e3a09032bed59fcd3c1a959a4f85`, while ordinary Lepton launches still default to the generic PyTorch image and must be labeled `env-parity-unverified`.
- `#126` is the correct keep-base for a parity confirmation lane because it is the last clean Family B closeout with visible causal-backoff exact markers; live `#129` is still an optimization probe and had not surfaced `final_int6_causal_backoff(_exact)` when this round was chosen.
- The queue is empty under the single-node budget, so the next highest-EV move is a bounded parity confirmation launch, not a new family or search pivot.

### Decision
- Keep base = `#126 / e8d6d047977a9a976a29153e38e7dc0827ad8952`.
- Do no new model-path or Family B algorithm change in this round.
- Apply only a minimal re-gate, then launch the clean `#126` surface under the official RunPod parity launcher mode and report:
  - job
  - job id
  - node group
  - direct state
  - parity image/label confirmation

### Codex Review
- Scope is launch-only on the already accepted `#126` surface plus the mandatory `Round 134` planning-file updates.
- Success condition is:
  - `python3 -m py_compile train_gpt.py`
  - clean parity dry-run resolving the official RunPod image + explicit parity label
  - launch/report with no family or legality drift

## [2026-03-30 14:35] Round 136

### Research Findings
- `#134` proved the first parity confirmation failure was launcher-side, not Family B method-side:
  - `fatal: destination path '/workspace/parameter-golf' already exists and is not an empty directory.`
- `#135` repaired that bounded launcher-path issue by changing parity-mode checkout to `/workspace/parameter-golf-parity` while keeping the same official RunPod image/label surface.
- The node is idle again, so the correct immediate next move is to rerun the same clean `#126` keep-base on the repaired parity launcher, not to change family or method.

### Decision
- Keep base = `#126 / e8d6d047977a9a976a29153e38e7dc0827ad8952`.
- Reuse the repaired parity launcher path:
  - same official RunPod digest image
  - same `runpod-template` / `runpod-template-targeted` surface
  - repaired checkout dir = `/workspace/parameter-golf-parity`
- Apply only a minimal re-gate, then relaunch and report:
  - job
  - job id
  - node group
  - direct state

### Codex Review
- Scope is launch-only on the already accepted `#126` surface plus the mandatory `Round 136` planning-file updates.
- Success condition is:
  - exact-branch compile check still passes on `#126`
  - parity dry-run resolves the repaired checkout dir `/workspace/parameter-golf-parity`
  - launch/report without new method or legality drift

## [2026-03-30 15:04] Round 142

### Research Findings
- `#136` proved the repaired parity launcher can carry the clean `#126` keep-base through real training and full control-plane completion.
- `#141` also proved `#136` was not a full Family B parity verdict, because the relaunch dropped the Family B env activation surface and therefore never entered `final_int6_causal_backoff(_exact)`.
- The next bounded move is not a new method or family change; it is the same clean `#126` keep-base under the same repaired parity path, but with the full Family B env surface restored explicitly.

### Decision
- Keep base = `#126 / e8d6d047977a9a976a29153e38e7dc0827ad8952`.
- Reuse the repaired parity launcher surface unchanged:
  - same official RunPod digest image
  - same `runpod-template` / `runpod-template-targeted` surface
  - same checkout dir `/workspace/parameter-golf-parity`
- Restore the full Family B env surface explicitly:
  - `USE_NGRAM_MIXER=1`
  - `NGRAM_ORDER=10`
  - `NGRAM_MIN_ORDER=2`
  - `NGRAM_BUCKETS=4194304`
  - `NGRAM_MIN_COUNT=1`
  - `ALPHA_BASE=0.20`
  - `ALPHA_RANGE=0.55`
  - `ALPHA_CENTER=3.0`
  - `NGRAM_BATCH_SEQS=128`
  - `TTT_ENABLED=0`
  - `TRIGRAM=0`
- Apply only a minimal re-gate, then relaunch and report:
  - job
  - job id
  - node group
  - direct state
  - confirmation that the full Family B env surface is present in the launch record

### Codex Review
- Scope is launch-only on the already accepted `#126` surface plus the mandatory `Round 142` planning-file updates.
- Success condition is:
  - exact-branch compile check still passes on `#126`
  - parity dry-run resolves the same repaired checkout dir `/workspace/parameter-golf-parity`
  - dry-run/launch record explicitly shows the full Family B env surface
  - launch/report without any new method or legality drift

## [2026-03-30 18:44] Round 151

### Research Findings
- `#147` and `#150` jointly showed that synchronized-authoritative reuse is no longer catastrophically wrong, but it still carries a small persistent semantic delta vs the `#142` base:
  - `#142`: `final_int6_causal_backoff_exact val_bpb:0.39344583`
  - `#147`: `0.39390852`
  - `#150`: `0.39386201`
- The accepted `#149` decision frame therefore says to stop treating synchronized reuse as a base-preserving repair path and switch next to the safer standalone-cache/state candidate on top of the same semantic base.
- The already-built standalone candidate from `#129` fits that boundary exactly:
  - base head is still `e8d6d047977a9a976a29153e38e7dc0827ad8952`
  - it only prefetches Family B count/state lookups inside the standalone scorer
  - it does not reuse sliding-pass control flow or distributed shared-state tricks

### Decision
- Revive the standalone Family B count-prefetch candidate as the next safer packet on top of semantic base `#142`.
- Keep the Family B formula unchanged.
- Keep the standalone causal-backoff evaluation path unchanged in structure; only prefetch count/state lookups that are already implied by the fixed batch key cache.

### Codex Review
- Scope is the pre-existing `train_gpt.py` patch on branch `autoresearch/exp-familyb-count-prefetch` plus the mandatory `Round 151` planning-file updates.
- Success condition is:
  - `python3 -m py_compile train_gpt.py`
  - clean parity-targeted `run_lepton.py --dry-run --round 151 ...`
  - packet writeup must state explicitly:
    - this is a switch away from synchronized reuse, not another repair on that family
    - semantic base remains `#142 / e8d6d04`
    - candidate is the safer standalone-cache/state branch, not a new formula/base change

## [2026-03-30 18:48] Round 152

### Research Findings
- `#151` was accepted as the next safer switch candidate after `#150`.
- The accepted branch is the pre-existing standalone count/state prefetch patch:
  - branch `autoresearch/exp-familyb-count-prefetch`
  - head `4dc08a7c53a3614bc1edc9b43ff9819eae1e7c10`
- The governing read is now:
  - semantic base stays `#142`
  - synchronized reuse is no longer the repair family
  - next live test should be this safer standalone-cache/state optimization on the corrected parity/full-Family-B surface

### Decision
- Treat `#151` as launch-authorized.
- Do only the minimal re-gate required for launch:
  - re-check `py_compile`
  - re-check the parity dry-run on the corrected full Family B env surface
  - if clean, launch on the single allowed AWS node and report identifiers

### Codex Review
- Scope is launch hygiene only; no new code delta beyond accepted head `4dc08a7`.
- Success condition is:
  - `python3 -m py_compile train_gpt.py`
  - clean `run_lepton.py --dry-run --round 152 --env USE_NGRAM_MIXER=1 ...`
  - launch report must include:
    - `job`
    - `job id`
    - `node group`
    - `direct state`
  - launch writeup must state that this is the safer standalone count/state prefetch candidate, not another synchronized-reuse lane

## [2026-03-30 19:53] Round 153

### Research Findings
- `#152` proved that the standalone count/state prefetch candidate can reach full strict-legal Family B closeout.
- But it did not clearly beat semantic base `#142`; it closed at `0.39363898` vs `0.39344583` and `1804.7s` vs `1790.8s`.
- The accepted next move is therefore one confirmation rerun to decide whether that residual semantic delta is noise or persistent.

### Decision
- Treat the accepted standalone count/state prefetch candidate as launch-authorized again for one confirmation rerun.
- Keep code head, parity surface, and full Family B env surface unchanged.
- Do only the minimal re-gate required for launch, then rerun and report identifiers.

### Codex Review
- Scope is launch hygiene only; no new code delta beyond accepted head `4dc08a7`.
- Success condition is:
  - `python3 -m py_compile train_gpt.py`
  - clean `run_lepton.py --dry-run --round 153 --env USE_NGRAM_MIXER=1 ...`
  - launch report must include:
    - `job`
    - `job id`
    - `node group`
    - `direct state`
  - launch writeup must state this is a confirmation rerun for noise-vs-persistence on `#152`, not a new method search

## [2026-03-30 20:54] Round 154

### Research Findings
- `#152/#153` confirm that standalone count/state prefetch is the current safe working line after the failed synchronized-reuse family.
- The remaining obvious cost center inside that safe line is the dense post-batch update path:
  - every order still allocates a full-bucket `np.bincount(..., minlength=4,194,304)` vector
  - that work is repeated batch after batch inside `final_int6_causal_backoff`
- A bounded next move is to keep the same standalone scorer semantics but replace those dense updates with sparse unique-key accumulation.

### Decision
- Build one bounded follow-up on top of the confirmed standalone-prefetch line:
  - keep `4dc08a7` semantics and control flow
  - patch only the post-batch update implementation in `update_from_cache(...)`
  - switch from dense full-bucket `np.bincount` updates to sparse per-batch unique-key accumulation
- Deliver patch + pre-launch packet only.
- No launch in this round.

### Codex Review
- Scope is:
  - `train_gpt.py`
  - mandatory `Round 154` planning-file updates
- Success condition is:
  - `python3 -m py_compile train_gpt.py`
  - clean parity-targeted `run_lepton.py --dry-run --round 154 ...`
  - packet writeup must state explicitly:
    - standalone count/state prefetch remains the keep-line
    - no Family B formula or control-flow change is introduced
    - the only delta is sparse post-batch update accumulation

## [2026-03-31 02:47] Round 163

### Research Findings
- Sparse-update `c6e77c4` remains the confirmed keep-line after the non-promoting `#159/#160` and `#161/#162` follow-ups.
- A remaining bounded hotspot on that keep-line is repeated candidate-index reconstruction in the cached scorer:
  - per-order `valid & ~mixed_mask`
  - followed by `np.nonzero(...)`
- Those valid positions are already determined by the cached target range, so we can cache them once and reuse them.

### Decision
- Build one bounded valid-index-cache follow-up on top of sparse-update keep-line `c6e77c4`.
- Patch only `train_gpt.py` plus mandatory `Round 163` planning-file updates.
- Keep:
  - semantic anchor `#142`
  - sparse-update keep-line behavior
  - full parity/full-Family-B launch surface
- No launch in this round.

### Codex Review
- Scope is:
  - `train_gpt.py`
  - mandatory `Round 163` planning-file updates
- Success condition is:
  - `python3 -m py_compile train_gpt.py`
  - clean parity/full-Family-B `run_lepton.py --dry-run --round 163 ...`
  - packet writeup must state explicitly:
    - keep-line remains `c6e77c4`
    - `508892d` is not promoted
    - the only delta is reusing per-order valid indices inside the cached scorer path
