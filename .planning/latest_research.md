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
