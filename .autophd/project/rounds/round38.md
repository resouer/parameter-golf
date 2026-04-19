# Round 38

**Status:** completed
**Plan:** `.autophd/lab/plans/round-38-plan.md`
**Spec:** `.autophd/lab/specs/round-38-spec.md`
**Results:** `.autophd/lab/analysis/round-38-retro.md`

## Baseline

- **Commit:** `605e2a0`
- **BPP:** `1.07671973`
- **Base PR:** `#1493`

## Frontier Context

- **Merged SOTA / baseline to beat:** `1.0810` (`#1493`)
- **Recent invalidated rounds:** `round35`, `round36`, `round37`
- **Known traps to avoid:** fake training scripts, scorer inheritance bugs, packed-wrapper autodetect drift, stale node-group assumptions

## Base Reproduction

- **Base reproduction artifact:** `.autophd/lab/analysis/round-37-retro.md`
- **Metric basis:** merged `#1493` public logs vs current calibration runs
- **Known config assumptions:** frontier search is blocked until drift source is isolated

## Goal

Determine whether the drift is node-group specific or shared across the
available Heimdall nodes, and isolate whether launcher/runtime-image changes
can recover merged-`#1493` behavior.

## Evaluator

| Command | Value |
|---------|-------|
| Script | `evaluate.py` |
| Threshold | `1.0810` |
| Eval | `python3 evaluate.py --threshold 1.0810 --node-group <worker-node-group> --timeout 3600` |
| Pre-flight | `python3 evaluate.py --preflight --commit <baseline> --node-group <ng>` |
| 3-seed | `python3 evaluate.py --3seed --commit <sha> --node-group <ng>` |
| Stop | `python3 evaluate.py --stop <job-id>` |

## Workers

| Worker | Repo | Node Group | Branch Prefix | Source Ref | Source Repo | Source PR |
|--------|------|------------|---------------|------------|-------------|-----------|
| worker-1 | `~/code/parameter-golf-w87-pr1712` | `heimdall-dev` | `exp/round-38/w108` | `` | `` | `1493` |
| worker-2 | `~/code/parameter-golf-w89-pr171x-followup` | `heimdall-dev-ayxxjemt` | `exp/round-38/w109` | `` | `` | `1493` |
| worker-3 | `~/code/parameter-golf-w88-pr1711` | `heimdall-dev` | `exp/round-38/w110` | `` | `` | `` |

### Lane Assessments

**Worker-1:**
- Novelty: none; node-group calibration of merged `#1493`
- Compliance: highest
- Submittable: yes
- Lane thesis: if `heimdall-dev` is healthy, merged `#1493` sliding control should recover toward its public timing/score band
- Scoring: novelty=1, legality=5, infra=5, cost=4, score_upside=5
- Cheap falsification: one completed sliding-only control run

**Worker-2:**
- Novelty: none; paired control on current problematic node group
- Compliance: highest
- Submittable: yes
- Lane thesis: same surface on `heimdall-dev-ayxxjemt` tells us whether the bad behavior is node-group specific
- Scoring: novelty=1, legality=5, infra=5, cost=4, score_upside=5
- Cheap falsification: one completed sliding-only control run

**Worker-3:**
- Novelty: follow-up only
- Compliance: high
- Submittable: yes
- Lane thesis: hold for container-image follow-up if both nodes drift
- Scoring: novelty=1, legality=4, infra=4, cost=2, score_upside=3
- Cheap falsification: hold

## Experiment Steps

### Step 1: merged `#1493` sliding-only on `heimdall-dev`
- Description: run the control on the alternate Heimdall node group
- Dependencies: none
- Parallelizable: yes

### Step 2: merged `#1493` sliding-only on `heimdall-dev-ayxxjemt`
- Description: run the paired control on the currently problematic node group
- Dependencies: none
- Parallelizable: yes

### Step 3: decide next repair lane
- Description: choose node-group fix vs container-image fix
- Dependencies: steps 1-2
- Parallelizable: no

## Cross-Pollination Rules

- Do not launch W110 until W108 and W109 finish.

## Outcome

- Heimdall current-image and alt-image both stayed in the bad `~1.14x` band.
- AWS 8x positive control remained unavailable because of `InsufficientQuota`.
- AWS 1x microbenchmark completed immediately and showed healthy FA3 + matmul.
- Heimdall 1x microbenchmark also worked, but was slower and less stable.
- Heimdall 2x failed quickly; Heimdall 4x/8x distributed diagnostics could
  start but stalled before first useful benchmark output.

## Stop Criteria

- Stop the round once the node-group / substrate fault domain is clearer than
  the model-code fault domain.
