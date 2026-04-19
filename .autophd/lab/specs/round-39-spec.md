# Round 39 Spec

## Objective

Turn the round38 drift findings into a focused substrate diagnosis round.

The new question is not "which model is best", but:

**Where does Heimdall's multi-GPU runtime path break relative to AWS and 1x
healthy baselines?**

## Hypothesis

- The current image and FA3 stack are not sufficient explanations.
- The strongest remaining fault domain is Heimdall multi-GPU substrate:
  scheduler, NCCL path, or unhealthy node state that only shows up at `2x+`.

## Scope

- No new model ideas.
- No frontier replays.
- Only calibration / microbenchmark / health evidence.

## Required Evidence

1. 1x AWS current-image healthy baseline
2. 1x Heimdall current-image baseline
3. 2x / 4x / 8x Heimdall distributed probes
4. Queue/quota health signals for AWS 8x
5. Node-health snapshots before launch

## Deliverable

A round retro that can justify:

- whether the fault is `1x`, `2x+`, or `8x-only`
- whether AWS is blocked by quota only or by deeper issues
- whether the outer loop should wait for infra repair before frontier search resumes
