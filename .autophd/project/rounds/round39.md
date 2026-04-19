# Round 39

**Status:** executing
**Plan:** `.autophd/lab/plans/round-39-plan.md`
**Spec:** `.autophd/lab/specs/round-39-spec.md`
**Results:** `.autophd/lab/analysis/round-38-retro.md`

## Baseline

- **Commit:** `605e2a0`
- **BPP:** `1.07671973`
- **Base PR:** `#1493`

## Goal

Map the failure threshold of Heimdall's multi-GPU substrate and distinguish it
from launcher/image issues.

## Current Evidence

- AWS 1x current-image microbenchmark is healthy.
- Heimdall 1x current-image microbenchmark works but is slower.
- Heimdall 2x fails quickly.
- Heimdall 4x/8x can start distributed init but stall before useful benchmark output.
- AWS 8x remains quota-blocked.

## Stop Criteria

- Stop when the substrate fault domain is narrower than the model-code fault
  domain, or when AWS 8x quota returns and provides a clean counterexample.
