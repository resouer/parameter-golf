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
- AWS 2x distributed microbenchmark is healthy.
- Heimdall 2x is unhealthy.
- Heimdall 4x/8x can start distributed init but stall before useful benchmark output unless pinned to a better node.
- Heimdall node-level training split is now visible:
  - `node-ip-10-0-83-32` stays in the bad `~1.139` band
  - `node-ip-10-0-80-97` restores healthy training speed but still lacks a proven final sliding tail
- AWS 8x remains quota-blocked.

## Stop Criteria

- Stop when the substrate fault domain is narrower than the model-code fault
  domain, or when AWS 8x quota returns and provides a clean counterexample.
