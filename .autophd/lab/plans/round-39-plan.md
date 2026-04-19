# Round 39 Plan

## Direction

Round 39 is a pure infrastructure / substrate round. It inherits the merged
`#1493` calibration context from round38 but stops treating model code as the
leading suspect.

## Priority Lanes

### 1. Heimdall scale ladder

- `1x` current-image microbenchmark
- `2x` current-image distributed microbenchmark
- `4x` current-image distributed microbenchmark
- `8x` current-image distributed microbenchmark

Goal:
- identify the smallest scale where the substrate becomes unstable

### 2. AWS control ladder

- `1x` current-image microbenchmark
- `8x` current-image calibration when quota appears

Goal:
- keep a clean control surface ready the moment quota returns

### 3. Health gate hardening

- fail fast on explicit `InsufficientQuota`
- record node-group health snapshot before launch
- avoid wasting long queue time on known-bad capacity states

## Execution Rules

- Prefer cheap diagnostics over submission-style training.
- Stop any distributed diagnostic that reaches rank0 setup but fails to emit
  benchmark lines in a reasonable window.
- Keep launcher fixes unified across main + worker evaluators.

## Stop Criteria

- Stop once the fault domain is localized to a substrate tier (`1x`, `2x+`,
  `8x`) or AWS quota returns and gives a clean 8x counterexample.
