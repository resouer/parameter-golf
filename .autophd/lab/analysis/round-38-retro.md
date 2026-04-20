# Round 38 Retro

## Current State

Round 38 has already answered the original "node-group only" question
partially, and it uncovered two stronger infrastructure facts:

1. Heimdall itself is currently degraded and should not be treated as a clean
   calibration surface.
2. The alt-image probe failures were initially confounded by our own launcher,
   not by the model code.

## Live Findings

- The intended AWS vs Heimdall node-group comparison was initially misconfigured:
  - `heimdall-dev` resolves to node group ID `heimdall-dev-ayxxjemt`
  - so the first pair of jobs were not actually cross-node-group
- Direct node inventory now shows the active Heimdall group is itself degraded:
  - `available = 0/3`
  - `unhealthy = True`
  - `diskpressure = True`
  - `initializing = True`
  - concrete states include:
    - one `NotReady, Unhealthy, Idle` node
    - one `Ready, DiskPressure, Initializing, Idle` node
    - two `Ready,Healthy, Used` nodes
- A real AWS-vs-Heimdall split was then launched:
  - `W108` → `aws-iad-leptondev-001`
  - `W109` → `heimdall-dev-ayxxjemt`
- AWS side never produced a usable control run:
  - `W108 r4` spent the full 3600s in `Queueing`
  - final reason remained `InsufficientQuota`
  - so round38 ended without a clean AWS positive-control measurement
- Separate alt-image probe on Heimdall (`runpod/pytorch:1.0.3-cu1290-torch280-ubuntu2204`)
  is active as an image-level isolation lane after the FA3 hypothesis weakened.
- A lighter-weight 1xH100 scheduler diagnostic was then launched on both groups:
  - AWS 1x completed immediately and produced usable FA3/matmul logs
  - Heimdall 1x initially stalled at `Starting`, but a live-mirrored rerun did complete
    and produced usable FA3/matmul logs too
- A follow-up 8x distributed diagnostic on Heimdall current-image started cleanly,
  printed rank0 setup (`WORLD_SIZE=8`), then stalled before emitting the first
  `MATMUL_SEC` / `ALLREDUCE_SEC_MEAN` lines and had to be manually stopped.
- Additional scale probes on Heimdall narrowed the failure band further:
  - a fresh `2x` distributed diagnostic failed quickly without ever producing
    benchmark output; the live mirror only captured an immediate
    `Connection stopped.`
  - a fresh `4x` distributed diagnostic reached rank0 setup (`WORLD_SIZE=4`)
    but also stalled before emitting `MATMUL_SEC` / `ALLREDUCE_SEC_MEAN`
    lines; the live mirror ended right after rank0 bootstrap output
- A stable runfile-based `2x` distributed diagnostic on AWS completed cleanly:
  - `python_exec = /usr/bin/python3`
  - `WORLD_SIZE = 2`
  - `MATMUL_SEC = 0.080145`
  - `ALLREDUCE_SEC_MEAN = 0.0003` for `64 MiB`
- The matching stable runfile-based `2x` diagnostic on Heimdall did not even
  leave `Starting` over the same observation window.

## Confirmed Runtime Findings

- Current-image Heimdall control (`W109 r3`) reproduces the bad band even with
  explicit FA3 and the merged `#1493` sliding-only surface:
  - `qk_gain_init = 5.25`
  - `ttt_enabled = False`
  - `train_shards = 128`
  - `500/20000 = 2.4m`
  - `1000/20000 = 6.0m`
  - `quantized_sliding_window = 1.14186327`
  - `bytes_total = 16,003,807`
- This means the drift is present before TTT and is not limited to the open PR
  frontier.

## FA3 Hypothesis Status

- A direct environment probe on the current default image completed and showed:
  - `torch_version = 2.9.1+cu128`
  - `cuda_available = True`
  - `cuda_device_count = 8`
  - `flash_attn_3_version = unknown`
  - `fa3_call_ok ... elapsed_sec = 0.8682`
- We also observed current-image training controls logging:
  - `Requirement already satisfied: flash_attn_3 ...`
- So "missing FA3" is no longer a strong primary-cause hypothesis.
- The AWS 1x diagnostic further confirms the current image itself is healthy on
  a clean scheduler path:
  - `python_exec = /usr/bin/python3`
  - `torch = 2.9.1+cu128`
  - `FA3_IMPORT ok`
  - `FA3_CALL_OK elapsed_sec = 0.796484`
  - `MATMUL_SEC = 0.082915`
- Heimdall 1x on the same image/diagnostic is not catastrophically broken:
  - `FA3_CALL_OK elapsed_sec = 0.783914`
  - `MATMUL_SEC = 0.11914`
- Heimdall 1x on the same image/diagnostic is not catastrophically broken:
  - `python_exec = /usr/bin/python3`
  - `torch = 2.9.1+cu128`
  - `FA3_IMPORT ok`
- So the simplest "FA3/import path broken on Heimdall" story is false. The
  remaining gap is more likely tied to multi-GPU runtime state, scheduler
  health, or other 8x-only effects.

## Alt-Image Probe Status

- The alternate-image training probes on
  `runpod/pytorch:1.0.3-cu1290-torch280-ubuntu2204` are currently inconclusive
  as performance evidence.
- They do successfully install the requested FA3 wheel:
  - `flash_attn_3-3.0.0+20260316.cu129torch280...`
- But the launcher then installs `torch-2.11.0` from `requirements.txt`,
  replacing the image-owned runtime stack before training starts.
- The run then dies during PyTorch startup (`torch/__init__.py` global CUDA
  dependency load), so those alt-image failures are harness-induced rather than
  model-induced.

## Alt-Image Launcher Repair Findings

- A dedicated interpreter probe on the alt image established:
  - `python = /usr/local/bin/python`
  - `python3 = /usr/bin/python3`
  - `torchrun = /usr/local/bin/torchrun`
  - `python` can import `torch 2.8.0+cu129`
  - `python3` cannot import `torch`
- This explains the earlier probe failures:
  - our launcher used `python3 -m pip ...` and later `python3 -m torch.distributed.run`
  - but the image-owned PyTorch stack lives under `/usr/local/bin/python`
- After fixing the launcher to auto-select the interpreter that can import
  `torch`, the alt-image probe (`W109 r9`) finally entered real training.
- Fresh performance evidence from that repaired probe:
  - `python_exec = /usr/local/bin/python`
  - `flash_attn_3` installs successfully
  - `500/20000 train_time = 2.3m`
  - `tok/s = 2,877,277`
  - `1000/20000 train_time = 5.7m`
  - `1500/20000 train_time = 9.8m`
  - `stopping_early step = 1504`
  - `pre-quantization post-ema val_bpb = 1.14964582`
  - `bytes_total = 16,002,639`
- So the alt image is no longer blocked on launcher bugs, but it still lands in
  an even worse drift band than the current image.

## Practical Conclusion

- Frontier search should stay paused.
- Current strongest evidence points to a mixed infra problem:
  - degraded Heimdall node health
  - plus launcher/runtime ownership bugs around environment composition
- The 1x diagnostics strengthen the scheduler interpretation:
  - AWS can run the current image and FA3 microbenchmark immediately
  - Heimdall can run the same 1x microbenchmark too, but startup/log behavior is
    less stable and matmul is somewhat slower
- The 8x distributed diagnostic sharpens that further:
  - Heimdall can start an 8x `torch.distributed` job
  - but it can hang before even finishing the first simple local matmul /
    allreduce benchmark block
  - so the remaining fault domain looks increasingly 8x-specific
- The 2x/4x reruns suggest the break is not only at the full 8x scale:
  - `1x` microbenchmarks work on both AWS and Heimdall
  - `2x` is healthy on AWS but already unhealthy on Heimdall
  - `4x` and `8x` can reach distributed init but stall before the first useful
    benchmark result
  - so the fault domain now looks like Heimdall multi-GPU / NCCL / scheduler
    substrate rather than model code
- Node-pinned submission-style controls narrow the fault domain even further:
  - `node-ip-10-0-83-32` (`node-B`) stays in the bad training band:
    - `500/20000 = 2.3m`
    - `1000/20000 = 5.7m`
    - `quantized_sliding_window = 1.13941579`
  - `node-ip-10-0-80-97` (`node-A`) restores healthy training speed:
    - `500/20000 = 0.9m`
    - `1000/20000 = 1.7m`
    - `1500/20000 = 2.6m`
    - `4000/20000 val_bpb = 1.1109`
    - `pre-quantization post-ema = 1.08767090`
    - `quantized = 1.09884460`
  - A final-gated rerun on `node-A` still did not produce a
    `quantized_sliding_window` tail, so it proves training/quantized recovery
    but does not yet prove full final-sliding recovery
- The alt-image path is now a valid probe surface, but it has not restored the
  expected speed band; simply switching to `cu129/torch280` is not sufficient.
- The next meaningful probe is not "new model idea", but:
  - continue refusing or deprioritizing launches when node-group health is degraded
  - recover a clean AWS calibration lane once quota returns
  - or pivot to explicit 8x distributed / NCCL evidence instead of more model runs
