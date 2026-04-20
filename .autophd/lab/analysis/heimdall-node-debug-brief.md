# Heimdall Node Debug Brief

## Goal

Explain why Heimdall node-level behavior diverges under the same merged `#1493`
sliding-only control.

## Suspected Bad Node

- `node-ip-10-0-83-32`

## Suspected Better Node

- `node-ip-10-0-80-97`

## Key Evidence

### Same training surface, different node, very different behavior

Common surface on both nodes:
- merged `#1493` sliding-only control
- `seed = 1337`
- `qk_gain_init = 5.25`
- `sliding_window_enabled = True`
- `ttt_enabled = False`
- `train_shards = 128`
- same image: `runpod/parameter-golf@sha256:394ad...`

Bad node (`10.0.83.32`):
- `500/20000 = 2.3m`
- `1000/20000 = 5.7m`
- `quantized_sliding_window = 1.13941579`
- `bytes_total = 16,004,041`

Better node (`10.0.80.97`):
- `500/20000 = 0.9m`
- `1000/20000 = 1.7m`
- `1500/20000 = 2.6m`
- `4000/20000 val_bpb = 1.1109`
- `pre-quantization post-ema = 1.08767090`
- `quantized = 1.09884460`
- `bytes_total = 15,986,756`
- final-gated rerun still did not emit `quantized_sliding_window`

### Scale ladder

Healthy controls:
- AWS `1x`: healthy
- AWS `2x`: healthy

Heimdall:
- `1x`: works, but slower than AWS
- `2x`: unhealthy
- `4x`: reaches distributed init, then stalls before benchmark output
- `8x`: node-B can complete a bare distributed benchmark, but real training still lands in the bad band

### Bare distributed benchmark

AWS `2x`:
- `MATMUL_SEC = 0.080145`
- `ALLREDUCE_SEC_MEAN = 0.0003` for `64 MiB`

Heimdall `8x` on node-B (`10.0.83.32`):
- `MATMUL_SEC = 0.380703`
- `ALLREDUCE_SEC_MEAN = 0.00039`

Interpretation:
- FA3 / import / basic NCCL path is not completely broken on node-B
- but real training still collapses there

## Not the cause

- Not missing FA3
- Not globally broken current image
- Not a launcher interpreter mismatch anymore
- Not the model code itself

## Most plausible fault domain

- node-local storage / page cache / dataset IO path
- host-level CPU / memory / NUMA contention
- scheduler placement / hidden colocated workloads
- training-specific runtime pressure that does not show up in the bare 8x benchmark

## Recommended next checks

1. Compare `node-ip-10-0-83-32` vs `node-ip-10-0-80-97` on:
   - local NVMe pressure
   - CPU steal / load
   - NUMA / topology
   - container runtime differences
   - cgroup limits / colocated jobs
2. Explain why node-B passes bare 8x distributed benchmark but fails real training throughput
3. Treat node-A as the preferred diagnosis/training node until node-B is explained
