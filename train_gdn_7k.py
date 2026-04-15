#!/usr/bin/env python3
"""GDN Hybrid Full Training Script — minimal feasibility port for round 23.

This is not yet a faithful reproduction of PR #1370's full non-record stack.
It is the narrowest scaffold that lets us answer the first round-23 question:
can a Gated DeltaNet lane import, train, quantize, and evaluate inside the
current parameter-golf harness on Heimdall?
"""
from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import sys
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from architectures import HybridGDN, CastedLinear
from configs import get_config


class Hyperparameters:
    arch_mode = os.environ.get("ARCH_MODE", "A")
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 42))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    iterations = int(os.environ.get("ITERATIONS", 7000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 2100))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 100))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    compile_enabled = bool(int(os.environ.get("COMPILE_ENABLED", "1")))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    w40_single_gpu_pilot = bool(int(os.environ.get("W40_SINGLE_GPU_PILOT", "0")))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = rank == 0
    grad_accum_steps = 1


def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype=np.uint32, count=256)
    assert header[0] == 20240520, f"Bad magic: {header[0]}"
    assert header[1] in (1, 7), f"Bad version: {header[1]}"
    ntok = int(header[2])
    return torch.from_numpy(np.fromfile(file, dtype=np.uint16, offset=256 * 4)[:ntok].astype(np.int64))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        assert self.files, f"No files matching {pattern}"
        self.idx = 0
        self.buf = load_data_shard(self.files[self.idx])
        self.pos = 0

    def _advance_file(self) -> None:
        self.idx = (self.idx + 1) % len(self.files)
        self.buf = load_data_shard(self.files[self.idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        parts = []
        remaining = n
        while remaining > 0:
            avail = self.buf.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            take_n = min(avail, remaining)
            parts.append(self.buf[self.pos:self.pos + take_n])
            self.pos += take_n
            remaining -= take_n
        return torch.cat(parts)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.stream = TokenStream(pattern)
        self.rank = rank
        self.world_size = world_size
        self.device = device

    def next_batch(self, global_tokens: int, seq_len: int) -> tuple[Tensor, Tensor]:
        tokens_per_rank = global_tokens // self.world_size
        seqs_per_rank = tokens_per_rank // seq_len
        total_seqs = seqs_per_rank * self.world_size
        total_needed = total_seqs * seq_len + 1
        all_tokens = self.stream.take(total_needed)
        start = self.rank * seqs_per_rank * seq_len
        chunk = all_tokens[start:start + seqs_per_rank * seq_len + 1]
        x = chunk[:-1].reshape(seqs_per_rank, seq_len)
        y = chunk[1:].reshape(seqs_per_rank, seq_len)
        return x.to(self.device), y.to(self.device)


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = sorted(glob.glob(pattern))
    parts = [load_data_shard(Path(f)) for f in files]
    combined = torch.cat(parts)
    return combined[:((combined.numel() - 1) // seq_len) * seq_len + 1]


def build_sentencepiece_luts(sp, vocab_size, device):
    base_bytes = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    has_space = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    is_boundary = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        raw = piece.encode("utf-8")
        base_bytes[i] = len(raw)
        if piece.startswith("\u2581"):
            has_space[i] = True
            base_bytes[i] = len(piece[1:].encode("utf-8")) + 1
        if sp.is_control(i) or sp.is_unknown(i):
            is_boundary[i] = True
    return base_bytes, has_space, is_boundary


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g + momentum * buf if nesterov else buf
                if g.ndim == 2 and min(g.shape) >= 2:
                    g = zeropower_via_newtonschulz5(g, steps=group["backend_steps"])
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(g, alpha=-lr)


def eval_val_sliding(model: nn.Module, val_tokens: Tensor, base_bytes_lut: Tensor, has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor, rank: int, world_size: int, device: torch.device, seq_len: int = 1024, stride: int = 64, batch_seqs: int = 128, xsa_eval: bool = False) -> tuple[float, float]:
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    base_model = model.module if hasattr(model, "module") else model
    if xsa_eval and hasattr(base_model, "set_xsa"):
        base_model.set_xsa(True)
    forward_fn = base_model.forward_logits
    try:
        compiled_logits = torch.compile(forward_fn, dynamic=False)
    except Exception:
        compiled_logits = forward_fn
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    if xsa_eval and hasattr(base_model, "set_xsa"):
        base_model.set_xsa(False)
    model.train()
    return val_loss, bits_per_token * tokens_per_byte


def main():
    args = Hyperparameters()
    pilot_key = (
        os.environ.get("W40_PILOT_KEY")
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or os.environ.get("MASTER_PORT")
        or "default"
    )
    pilot_done = Path(f"/tmp/w40_done_{pilot_key}")
    if args.w40_single_gpu_pilot and args.distributed and args.local_rank != 0:
        print(
            f"w40_pilot: rank {args.local_rank} waiting on {pilot_done} while rank0 runs single-GPU feasibility",
            flush=True,
        )
        while not pilot_done.exists():
            time.sleep(5)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    if args.distributed and not args.w40_single_gpu_pilot:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl")
    else:
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
    if args.w40_single_gpu_pilot:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.train_batch_tokens = max(args.train_batch_tokens // 8, args.train_seq_len)
        args.val_batch_size = max(args.val_batch_size // 8, args.eval_seq_len)
    rank = args.rank
    world_size = args.world_size
    if rank == 0:
        if args.w40_single_gpu_pilot:
            print(
                f"w40_pilot: single-GPU feasibility mode train_batch_tokens={args.train_batch_tokens} "
                f"val_batch_size={args.val_batch_size} sentinel={pilot_done}",
                flush=True,
            )
        print(f"Arch: {args.arch_mode}", flush=True)
    cfg = get_config(args.arch_mode)
    model = HybridGDN(cfg, vocab_size=args.vocab_size, tie_embeddings=True, logit_softcap=args.logit_softcap).to(device).bfloat16()
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)
    loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    val_tokens = load_validation_tokens(args.val_files, args.eval_seq_len)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    matrix_params, scalar_params = [], []
    for name, p in (model.module if hasattr(model, "module") else model).named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2:
            matrix_params.append(p)
        else:
            scalar_params.append(p)
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    opt_adam = torch.optim.AdamW(scalar_params, lr=args.scalar_lr, betas=(args.beta1, args.beta2), weight_decay=args.adam_wd)
    ema_state = {name: t.detach().float().clone() for name, t in (model.module if hasattr(model, "module") else model).state_dict().items()}
    t0 = time.perf_counter()
    step = 0
    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= args.max_wallclock_seconds:
            break
        x, y = loader.next_batch(args.train_batch_tokens, args.train_seq_len)
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        opt_muon.step()
        opt_adam.step()
        with torch.no_grad():
            state_dict = (model.module if hasattr(model, "module") else model).state_dict()
            for name, t in state_dict.items():
                ema_state[name].mul_(args.ema_decay).add_(t.detach().float(), alpha=1.0 - args.ema_decay)
        step += 1
        if rank == 0 and (step <= 5 or step % args.train_log_every == 0):
            tok_s = step * args.train_batch_tokens / max(time.perf_counter() - t0, 1e-9)
            print(f"{step}/{args.iterations} train_loss: {loss.item():.4f} train_time: {(time.perf_counter()-t0)/60:.1f}m tok/s: {tok_s:.0f}", flush=True)
        if args.val_loss_every > 0 and step % args.val_loss_every == 0:
            val_loss, val_bpb = eval_val_sliding(model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, rank, world_size, device, seq_len=args.eval_seq_len, stride=args.eval_stride, batch_seqs=64)
            if rank == 0:
                print(f"{step}/{args.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}", flush=True)
    if rank == 0:
        print(f"stopping_early: wallclock_cap train_time: {1e3*(time.perf_counter()-t0):.0f}ms step: {step}/{args.iterations}", flush=True)
    base = model.module if hasattr(model, "module") else model
    try:
        if rank == 0:
            print("final_eval:barrier_enter", flush=True)
        if args.distributed:
            dist.barrier()
        if rank == 0:
            print("final_eval:barrier_done", flush=True)
            print("final_eval:start", flush=True)
        base.load_state_dict({name: t.to(dtype=base.state_dict()[name].dtype) for name, t in ema_state.items()}, strict=True)
        if rank == 0:
            print("final_eval:ema_loaded", flush=True)
        val_loss, val_bpb = eval_val_sliding(base, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, rank, world_size, device, seq_len=args.eval_seq_len, stride=args.eval_stride, batch_seqs=64)
        if rank == 0:
            print("final_eval:done", flush=True)
            print(f"pre-quantization post-ema val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}", flush=True)
            print("FLA feasibility pilot finished before quantization integration", flush=True)
            print(
                f'results_json: {{"val_bpb": {val_bpb:.8f}, "val_loss": {val_loss:.8f}, '
                f'"bytes_total": 0, "peak_memory_mib": {torch.cuda.max_memory_allocated()//1024//1024}, '
                f'"w40_feasibility_only": true}}',
                flush=True,
            )
            pilot_done.touch()
    except Exception:
        if rank == 0:
            print("final_eval:exception", flush=True)
            traceback.print_exc()
        raise
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
