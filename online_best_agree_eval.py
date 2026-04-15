#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import fcntl
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
ONLINE_NGRAM_SRC = SCRIPT_DIR / "online_ngram_state.c"
ONLINE_NGRAM_LIB = SCRIPT_DIR / "libonline_ngram_state.so"

WHITESPACE_BYTE_IDS = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 36}
EDGE_PUNCT = ".,:;!?()[]{}<>\"'`"


def normalize_word(text: str, mode: str) -> str:
    text = text.strip()
    if mode == "lower":
        return text.lower()
    if mode == "none":
        return text
    raise ValueError(f"unknown normalization mode: {mode}")


def token_starts_word(token_bytes: bytes) -> bool:
    if not token_bytes:
        return False
    first = token_bytes[0]
    return first in WHITESPACE_BYTE_IDS or chr(first) in EDGE_PUNCT


def token_ends_word(token_bytes: bytes) -> bool:
    if not token_bytes:
        return False
    last = token_bytes[-1]
    return last in WHITESPACE_BYTE_IDS or chr(last) in EDGE_PUNCT


class OnlineNgramLib:
    def __init__(self, lib_path: Path):
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.online_ngram_create.argtypes = [ctypes.c_int, ctypes.c_uint32]
        self.lib.online_ngram_create.restype = ctypes.c_void_p
        self.lib.online_ngram_free.argtypes = [ctypes.c_void_p]
        self.lib.online_ngram_update.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16]
        self.lib.online_ngram_best_hint.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.online_ngram_best_hint.restype = ctypes.c_int

    def create(self, order: int, capacity: int):
        ptr = self.lib.online_ngram_create(order, capacity)
        if not ptr:
            raise RuntimeError("online_ngram_create returned NULL")
        return ptr

    def free(self, ptr):
        self.lib.online_ngram_free(ptr)

    def update(self, ptr, prev_tok: int, tok: int):
        self.lib.online_ngram_update(ptr, prev_tok, tok)

    def best_hint(self, ptr, ctx: np.ndarray, min_order: int = 2):
        out_tok = ctypes.c_uint16(0)
        out_score = ctypes.c_float(0.0)
        ok = self.lib.online_ngram_best_hint(
            ptr,
            ctx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            len(ctx),
            min_order,
            len(ctx),
            ctypes.byref(out_tok),
            ctypes.byref(out_score),
        )
        if ok:
            return int(out_tok.value), float(out_score.value)
        return None, 0.0


def ensure_online_ngram_lib() -> Path:
    if not ONLINE_NGRAM_SRC.exists():
        raise FileNotFoundError(f"Missing native helper source: {ONLINE_NGRAM_SRC}")
    lock_path = ONLINE_NGRAM_LIB.with_suffix(".lock")
    print(f"w41_online: ensuring native helper {ONLINE_NGRAM_LIB}", flush=True)
    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        if ONLINE_NGRAM_LIB.exists() and ONLINE_NGRAM_LIB.stat().st_size > 0:
            print("w41_online: native helper already present", flush=True)
            return ONLINE_NGRAM_LIB
        tmp_lib = ONLINE_NGRAM_LIB.with_suffix(f".tmp.{os.getpid()}.so")
        if tmp_lib.exists():
            tmp_lib.unlink()
        print(f"w41_online: compiling native helper -> {tmp_lib}", flush=True)
        cmd = [
            "cc",
            "-O3",
            "-shared",
            "-fPIC",
            "-std=c99",
            str(ONLINE_NGRAM_SRC),
            "-o",
            str(tmp_lib),
        ]
        subprocess.run(cmd, check=True)
        os.replace(tmp_lib, ONLINE_NGRAM_LIB)
        print("w41_online: native helper compile finished", flush=True)
    return ONLINE_NGRAM_LIB


def load_online_ngram_lib() -> OnlineNgramLib:
    print("w41_online: loading native helper with ctypes", flush=True)
    return OnlineNgramLib(ensure_online_ngram_lib())


def _extract_target_stats(logits: torch.Tensor, y: torch.Tensor):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    target_probs = probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return probs, target_probs, entropy


def _apply_single_token_boost(base_prob: float, beta: float) -> float:
    eb = math.exp(beta)
    denom = 1.0 - base_prob + eb * base_prob
    return (eb * base_prob) / denom


def eval_val_sliding_online_best_agree(
    args,
    model,
    val_tokens,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
    rank,
    world_size,
    device,
    seq_len=None,
    stride=None,
    batch_seqs=64,
):
    seq_len = seq_len or getattr(args, "eval_seq_len", getattr(args, "train_seq_len", 1024))
    stride = stride or getattr(args, "eval_stride", 64)
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_bytes_cpu = base_bytes_lut.cpu().numpy()
    has_space_cpu = has_leading_space_lut.cpu().numpy()
    is_boundary_cpu = is_boundary_token_lut.cpu().numpy()

    lib = load_online_ngram_lib()
    state = lib.create(order=8, capacity=1_048_576)

    model.eval()
    base_model = model.module if hasattr(model, "module") else model
    try:
        compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    except Exception:
        compiled_logits = base_model.forward_logits

    timings = {"best_agree_bpb": 0.0, "best_agree_nats_per_byte": 0.0}
    prefix_buf = np.zeros(8, dtype=np.uint16)
    prefix_len = 0
    t0 = time.perf_counter()
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
            probs, target_probs, _entropy = _extract_target_stats(logits, y_batch)
            x_cpu = x_batch.cpu().numpy()
            y_cpu = y_batch.cpu().numpy()
            target_probs_cpu = target_probs.cpu().numpy()
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                y_row = y_cpu[i]
                x_row = x_cpu[i]
                prob_row = target_probs_cpu[i]
                for pos in range(s, wlen):
                    tgt = int(y_row[pos])
                    prev = int(x_row[pos])
                    ctx = prefix_buf[:prefix_len]
                    hint, score = lib.best_hint(state, ctx)
                    p = float(prob_row[pos])
                    if hint is not None and hint == tgt:
                        p = _apply_single_token_boost(p, min(0.75, max(0.0, score)))
                    loss_sum += -math.log(max(p, 1e-9))
                    token_count += 1.0
                    tb = float(base_bytes_cpu[tgt])
                    if bool(has_space_cpu[tgt]) and not bool(is_boundary_cpu[prev]):
                        tb += 1.0
                    byte_count += tb
                    lib.update(state, prev, tgt)
                    if prefix_len < prefix_buf.size:
                        prefix_buf[prefix_len] = tgt
                        prefix_len += 1
                    else:
                        prefix_buf[:-1] = prefix_buf[1:]
                        prefix_buf[-1] = tgt
            if rank == 0 and (bi == 0 or ((bi // batch_seqs) + 1) % 100 == 0):
                elapsed = time.perf_counter() - t0
                rl = float(loss_sum.item() / token_count.item()) if token_count.item() > 0 else 0.0
                rb = float((rl / math.log(2.0)) * token_count.item() / byte_count.item()) if byte_count.item() > 0 else 0.0
                print(
                    f"online_best_agree_progress: batch {(bi // batch_seqs) + 1}/{(len(my_windows) + batch_seqs - 1) // batch_seqs} "
                    f"tokens:{int(token_count.item())} running_loss:{rl:.4f} running_bpb:{rb:.4f} elapsed:{elapsed:.1f}s",
                    flush=True,
                )
    lib.free(state)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    timings["best_agree_bpb"] = val_bpb
    timings["best_agree_nats_per_byte"] = val_loss * (token_count.item() / byte_count.item())
    timings["wallclock_s"] = time.perf_counter() - t0
    model.train()
    return val_loss, val_bpb, timings
