#!/usr/bin/env python3
"""Minimal placeholder for the fetched #1370 eval helper.

Round 23 uses this only as an implementation reference during the feasibility
stage; the main entrypoint remains train_gpt.py -> train_gdn_7k.py.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def score_chunk(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    bsz, sl, vocab = logits.shape
    return F.cross_entropy(
        logits.float().reshape(-1, vocab),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape(bsz, sl)


def loss_bpb(loss_sum: torch.Tensor, token_count: torch.Tensor, byte_count: torch.Tensor) -> tuple[float, float]:
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return val_loss, val_bpb
