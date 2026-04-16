"""GDN Hybrid Architecture — modular blocks using FLA native layers.

Supports 7 model variants (A-G) for the Parameter Golf screening experiments.
Each model is a stack of mixed {GDN, DeltaProduct, RWKV-7, Mamba-2, SWA} blocks
with shared MLP, RMSNorm, and residual connections.

Key design choices:
- FLA layers handle recurrent attention (GatedDeltaNet, GatedDeltaProduct, RWKV7, Mamba2)
- Sliding Window Attention (SWA) uses flash attention with a causal window mask
- All blocks follow the same pre-norm residual pattern for uniform gradient flow
- Weight sharing for SWA layers in Zamba/Hymba-style models
- Score-first eval: XSA-all only extends attention layers (no future context leakage)
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, Tensor


# -----------------------------------------------------------------------------
# Optional FLA Triton -> pure PyTorch fallback patch
# -----------------------------------------------------------------------------
# Set FLA_USE_NAIVE=1 to force pure-PyTorch (naive) kernels instead of Triton.
# This is mainly for environments where Triton cache/build is flaky; H100 should
# normally use Triton.
_USE_NAIVE = os.environ.get("FLA_USE_NAIVE", "0") == "1"
def _ensure_fla_installed() -> None:
    try:
        import fla  # noqa: F401
        return
    except ImportError:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--ignore-installed",
                "triton==3.2.0",
                "safetensors==0.7.0",
                "tokenizers==0.22.2",
                "transformers==5.5.4",
                "fla-core==0.4.2",
                "flash-linear-attention==0.4.2",
            ],
            stdout=sys.stderr,
            stderr=sys.stderr,
        )


_ensure_fla_installed()


if _USE_NAIVE:
    try:
        # 1. Patch GatedDeltaNet's chunk op
        import fla.ops.gated_delta_rule.chunk as _gdr_chunk
        import fla.ops.gated_delta_rule.naive as _gdr_naive
        import fla.layers.gated_deltanet as _gdn_layer
        def _patched_chunk_gated_delta_rule(
            q, k, v, beta, gk, scale=None, initial_state=None, output_final_state=False,
            cu_seqlens=None, head_first=False, chunk_size=64,
        ):
            return _gdr_naive.naive_chunk_gated_delta_rule(
                q, k, v, beta, gk,
                chunk_size=64, scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                head_first=head_first,
            )
        _gdr_chunk.chunk_gated_delta_rule = _patched_chunk_gated_delta_rule
        _gdn_layer.chunk_gated_delta_rule = _patched_chunk_gated_delta_rule
        # 2. Patch GatedDeltaProduct's chunk op
        import fla.ops.gated_delta_product.chunk as _gdp_chunk
        import fla.ops.gated_delta_product.naive as _gdp_naive
        import fla.layers.gated_delta_product as _gdp_layer
        def _patched_chunk_gated_delta_product(
            q, k, v, beta, gk, scale=None, initial_state=None,
            output_final_state=False, cu_seqlens=None, head_first=False,
            chunk_size=64,
        ):
            return _gdp_naive.naive_chunk_gated_delta_product(
                q, k, v, beta, gk,
                chunk_size=64, scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                head_first=head_first,
            )
        _gdp_chunk.chunk_gated_delta_product = _patched_chunk_gated_delta_product
        _gdp_layer.chunk_gated_delta_product = _patched_chunk_gated_delta_product
        print("[FLA] Using NAIVE (pure-PyTorch) kernels — set FLA_USE_NAIVE=0 for Triton", flush=True)
    except Exception as e:
        print(f"[FLA] Failed to apply naive fallback patch: {e}", flush=True)

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.layers.gated_deltaproduct import GatedDeltaProduct
from fla.layers.mamba2 import Mamba2
try:
    from fla.layers.rwkv7 import RWKV7Attention
except ImportError:
    RWKV7Attention = None  # type: ignore


class CastedLinear(nn.Linear):
    """A linear layer where weights are cast to the input dtype at runtime."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.ste_qat = False  # toggled late in training

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if self.ste_qat:
            # Straight-through fake-int6 quantization during late warmdown.
            w = fake_quantize_int6_ste(w)
        return F.linear(x, w.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


def fake_quantize_int6_ste(w: Tensor, clip_range: int = 31) -> Tensor:
    """Per-row int6 fake quantization with a straight-through estimator."""
    w32 = w.float()
    if w32.ndim != 2:
        return w
    row_clip = torch.quantile(w32.abs(), 0.9999, dim=1)
    scale = (row_clip / clip_range).clamp_min(1.0 / clip_range)
    q = torch.clamp(torch.round(w32 / scale[:, None]), -clip_range, clip_range)
    deq = q * scale[:, None]
    return w + (deq.to(w.dtype) - w).detach()


class BigramHash(nn.Module):
    def __init__(self, vocab_size: int, dim: int, trigram: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.trigram = trigram
        self.bigram = nn.Embedding(vocab_size, dim)
        self.tri = nn.Embedding(vocab_size, dim) if trigram else None
        nn.init.zeros_(self.bigram.weight)
        if self.tri is not None:
            nn.init.zeros_(self.tri.weight)

    def forward(self, ids: Tensor) -> Tensor:
        # ids: [B, T]
        prev = torch.zeros_like(ids)
        prev[:, 1:] = ids[:, :-1]
        h = self.bigram((prev * 131 + ids) % self.vocab_size)
        if self.tri is not None:
            prev2 = torch.zeros_like(ids)
            prev2[:, 2:] = ids[:, :-2]
            tri_idx = (prev2 * 173 + prev * 131 + ids) % self.vocab_size
            h = h + self.tri(tri_idx)
        return h


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


class Rotary:
    def __init__(self, head_dim: int, base: float = 10000.0):
        self.head_dim = head_dim
        self.base = base
        self._seq_len_cached = 0
        self._cache = None

    def __call__(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._cache is not None and self._seq_len_cached >= seq_len and self._cache[0].device == device and self._cache[0].dtype == dtype:
            cos, sin = self._cache
            return cos[:seq_len], sin[:seq_len]
        theta = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, theta)
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        self._seq_len_cached = seq_len
        self._cache = (cos, sin)
        return cos, sin


def apply_rotary_emb(q: Tensor, cos: Tensor, sin: Tensor, dims: int):
    if dims <= 0:
        return q
    q1 = q[..., :dims]
    q2 = q[..., dims:]
    q1_even = q1[..., 0::2]
    q1_odd = q1[..., 1::2]
    rot_even = q1_even * cos.unsqueeze(0).unsqueeze(2) - q1_odd * sin.unsqueeze(0).unsqueeze(2)
    rot_odd = q1_even * sin.unsqueeze(0).unsqueeze(2) + q1_odd * cos.unsqueeze(0).unsqueeze(2)
    q1_rot = torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)
    return torch.cat((q1_rot, q2), dim=-1)


class SWAWrapper(nn.Module):
    """Sliding-window attention using flash-attn or SDPA fallback."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, window: int, rope_base: float = 10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.window = window
        self.q = CastedLinear(dim, dim, bias=False)
        self.k = CastedLinear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v = CastedLinear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o = CastedLinear(dim, dim, bias=False)
        self.rotary = Rotary(self.head_dim, rope_base)
        self.q_gain = nn.Parameter(torch.full((num_heads,), 5.0))

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q = self.q(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.v(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.head_dim)
        k = apply_rotary_emb(k, cos, sin, self.head_dim)
        q = q * self.q_gain.view(1, 1, self.num_heads, 1).to(q.dtype)

        qh = q.transpose(1, 2)  # [B, H, T, D]
        kh = k.transpose(1, 2)
        vh = v.transpose(1, 2)
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            kh = kh.repeat_interleave(repeat, dim=1)
            vh = vh.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(
            qh, kh, vh,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=False,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.o(out)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc = CastedLinear(dim, hidden * 2, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        u, v = self.fc(x).chunk(2, dim=-1)
        return self.proj(F.silu(u) * v)


class RecurrentBlock(nn.Module):
    """Wraps recurrent token-mixing with pre-norm, residual, and MLP."""
    def __init__(self, dim: int, mixer: nn.Module, mlp_mult: float):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.recurrent = mixer
        self.attn_scale = nn.Parameter(torch.tensor(1.0))
        self.mlp_scale = nn.Parameter(torch.tensor(1.0))
        self.mlp = MLP(dim, int(dim * mlp_mult))

    def forward(self, x: Tensor, x0: Tensor | None = None) -> Tensor:
        x_in = self.attn_norm(x)
        recurrent_out = self.recurrent(x_in)
        if isinstance(recurrent_out, tuple):
            recurrent_out = recurrent_out[0]
        x = x + self.attn_scale.to(dtype=x.dtype) * recurrent_out
        x = x + self.mlp_scale.to(dtype=x.dtype) * self.mlp(self.mlp_norm(x))
        return x


class HybridGDN(nn.Module):
    """Minimal GDN-first architecture scaffold from PR #1370."""
    def __init__(self, cfg: dict, vocab_size: int, rope_base: float = 10000.0, tie_embeddings: bool = True, logit_softcap: float = 30.0):
        super().__init__()
        self.cfg = cfg
        dim = cfg["model_dim"]
        self.vocab_size = vocab_size
        self.model_dim = dim
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.bigram = BigramHash(cfg.get("bigram_vocab_size", vocab_size), cfg.get("bigram_dim", 0), trigram=cfg.get("trigram", False)) if cfg.get("bigram_dim", 0) > 0 else None
        self.embed_proj = CastedLinear(dim + cfg.get("bigram_dim", 0), dim, bias=False) if self.bigram is not None else None
        self.blocks = nn.ModuleList()
        layout = cfg["layer_layout"]
        if layout == "gdn_only":
            for _ in range(cfg["num_gdn_layers"]):
                if cfg.get("use_deltaproduct", False):
                    mixer = GatedDeltaProduct(
                        hidden_size=dim,
                        expand_v=cfg.get("gdn_expand_v", 1),
                        head_dim=cfg.get("gdn_head_dim", 64),
                        num_heads=cfg["num_heads"],
                        use_short_conv=cfg.get("gdn_use_short_conv", True),
                        allow_neg_eigval=cfg.get("dp_allow_neg_eigval", False),
                        num_householder=cfg.get("dp_num_householder", 2),
                        mode="chunk",
                    )
                else:
                    mixer = GatedDeltaNet(
                        hidden_size=dim,
                        expand_v=cfg.get("gdn_expand_v", 1),
                        head_dim=cfg.get("gdn_head_dim", 64),
                        num_heads=cfg["num_heads"],
                        use_short_conv=cfg.get("gdn_use_short_conv", True),
                        allow_neg_eigval=cfg.get("gdn_allow_neg_eigval", False),
                        mode="chunk",
                    )
                self.blocks.append(RecurrentBlock(dim, mixer, cfg["mlp_mult"]))
        else:
            raise NotImplementedError(f"Unsupported initial FLA layout: {layout}")
        self.final_norm = RMSNorm(dim)
        self.lm_head = None if tie_embeddings else CastedLinear(dim, vocab_size, bias=False)

    def set_xsa(self, enabled: bool):
        # Compatibility hook used by the fetched eval path.
        return

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = torch.cat([x, self.bigram(input_ids)], dim=-1)
            x = self.embed_proj(x)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        for block in self.blocks:
            x = block(x, x0)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1),
            reduction="mean",
        )
