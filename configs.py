"""Model architecture configurations for GDN Hybrid experiments.

Each config returns a dict consumed by HybridGDN.__init__.
All models are sized to fit ~16MB at int6+zstd-22.
"""
from __future__ import annotations


def model_a_pure_gdn() -> dict:
    """Model A: Pure GDN (Baseline) — 10 layers Gated DeltaNet."""
    return dict(
        arch_name="A_PureGDN",
        num_gdn_layers=10,
        num_mamba_layers=0,
        num_swa_layers=0,
        swa_shared=False,
        model_dim=512,
        num_heads=8,
        mlp_mult=3.0,
        gdn_expand_v=1,
        gdn_head_dim=64,
        gdn_allow_neg_eigval=False,
        gdn_use_short_conv=True,
        swa_window=512,
        swa_num_kv_heads=4,
        meta_tokens=0,
        layer_layout="gdn_only",
        bigram_vocab_size=3072,
        bigram_dim=112,
        trigram=True,
    )


def model_b_deltaproduct() -> dict:
    """Model B: Gated DeltaProduct n_h=2 — rank-2 state transitions."""
    return dict(
        arch_name="B_DeltaProduct",
        num_gdn_layers=10,
        num_mamba_layers=0,
        num_swa_layers=0,
        swa_shared=False,
        model_dim=480,
        num_heads=8,
        mlp_mult=3.0,
        use_deltaproduct=True,
        dp_num_householder=2,
        dp_allow_neg_eigval=False,
        gdn_expand_v=1,
        gdn_head_dim=64,
        gdn_use_short_conv=True,
        swa_window=512,
        swa_num_kv_heads=4,
        meta_tokens=0,
        layer_layout="gdn_only",
    )


def model_b2_deltaproduct_neg() -> dict:
    """Model B2: DeltaProduct + negative eigenvalues."""
    cfg = model_b_deltaproduct()
    cfg["arch_name"] = "B2_DeltaProduct_NegEig"
    cfg["dp_allow_neg_eigval"] = True
    return cfg


def model_c_gdn_neg() -> dict:
    """Model C: GDN with negative eigenvalues — richer state dynamics."""
    return dict(
        arch_name="C_GDN_NegEig",
        num_gdn_layers=11,
        num_mamba_layers=0,
        num_swa_layers=0,
        swa_shared=False,
        model_dim=512,
        num_heads=8,
        mlp_mult=3.0,
        gdn_expand_v=1,
        gdn_head_dim=64,
        gdn_allow_neg_eigval=True,
        gdn_use_short_conv=True,
        swa_window=512,
        swa_num_kv_heads=4,
        meta_tokens=0,
        layer_layout="gdn_only",
        bigram_vocab_size=3072,
        bigram_dim=112,
        trigram=True,
    )


ALL_CONFIGS = {
    "A": model_a_pure_gdn,
    "B": model_b_deltaproduct,
    "B2": model_b2_deltaproduct_neg,
    "C": model_c_gdn_neg,
}


def get_config(model_id: str) -> dict:
    if model_id not in ALL_CONFIGS:
        raise ValueError(f"Unknown model ID '{model_id}'. Choose from {list(ALL_CONFIGS.keys())}")
    return ALL_CONFIGS[model_id]()
