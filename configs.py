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


def model_d_gdn_1swa() -> dict:
    """Model D: GDN + 1 shared SWA (Zamba-style)."""
    return dict(
        arch_name="D_GDN_1SWA",
        num_gdn_layers=10,
        num_mamba_layers=0,
        num_swa_layers=1,
        swa_shared=True,
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
        layer_layout="gdn5_swa_gdn5_swa_shared",
        bigram_vocab_size=3072,
        bigram_dim=112,
        trigram=True,
    )


def model_i_kv_share() -> dict:
    """Model I: GDN + KV share — same as A but with kv_sharing_stride=2."""
    return dict(
        arch_name="I_KVShare",
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
        kv_sharing_stride=2,
    )


def model_j_kv_share_deeper() -> dict:
    """Model J: GDN + KV share + deeper — 12L dim=480 near iso-param to A."""
    return dict(
        arch_name="J_KVShare_Deeper",
        num_gdn_layers=12,
        num_mamba_layers=0,
        num_swa_layers=0,
        swa_shared=False,
        model_dim=480,
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
        kv_sharing_stride=2,
    )


def model_k_kv_share_wider() -> dict:
    """Model K: GDN + KV share + wider — 10L dim=544 near iso-param to A."""
    return dict(
        arch_name="K_KVShare_Wider",
        num_gdn_layers=10,
        num_mamba_layers=0,
        num_swa_layers=0,
        swa_shared=False,
        model_dim=544,
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
        kv_sharing_stride=2,
    )


def model_h_pure_swa() -> dict:
    """Model H: Pure sliding-window attention control."""
    return dict(
        arch_name="H_PureSWA",
        num_gdn_layers=0,
        num_mamba_layers=0,
        num_swa_layers=10,
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
        layer_layout="swa_only",
        bigram_vocab_size=3072,
        bigram_dim=112,
        trigram=True,
    )


def model_l_kv_share_wider_neg() -> dict:
    """Model L: KV-share wider + negative eigenvalues."""
    cfg = model_k_kv_share_wider()
    cfg["arch_name"] = "L_KVShare_Wider_NegEig"
    cfg["gdn_allow_neg_eigval"] = True
    return cfg


def model_m_gdn_1swa_wider() -> dict:
    """Model M: wider shared-SWA hybrid."""
    cfg = model_d_gdn_1swa()
    cfg["arch_name"] = "M_GDN_1SWA_Wider"
    cfg["model_dim"] = 544
    return cfg


def model_f_mamba2() -> dict:
    """Model F: Mamba-2 only."""
    return dict(
        arch_name="F_Mamba2",
        num_gdn_layers=0,
        num_mamba_layers=11,
        num_swa_layers=0,
        swa_shared=False,
        model_dim=512,
        num_heads=8,
        mlp_mult=3.0,
        mamba_state_size=64,
        mamba_expand=2,
        gdn_expand_v=1,
        gdn_head_dim=64,
        gdn_use_short_conv=True,
        swa_window=512,
        swa_num_kv_heads=4,
        meta_tokens=0,
        layer_layout="mamba_only",
        bigram_vocab_size=3072,
        bigram_dim=112,
        trigram=True,
    )


def model_g_hybrid() -> dict:
    """Model G: GDN + Mamba2 + SWA triple hybrid."""
    return dict(
        arch_name="G_GDN_Mamba_SWA",
        num_gdn_layers=6,
        num_mamba_layers=4,
        num_swa_layers=1,
        swa_shared=False,
        model_dim=512,
        num_heads=8,
        mlp_mult=3.0,
        mamba_state_size=64,
        mamba_expand=2,
        gdn_expand_v=1,
        gdn_head_dim=64,
        gdn_allow_neg_eigval=False,
        gdn_use_short_conv=True,
        swa_window=512,
        swa_num_kv_heads=4,
        meta_tokens=0,
        layer_layout="gdn3_mamba2_swa_gdn3_mamba2",
        bigram_vocab_size=3072,
        bigram_dim=112,
        trigram=True,
    )


ALL_CONFIGS = {
    "A": model_a_pure_gdn,
    "B": model_b_deltaproduct,
    "B2": model_b2_deltaproduct_neg,
    "C": model_c_gdn_neg,
    "D": model_d_gdn_1swa,
    "F": model_f_mamba2,
    "G": model_g_hybrid,
    "H": model_h_pure_swa,
    "I": model_i_kv_share,
    "J": model_j_kv_share_deeper,
    "K": model_k_kv_share_wider,
    "L": model_l_kv_share_wider_neg,
    "M": model_m_gdn_1swa_wider,
}


def get_config(model_id: str) -> dict:
    if model_id not in ALL_CONFIGS:
        raise ValueError(f"Unknown model ID '{model_id}'. Choose from {list(ALL_CONFIGS.keys())}")
    return ALL_CONFIGS[model_id]()
