# Copyright 2025 - Multi-Scale Spatial Latent Attention (MSLA)
# A novel attention mechanism for visual tasks combining:
#   - Latent KV compression (inspired by MLA / DeepSeek V2)
#   - 2D relative position bias for spatial awareness
#   - Multi-scale latent pooling for hierarchical context
#

"""
Multi-Scale Spatial Latent Attention (MSLA)
============================================
MSLA is designed for vision transformer (ViT-style) architectures that process
2D image patches. It extends Multi-Head Latent Attention (MLA) with two
vision-specific innovations:

1. **2D Relative Position Bias (RPB):** Rather than using absolute positional
   embeddings, MSLA learns a relative position bias table indexed by the
   (row_delta, col_delta) between every query-patch and key-patch pair.
   This gives the model explicit 2D spatial awareness.

2. **Multi-Scale Latent Pooling (MSLP):** The compressed KV latent is pooled
   at multiple spatial scales (1×, 2×, 4×) before being up-projected into
   keys and values. The multi-scale keys/values are concatenated, giving each
   query access to both fine-grained local and coarse global context in a
   single attention pass — without extra attention layers.

Memory savings vs standard MHA:
  KV cache (MHA)  ≈ B × L × n_layers × emb_dim × 2 × dtype_bytes
  KV cache (MSLA) ≈ B × L × n_layers × latent_dim × dtype_bytes
  (latent_dim << emb_dim × 2  ⟹  significant saving)

Usage
-----
    from msla_attention import MSLABlock, MSLAViT

    model = MSLAViT(
        img_size=224, patch_size=16, in_channels=3,
        emb_dim=768, n_heads=12, n_layers=12,
        latent_dim=192,           # KV compression (4× savings vs MHA)
        pool_scales=(1, 2, 4),    # Multi-scale spatial pooling
        num_classes=1000,
    )
    logits = model(images)        # images: (B, 3, H, W)
"""

import math
import argparse
import time
from typing import Tuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_2d_relative_position_index(grid_h: int, grid_w: int) -> torch.Tensor:
    """
    Build an integer relative-position index of shape (H*W, H*W).
    Each entry (i, j) encodes a flat index into a bias table of size
    (2*H-1, 2*W-1).  Inspired by Swin Transformer's RPB.
    """
    coords_h = torch.arange(grid_h)
    coords_w = torch.arange(grid_w)
    # coords: (2, H, W)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
    # (2, H*W)
    coords_flat = coords.flatten(1)
    # (2, H*W, H*W)  — pairwise differences
    relative = coords_flat[:, :, None] - coords_flat[:, None, :]
    # shift to non-negative
    relative[0] += grid_h - 1
    relative[1] += grid_w - 1
    # combine into a single flat index
    relative[0] *= 2 * grid_w - 1
    return relative.sum(0)  # (H*W, H*W)


# ---------------------------------------------------------------------------
# Core: Multi-Scale Spatial Latent Attention
# ---------------------------------------------------------------------------

class MultiScaleSpatialLatentAttention(nn.Module):
    """
    Multi-Scale Spatial Latent Attention (MSLA).

    Parameters
    ----------
    d_in        : int  — input feature dimension
    d_out       : int  — output feature dimension (= emb_dim, must be divisible by n_heads)
    n_heads     : int  — number of attention heads
    latent_dim  : int  — dimension of the compressed KV latent vector
    grid_size   : (H, W) — spatial grid of patches (e.g. (14,14) for 224px / 16px patches)
    pool_scales : sequence of ints — spatial pooling factors for multi-scale context
                  e.g. (1, 2, 4) keeps full resolution, 2× downsampled, 4× downsampled
    dropout     : float — attention dropout
    qkv_bias    : bool
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_heads: int,
        latent_dim: int,
        grid_size: Tuple[int, int],
        pool_scales: Sequence[int] = (1, 2, 4),
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.latent_dim = latent_dim
        self.grid_h, self.grid_w = grid_size
        self.pool_scales = pool_scales
        self.n_scales = len(pool_scales)

        # ── Query projection ──────────────────────────────────────────────
        self.W_Q = nn.Linear(d_in, d_out, bias=qkv_bias)

        # ── KV latent compression / up-projection ─────────────────────────
        self.W_DKV = nn.Linear(d_in, latent_dim, bias=qkv_bias)    # down-project
        # Each scale gets its own up-projection for K and V
        self.W_UK = nn.ModuleList(
            [nn.Linear(latent_dim, d_out, bias=qkv_bias) for _ in pool_scales]
        )
        self.W_UV = nn.ModuleList(
            [nn.Linear(latent_dim, d_out, bias=qkv_bias) for _ in pool_scales]
        )

        # ── 2D Relative Position Bias ─────────────────────────────────────
        # One bias table per scale (each scale has its own effective grid)
        self.rel_pos_bias_tables = nn.ParameterList()
        self.register_buffer("rel_pos_indices", None, persistent=False)  # placeholder

        for s in pool_scales:
            gh = math.ceil(self.grid_h / s)
            gw = math.ceil(self.grid_w / s)
            table_h = 2 * self.grid_h - 1   # query grid is always full
            table_w = 2 * gw - 1
            # shape: (n_heads, table_h * table_w)
            table = nn.Parameter(torch.zeros(n_heads, table_h * table_w))
            nn.init.trunc_normal_(table, std=0.02)
            self.rel_pos_bias_tables.append(table)

        # Pre-compute relative position indices for each scale
        self._build_rel_pos_indices()

        # ── Output projection ─────────────────────────────────────────────
        # After multi-scale concat the key/value sequence is n_scales × longer,
        # but context_vec is still (B, T_q, d_out) — no change needed here.
        self.out_proj = nn.Linear(d_out, d_out)
        self.attn_drop = nn.Dropout(dropout)

        # ── KV cache (stores compressed latent only) ───────────────────────
        self.register_buffer("cache_latent", None, persistent=False)
        self.cache_ptr = 0

    # ------------------------------------------------------------------
    def _build_rel_pos_indices(self):
        """Pre-compute per-scale (q_len, k_len) relative position index."""
        indices = []
        for s in self.pool_scales:
            gh_k = math.ceil(self.grid_h / s)
            gw_k = math.ceil(self.grid_w / s)
            # query coords (full grid)
            coords_q_h = torch.arange(self.grid_h)
            coords_q_w = torch.arange(self.grid_w)
            gq = torch.stack(torch.meshgrid(coords_q_h, coords_q_w, indexing="ij")).flatten(1)  # (2, H*W)
            # key coords (downsampled grid, then scaled back to query space)
            coords_k_h = torch.arange(gh_k).float() * s + (s - 1) / 2.0
            coords_k_w = torch.arange(gw_k).float() * s + (s - 1) / 2.0
            gk_mesh = torch.stack(torch.meshgrid(coords_k_h, coords_k_w, indexing="ij")).flatten(1)  # (2, Hk*Wk)
            # relative offset (query - key), rounded to int, then shifted
            rel_h = (gq[0, :, None] - gk_mesh[0, None, :]).long() + (self.grid_h - 1)
            rel_w = (gq[1, :, None] - gk_mesh[1, None, :]).long() + (gw_k - 1)
            table_w = 2 * gw_k - 1
            flat_idx = rel_h * table_w + rel_w  # (H*W, Hk*Wk)
            indices.append(flat_idx)
        self._rpb_indices = indices  # list of tensors, not buffers (they're small)

    # ------------------------------------------------------------------
    def reset_cache(self):
        self.cache_latent = None
        self.cache_ptr = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _to_heads(x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
        """(B, T, d_out) -> (B, n_heads, T, head_dim)"""
        b, t, _ = x.shape
        return x.view(b, t, n_heads, head_dim).transpose(1, 2).contiguous()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """
        x : (B, T, d_in)   T = grid_h * grid_w (sequence of image patches)
        """
        B, T, _ = x.shape

        # 1) Query
        Q = self._to_heads(self.W_Q(x), self.n_heads, self.head_dim)   # (B, H, T, hd)

        # 2) Compress to latent
        latent_new = self.W_DKV(x)  # (B, T, latent_dim)

        # 3) KV cache handling
        if use_cache:
            latent_total = latent_new if self.cache_latent is None else \
                torch.cat([self.cache_latent, latent_new], dim=1)
            self.cache_latent = latent_total
        else:
            latent_total = latent_new  # (B, T_total, latent_dim)

        T_total = latent_total.shape[1]
        Hg, Wg = self.grid_h, self.grid_w
        n_patches = Hg * Wg

        # If the sequence has a CLS token prepended (T = n_patches+1), peel it off
        # before spatial reshaping and re-attach its latent separately.
        has_cls = (T_total != n_patches)
        if has_cls:
            latent_cls = latent_total[:, :1, :]        # (B, 1, latent_dim) — CLS latent
            latent_patches = latent_total[:, 1:, :]    # (B, n_patches, latent_dim)
        else:
            latent_patches = latent_total

        # 4) Multi-scale up-projection + attention per scale, then sum
        attn_out = torch.zeros(B, self.n_heads, T, self.head_dim, device=x.device, dtype=x.dtype)
        scale_factor = 1.0 / math.sqrt(self.head_dim)

        for i, s in enumerate(self.pool_scales):
            # Reshape latent to spatial grid for pooling
            L_spatial = latent_patches.view(B, Hg, Wg, self.latent_dim).permute(0, 3, 1, 2)
            # (B, latent_dim, Hg, Wg)

            if s > 1:
                gh_k = math.ceil(Hg / s)
                gw_k = math.ceil(Wg / s)
                L_pooled = F.adaptive_avg_pool2d(L_spatial, (gh_k, gw_k))
            else:
                L_pooled = L_spatial  # (B, latent_dim, Hg, Wg)

            # Flatten spatial dims back to sequence
            L_seq = L_pooled.flatten(2).transpose(1, 2)  # (B, Tk, latent_dim)
            if has_cls:
                L_seq = torch.cat([latent_cls, L_seq], dim=1)  # prepend CLS latent

            # Up-project to K and V
            K_s = self._to_heads(self.W_UK[i](L_seq), self.n_heads, self.head_dim)
            V_s = self._to_heads(self.W_UV[i](L_seq), self.n_heads, self.head_dim)

            # Scaled dot-product scores
            scores = torch.matmul(Q, K_s.transpose(-2, -1)) * scale_factor
            # (B, n_heads, T_q, T_k)

            # Add 2D relative position bias (patch tokens only)
            # T_q and T_k may include a CLS token at position 0 — skip bias for CLS rows/cols
            if has_cls:
                T_q_patch = T - 1           # query patch tokens
                T_k_patch = L_seq.shape[1] - 1   # key patch tokens (after CLS)
                idx = self._rpb_indices[i].to(x.device)[:T_q_patch, :T_k_patch]
                bias = self.rel_pos_bias_tables[i][:, idx]  # (n_heads, T_q_patch, T_k_patch)
                # Pad with zeros for CLS rows/cols
                pad_row = torch.zeros(self.n_heads, 1, T_k_patch + 1, device=x.device, dtype=bias.dtype)
                pad_col = torch.zeros(self.n_heads, T_q_patch, 1, device=x.device, dtype=bias.dtype)
                bias_full = torch.cat([
                    pad_row,
                    torch.cat([pad_col, bias], dim=2)
                ], dim=1)  # (n_heads, T_q, T_k)
                scores = scores + bias_full.unsqueeze(0)
            else:
                idx = self._rpb_indices[i].to(x.device)
                bias = self.rel_pos_bias_tables[i][:, idx]  # (n_heads, T_q, T_k)
                scores = scores + bias.unsqueeze(0)

            attn_w = torch.softmax(scores, dim=-1)
            attn_w = self.attn_drop(attn_w)

            attn_out += attn_w @ V_s   # accumulate across scales

        # Average across scales (uniform weighting)
        attn_out = attn_out / self.n_scales

        # 5) Merge heads
        ctx = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.out_proj(ctx)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class MSLABlock(nn.Module):
    """
    A single Transformer block using MSLA instead of standard MHA.

    Parameters
    ----------
    emb_dim     : int
    n_heads     : int
    latent_dim  : int
    grid_size   : (H, W) patch grid
    pool_scales : multi-scale factors
    mlp_ratio   : hidden-dim multiplier for the MLP
    dropout     : float
    """

    def __init__(
        self,
        emb_dim: int,
        n_heads: int,
        latent_dim: int,
        grid_size: Tuple[int, int],
        pool_scales: Sequence[int] = (1, 2, 4),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = MultiScaleSpatialLatentAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            n_heads=n_heads,
            latent_dim=latent_dim,
            grid_size=grid_size,
            pool_scales=pool_scales,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(emb_dim)
        hidden = int(emb_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), use_cache=use_cache)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full ViT-style model using MSLA
# ---------------------------------------------------------------------------

class MSLAViT(nn.Module):
    """
    Vision Transformer with Multi-Scale Spatial Latent Attention (MSLA).

    Parameters
    ----------
    img_size    : int   — square input image size (e.g. 224)
    patch_size  : int   — patch size (e.g. 16)
    in_channels : int   — input channels (3 for RGB)
    emb_dim     : int   — embedding dimension
    n_heads     : int   — attention heads
    n_layers    : int   — number of MSLA blocks
    latent_dim  : int   — KV latent compression dim
    pool_scales : tuple — multi-scale pooling factors
    mlp_ratio   : float — MLP hidden dim multiplier
    dropout     : float
    num_classes : int   — classification head output size (0 = no head)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        emb_dim: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        latent_dim: int = 192,
        pool_scales: Sequence[int] = (1, 2, 4),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_classes: int = 1000,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        grid = img_size // patch_size
        self.grid_size = (grid, grid)
        self.n_patches = grid * grid
        self.emb_dim = emb_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, emb_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # [CLS] token + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, emb_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.drop_embed = nn.Dropout(dropout)

        # MSLA blocks (CLS token is prepended, so grid for attention = original grid)
        self.blocks = nn.ModuleList([
            MSLABlock(
                emb_dim=emb_dim,
                n_heads=n_heads,
                latent_dim=latent_dim,
                grid_size=self.grid_size,
                pool_scales=pool_scales,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(emb_dim)

        # Classification head
        self.head = nn.Linear(emb_dim, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)
        Returns logits (B, num_classes) or features (B, emb_dim) if num_classes=0.
        """
        B = x.shape[0]

        # Patch embedding: (B, emb_dim, grid, grid) -> (B, n_patches, emb_dim)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Prepend [CLS]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)           # (B, n_patches+1, emb_dim)
        x = self.drop_embed(x + self.pos_embed)

        # NOTE: MSLA spatial pooling operates only on the patch tokens (no CLS).
        # We pass the full sequence; the CLS token's spatial position is
        # handled gracefully because the attention is unconstrained for it.
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])   # CLS token


# ---------------------------------------------------------------------------
# Memory estimation utility
# ---------------------------------------------------------------------------

def estimate_kv_memory(
    batch_size: int,
    seq_len: int,
    n_layers: int,
    emb_dim: int,
    latent_dim: int,
    bytes_per_elem: int = 2,   # bf16
) -> dict:
    """Compare KV cache memory between MHA and MSLA."""
    mha = batch_size * seq_len * n_layers * emb_dim * 2 * bytes_per_elem
    msla = batch_size * seq_len * n_layers * latent_dim * bytes_per_elem
    return {
        "MHA_GB":  mha  / 1024**3,
        "MSLA_GB": msla / 1024**3,
        "ratio":   mha  / msla,
        "savings_pct": (1 - msla / mha) * 100,
    }


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MSLA — Multi-Scale Spatial Latent Attention demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--img_size",    type=int, default=224)
    parser.add_argument("--patch_size",  type=int, default=16)
    parser.add_argument("--emb_dim",     type=int, default=768)
    parser.add_argument("--n_heads",     type=int, default=12)
    parser.add_argument("--n_layers",    type=int, default=6,
                        help="Use fewer layers for a quick test")
    parser.add_argument("--latent_dim",  type=int, default=192,
                        help="KV latent dim (emb_dim*2 / latent_dim = compression ratio)")
    parser.add_argument("--pool_scales", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--batch_size",  type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print("  Multi-Scale Spatial Latent Attention (MSLA) — Demo")
    print(f"{'='*55}")
    print(f"  Device      : {device}")
    print(f"  img_size    : {args.img_size}")
    print(f"  patch_size  : {args.patch_size}")
    print(f"  emb_dim     : {args.emb_dim}")
    print(f"  n_heads     : {args.n_heads}")
    print(f"  n_layers    : {args.n_layers}")
    print(f"  latent_dim  : {args.latent_dim}  "
          f"(compression {args.emb_dim * 2 / args.latent_dim:.1f}×)")
    print(f"  pool_scales : {args.pool_scales}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  num_classes : {args.num_classes}")
    print(f"{'='*55}\n")

    model = MSLAViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,
        emb_dim=args.emb_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        latent_dim=args.latent_dim,
        pool_scales=args.pool_scales,
        num_classes=args.num_classes,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters: {n_params:.1f} M\n")

    images = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        logits = model(images)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    print(f"  Forward pass : {elapsed*1000:.1f} ms  "
          f"(batch={args.batch_size})")
    print(f"  Output shape : {tuple(logits.shape)}")

    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak GPU mem : {mem:.3f} GB")

    # KV memory comparison
    grid = args.img_size // args.patch_size
    seq_len = grid * grid
    mem_stats = estimate_kv_memory(
        batch_size=args.batch_size,
        seq_len=seq_len,
        n_layers=args.n_layers,
        emb_dim=args.emb_dim,
        latent_dim=args.latent_dim,
    )
    print(f"\n{'─'*55}")
    print("  KV-cache memory estimate (bf16)")
    print(f"  MHA  : {mem_stats['MHA_GB']:.4f} GB")
    print(f"  MSLA : {mem_stats['MSLA_GB']:.4f} GB")
    print(f"  Savings : {mem_stats['savings_pct']:.1f}%  "
          f"({mem_stats['ratio']:.2f}× less)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
