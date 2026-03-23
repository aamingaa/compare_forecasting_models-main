"""
PatchTST-style cross-asset position model.

This is a minimal, production-oriented skeleton for cross-asset allocation:
- Input:  (B, W, N, F)
- Output: (B, N, H) or (B, N) when H=1

Design:
1) Per-asset feature projection
2) Time patching + temporal Transformer (shared across assets)
3) Asset-level Transformer for cross-asset interaction
4) Per-asset position head with optional tanh constraint
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.base import BaseForecaster
from src.models import register_model


@register_model("PatchTSTCrossAsset")
class PatchTSTCrossAsset(BaseForecaster):
    """Minimal PatchTST-based cross-asset position model.

    Args:
        input_size: Number of features per asset (F).
        window_size: Lookback length (W).
        horizon: Number of forecast/position steps (H).
        num_assets: Number of assets in the joint universe (N).
        d_model: Hidden dimension.
        patch_len: Patch length over the time axis.
        stride: Patch stride over the time axis.
        n_heads: Attention heads for both temporal and asset encoders.
        time_layers: Number of temporal TransformerEncoder layers.
        asset_layers: Number of cross-asset TransformerEncoder layers.
        d_ff: Feed-forward hidden size in Transformer blocks.
        dropout: Dropout used by Transformer blocks and head.
        use_tanh_output: If True, map output to [-1, 1] as positions.
    """

    def __init__(
        self,
        input_size: int,
        window_size: int,
        horizon: int,
        num_assets: int = 4,
        d_model: int = 64,
        patch_len: int = 16,
        stride: int = 8,
        n_heads: int = 4,
        time_layers: int = 2,
        asset_layers: int = 1,
        d_ff: int = 128,
        dropout: float = 0.1,
        use_tanh_output: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name="PatchTSTCrossAsset",
            input_size=input_size,
            window_size=window_size,
            horizon=horizon,
            num_assets=num_assets,
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            n_heads=n_heads,
            time_layers=time_layers,
            asset_layers=asset_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_tanh_output=use_tanh_output,
            **kwargs,
        )

        if patch_len > window_size:
            raise ValueError(
                f"patch_len ({patch_len}) must be <= window_size ({window_size})."
            )

        self.num_assets = int(num_assets)
        self.d_model = int(d_model)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.use_tanh_output = bool(use_tanh_output)

        self.n_patches = (window_size - self.patch_len) // self.stride + 1
        if self.n_patches < 1:
            raise ValueError(
                "Invalid patch configuration: computed number of patches is < 1."
            )

        # 1) Per-asset feature projection: (B, W, N, F) -> (B, W, N, D)
        self.feature_proj = nn.Linear(input_size, self.d_model)

        # 2) Patch token projection: (patch_len * D) -> D
        self.patch_proj = nn.Linear(self.patch_len * self.d_model, self.d_model)
        self.time_pos_emb = nn.Parameter(
            torch.zeros(1, 1, self.n_patches, self.d_model)
        )

        # Temporal encoder on patch tokens (runs on B*N sequences).
        time_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.time_encoder = nn.TransformerEncoder(time_layer, num_layers=time_layers)

        # 3) Cross-asset encoder on N dimension.
        asset_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.asset_encoder = nn.TransformerEncoder(asset_layer, num_layers=asset_layers)

        # 4) Position head: (B, N, D) -> (B, N, H)
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor with shape (B, W, N, F).

        Returns:
            Tensor with shape (B, N, H), or (B, N) when H=1.
        """
        assert x.ndim == 4, f"Expected 4D input (B, W, N, F), got {x.ndim}D."
        bsz, win, n_assets, n_feat = x.shape
        assert win == self.window_size, f"Expected W={self.window_size}, got {win}."
        assert n_assets == self.num_assets, f"Expected N={self.num_assets}, got {n_assets}."
        assert n_feat == self.input_size, f"Expected F={self.input_size}, got {n_feat}."

        # (B, W, N, F) -> (B, W, N, D) -> (B, N, W, D)
        h = self.feature_proj(x).permute(0, 2, 1, 3).contiguous()

        # Unfold time axis into patches:
        # (B, N, W, D) -> (B, N, M, P, D)
        patches = h.unfold(dimension=2, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 2, 4, 3).contiguous()

        # (B, N, M, P*D) -> (B, N, M, D)
        patch_tokens = patches.view(bsz, n_assets, self.n_patches, self.patch_len * self.d_model)
        patch_tokens = self.patch_proj(patch_tokens)
        patch_tokens = patch_tokens + self.time_pos_emb

        # Temporal encoding over patches for each asset:
        # (B, N, M, D) -> (B*N, M, D) -> (B*N, M, D)
        t = patch_tokens.view(bsz * n_assets, self.n_patches, self.d_model)
        t = self.time_encoder(t)

        # Pool over M patch tokens -> per-asset representation (B, N, D)
        t = t.mean(dim=1).view(bsz, n_assets, self.d_model)

        # Cross-asset interaction on N dimension.
        z = self.asset_encoder(t)  # (B, N, D)

        # Position head.
        out = self.head(z)  # (B, N, H)
        if self.use_tanh_output:
            out = torch.tanh(out)

        expected_shape = (bsz, n_assets, self.horizon)
        assert out.shape == expected_shape, (
            f"Expected output shape {expected_shape}, got {tuple(out.shape)}"
        )
        assert torch.isfinite(out).all(), "Output contains NaN/Inf."
        return out.squeeze(-1) if self.horizon == 1 else out
