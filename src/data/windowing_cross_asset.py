"""
Cross-asset windowing utilities.

Creates aligned multi-asset datasets for joint modeling:
- Input X shape: (n_samples, window_size, n_assets, n_features)
- Target y shape: (n_samples, n_assets, horizon)

This module is intentionally isolated from src.data.windowing to avoid changing
the existing single-asset pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.data.loader import load_raw_csv
from src.data.preprocessing import preprocess_data, split_data
from src.data.scaler import create_scaler
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_aligned_panel(
    file_paths: Sequence[Union[str, Path]],
    asset_names: Sequence[str],
    features: List[str],
    csv_columns: Optional[List[str]],
    datetime_format: str,
) -> np.ndarray:
    """Load all assets and align on common datetime index.

    Returns:
        3D array with shape (timesteps, n_assets, n_features).
    """
    if len(file_paths) != len(asset_names):
        raise ValueError("file_paths and asset_names must have the same length.")
    if len(file_paths) < 2:
        raise ValueError("Cross-asset dataset requires at least two assets.")

    per_asset: List[pd.DataFrame] = []
    renamed_cols: List[List[str]] = []

    for asset_name, file_path in zip(asset_names, file_paths):
        df = load_raw_csv(file_path, columns=csv_columns, datetime_format=datetime_format)
        df = preprocess_data(df, features=features)
        cols = [f"{asset_name}__{f}" for f in features]
        renamed_cols.append(cols)
        per_asset.append(df.rename(columns=dict(zip(features, cols))))

    # Inner join keeps only common timestamps across all assets.
    merged = per_asset[0]
    for df in per_asset[1:]:
        merged = merged.join(df, how="inner")

    merged = merged.sort_index().dropna()
    if merged.empty:
        raise ValueError("No overlapping timestamps after cross-asset alignment.")

    n_assets = len(asset_names)
    n_features = len(features)
    panel = np.zeros((len(merged), n_assets, n_features), dtype=np.float32)
    for i, cols in enumerate(renamed_cols):
        panel[:, i, :] = merged[cols].to_numpy(dtype=np.float32)

    logger.info(
        "Aligned cross-asset panel: T=%s, N=%s, F=%s",
        panel.shape[0], panel.shape[1], panel.shape[2],
    )
    return panel


def _scale_panel_by_asset(
    train_panel: np.ndarray,
    val_panel: np.ndarray,
    test_panel: np.ndarray,
    scaler_type: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[object]]:
    """Fit one scaler per asset (using train split only)."""
    n_assets = train_panel.shape[1]
    train_scaled = np.zeros_like(train_panel, dtype=np.float32)
    val_scaled = np.zeros_like(val_panel, dtype=np.float32)
    test_scaled = np.zeros_like(test_panel, dtype=np.float32)
    scalers: List[object] = []

    for i in range(n_assets):
        scaler = create_scaler(scaler_type)
        train_scaled[:, i, :] = scaler.fit_transform(train_panel[:, i, :]).astype(np.float32)
        val_scaled[:, i, :] = scaler.transform(val_panel[:, i, :]).astype(np.float32)
        test_scaled[:, i, :] = scaler.transform(test_panel[:, i, :]).astype(np.float32)
        scalers.append(scaler)

    return train_scaled, val_scaled, test_scaled, scalers


def _create_windows_cross_asset(
    panel: np.ndarray,
    window_size: int,
    horizon: int,
    target_idx: int,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows for cross-asset panel data."""
    # panel shape: (T, N, F)
    n_total = panel.shape[0]
    n_samples = n_total - window_size - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Insufficient data: T={n_total}, window={window_size}, horizon={horizon}"
        )

    effective = min(n_samples, max_samples) if max_samples else n_samples
    start_idx = n_samples - effective  # keep most recent windows if capped

    n_assets = panel.shape[1]
    n_features = panel.shape[2]
    x = np.zeros((effective, window_size, n_assets, n_features), dtype=np.float32)
    y = np.zeros((effective, n_assets, horizon), dtype=np.float32)

    for i in range(effective):
        s = start_idx + i
        x[i] = panel[s : s + window_size]
        target_slice = panel[s + window_size : s + window_size + horizon, :, target_idx]
        y[i] = target_slice.T  # (N, H)

    return x, y


def create_cross_asset_dataset(
    file_paths: Sequence[Union[str, Path]],
    asset_names: Sequence[str],
    output_dir: Union[str, Path],
    window_size: int,
    horizon: int,
    features: Optional[List[str]] = None,
    target: str = "close",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    scaler_type: str = "standard",
    csv_columns: Optional[List[str]] = None,
    datetime_format: str = "%Y-%m-%d %H:%M",
    force_recreate: bool = False,
    max_samples: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Create and save a cross-asset dataset for joint training."""
    output_dir = Path(output_dir)
    required = [
        "train_x.npy", "train_y.npy", "val_x.npy", "val_y.npy",
        "test_x.npy", "test_y.npy", "metadata.json",
    ]
    if not force_recreate and all((output_dir / f).exists() for f in required):
        logger.info("Cross-asset processed data exists at %s, loading...", output_dir)
        return load_cross_asset_data(output_dir)

    if features is None:
        features = ["open", "high", "low", "close", "volume"]
    if target not in features:
        raise ValueError(f"Target '{target}' not found in features {features}")
    target_idx = features.index(target)

    panel = _build_aligned_panel(
        file_paths=file_paths,
        asset_names=asset_names,
        features=features,
        csv_columns=csv_columns,
        datetime_format=datetime_format,
    )

    # Split along time axis.
    df_for_split = pd.DataFrame(index=np.arange(panel.shape[0]), data={"t": np.arange(panel.shape[0])})
    train_df, val_df, test_df = split_data(
        df_for_split, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
    )
    train_panel = panel[train_df.index]
    val_panel = panel[val_df.index]
    test_panel = panel[test_df.index]

    train_scaled, val_scaled, test_scaled, _ = _scale_panel_by_asset(
        train_panel, val_panel, test_panel, scaler_type=scaler_type
    )

    train_x, train_y = _create_windows_cross_asset(
        train_scaled, window_size=window_size, horizon=horizon, target_idx=target_idx
    )
    val_x, val_y = _create_windows_cross_asset(
        val_scaled, window_size=window_size, horizon=horizon, target_idx=target_idx
    )
    test_x, test_y = _create_windows_cross_asset(
        test_scaled, window_size=window_size, horizon=horizon, target_idx=target_idx
    )

    # Optional global cap distributed proportionally across splits.
    if max_samples is not None:
        desired = int(max_samples)
        total = train_x.shape[0] + val_x.shape[0] + test_x.shape[0]
        if total > desired:
            train_cap = int(desired * train_ratio)
            val_cap = int(desired * val_ratio)
            test_cap = max(0, desired - train_cap - val_cap)
            train_x, train_y = train_x[-train_cap:], train_y[-train_cap:]
            val_x, val_y = val_x[-val_cap:], val_y[-val_cap:]
            test_x, test_y = test_x[-test_cap:], test_y[-test_cap:]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_x.npy", train_x)
    np.save(output_dir / "train_y.npy", train_y)
    np.save(output_dir / "val_x.npy", val_x)
    np.save(output_dir / "val_y.npy", val_y)
    np.save(output_dir / "test_x.npy", test_x)
    np.save(output_dir / "test_y.npy", test_y)

    metadata = {
        "asset_names": list(asset_names),
        "features": features,
        "target": target,
        "window_size": int(window_size),
        "horizon": int(horizon),
        "shape": {
            "train_x": list(train_x.shape),
            "train_y": list(train_y.shape),
            "val_x": list(val_x.shape),
            "val_y": list(val_y.shape),
            "test_x": list(test_x.shape),
            "test_y": list(test_y.shape),
        },
        "scaler_type": scaler_type,
        "split": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "max_samples": max_samples,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Saved cross-asset dataset to %s | train=%s val=%s test=%s",
        output_dir, train_x.shape[0], val_x.shape[0], test_x.shape[0]
    )
    return {
        "train_x": train_x,
        "train_y": train_y.squeeze(-1) if horizon == 1 else train_y,
        "val_x": val_x,
        "val_y": val_y.squeeze(-1) if horizon == 1 else val_y,
        "test_x": test_x,
        "test_y": test_y.squeeze(-1) if horizon == 1 else test_y,
    }


def load_cross_asset_data(data_dir: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load saved cross-asset arrays."""
    data_dir = Path(data_dir)
    required = ["train_x.npy", "train_y.npy", "val_x.npy", "val_y.npy", "test_x.npy", "test_y.npy"]
    for name in required:
        if not (data_dir / name).exists():
            raise FileNotFoundError(f"Missing processed file: {data_dir / name}")

    train_y = np.load(data_dir / "train_y.npy")
    val_y = np.load(data_dir / "val_y.npy")
    test_y = np.load(data_dir / "test_y.npy")
    horizon = train_y.shape[-1] if train_y.ndim == 3 else 1

    return {
        "train_x": np.load(data_dir / "train_x.npy"),
        "train_y": train_y.squeeze(-1) if horizon == 1 else train_y,
        "val_x": np.load(data_dir / "val_x.npy"),
        "val_y": val_y.squeeze(-1) if horizon == 1 else val_y,
        "test_x": np.load(data_dir / "test_x.npy"),
        "test_y": test_y.squeeze(-1) if horizon == 1 else test_y,
    }

