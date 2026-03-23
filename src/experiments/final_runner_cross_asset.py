"""
Cross-asset final training runner.

This runner is isolated from the existing single-asset final pipeline.
It trains one model per (model, category, horizon) on the joint cross-asset
dataset and evaluates over multiple seeds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.windowing_cross_asset import load_cross_asset_data
from src.evaluation.aggregate import aggregate_seed_metrics, save_aggregated_metrics
from src.models import get_model_class
from src.training.trainer import Trainer
from src.utils.config import ProjectConfig
from src.utils.logger import get_logger, setup_experiment_logger
from src.utils.seed import get_device, set_seed

logger = get_logger(__name__)


def _compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Basic regression metrics for cross-asset tensors."""
    diff = y_true - y_pred
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    smape = float(100.0 * np.mean(2.0 * np.abs(diff) / denom))
    return {"rmse": rmse, "mae": mae, "mse": mse, "smape": smape}


def _load_asset_names(data_dir: Path, n_assets: int) -> List[str]:
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        names = meta.get("asset_names", [])
        if isinstance(names, list) and len(names) == n_assets:
            return [str(n) for n in names]
    return [f"asset_{i}" for i in range(n_assets)]


def run_final_cross_asset_single(
    config: ProjectConfig,
    model_name: str,
    category: str,
    horizon: int,
    seed: int,
    processed_subdir: str = "data/processed_cross_asset",
) -> Dict[str, Any]:
    """Run one seed for cross-asset final training."""
    set_seed(seed)
    device = get_device(config.base.get("device", "auto"), model_name=model_name)

    exp_logger = setup_experiment_logger(
        "final_cross_asset", model_name, category, "__cross_asset__", horizon, config.get_path("logs_dir")
    )
    exp_logger.info(
        "Cross-asset final: model=%s category=%s horizon=%s seed=%s",
        model_name, category, horizon, seed
    )

    data_dir = config.project_root / processed_subdir / category / str(horizon)
    dataset = load_cross_asset_data(data_dir)
    n_assets = dataset["train_x"].shape[2]
    asset_names = _load_asset_names(data_dir, n_assets)

    best_hparams_raw = config.get_model_best_config(model_name, category=category, horizon=horizon)
    best_hparams = best_hparams_raw.get("best_params", best_hparams_raw)

    model_hparams = {k: v for k, v in best_hparams.items() if k not in ("learning_rate", "batch_size")}
    model_hparams["num_assets"] = model_hparams.get("num_assets", n_assets)
    train_overrides = {
        "learning_rate": best_hparams.get("learning_rate", config.training["learning_rate"]),
        "batch_size": best_hparams.get("batch_size", config.training["batch_size"]),
    }

    model_cls = get_model_class(model_name)
    model = model_cls(
        input_size=len(config.dataset["features"]),
        window_size=config.get_window_size(horizon),
        horizon=horizon,
        **model_hparams,
    )
    train_config = {**config.training, **train_overrides}

    checkpoint_dir = (
        config.get_path("models_dir")
        / "final_cross_asset"
        / model_name
        / category
        / str(horizon)
        / str(seed)
    )
    results_dir = (
        config.get_path("results_dir")
        / "final_cross_asset"
        / model_name
        / category
        / str(horizon)
        / str(seed)
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(model, device, train_config, seed=seed)
    train_results = trainer.fit(
        train_x=dataset["train_x"],
        train_y=dataset["train_y"],
        val_x=dataset["val_x"],
        val_y=dataset["val_y"],
        checkpoint_dir=checkpoint_dir,
        resume=config.training.get("checkpoint", {}).get("resume", True),
        model_name=model_name,
        category=category,
        asset="__cross_asset__",
        horizon=horizon,
        hyperparameters=best_hparams,
    )

    predictions = trainer.predict(dataset["test_x"])
    metrics = _compute_basic_metrics(dataset["test_y"], predictions)
    metrics.update(
        {
            "seed": seed,
            "model": model_name,
            "category": category,
            "asset": "__cross_asset__",
            "horizon": horizon,
            "epochs_trained": train_results["epochs_trained"],
            "n_parameters": train_results["n_parameters"],
            "best_val_loss": train_results["best_val_loss"],
            "n_assets": n_assets,
        }
    )

    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions in flattened table for easy inspection.
    pred_arr = predictions if predictions.ndim == 3 else predictions[:, :, np.newaxis]
    cols = [f"{asset_names[i]}_h{h + 1}" for i in range(n_assets) for h in range(pred_arr.shape[2])]
    pred_df = pd.DataFrame(pred_arr.reshape(pred_arr.shape[0], -1), columns=cols)
    pred_df.to_csv(results_dir / "predictions.csv", index=False)

    return metrics


def run_final_for_model_category_cross_asset(
    config: ProjectConfig,
    model_name: str,
    category: str,
    horizon: int,
    processed_subdir: str = "data/processed_cross_asset",
) -> Dict[str, Any]:
    """Run final cross-asset training for all evaluation seeds."""
    _ = processed_subdir  # Kept for API symmetry and future use.
    seeds = config.get_eval_seeds()
    all_metrics: List[Dict[str, Any]] = []

    for seed in seeds:
        try:
            metrics = run_final_cross_asset_single(
                config=config,
                model_name=model_name,
                category=category,
                horizon=horizon,
                seed=seed,
                processed_subdir=processed_subdir,
            )
            all_metrics.append(metrics)
        except Exception as exc:
            logger.error(
                "Cross-final failed: %s/%s/h%s/s%s -> %s",
                model_name, category, horizon, seed, exc
            )

    if all_metrics:
        aggregated = aggregate_seed_metrics(all_metrics)
        agg_path = (
            config.get_path("results_dir")
            / "final_cross_asset"
            / model_name
            / category
            / str(horizon)
            / "aggregated.json"
        )
        save_aggregated_metrics(aggregated, agg_path)
        return aggregated
    return {}


def run_all_final_cross_asset(
    config: ProjectConfig,
    models_filter: Optional[List[str]] = None,
    processed_subdir: str = "data/processed_cross_asset",
) -> None:
    """Run cross-asset final training for all selected combinations."""
    models = models_filter or config.get_available_models()
    categories = config.get_categories()
    horizons = config.get_horizons()
    total = len(models) * len(categories) * len(horizons)
    done = 0

    logger.info("Starting cross-asset final training: %s combinations", total)
    for model_name in models:
        for category in categories:
            for horizon in horizons:
                done += 1
                logger.info("[%s/%s] cross-final %s/%s/h%s", done, total, model_name, category, horizon)
                try:
                    run_final_for_model_category_cross_asset(
                        config=config,
                        model_name=model_name,
                        category=category,
                        horizon=horizon,
                        processed_subdir=processed_subdir,
                    )
                except Exception as exc:
                    logger.error("Cross-final failed for %s/%s/h%s: %s", model_name, category, horizon, exc)

