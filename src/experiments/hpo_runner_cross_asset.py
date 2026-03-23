"""
Cross-asset hyperparameter optimization runner.

This runner is isolated from the existing single-asset HPO pipeline.
It expects datasets generated under: data/processed_cross_asset/<category>/<horizon>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
import pandas as pd
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

from src.data.windowing_cross_asset import load_cross_asset_data
from src.models import get_model_class
from src.training.trainer import Trainer
from src.utils.config import ProjectConfig
from src.utils.logger import get_logger, setup_experiment_logger
from src.utils.seed import get_device, set_seed

logger = get_logger(__name__)

SAMPLER_MAP = {
    "TPE": TPESampler,
    "Random": RandomSampler,
    "CmaEs": CmaEsSampler,
}

PRUNER_MAP = {
    "median": MedianPruner,
    "hyperband": HyperbandPruner,
}


def _suggest_param(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
    param_type = spec["type"]
    if param_type == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    if param_type == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], step=spec.get("step"))
    if param_type == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if param_type == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown parameter type: {param_type}")


def run_hpo_cross_asset(
    config: ProjectConfig,
    model_name: str,
    category: str,
    horizon: int,
    processed_subdir: str = "data/processed_cross_asset",
) -> Dict[str, Any]:
    """Run HPO for one (model, category, horizon) in cross-asset mode."""
    fixed_seed = config.hpo.get("fixed_seed", 42)
    set_seed(fixed_seed)
    device = get_device(config.base.get("device", "auto"), model_name=model_name)

    data_dir = config.project_root / processed_subdir / category / str(horizon)
    dataset = load_cross_asset_data(data_dir)
    n_assets = dataset["train_x"].shape[2]

    exp_logger = setup_experiment_logger(
        "hpo_cross_asset", model_name, category, "__cross_asset__", horizon, config.get_path("logs_dir")
    )
    exp_logger.info("Cross-asset HPO: model=%s category=%s horizon=%s", model_name, category, horizon)

    search_space = config.get_model_search_space(model_name)["search_space"]
    hpo_models_dir = config.get_path("models_dir") / "hpo_cross_asset" / model_name / category / str(horizon)

    def objective(trial: optuna.Trial) -> float:
        hparams = {name: _suggest_param(trial, name, spec) for name, spec in search_space.items()}
        trial_lr = hparams.pop("learning_rate", config.hpo.get("learning_rate", 0.001))
        trial_bs = hparams.pop("batch_size", config.hpo.get("batch_size", 64))
        hparams["num_assets"] = hparams.get("num_assets", n_assets)

        model_cls = get_model_class(model_name)
        model = model_cls(
            input_size=len(config.dataset["features"]),
            window_size=config.get_window_size(horizon),
            horizon=horizon,
            **hparams,
        )

        trial_config = {
            **config.training,
            "epochs": config.hpo.get("epochs", 50),
            "learning_rate": trial_lr,
            "batch_size": trial_bs,
        }
        trainer = Trainer(model, device, trial_config, seed=fixed_seed)

        trial_dir = hpo_models_dir / f"trial_{trial.number:03d}"
        all_hparams = {**hparams, "learning_rate": trial_lr, "batch_size": trial_bs}
        results = trainer.fit(
            train_x=dataset["train_x"],
            train_y=dataset["train_y"],
            val_x=dataset["val_x"],
            val_y=dataset["val_y"],
            checkpoint_dir=trial_dir,
            resume=True,
            model_name=model_name,
            category=category,
            asset="__cross_asset__",
            horizon=horizon,
            hyperparameters=all_hparams,
        )
        return float(results["best_val_loss"])

    sampler_cls = SAMPLER_MAP.get(config.hpo.get("sampler", "TPE"), TPESampler)
    sampler = sampler_cls(seed=fixed_seed)
    pruner_cfg = config.hpo.get("pruner", {})
    pruner_cls = PRUNER_MAP.get(pruner_cfg.get("type", "median"), MedianPruner)
    pruner_kwargs: Dict[str, Any] = {}
    if "n_startup_trials" in pruner_cfg:
        pruner_kwargs["n_startup_trials"] = pruner_cfg["n_startup_trials"]
    if "n_warmup_steps" in pruner_cfg:
        pruner_kwargs["n_warmup_steps"] = pruner_cfg["n_warmup_steps"]
    pruner = pruner_cls(**pruner_kwargs)

    study = optuna.create_study(
        direction=config.hpo.get("direction", "minimize"),
        sampler=sampler,
        pruner=pruner,
    )
    n_trials = config.hpo.get("n_trials", 5)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    trials_df = pd.DataFrame(
        [
            {"number": t.number, "value": t.value, "state": str(t.state), **t.params}
            for t in study.trials
        ]
    )
    hpo_models_dir.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(hpo_models_dir / "trials.csv", index=False)

    best_result = {
        "trial_number": study.best_trial.number,
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
        "model_name": model_name,
        "category": category,
        "asset": "__cross_asset__",
        "horizon": horizon,
        "mode": "cross_asset",
    }
    with open(hpo_models_dir / "best_trial.json", "w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=2)

    config.save_model_best_config(model_name, study.best_trial.params, category=category, horizon=horizon)
    return best_result


def run_all_hpo_cross_asset(
    config: ProjectConfig,
    models_filter: Optional[List[str]] = None,
    processed_subdir: str = "data/processed_cross_asset",
) -> None:
    """Run cross-asset HPO for all selected models/categories/horizons."""
    models = models_filter or config.get_available_models()
    categories = config.get_categories()
    horizons = config.get_horizons()

    total = len(models) * len(categories) * len(horizons)
    done = 0
    logger.info("Starting cross-asset HPO: %s combinations", total)
    for model_name in models:
        for category in categories:
            for horizon in horizons:
                done += 1
                logger.info("[%s/%s] cross-HPO %s/%s/h%s", done, total, model_name, category, horizon)
                try:
                    run_hpo_cross_asset(
                        config=config,
                        model_name=model_name,
                        category=category,
                        horizon=horizon,
                        processed_subdir=processed_subdir,
                    )
                except Exception as exc:
                    logger.error("Cross-HPO failed for %s/%s/h%s: %s", model_name, category, horizon, exc)

