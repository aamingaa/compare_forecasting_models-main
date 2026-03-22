"""
Batch-generate processed datasets from raw CSV files.

Usage examples:
    python scripts/generate_processed.py
    python scripts/generate_processed.py --force
    python scripts/generate_processed.py --category crypto --horizon 4
    python scripts/generate_processed.py --asset BTCUSDT ETHUSDT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Set

# Ensure project root is on Python path when running from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.windowing import create_dataset
from src.utils.config import ProjectConfig
from src.utils.logger import get_logger


logger = get_logger("generate_processed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate data/processed for all configured assets/horizons.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(PROJECT_ROOT),
        help="Project root path.",
    )
    parser.add_argument(
        "--category",
        nargs="+",
        default=None,
        help="Only process selected categories (e.g. crypto forex).",
    )
    parser.add_argument(
        "--asset",
        nargs="+",
        default=None,
        help="Only process selected asset names (e.g. BTCUSDT EURUSD).",
    )
    parser.add_argument(
        "--horizon",
        nargs="+",
        type=int,
        default=None,
        help="Only process selected horizons (e.g. 4 24).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate processed data even if target files already exist.",
    )
    return parser.parse_args()


def _validate_filters(
    config: ProjectConfig,
    categories_filter: Optional[List[str]],
    assets_filter: Optional[List[str]],
    horizons_filter: Optional[List[int]],
) -> tuple[List[str], Set[str], List[int]]:
    categories = config.get_categories()
    horizons = config.get_horizons()

    selected_categories = categories_filter or categories
    invalid_categories = [c for c in selected_categories if c not in categories]
    if invalid_categories:
        raise ValueError(f"Unknown categories: {invalid_categories}. Available: {categories}")

    selected_horizons = horizons_filter or horizons
    invalid_horizons = [h for h in selected_horizons if h not in horizons]
    if invalid_horizons:
        raise ValueError(f"Unknown horizons: {invalid_horizons}. Available: {horizons}")

    if assets_filter:
        all_assets = {
            a["name"]
            for c in categories
            for a in config.get_assets_for_category(c)
        }
        invalid_assets = [a for a in assets_filter if a not in all_assets]
        if invalid_assets:
            raise ValueError(
                f"Unknown assets: {invalid_assets}. "
                f"Available: {sorted(all_assets)}"
            )
        selected_assets = set(assets_filter)
    else:
        selected_assets = set()

    return selected_categories, selected_assets, selected_horizons


def main() -> None:
    args = parse_args()
    config = ProjectConfig(args.project_root)

    selected_categories, selected_assets, selected_horizons = _validate_filters(
        config=config,
        categories_filter=args.category,
        assets_filter=args.asset,
        horizons_filter=args.horizon,
    )

    data_raw_dir = config.get_path("data_raw")
    data_processed_dir = config.get_path("data_processed")

    features = config.dataset["features"]
    target = config.dataset["target"]
    split_cfg = config.dataset["split"]
    scaler_type = config.dataset.get("scaler_type", "standard")
    csv_columns = config.dataset.get("csv_columns")
    datetime_format = config.dataset.get("datetime_format", "%Y-%m-%d %H:%M")
    max_samples = config.dataset.get("max_samples")

    total_jobs = 0
    success_jobs = 0
    failed_jobs = 0

    logger.info("=" * 60)
    logger.info("Generating processed datasets")
    logger.info("=" * 60)

    for category in selected_categories:
        assets = config.get_assets_for_category(category)
        for asset_info in assets:
            asset_name = asset_info["name"]
            if selected_assets and asset_name not in selected_assets:
                continue

            raw_file = data_raw_dir / asset_info["file"]
            if not raw_file.exists():
                logger.error(f"Missing raw CSV: {raw_file}")
                failed_jobs += len(selected_horizons)
                continue

            for horizon in selected_horizons:
                total_jobs += 1
                window_size = config.get_window_size(horizon)
                output_dir = data_processed_dir / category / asset_name / str(horizon)

                logger.info(
                    f"[{total_jobs}] {category}/{asset_name}/h{horizon} "
                    f"(window={window_size})"
                )
                try:
                    create_dataset(
                        file_path=raw_file,
                        output_dir=output_dir,
                        window_size=window_size,
                        horizon=horizon,
                        features=features,
                        target=target,
                        train_ratio=split_cfg["train"],
                        val_ratio=split_cfg["val"],
                        test_ratio=split_cfg["test"],
                        scaler_type=scaler_type,
                        csv_columns=csv_columns,
                        datetime_format=datetime_format,
                        force_recreate=args.force,
                        max_samples=max_samples,
                    )
                    success_jobs += 1
                except Exception as exc:
                    failed_jobs += 1
                    logger.error(
                        f"Failed: {category}/{asset_name}/h{horizon} -> {exc}"
                    )

    logger.info("=" * 60)
    logger.info(
        f"Done. total={total_jobs}, success={success_jobs}, failed={failed_jobs}"
    )
    logger.info("=" * 60)

    if failed_jobs > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
