"""
Generate cross-asset processed datasets.

This script creates one joint dataset per (category, horizon), without touching
the existing single-asset processed-data pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.windowing_cross_asset import create_cross_asset_dataset
from src.utils.config import ProjectConfig
from src.utils.logger import get_logger

logger = get_logger("generate_processed_cross_asset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cross-asset processed datasets.")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--category", nargs="+", default=None, help="Categories to process.")
    parser.add_argument("--horizon", nargs="+", type=int, default=None, help="Horizons to process.")
    parser.add_argument("--force", action="store_true", help="Recreate existing outputs.")
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="data/processed_cross_asset",
        help="Output sub-directory (relative to project root).",
    )
    return parser.parse_args()


def _validate_filters(
    config: ProjectConfig,
    categories_filter: Optional[List[str]],
    horizons_filter: Optional[List[int]],
) -> tuple[List[str], List[int]]:
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

    return selected_categories, selected_horizons


def main() -> None:
    args = parse_args()
    config = ProjectConfig(args.project_root)
    selected_categories, selected_horizons = _validate_filters(
        config=config, categories_filter=args.category, horizons_filter=args.horizon
    )

    data_raw_dir = config.get_path("data_raw")
    output_root = Path(args.project_root) / args.output_subdir

    features = config.dataset["features"]
    target = config.dataset["target"]
    split_cfg = config.dataset["split"]
    scaler_type = config.dataset.get("scaler_type", "standard")
    csv_columns = config.dataset.get("csv_columns")
    datetime_format = config.dataset.get("datetime_format", "%Y-%m-%d %H:%M")
    max_samples = config.dataset.get("max_samples")

    total = 0
    success = 0
    failed = 0

    logger.info("=" * 60)
    logger.info("Generating cross-asset processed datasets")
    logger.info("=" * 60)

    for category in selected_categories:
        assets = config.get_assets_for_category(category)
        asset_names = [a["name"] for a in assets]
        file_paths = [data_raw_dir / a["file"] for a in assets]
        missing = [str(p) for p in file_paths if not p.exists()]
        if missing:
            logger.error("Missing raw files for category %s: %s", category, missing)
            failed += len(selected_horizons)
            continue

        for horizon in selected_horizons:
            total += 1
            window_size = config.get_window_size(horizon)
            out_dir = output_root / category / str(horizon)
            logger.info(
                "[%s] cross-asset %s/h%s | assets=%s window=%s",
                total, category, horizon, len(asset_names), window_size
            )
            try:
                create_cross_asset_dataset(
                    file_paths=file_paths,
                    asset_names=asset_names,
                    output_dir=out_dir,
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
                success += 1
            except Exception as exc:
                failed += 1
                logger.error("Failed cross-asset %s/h%s: %s", category, horizon, exc)

    logger.info("=" * 60)
    logger.info("Done. total=%s success=%s failed=%s", total, success, failed)
    logger.info("=" * 60)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

