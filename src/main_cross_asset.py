"""
CLI entry for cross-asset branch (isolated from existing src/main.py).

Usage:
  python src/main_cross_asset.py --mode hpo --models PatchTSTCrossAsset
  python src/main_cross_asset.py --mode final --models PatchTSTCrossAsset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ProjectConfig
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-asset training branch.")
    parser.add_argument("--mode", choices=["hpo", "final"], required=True)
    parser.add_argument("--models", nargs="+", default=["PatchTSTCrossAsset"])
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--processed-subdir",
        type=str,
        default="data/processed_cross_asset",
        help="Cross-asset processed dataset sub-directory (relative to project root).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig(args.project_root)
    logger = get_logger("main_cross_asset", log_file=config.get_path("logs_dir") / "system_cross_asset.log")
    set_seed(args.seed)

    available = config.get_available_models()
    invalid = [m for m in args.models if m not in available]
    if invalid:
        logger.error("Unknown models: %s. Available: %s", invalid, available)
        sys.exit(1)

    if args.mode == "hpo":
        from src.experiments.hpo_runner_cross_asset import run_all_hpo_cross_asset
        run_all_hpo_cross_asset(
            config=config, models_filter=args.models, processed_subdir=args.processed_subdir
        )
    else:
        from src.experiments.final_runner_cross_asset import run_all_final_cross_asset
        run_all_final_cross_asset(
            config=config, models_filter=args.models, processed_subdir=args.processed_subdir
        )

    logger.info("Cross-asset branch done.")


if __name__ == "__main__":
    main()

