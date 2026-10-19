"""Main Application CLI Entry Point for Stage 0.

Provides commands to run environment validation and image pipeline tests.
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is available in module path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import ensure_project_directories
from scripts.validate_env import validate_environment, print_environment_report
from scripts.test_pipeline import run_pipeline_test
from src.utils.logger import get_logger

logger = get_logger("app.main")


def main() -> int:
    """Parse CLI arguments and dispatch commands."""
    ensure_project_directories()

    parser = argparse.ArgumentParser(
        description="AI Atmospheric Visibility Enhancement & Haze Detection System - Stage 0 Foundation"
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Run environment and dependency diagnostics.",
    )
    parser.add_argument(
        "--test-pipeline",
        action="store_true",
        help="Run the Stage 0 basic image loading, tensor conversion, and saving test.",
    )
    parser.add_argument(
        "--sample-image",
        type=str,
        default="sample_haze.png",
        help="Sample image filename in data/samples/ for the pipeline test.",
    )

    args = parser.parse_args()

    # If neither specific flag is given, run full Stage 0 validation suite
    run_all = not args.check_env and not args.test_pipeline

    overall_success = True

    if args.check_env or run_all:
        logger.info("Starting environment validation check...")
        env_results = validate_environment()
        env_ok = print_environment_report(env_results)
        if not env_ok:
            overall_success = False
        print()

    if args.test_pipeline or run_all:
        logger.info("Executing image pipeline test...")
        pipeline_ok = run_pipeline_test(sample_filename=args.sample_image)
        if not pipeline_ok:
            overall_success = False

    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
