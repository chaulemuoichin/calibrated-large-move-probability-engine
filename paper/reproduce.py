"""
Reproducibility script for the paper:
"Calibrated Large-Move Probability Estimation via Walk-Forward Monte Carlo
 with Online Recalibration"

This script regenerates all tables and figures from the paper.
Expected runtime: ~2-4 hours depending on hardware.

Usage:
    # Full reproduction (all experiments)
    python paper/reproduce.py --all

    # Individual components
    python paper/reproduce.py --main-results
    python paper/reproduce.py --baselines
    python paper/reproduce.py --ablation
    python paper/reproduce.py --holdout
    python paper/reproduce.py --economic
    python paper/reproduce.py --figures
"""

import argparse
import logging
import os
import sys
import time

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

os.makedirs("outputs/paper", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/paper/reproduction.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

TICKERS = ["spy", "googl", "amzn", "nvda"]


def run_main_results():
    """Reproduce Table 1: Main CV results."""
    logger.info("=" * 60)
    logger.info("REPRODUCING: Main CV Results (Table 1)")
    logger.info("=" * 60)
    from scripts.run_paper_results import generate_main_results_table
    import pandas as pd

    all_results = []
    for ticker in TICKERS:
        logger.info("Processing %s...", ticker.upper())
        try:
            df = generate_main_results_table(ticker, n_folds=5)
            all_results.append(df)
        except Exception as e:
            logger.error("Failed for %s: %s", ticker, e)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        os.makedirs("outputs/paper/tables", exist_ok=True)
        combined.to_csv("outputs/paper/tables/main_results.csv", index=False)
        logger.info("Main results saved to outputs/paper/tables/main_results.csv")
        print(combined.to_string(index=False))


def run_baselines():
    """Reproduce Table 2: Baseline comparison."""
    logger.info("=" * 60)
    logger.info("REPRODUCING: Baseline Comparison (Table 2)")
    logger.info("=" * 60)
    from scripts.run_paper_results import generate_baseline_comparison_table
    import pandas as pd

    all_results = []
    for ticker in TICKERS:
        logger.info("Processing %s...", ticker.upper())
        try:
            df = generate_baseline_comparison_table(ticker)
            all_results.append(df)
        except Exception as e:
            logger.error("Failed for %s: %s", ticker, e)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        os.makedirs("outputs/paper/tables", exist_ok=True)
        combined.to_csv("outputs/paper/tables/baseline_comparison.csv", index=False)
        logger.info("Baseline comparison saved")
        print(combined.to_string(index=False))


def run_ablation():
    """Reproduce Table 3: Ablation study."""
    logger.info("=" * 60)
    logger.info("REPRODUCING: Ablation Study (Table 3)")
    logger.info("=" * 60)
    from scripts.run_ablation_study import run_ablation as _run_ablation
    import pandas as pd

    all_results = []
    for ticker in TICKERS:
        logger.info("Processing %s...", ticker.upper())
        try:
            df = _run_ablation(ticker, n_folds=5)
            all_results.append(df)
        except Exception as e:
            logger.error("Failed for %s: %s", ticker, e)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv("outputs/paper/ablation_all.csv", index=False)
        logger.info("Ablation results saved")


def run_holdout():
    """Reproduce Table 4: Temporal hold-out."""
    logger.info("=" * 60)
    logger.info("REPRODUCING: Temporal Hold-Out (Table 4)")
    logger.info("=" * 60)
    from scripts.run_temporal_holdout import run_temporal_holdout
    import pandas as pd

    all_results = []
    for ticker in TICKERS:
        logger.info("Processing %s...", ticker.upper())
        try:
            df = run_temporal_holdout(ticker, cutoff="2019-12-31")
            all_results.append(df)
        except Exception as e:
            logger.error("Failed for %s: %s", ticker, e)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv("outputs/paper/holdout_all.csv", index=False)
        logger.info("Hold-out results saved")


def run_economic():
    """Reproduce Table 5: Economic significance."""
    logger.info("=" * 60)
    logger.info("REPRODUCING: Economic Significance (Table 5)")
    logger.info("=" * 60)
    from scripts.run_economic_significance import run_economic_analysis
    import pandas as pd

    all_results = []
    for ticker in TICKERS:
        logger.info("Processing %s...", ticker.upper())
        try:
            df = run_economic_analysis(ticker)
            all_results.append(df)
        except Exception as e:
            logger.error("Failed for %s: %s", ticker, e)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv("outputs/paper/economic_all.csv", index=False)
        logger.info("Economic results saved")


def run_figures():
    """Reproduce all paper figures."""
    logger.info("=" * 60)
    logger.info("REPRODUCING: Paper Figures")
    logger.info("=" * 60)
    from scripts.generate_paper_figures import main as gen_figures
    gen_figures()


def main():
    parser = argparse.ArgumentParser(description="Reproduce all paper results")
    parser.add_argument("--all", action="store_true", help="Run everything")
    parser.add_argument("--main-results", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--economic", action="store_true")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()

    os.makedirs("outputs/paper", exist_ok=True)

    start = time.time()

    if args.all or args.main_results:
        run_main_results()
    if args.all or args.baselines:
        run_baselines()
    if args.all or args.ablation:
        run_ablation()
    if args.all or args.holdout:
        run_holdout()
    if args.all or args.economic:
        run_economic()
    if args.all or args.figures:
        run_figures()

    elapsed = time.time() - start
    logger.info("Total reproduction time: %.1f minutes", elapsed / 60)

    if not any(vars(args).values()):
        parser.print_help()
        print("\nRun with --all to reproduce everything, or select individual components.")


if __name__ == "__main__":
    main()
