"""
Temporal hold-out evaluation: the gold standard for time-series forecasting.

Train on data through a cutoff date, test on all subsequent data.
Default split: train through 2019-12-31, test 2020-01-01 to end.

This covers three distinct regimes the model has never seen during training:
  - COVID crash and recovery (2020)
  - Fed tightening cycle (2021-2022)
  - AI-driven rally (2023-2025)

Also runs baselines on the same split for fair comparison.

Usage:
    python scripts/run_temporal_holdout.py spy
    python scripts/run_temporal_holdout.py all
    python scripts/run_temporal_holdout.py spy --cutoff 2021-12-31
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from em_sde.config import load_config
from em_sde.data_layer import load_data
from em_sde.backtest import run_walkforward
from em_sde.evaluation import (
    brier_score, brier_skill_score, auc_roc, expected_calibration_error,
    effective_sample_size, paired_bootstrap_loss_diff_pvalue,
)
from scripts.baselines import (
    run_all_baselines, evaluate_baseline,
    historical_frequency_baseline, garch_cdf_baseline,
    implied_vol_baseline, feature_logistic_baseline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TICKER_CONFIGS = {
    "spy": "configs/exp_suite/exp_spy_regime_gated.yaml",
    "googl": "configs/exp_suite/exp_googl_regime_gated.yaml",
    "amzn": "configs/exp_suite/exp_amzn_regime_gated.yaml",
    "nvda": "configs/exp_suite/exp_nvda_regime_gated.yaml",
}


def _era_label(dt: pd.Timestamp) -> str:
    """Assign era label for regime-conditional analysis."""
    if dt <= pd.Timestamp("2020-02-19"):
        return "pre_covid"
    if dt <= pd.Timestamp("2020-12-31"):
        return "covid_2020"
    if dt <= pd.Timestamp("2022-12-31"):
        return "tightening"
    return "post_2023"


def run_temporal_holdout(ticker: str, cutoff: str = "2019-12-31") -> pd.DataFrame:
    """Run temporal hold-out evaluation for a single ticker."""
    config_path = TICKER_CONFIGS.get(ticker.lower())
    if not config_path:
        raise ValueError(f"Unknown ticker: {ticker}")

    cfg = load_config(config_path)
    df, _ = load_data(cfg)
    cutoff_date = pd.Timestamp(cutoff)
    horizons = cfg.model.horizons

    # Split
    train_mask = df.index <= cutoff_date
    test_mask = df.index > cutoff_date
    n_train = train_mask.sum()
    n_test = test_mask.sum()

    logger.info("=== Temporal Hold-out: %s ===", ticker.upper())
    logger.info("Train: %d rows (through %s)", n_train, cutoff)
    logger.info("Test:  %d rows (%s to %s)", n_test,
                df.index[test_mask][0].strftime("%Y-%m-%d") if n_test > 0 else "N/A",
                df.index[test_mask][-1].strftime("%Y-%m-%d") if n_test > 0 else "N/A")

    if n_test < 100:
        logger.warning("Test set too small (%d rows). Need at least 100.", n_test)
        return pd.DataFrame()

    # --- Run full model on entire dataset (walk-forward) ---
    logger.info("Running full model walk-forward...")
    results = run_walkforward(df, cfg)

    # Extract test portion
    test_dates = list(df.index[test_mask])
    test_results = results[results["date"].isin(test_dates)]
    logger.info("Test predictions: %d rows", len(test_results))

    # --- Get thresholds ---
    thresholds = {}
    fixed_pct_by_horizon = cfg.model.regime_gated_fixed_pct_by_horizon or {}
    for H in horizons:
        thresholds[H] = fixed_pct_by_horizon.get(H, cfg.model.fixed_threshold_pct)

    prices = df["price"].to_numpy(dtype=float)
    dates_idx = df.index

    # --- Run baselines ---
    logger.info("Running baselines...")
    baseline_results = run_all_baselines(
        prices, dates_idx, horizons, thresholds,
        iv_csv_path=cfg.model.implied_vol_csv_path if cfg.model.implied_vol_enabled else None,
        garch_window=cfg.model.garch_window,
        t_df=cfg.model.t_df,
    )

    # --- Evaluate everything on test set ---
    all_rows = []

    # Full model evaluation
    for H in horizons:
        y_col = f"y_{H}"
        p_col = f"p_cal_{H}"
        if y_col not in test_results.columns or p_col not in test_results.columns:
            continue

        p = test_results[p_col].to_numpy(dtype=float)
        y = test_results[y_col].to_numpy(dtype=float)
        mask = np.isfinite(p) & np.isfinite(y)
        if mask.sum() < 30:
            continue

        p_m, y_m = p[mask], y[mask]

        # Era breakdown
        test_dates_arr = pd.to_datetime(test_results["date"].to_numpy())
        eras = [_era_label(d) for d in test_dates_arr[mask]]

        # Significance vs climatology
        clim = float(np.mean(y_m))
        model_losses = (p_m - y_m) ** 2
        clim_losses = (np.full_like(y_m, clim) - y_m) ** 2
        pval_clim = paired_bootstrap_loss_diff_pvalue(model_losses, clim_losses, n_boot=2000)

        n_eff = effective_sample_size(y_m, H)

        all_rows.append({
            "ticker": ticker.upper(),
            "method": "Full Model",
            "horizon": H,
            "bss": brier_skill_score(p_m, y_m),
            "auc": auc_roc(p_m, y_m),
            "ece": expected_calibration_error(p_m, y_m, adaptive=False),
            "brier": brier_score(p_m, y_m),
            "n": int(mask.sum()),
            "n_eff": round(n_eff, 1),
            "event_rate": clim,
            "p_value_vs_clim": pval_clim,
            "test_start": cutoff,
        })

        # Per-era breakdown
        for era in sorted(set(eras)):
            era_mask = np.array([e == era for e in eras])
            if era_mask.sum() < 20:
                continue
            p_era, y_era = p_m[era_mask], y_m[era_mask]
            all_rows.append({
                "ticker": ticker.upper(),
                "method": f"Full Model ({era})",
                "horizon": H,
                "bss": brier_skill_score(p_era, y_era),
                "auc": auc_roc(p_era, y_era),
                "ece": expected_calibration_error(p_era, y_era, adaptive=False),
                "brier": brier_score(p_era, y_era),
                "n": int(era_mask.sum()),
                "n_eff": round(effective_sample_size(y_era, H), 1),
                "event_rate": float(np.mean(y_era)),
                "p_value_vs_clim": np.nan,
                "test_start": cutoff,
            })

    # Baseline evaluations (restrict to test period)
    test_start_idx = int(np.searchsorted(np.arange(len(prices)), n_train))
    for baseline_name, bl_df in baseline_results.items():
        if len(bl_df) == 0:
            continue

        # Filter to test period indices
        bl_test = bl_df[bl_df["idx"] >= test_start_idx]
        if len(bl_test) < 30:
            continue

        for H in horizons:
            p_col = f"p_baseline_{H}"
            y_col = f"y_{H}"
            if p_col not in bl_test.columns:
                continue

            p = bl_test[p_col].to_numpy(dtype=float)
            y = bl_test[y_col].to_numpy(dtype=float)
            mask = np.isfinite(p) & np.isfinite(y)
            if mask.sum() < 30:
                continue

            p_m, y_m = p[mask], y[mask]
            clim = float(np.mean(y_m))
            model_losses = (p_m - y_m) ** 2
            clim_losses = (np.full_like(y_m, clim) - y_m) ** 2
            pval = paired_bootstrap_loss_diff_pvalue(model_losses, clim_losses, n_boot=2000)

            all_rows.append({
                "ticker": ticker.upper(),
                "method": baseline_name,
                "horizon": H,
                "bss": brier_skill_score(p_m, y_m),
                "auc": auc_roc(p_m, y_m),
                "ece": expected_calibration_error(p_m, y_m, adaptive=False),
                "brier": brier_score(p_m, y_m),
                "n": int(mask.sum()),
                "n_eff": round(effective_sample_size(y_m, H), 1),
                "event_rate": clim,
                "p_value_vs_clim": pval,
                "test_start": cutoff,
            })

    results_df = pd.DataFrame(all_rows)

    # Save
    os.makedirs("outputs/paper", exist_ok=True)
    outpath = f"outputs/paper/holdout_{ticker.lower()}.csv"
    results_df.to_csv(outpath, index=False)
    logger.info("Hold-out results saved to %s", outpath)

    # Print summary
    print(f"\n{'='*80}")
    print(f"TEMPORAL HOLD-OUT: {ticker.upper()} (train through {cutoff})")
    print(f"{'='*80}")
    main_results = results_df[~results_df["method"].str.contains(r"\(")]
    for H in horizons:
        print(f"\n--- Horizon H={H} ---")
        h_res = main_results[main_results["horizon"] == H]
        if len(h_res) > 0:
            print(h_res[["method", "bss", "auc", "ece", "n", "p_value_vs_clim"]].to_string(index=False))

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Temporal hold-out evaluation")
    parser.add_argument("ticker", help="Ticker or 'all'")
    parser.add_argument("--cutoff", default="2019-12-31", help="Train/test cutoff date")
    args = parser.parse_args()

    if args.ticker.lower() == "all":
        all_results = []
        for ticker in TICKER_CONFIGS:
            try:
                result = run_temporal_holdout(ticker, cutoff=args.cutoff)
                all_results.append(result)
            except Exception as e:
                logger.error("Hold-out failed for %s: %s", ticker, e)
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            combined.to_csv("outputs/paper/holdout_all.csv", index=False)
    else:
        run_temporal_holdout(args.ticker, cutoff=args.cutoff)


if __name__ == "__main__":
    main()
