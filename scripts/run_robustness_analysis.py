"""
Robustness analysis: addresses blind spots identified in critical review.

1. VIX threshold rule baseline (practitioner comparison)
2. Market-implied probability baseline (straddle comparison)
3. Prediction sharpness analysis
4. Conditional failure analysis (ECE by vol regime, era)
5. Universal threshold test (5% and 10% for all tickers)
6. Cross-asset prediction correlation
7. AAPL honest failure appendix
8. Pre-registered framing for NVDA

Usage:
    python scripts/run_robustness_analysis.py all
    python scripts/run_robustness_analysis.py spy
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
    brier_skill_score, auc_roc, expected_calibration_error,
    prediction_sharpness, conditional_ece,
)
from scripts.baselines import (
    vix_threshold_baseline, market_implied_probability_baseline,
    evaluate_baseline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TICKER_CONFIGS = {
    "spy": "configs/exp_suite/exp_spy_regime_gated.yaml",
    "googl": "configs/exp_suite/exp_googl_regime_gated.yaml",
    "amzn": "configs/exp_suite/exp_amzn_regime_gated.yaml",
    "nvda": "configs/exp_suite/exp_nvda_regime_gated.yaml",
}

# AAPL included for honest failure analysis
AAPL_CONFIG = "configs/archive/exp_aapl_regime_gated.yaml"


def run_sharpness_analysis(ticker: str) -> pd.DataFrame:
    """Analyze prediction sharpness -- is the model saying anything beyond the base rate?"""
    config_path = TICKER_CONFIGS[ticker.lower()]
    cfg = load_config(config_path)
    df, _ = load_data(cfg)

    logger.info("Running walk-forward for sharpness analysis: %s", ticker.upper())
    results = run_walkforward(df, cfg)

    rows = []
    for H in cfg.model.horizons:
        p_col = f"p_cal_{H}"
        y_col = f"y_{H}"
        if p_col not in results.columns:
            continue

        p = results[p_col].to_numpy(dtype=float)
        y = results[y_col].to_numpy(dtype=float)
        sharp = prediction_sharpness(p, y)

        rows.append({
            "ticker": ticker.upper(),
            "horizon": H,
            "p_std": sharp["std"],
            "p_iqr": sharp["iqr"],
            "p_range": sharp["range"],
            "p_min": sharp["min"],
            "p_max": sharp["max"],
            "pct_deviate_5pp": sharp["pct_deviate_5pp"],
            "base_rate": sharp["base_rate"],
        })

    return pd.DataFrame(rows)


def run_conditional_failure_analysis(ticker: str) -> pd.DataFrame:
    """ECE conditioned on vol regime and time era."""
    config_path = TICKER_CONFIGS[ticker.lower()]
    cfg = load_config(config_path)
    df, _ = load_data(cfg)

    logger.info("Running walk-forward for conditional analysis: %s", ticker.upper())
    results = run_walkforward(df, cfg)
    prices = df["price"].to_numpy(dtype=float)
    returns = np.diff(prices) / prices[:-1]

    # Compute rolling 20d realized vol for each prediction row
    n_results = len(results)
    vol_20d = np.full(n_results, np.nan)
    for i in range(20, len(returns)):
        if i < n_results:
            vol_20d[i] = float(np.std(returns[max(0, i-20):i]) * np.sqrt(252))

    # Time eras
    dates = df.index[:n_results]
    era = np.full(n_results, np.nan)
    for i in range(n_results):
        if i < len(dates):
            yr = dates[i].year
            if yr < 2020:
                era[i] = 0  # Pre-COVID
            elif yr == 2020:
                era[i] = 1  # COVID
            elif yr <= 2022:
                era[i] = 2  # Tightening
            else:
                era[i] = 3  # Post-2023

    era_bins = [
        ("Pre-2020", -0.5, 0.5),
        ("COVID-2020", 0.5, 1.5),
        ("Tightening 2021-22", 1.5, 2.5),
        ("Post-2023", 2.5, 4.0),
    ]

    rows = []
    for H in cfg.model.horizons:
        p_col = f"p_cal_{H}"
        y_col = f"y_{H}"
        if p_col not in results.columns:
            continue

        p = results[p_col].to_numpy(dtype=float)
        y = results[y_col].to_numpy(dtype=float)

        # By vol regime
        vol_results = conditional_ece(p, y, vol_20d)
        for r in vol_results:
            rows.append({
                "ticker": ticker.upper(), "horizon": H,
                "condition": f"Vol-{r['name']}", "ece": r["ece"],
                "n": r["n"], "base_rate": r["base_rate"],
            })

        # By time era
        era_results = conditional_ece(p, y, era, condition_bins=era_bins)
        for r in era_results:
            rows.append({
                "ticker": ticker.upper(), "horizon": H,
                "condition": r["name"], "ece": r["ece"],
                "n": r["n"], "base_rate": r["base_rate"],
            })

    return pd.DataFrame(rows)


def run_universal_threshold_test() -> pd.DataFrame:
    """Test 5% and 10% fixed thresholds across all tickers without BO tuning."""
    rows = []
    universal_thresholds = [0.05, 0.10]

    for ticker, config_path in TICKER_CONFIGS.items():
        cfg = load_config(config_path)
        df, _ = load_data(cfg)

        for univ_thr in universal_thresholds:
            # Override thresholds
            import copy
            cfg_univ = copy.deepcopy(cfg)
            cfg_univ.model.threshold_mode = "fixed_pct"
            cfg_univ.model.fixed_threshold_pct = univ_thr
            cfg_univ.model.regime_gated_fixed_pct_by_horizon = {}
            cfg_univ.output.charts = False

            logger.info("Universal threshold test: %s @ %.0f%%", ticker.upper(), univ_thr * 100)
            try:
                results = run_walkforward(df, cfg_univ)
                for H in cfg.model.horizons:
                    p_col = f"p_cal_{H}"
                    y_col = f"y_{H}"
                    if p_col not in results.columns:
                        continue

                    p = results[p_col].to_numpy(dtype=float)
                    y = results[y_col].to_numpy(dtype=float)
                    mask = np.isfinite(p) & np.isfinite(y)
                    if mask.sum() < 50:
                        continue
                    p_m, y_m = p[mask], y[mask]

                    rows.append({
                        "ticker": ticker.upper(),
                        "threshold": f"{univ_thr*100:.0f}%",
                        "horizon": H,
                        "bss": brier_skill_score(p_m, y_m),
                        "auc": auc_roc(p_m, y_m),
                        "ece": expected_calibration_error(p_m, y_m, adaptive=False),
                        "event_rate": float(np.mean(y_m)),
                        "n": int(mask.sum()),
                        "gates": "PASS" if (brier_skill_score(p_m, y_m) > 0 and
                                            auc_roc(p_m, y_m) >= 0.55 and
                                            expected_calibration_error(p_m, y_m, adaptive=False) <= 0.02) else "FAIL",
                    })
            except Exception as e:
                logger.error("Failed for %s @ %.0f%%: %s", ticker, univ_thr * 100, e)

    return pd.DataFrame(rows)


def run_cross_asset_correlation() -> pd.DataFrame:
    """Compute correlation of p_cal predictions across tickers."""
    all_preds = {}

    for ticker, config_path in TICKER_CONFIGS.items():
        cfg = load_config(config_path)
        df, _ = load_data(cfg)

        logger.info("Running walk-forward for cross-asset: %s", ticker.upper())
        results = run_walkforward(df, cfg)

        for H in cfg.model.horizons:
            p_col = f"p_cal_{H}"
            if p_col in results.columns:
                key = f"{ticker.upper()}_H{H}"
                # Use date as index for alignment
                preds = results[p_col].copy()
                preds.index = df.index[:len(preds)]
                all_preds[key] = preds

    if len(all_preds) < 2:
        return pd.DataFrame()

    # Align by date and compute correlations
    pred_df = pd.DataFrame(all_preds)
    corr = pred_df.corr()

    return corr


def run_aapl_failure_analysis() -> pd.DataFrame:
    """Run AAPL as honest failure analysis for appendix."""
    if not os.path.exists(AAPL_CONFIG):
        logger.warning("AAPL config not found at %s", AAPL_CONFIG)
        return pd.DataFrame()

    cfg = load_config(AAPL_CONFIG)
    df, _ = load_data(cfg)

    logger.info("Running AAPL failure analysis (%d rows)...", len(df))
    results = run_walkforward(df, cfg)

    rows = []
    for H in cfg.model.horizons:
        p_col = f"p_cal_{H}"
        y_col = f"y_{H}"
        if p_col not in results.columns:
            continue

        p = results[p_col].to_numpy(dtype=float)
        y = results[y_col].to_numpy(dtype=float)
        mask = np.isfinite(p) & np.isfinite(y)
        if mask.sum() < 50:
            continue
        p_m, y_m = p[mask], y[mask]
        sharp = prediction_sharpness(p, y)

        rows.append({
            "ticker": "AAPL",
            "horizon": H,
            "bss": brier_skill_score(p_m, y_m),
            "auc": auc_roc(p_m, y_m),
            "ece": expected_calibration_error(p_m, y_m, adaptive=False),
            "event_rate": float(np.mean(y_m)),
            "n": int(mask.sum()),
            "p_std": sharp["std"],
            "pct_deviate_5pp": sharp["pct_deviate_5pp"],
            "gates": "PASS" if (brier_skill_score(p_m, y_m) > 0 and
                                auc_roc(p_m, y_m) >= 0.55 and
                                expected_calibration_error(p_m, y_m, adaptive=False) <= 0.02) else "FAIL",
        })

    return pd.DataFrame(rows)


def run_vix_and_straddle_baselines() -> pd.DataFrame:
    """Run VIX rule and market-implied baselines for all tickers."""
    all_rows = []

    for ticker, config_path in TICKER_CONFIGS.items():
        cfg = load_config(config_path)
        df, _ = load_data(cfg)
        prices = df["price"].to_numpy(dtype=float)
        dates_idx = df.index
        horizons = cfg.model.horizons

        thresholds = {}
        fixed_pct = cfg.model.regime_gated_fixed_pct_by_horizon or {}
        for H in horizons:
            thresholds[H] = fixed_pct.get(H, cfg.model.fixed_threshold_pct)

        iv_path = cfg.model.implied_vol_csv_path if cfg.model.implied_vol_enabled else "data/vix_history.csv"

        # VIX rule
        vix_df = vix_threshold_baseline(prices, dates_idx, horizons, thresholds, iv_csv_path=iv_path)
        if len(vix_df) > 0:
            for H in horizons:
                p_col = f"p_baseline_{H}"
                y_col = f"y_{H}"
                if p_col not in vix_df.columns:
                    continue
                p = vix_df[p_col].to_numpy(dtype=float)
                y = vix_df[y_col].to_numpy(dtype=float)
                mask = np.isfinite(p) & np.isfinite(y)
                if mask.sum() < 50:
                    continue
                p_m, y_m = p[mask], y[mask]
                all_rows.append({
                    "ticker": ticker.upper(), "baseline": "VIX Threshold Rule",
                    "horizon": H, "bss": brier_skill_score(p_m, y_m),
                    "auc": auc_roc(p_m, y_m),
                    "ece": expected_calibration_error(p_m, y_m, adaptive=False),
                })

        # Market-implied
        mkt_df = market_implied_probability_baseline(prices, dates_idx, horizons, thresholds, iv_csv_path=iv_path)
        if len(mkt_df) > 0:
            for H in horizons:
                p_col = f"p_baseline_{H}"
                y_col = f"y_{H}"
                if p_col not in mkt_df.columns:
                    continue
                p = mkt_df[p_col].to_numpy(dtype=float)
                y = mkt_df[y_col].to_numpy(dtype=float)
                mask = np.isfinite(p) & np.isfinite(y)
                if mask.sum() < 50:
                    continue
                p_m, y_m = p[mask], y[mask]
                all_rows.append({
                    "ticker": ticker.upper(), "baseline": "Market-Implied (Straddle)",
                    "horizon": H, "bss": brier_skill_score(p_m, y_m),
                    "auc": auc_roc(p_m, y_m),
                    "ece": expected_calibration_error(p_m, y_m, adaptive=False),
                })

    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description="Robustness analysis for blind spots")
    parser.add_argument("ticker", help="Ticker or 'all'")
    parser.add_argument("--skip-heavy", action="store_true", help="Skip cross-asset and universal threshold tests")
    args = parser.parse_args()

    os.makedirs("outputs/paper/robustness", exist_ok=True)

    tickers = list(TICKER_CONFIGS.keys()) if args.ticker.lower() == "all" else [args.ticker.lower()]

    # 1 & 2: VIX rule and straddle baselines
    logger.info("=" * 60)
    logger.info("1-2. VIX Threshold Rule + Market-Implied Baselines")
    logger.info("=" * 60)
    bl_df = run_vix_and_straddle_baselines()
    if len(bl_df) > 0:
        bl_df.to_csv("outputs/paper/robustness/vix_straddle_baselines.csv", index=False)
        print("\nVIX & Straddle Baselines:")
        print(bl_df.to_string(index=False))

    # 3: Sharpness
    logger.info("=" * 60)
    logger.info("3. Prediction Sharpness Analysis")
    logger.info("=" * 60)
    all_sharp = []
    for ticker in tickers:
        try:
            sharp_df = run_sharpness_analysis(ticker)
            all_sharp.append(sharp_df)
        except Exception as e:
            logger.error("Sharpness failed for %s: %s", ticker, e)
    if all_sharp:
        combined = pd.concat(all_sharp, ignore_index=True)
        combined.to_csv("outputs/paper/robustness/sharpness.csv", index=False)
        print("\nSharpness Analysis:")
        print(combined.to_string(index=False))

    # 4: Conditional failure analysis
    logger.info("=" * 60)
    logger.info("4. Conditional ECE Analysis")
    logger.info("=" * 60)
    all_cond = []
    for ticker in tickers:
        try:
            cond_df = run_conditional_failure_analysis(ticker)
            all_cond.append(cond_df)
        except Exception as e:
            logger.error("Conditional analysis failed for %s: %s", ticker, e)
    if all_cond:
        combined = pd.concat(all_cond, ignore_index=True)
        combined.to_csv("outputs/paper/robustness/conditional_ece.csv", index=False)
        print("\nConditional ECE:")
        print(combined.to_string(index=False))

    if not args.skip_heavy:
        # 5: Universal threshold test
        logger.info("=" * 60)
        logger.info("5. Universal Threshold Test (5%%, 10%%)")
        logger.info("=" * 60)
        univ_df = run_universal_threshold_test()
        if len(univ_df) > 0:
            univ_df.to_csv("outputs/paper/robustness/universal_thresholds.csv", index=False)
            print("\nUniversal Thresholds:")
            print(univ_df.to_string(index=False))

        # 6: Cross-asset correlation
        logger.info("=" * 60)
        logger.info("6. Cross-Asset Prediction Correlation")
        logger.info("=" * 60)
        corr_df = run_cross_asset_correlation()
        if len(corr_df) > 0:
            corr_df.to_csv("outputs/paper/robustness/cross_asset_correlation.csv")
            print("\nCross-Asset Correlation:")
            print(corr_df.to_string())

    # 7: AAPL failure analysis
    logger.info("=" * 60)
    logger.info("7. AAPL Honest Failure Analysis")
    logger.info("=" * 60)
    aapl_df = run_aapl_failure_analysis()
    if len(aapl_df) > 0:
        aapl_df.to_csv("outputs/paper/robustness/aapl_failure.csv", index=False)
        print("\nAAPL Failure Analysis:")
        print(aapl_df.to_string(index=False))

    print("\n=== Robustness analysis saved to outputs/paper/robustness/ ===")


if __name__ == "__main__":
    main()
