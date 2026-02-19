"""
Model selection framework: expanding-window cross-validation
and information criteria for systematic model comparison.

Usage:
    python -m em_sde.run --compare configs/spy_fixed.yaml configs/spy.yaml
"""

import logging
from typing import List

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import PipelineConfig
from .backtest import run_walkforward
from .evaluation import (
    brier_score, brier_skill_score, log_loss, auc_roc, separation,
    expected_calibration_error,
)

logger = logging.getLogger(__name__)


def _as_float_array(values: object) -> NDArray[np.float64]:
    """Return a 1D float64 numpy array from pandas/numpy array-like inputs."""
    return np.asarray(values, dtype=np.float64).reshape(-1)


def expanding_window_cv(
    df: pd.DataFrame,
    configs: List[PipelineConfig],
    config_names: List[str],
    n_folds: int = 5,
    min_train_pct: float = 0.4,
) -> pd.DataFrame:
    """
    Run expanding-window cross-validation over multiple configs.

    The data is split into n_folds test windows. For each fold,
    the model trains on all data up to the test start, then
    is evaluated on the test window only.

    Parameters
    ----------
    df : pd.DataFrame
        Full price data with DatetimeIndex and 'price' column.
    configs : list of PipelineConfig
        Model configurations to compare.
    config_names : list of str
        Human-readable names for each config.
    n_folds : int
        Number of expanding-window folds.
    min_train_pct : float
        Minimum fraction of data used for training in the first fold.

    Returns
    -------
    results : pd.DataFrame
        Columns: config_name, fold, horizon, brier, bss, auc, logloss,
                 separation, event_rate, n
    """
    n = len(df)
    min_train = int(n * min_train_pct)
    remaining = n - min_train
    fold_size = remaining // n_folds

    if fold_size < 50:
        logger.warning("Fold size %d is very small; results may be noisy", fold_size)

    rows = []

    for fold_i in range(n_folds):
        train_end = min_train + fold_i * fold_size
        test_end = min(train_end + fold_size, n)

        logger.info(
            "CV fold %d/%d: train=[0:%d], test=[%d:%d]",
            fold_i + 1, n_folds, train_end, train_end, test_end,
        )

        for cfg, name in zip(configs, config_names):
            # Run walk-forward on data up to test_end
            df_fold = df.iloc[:test_end].copy()
            results = run_walkforward(df_fold, cfg)

            # Extract test portion only (rows generated from train_end onward)
            test_dates = list(df.index[train_end:test_end])
            test_mask = results["date"].isin(test_dates)
            test_res = results[test_mask]

            if len(test_res) == 0:
                continue

            for H in cfg.model.horizons:
                y_col = f"y_{H}"
                if y_col not in test_res.columns:
                    continue

                y = _as_float_array(test_res[y_col])
                p_raw = _as_float_array(test_res[f"p_raw_{H}"])
                p_cal = _as_float_array(test_res[f"p_cal_{H}"])

                mask = ~np.isnan(y)
                if mask.sum() < 10:
                    continue

                y_m = y[mask]
                p_raw_m = p_raw[mask]
                p_cal_m = p_cal[mask]

                # Sigma mean for regime bucketing
                sigma_mean = np.nan
                if "sigma_garch_1d" in test_res.columns:
                    sigma_mean = float(np.nanmean(
                        _as_float_array(test_res["sigma_garch_1d"])
                    ))

                rows.append({
                    "config_name": name,
                    "fold": fold_i,
                    "horizon": H,
                    "brier_raw": brier_score(p_raw_m, y_m),
                    "brier_cal": brier_score(p_cal_m, y_m),
                    "bss_raw": brier_skill_score(p_raw_m, y_m),
                    "bss_cal": brier_skill_score(p_cal_m, y_m),
                    "auc_raw": auc_roc(p_raw_m, y_m),
                    "auc_cal": auc_roc(p_cal_m, y_m),
                    "logloss_cal": log_loss(p_cal_m, y_m),
                    "separation_cal": separation(p_cal_m, y_m),
                    "ece_cal": expected_calibration_error(p_cal_m, y_m),
                    "event_rate": float(np.nanmean(y_m)),
                    "n": int(mask.sum()),
                    "sigma_mean": sigma_mean,
                })

    return pd.DataFrame(rows)


def compare_models(cv_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate CV results: mean +/- std across folds.
    Rank models by BSS (primary) and AUC (secondary).

    Parameters
    ----------
    cv_results : pd.DataFrame
        Output of expanding_window_cv.

    Returns
    -------
    summary : pd.DataFrame
        Aggregated comparison with ranking.
    """
    if len(cv_results) == 0:
        return pd.DataFrame()

    agg = cv_results.groupby(["config_name", "horizon"]).agg(
        bss_cal_mean=("bss_cal", "mean"),
        bss_cal_std=("bss_cal", "std"),
        auc_cal_mean=("auc_cal", "mean"),
        auc_cal_std=("auc_cal", "std"),
        brier_cal_mean=("brier_cal", "mean"),
        logloss_cal_mean=("logloss_cal", "mean"),
        n_folds=("fold", "count"),
        total_n=("n", "sum"),
    ).reset_index()

    # Rank by BSS descending (per horizon)
    agg["rank"] = agg.groupby("horizon")["bss_cal_mean"].rank(
        ascending=False, method="min"
    ).astype(int)

    return agg.sort_values(["horizon", "rank"])


def calibration_aic(p_cal: np.ndarray, y: np.ndarray, n_params: int) -> float:
    """AIC for calibration model: AIC = 2k - 2*LL."""
    mask = ~np.isnan(y) & ~np.isnan(p_cal)
    n = mask.sum()
    if n == 0:
        return np.nan
    ll = -log_loss(p_cal[mask], y[mask]) * n
    return float(2 * n_params - 2 * ll)


def calibration_bic(p_cal: np.ndarray, y: np.ndarray, n_params: int) -> float:
    """BIC for calibration model: BIC = k*ln(n) - 2*LL."""
    mask = ~np.isnan(y) & ~np.isnan(p_cal)
    n = mask.sum()
    if n == 0:
        return np.nan
    ll = -log_loss(p_cal[mask], y[mask]) * n
    return float(n_params * np.log(n) - 2 * ll)


def apply_promotion_gates(
    cv_results: pd.DataFrame,
    gates: dict = None,
) -> pd.DataFrame:
    """
    Apply hard promotion gates per regime bucket.

    Splits CV results by vol regime (sigma_mean terciles across folds),
    then checks that every gate passes in every regime bucket.
    A model fails if ANY gate fails in ANY bucket.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Output of expanding_window_cv (must include ece_cal and sigma_mean).
    gates : dict, optional
        Gate thresholds. Default: {"bss_cal": 0.0, "auc_cal": 0.55, "ece_cal": 0.02}

    Returns
    -------
    gate_report : pd.DataFrame
        Columns: config_name, horizon, regime, metric, value, threshold,
                 passed, margin, all_gates_passed
    """
    if gates is None:
        gates = {"bss_cal": 0.0, "auc_cal": 0.55, "ece_cal": 0.02}

    if len(cv_results) == 0:
        return pd.DataFrame()

    rows = []

    for (config_name, horizon), group in cv_results.groupby(["config_name", "horizon"]):
        # Bucket folds by vol regime using sigma_mean terciles
        if "sigma_mean" in group.columns and group["sigma_mean"].notna().any():
            sigma_vals = group["sigma_mean"].values
            p33 = float(np.nanpercentile(sigma_vals, 33))
            p66 = float(np.nanpercentile(sigma_vals, 66))

            regimes = []
            for s in sigma_vals:
                if np.isnan(s):
                    regimes.append("all")
                elif s < p33:
                    regimes.append("low_vol")
                elif s > p66:
                    regimes.append("high_vol")
                else:
                    regimes.append("mid_vol")
            group = group.copy()
            group["regime"] = regimes
        else:
            group = group.copy()
            group["regime"] = "all"

        for regime, regime_group in group.groupby("regime"):
            for metric, threshold in gates.items():
                if metric not in regime_group.columns:
                    continue
                value = float(regime_group[metric].mean())

                if metric == "ece_cal":
                    passed = value <= threshold
                    margin = round(threshold - value, 6)
                else:
                    passed = value >= threshold
                    margin = round(value - threshold, 6)

                rows.append({
                    "config_name": config_name,
                    "horizon": horizon,
                    "regime": regime,
                    "metric": metric,
                    "value": round(value, 6),
                    "threshold": threshold,
                    "passed": bool(passed),
                    "margin": margin,
                })

    report = pd.DataFrame(rows)

    if len(report) > 0:
        summary = report.groupby(["config_name", "horizon"])["passed"].all().reset_index()
        summary.rename(columns={"passed": "all_gates_passed"}, inplace=True)
        report = report.merge(summary, on=["config_name", "horizon"])

    return report
