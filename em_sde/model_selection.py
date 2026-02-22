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
) -> tuple:
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
    cv_results : pd.DataFrame
        Fold-level summary metrics. Columns: config_name, fold, horizon,
        brier, bss, auc, logloss, separation, event_rate, n, sigma_mean.
    oof_df : pd.DataFrame
        Row-level out-of-fold predictions pooled across all folds.
        Columns: config_name, fold, horizon, p_cal, y, sigma_1d.
    """
    n = len(df)
    min_train = int(n * min_train_pct)
    remaining = n - min_train
    fold_size = remaining // n_folds

    if fold_size < 50:
        logger.warning("Fold size %d is very small; results may be noisy", fold_size)

    rows = []
    oof_parts = []

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

            # Per-row sigma for OOF collection
            sigma_1d_all = (
                _as_float_array(test_res["sigma_garch_1d"])
                if "sigma_garch_1d" in test_res.columns
                else np.full(len(test_res), np.nan)
            )

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
                sigma_1d_m = sigma_1d_all[mask]

                # Collect OOF row-level predictions
                oof_parts.append(pd.DataFrame({
                    "config_name": name,
                    "fold": fold_i,
                    "horizon": H,
                    "p_cal": p_cal_m,
                    "y": y_m,
                    "sigma_1d": sigma_1d_m,
                }))

                # Sigma mean for legacy regime bucketing
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
                    "ece_cal": expected_calibration_error(p_cal_m, y_m, adaptive=False),
                    "ece_cal_adaptive": expected_calibration_error(p_cal_m, y_m, adaptive=True),
                    "event_rate": float(np.nanmean(y_m)),
                    "n": int(mask.sum()),
                    "sigma_mean": sigma_mean,
                })

    cv_results = pd.DataFrame(rows)
    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame(
        columns=["config_name", "fold", "horizon", "p_cal", "y", "sigma_1d"]
    )
    return cv_results, oof_df


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


def _bootstrap_ece_ci(
    p: np.ndarray, y: np.ndarray,
    n_bins: int = 10, n_boot: int = 1000,
    alpha: float = 0.05, seed: int = 42,
) -> tuple:
    """Bootstrap confidence interval for equal-width ECE."""
    rng = np.random.default_rng(seed)
    n = len(p)
    ece_samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ece_samples[b] = expected_calibration_error(
            p[idx], y[idx], n_bins=n_bins, adaptive=False,
        )
    lo = float(np.percentile(ece_samples, 100 * alpha / 2))
    hi = float(np.percentile(ece_samples, 100 * (1 - alpha / 2)))
    return lo, hi


def apply_promotion_gates_oof(
    oof_df: pd.DataFrame,
    gates: dict = None,
    min_samples: int = 30,
    min_events: int = 5,
) -> pd.DataFrame:
    """
    Apply promotion gates on pooled out-of-fold row-level predictions.

    Instead of bucketing folds by sigma_mean (noisy with 5 folds), this
    assigns each OOF row to a vol regime based on its own sigma_1d, then
    computes metrics on the pooled rows per regime. Provides much more
    statistical power and defensible pass/fail decisions.

    Parameters
    ----------
    oof_df : pd.DataFrame
        Row-level OOF predictions from expanding_window_cv.
        Required columns: config_name, horizon, p_cal, y, sigma_1d.
    gates : dict, optional
        Gate thresholds. Default: {"bss_cal": 0.0, "auc_cal": 0.55, "ece_cal": 0.02}
    min_samples : int
        Minimum rows in a regime bucket for evaluation (else insufficient_data).
    min_events : int
        Minimum positive labels in a regime bucket for evaluation.

    Returns
    -------
    gate_report : pd.DataFrame
        Columns: config_name, horizon, regime, metric, value, threshold,
                 passed, margin, n_samples, n_events, status,
                 ece_ci_low, ece_ci_high, all_gates_passed
    """
    if gates is None:
        gates = {"bss_cal": 0.0, "auc_cal": 0.55, "ece_cal": 0.02}

    if len(oof_df) == 0:
        return pd.DataFrame()

    rows = []

    for (config_name, horizon), group in oof_df.groupby(["config_name", "horizon"]):
        p_cal = _as_float_array(group["p_cal"])
        y = _as_float_array(group["y"])
        sigma = _as_float_array(group["sigma_1d"])

        # Row-level regime assignment via sigma_1d terciles
        valid_sigma = sigma[~np.isnan(sigma)]
        if len(valid_sigma) < 3:
            # Not enough sigma data; evaluate as single "all" bucket
            regime_labels = np.full(len(sigma), "all")
        else:
            p33 = float(np.nanpercentile(valid_sigma, 33))
            p66 = float(np.nanpercentile(valid_sigma, 66))
            regime_labels = np.array([
                "all" if np.isnan(s)
                else "low_vol" if s < p33
                else "high_vol" if s > p66
                else "mid_vol"
                for s in sigma
            ])

        for regime in sorted(set(regime_labels)):
            rmask = regime_labels == regime
            p_r = p_cal[rmask]
            y_r = y[rmask]
            n_samples = len(y_r)
            n_events = int(np.sum(y_r))

            insufficient = n_samples < min_samples or n_events < min_events

            # Compute bootstrap CI for ECE (always, if enough data)
            ece_ci_low, ece_ci_high = np.nan, np.nan
            if not insufficient:
                ece_ci_low, ece_ci_high = _bootstrap_ece_ci(p_r, y_r)

            for metric, threshold in gates.items():
                if insufficient:
                    rows.append({
                        "config_name": config_name,
                        "horizon": horizon,
                        "regime": regime,
                        "metric": metric,
                        "value": np.nan,
                        "threshold": threshold,
                        "passed": None,
                        "margin": np.nan,
                        "n_samples": n_samples,
                        "n_events": n_events,
                        "status": "insufficient_data",
                        "ece_ci_low": np.nan,
                        "ece_ci_high": np.nan,
                    })
                    continue

                if metric == "bss_cal":
                    value = float(brier_skill_score(p_r, y_r))
                elif metric == "auc_cal":
                    value = float(auc_roc(p_r, y_r))
                elif metric == "ece_cal":
                    value = float(expected_calibration_error(
                        p_r, y_r, adaptive=False,
                    ))
                else:
                    continue

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
                    "n_samples": n_samples,
                    "n_events": n_events,
                    "status": "evaluated",
                    "ece_ci_low": round(ece_ci_low, 6) if not np.isnan(ece_ci_low) else np.nan,
                    "ece_ci_high": round(ece_ci_high, 6) if not np.isnan(ece_ci_high) else np.nan,
                })

    report = pd.DataFrame(rows)

    if len(report) > 0:
        # all_gates_passed: True only if all evaluated metrics pass in all regimes
        def _all_pass(grp):
            evaluated = grp[grp["status"] == "evaluated"]
            if len(evaluated) == 0:
                return False
            return bool(evaluated["passed"].all())

        summary = report.groupby(
            ["config_name", "horizon"]
        ).apply(_all_pass, include_groups=False).reset_index(name="all_gates_passed")
        report = report.merge(summary, on=["config_name", "horizon"])

    return report


def apply_promotion_gates(
    cv_results: pd.DataFrame,
    gates: dict = None,
) -> pd.DataFrame:
    """
    Apply hard promotion gates per regime bucket (legacy fold-level method).

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
