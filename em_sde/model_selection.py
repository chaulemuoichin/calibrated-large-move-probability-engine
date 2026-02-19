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
                    "event_rate": float(np.nanmean(y_m)),
                    "n": int(mask.sum()),
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
