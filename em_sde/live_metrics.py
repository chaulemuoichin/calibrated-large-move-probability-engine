"""
Compute cumulative out-of-sample metrics from resolved live forecasts.

Uses only resolved forecasts — never pending ones. All metrics are
computed from the joined forecast+resolution ledger.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .evaluation import (
    brier_score,
    brier_skill_score,
    auc_roc,
    expected_calibration_error,
    expected_calibration_error_detailed,
    effective_sample_size,
    bootstrap_metric_ci,
)
from .ledger import load_joined

logger = logging.getLogger(__name__)


def compute_live_metrics(
    joined: Optional[pd.DataFrame] = None,
    forecast_path=None,
    resolution_path=None,
    min_resolved: int = 10,
) -> dict:
    """
    Compute pooled live metrics from all resolved forecasts.

    Returns a dict with metrics, sample sizes, and warnings.
    """
    if joined is None:
        joined = load_joined(forecast_path, resolution_path)

    if "event_occurred" not in joined.columns:
        resolved = pd.DataFrame()
    else:
        resolved = joined.dropna(subset=["event_occurred"]).copy()
    n_total = len(joined)
    n_resolved = len(resolved)
    n_pending = n_total - n_resolved

    result = {
        "n_total_forecasts": n_total,
        "n_resolved": n_resolved,
        "n_pending": n_pending,
        "metrics": None,
        "warnings": [],
    }

    if n_resolved < min_resolved:
        result["warnings"].append(
            f"Only {n_resolved} resolved forecasts (minimum {min_resolved} "
            f"needed for meaningful metrics). Metrics are unreliable at this sample size."
        )
        if n_resolved < 3:
            return result

    p_cal = resolved["p_cal"].to_numpy(dtype=float)
    y = resolved["event_occurred"].to_numpy(dtype=float)
    event_rate = float(np.mean(y))

    brier = brier_score(p_cal, y)
    bss = brier_skill_score(p_cal, y)
    auc = auc_roc(p_cal, y)

    n_events = int(np.sum(y))
    n_nonevents = n_resolved - n_events
    n_bins = max(3, min(10, n_resolved // 15))
    ece = expected_calibration_error(p_cal, y, n_bins=n_bins, adaptive=True)

    # ECE detailed with bin info
    ece_detail = expected_calibration_error_detailed(
        p_cal, y, n_bins=n_bins, adaptive=True,
    )

    # Sharpness
    p_std = float(np.std(p_cal))
    p_min = float(np.min(p_cal))
    p_max = float(np.max(p_cal))
    p_mean = float(np.mean(p_cal))

    # Calibration bins for reliability curve
    cal_bins = _compute_calibration_bins(p_cal, y, n_bins=n_bins)

    metrics = {
        "brier_score": round(brier, 6),
        "brier_skill_score": round(bss, 6),
        "auc": round(auc, 4) if not np.isnan(auc) else None,
        "ece": round(ece, 6),
        "n_resolved": n_resolved,
        "n_events": n_events,
        "n_nonevents": n_nonevents,
        "event_rate": round(event_rate, 4),
        "p_cal_mean": round(p_mean, 4),
        "p_cal_std": round(p_std, 4),
        "p_cal_min": round(p_min, 4),
        "p_cal_max": round(p_max, 4),
        "calibration_bins": cal_bins,
        "ece_detail": ece_detail,
    }

    # Bootstrap CIs (only if enough data)
    if n_resolved >= 30 and n_events >= 3 and n_nonevents >= 3:
        try:
            _point, bss_lo, bss_hi = bootstrap_metric_ci(
                y, p_cal, brier_skill_score, n_boot=1000, alpha=0.05,
            )
            if not (np.isnan(bss_lo) or np.isnan(bss_hi)):
                metrics["bss_ci_95"] = [round(bss_lo, 4), round(bss_hi, 4)]
        except Exception:
            pass

    # Warnings for small sample sizes
    if n_resolved < 50:
        result["warnings"].append(
            f"Sample size ({n_resolved}) is small. Metrics are preliminary "
            f"and should not be treated as statistically conclusive."
        )
    if n_events < 5:
        result["warnings"].append(
            f"Only {n_events} events observed. AUC and event-conditional "
            f"metrics are particularly unreliable."
        )

    result["metrics"] = metrics
    return result


def compute_per_group_metrics(
    joined: Optional[pd.DataFrame] = None,
    group_col: str = "ticker",
    min_resolved: int = 5,
) -> Dict[str, dict]:
    """Compute metrics grouped by ticker, horizon, or model_version."""
    if joined is None:
        joined = load_joined()

    if "event_occurred" not in joined.columns:
        return {}
    resolved = joined.dropna(subset=["event_occurred"])
    results = {}

    for group_val, group_df in resolved.groupby(group_col):
        p_cal = group_df["p_cal"].to_numpy(dtype=float)
        y = group_df["event_occurred"].to_numpy(dtype=float)

        n = len(y)
        n_events = int(np.sum(y))

        if n < min_resolved:
            results[str(group_val)] = {
                "n": n, "n_events": n_events,
                "warning": f"Too few samples ({n}) for reliable metrics",
            }
            continue

        n_bins = max(3, min(10, n // 15))
        results[str(group_val)] = {
            "n": n,
            "n_events": n_events,
            "event_rate": round(float(np.mean(y)), 4),
            "brier_score": round(brier_score(p_cal, y), 6),
            "bss": round(brier_skill_score(p_cal, y), 6),
            "auc": round(auc_roc(p_cal, y), 4) if n_events >= 2 and (n - n_events) >= 2 else None,
            "ece": round(expected_calibration_error(p_cal, y, n_bins=n_bins), 6),
            "p_cal_mean": round(float(np.mean(p_cal)), 4),
        }

    return results


def compute_rolling_metrics(
    joined: Optional[pd.DataFrame] = None,
    window: int = 50,
    step: int = 10,
) -> pd.DataFrame:
    """Compute rolling metrics over resolved forecasts (by resolution order)."""
    if joined is None:
        joined = load_joined()

    if "event_occurred" not in joined.columns:
        return pd.DataFrame()
    resolved = joined.dropna(subset=["event_occurred"]).copy()
    resolved = resolved.sort_values("forecast_date_market").reset_index(drop=True)

    n = len(resolved)
    if n < window:
        return pd.DataFrame()

    rows = []
    for start in range(0, n - window + 1, step):
        chunk = resolved.iloc[start:start + window]
        p_cal = chunk["p_cal"].to_numpy(dtype=float)
        y = chunk["event_occurred"].to_numpy(dtype=float)
        n_events = int(np.sum(y))

        row = {
            "window_start": chunk.iloc[0]["forecast_date_market"],
            "window_end": chunk.iloc[-1]["forecast_date_market"],
            "n": window,
            "n_events": n_events,
            "brier_score": round(brier_score(p_cal, y), 6),
            "bss": round(brier_skill_score(p_cal, y), 6),
            "ece": round(expected_calibration_error(p_cal, y, n_bins=5, adaptive=True), 6),
            "event_rate": round(float(np.mean(y)), 4),
            "p_cal_mean": round(float(np.mean(p_cal)), 4),
        }
        if n_events >= 2 and (window - n_events) >= 2:
            row["auc"] = round(auc_roc(p_cal, y), 4)
        else:
            row["auc"] = None
        rows.append(row)

    return pd.DataFrame(rows)


def _compute_calibration_bins(
    p_cal: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> List[dict]:
    """Compute calibration bins for reliability curve."""
    mask = ~np.isnan(y) & ~np.isnan(p_cal)
    p_m, y_m = p_cal[mask], y[mask]

    bin_edges = np.unique(np.quantile(p_m, np.linspace(0, 1, n_bins + 1)))
    n_actual = len(bin_edges) - 1

    bins = []
    for i in range(n_actual):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_actual - 1:
            in_bin = (p_m >= lo) & (p_m < hi)
        else:
            in_bin = (p_m >= lo) & (p_m <= hi)

        count = int(in_bin.sum())
        if count > 0:
            mean_pred = float(np.mean(p_m[in_bin]))
            mean_obs = float(np.mean(y_m[in_bin]))
            bins.append({
                "bin_lo": round(float(lo), 4),
                "bin_hi": round(float(hi), 4),
                "n": count,
                "mean_predicted": round(mean_pred, 4),
                "mean_observed": round(mean_obs, 4),
            })

    return bins
