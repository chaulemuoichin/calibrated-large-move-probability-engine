"""
Evaluation metrics: Brier score, LogLoss, AUC-ROC, and separation for both
overlapping (all days) and non-overlapping (every H days) windows.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

_EPS = 1e-15


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and outcomes."""
    mask = ~np.isnan(y) & ~np.isnan(p)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean((p[mask] - y[mask]) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Binary cross-entropy loss."""
    mask = ~np.isnan(y) & ~np.isnan(p)
    if mask.sum() == 0:
        return np.nan
    p_clip = np.clip(p[mask], _EPS, 1.0 - _EPS)
    y_m = y[mask]
    return float(-np.mean(y_m * np.log(p_clip) + (1 - y_m) * np.log(1 - p_clip)))


def auc_roc(p: np.ndarray, y: np.ndarray) -> float:
    """Area under ROC curve (manual implementation, no sklearn dependency)."""
    mask = ~np.isnan(y) & ~np.isnan(p)
    if mask.sum() == 0:
        return np.nan
    p_m, y_m = p[mask], y[mask]
    n_pos = y_m.sum()
    n_neg = len(y_m) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan

    # Sort by predicted probability descending
    order = np.argsort(-p_m)
    y_sorted = y_m[order]

    # Trapezoidal integration of ROC curve
    tp, fp = 0.0, 0.0
    tpr_prev, fpr_prev = 0.0, 0.0
    auc = 0.0
    for yi in y_sorted:
        if yi == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += 0.5 * (tpr + tpr_prev) * (fpr - fpr_prev)
        tpr_prev, fpr_prev = tpr, fpr
    return float(auc)


def separation(p: np.ndarray, y: np.ndarray) -> float:
    """Mean predicted probability for events minus non-events."""
    mask = ~np.isnan(y) & ~np.isnan(p)
    if mask.sum() == 0:
        return np.nan
    p_m, y_m = p[mask], y[mask]
    pos = y_m == 1
    neg = y_m == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    return float(p_m[pos].mean() - p_m[neg].mean())


def brier_skill_score(p: np.ndarray, y: np.ndarray) -> float:
    """
    Brier Skill Score: BSS = 1 - Brier_model / Brier_climatology.

    BSS > 0 means the model beats always predicting the historical event rate.
    BSS = 0 means no skill (equivalent to climatology).
    BSS < 0 means the model is worse than climatology.
    """
    mask = ~np.isnan(y) & ~np.isnan(p)
    if mask.sum() == 0:
        return np.nan
    y_m = y[mask]
    p_m = p[mask]
    brier_model = float(np.mean((p_m - y_m) ** 2))
    p_bar = float(np.mean(y_m))
    brier_clim = p_bar * (1.0 - p_bar)
    if brier_clim < 1e-15:
        return np.nan
    return 1.0 - brier_model / brier_clim


def expected_calibration_error(
    p: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    adaptive: bool = True,
) -> float:
    """
    Expected Calibration Error (ECE).

    Bins predicted probabilities, computes the absolute difference between
    mean predicted and mean observed in each bin, weighted by bin count.

    Parameters
    ----------
    p : np.ndarray
        Predicted probabilities.
    y : np.ndarray
        Binary outcomes (0 or 1).
    n_bins : int
        Number of calibration bins.
    adaptive : bool
        If True (default), use quantile-based (equal-mass) bins derived from
        the prediction distribution.  This prevents a single overloaded bin
        from dominating the score when predictions cluster in a narrow range
        (common for rare events).  If False, use equal-width bins over [0, 1].

    Returns
    -------
    ece : float
        Expected calibration error. Lower is better. 0 = perfectly calibrated.
    """
    mask = ~np.isnan(y) & ~np.isnan(p)
    if mask.sum() == 0:
        return np.nan
    p_m, y_m = p[mask], y[mask]
    n = len(p_m)

    if adaptive:
        bin_edges = np.unique(np.quantile(p_m, np.linspace(0, 1, n_bins + 1)))
        if len(bin_edges) < 2:
            # All predictions identical — ECE is simply |mean_pred - mean_obs|
            return float(abs(np.mean(p_m) - np.mean(y_m)))
        n_actual_bins = len(bin_edges) - 1
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)
        n_actual_bins = n_bins

    ece = 0.0
    for i in range(n_actual_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_actual_bins - 1:
            in_bin = (p_m >= lo) & (p_m < hi)
        else:
            in_bin = (p_m >= lo) & (p_m <= hi)
        n_bin = in_bin.sum()
        if n_bin > 0:
            mean_pred = float(np.mean(p_m[in_bin]))
            mean_obs = float(np.mean(y_m[in_bin]))
            ece += (n_bin / n) * abs(mean_pred - mean_obs)

    return float(ece)


def effective_sample_size(y: np.ndarray, H: int) -> float:
    """
    Estimate effective sample size accounting for H-step overlap autocorrelation.

    Uses Bartlett-type correction: N_eff = N / (1 + 2 * sum of weighted ACF).
    For non-overlapping (H=1), returns N.
    """
    n = len(y)
    mask = ~np.isnan(y)
    y_clean = y[mask]
    n_clean = len(y_clean)
    if n_clean <= H or H <= 1:
        return float(n_clean)
    y_centered = y_clean - np.mean(y_clean)
    var_y = float(np.var(y_clean))
    if var_y < 1e-15:
        return float(n_clean)
    acf_sum = 0.0
    for lag in range(1, H):
        acf_lag = float(np.mean(y_centered[lag:] * y_centered[:-lag])) / var_y
        acf_sum += (1.0 - lag / H) * acf_lag
    denom = 1.0 + 2.0 * acf_sum
    if denom <= 0:
        return float(n_clean)
    return float(n_clean) / denom


def crps_from_quantiles(
    quantiles: np.ndarray,
    q_levels: np.ndarray,
    realized: np.ndarray,
) -> float:
    """
    Approximate CRPS from stored quantile summaries using trapezoidal integration.

    Parameters
    ----------
    quantiles : np.ndarray, shape (n_samples, n_quantiles)
        Predicted quantiles per observation.
    q_levels : np.ndarray, shape (n_quantiles,)
        Quantile levels (e.g. [0.01, 0.05, ..., 0.99]).
    realized : np.ndarray, shape (n_samples,)
        Realized values.

    Returns
    -------
    crps : float
        Mean CRPS across all observations.
    """
    mask = ~np.isnan(realized)
    if mask.sum() == 0:
        return np.nan
    quantiles_m = quantiles[mask]
    realized_m = realized[mask]
    n = len(realized_m)
    total_crps = 0.0
    for i in range(n):
        obs = realized_m[i]
        q = quantiles_m[i]
        # Build piecewise CDF: at level q_levels[j], CDF = q_levels[j]
        # CRPS = integral [F(z) - I(z >= obs)]^2 dz
        # Approximate via trapezoidal rule between quantile points
        crps_i = 0.0
        for j in range(len(q_levels) - 1):
            z_lo, z_hi = q[j], q[j + 1]
            cdf_lo, cdf_hi = q_levels[j], q_levels[j + 1]
            width = z_hi - z_lo
            if width <= 0:
                continue
            # Indicator value at midpoint
            z_mid = (z_lo + z_hi) / 2.0
            ind = 1.0 if z_mid >= obs else 0.0
            cdf_mid = (cdf_lo + cdf_hi) / 2.0
            crps_i += (cdf_mid - ind) ** 2 * width
        total_crps += crps_i
    return float(total_crps / n)


# ---------------------------------------------------------------------------
# Risk Analytics
# ---------------------------------------------------------------------------


def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Historical Value-at-Risk at given confidence level.

    VaR is the negative of the (1-confidence) quantile of returns.
    E.g., VaR(95%) is the 5th percentile loss magnitude.
    """
    mask = ~np.isnan(returns)
    if mask.sum() < 10:
        return np.nan
    return float(-np.percentile(returns[mask], (1.0 - confidence) * 100))


def conditional_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Conditional Value-at-Risk (Expected Shortfall / CVaR).

    Mean of losses beyond VaR — measures average tail loss.
    """
    mask = ~np.isnan(returns)
    if mask.sum() < 10:
        return np.nan
    cutoff = np.percentile(returns[mask], (1.0 - confidence) * 100)
    tail = returns[mask][returns[mask] <= cutoff]
    if len(tail) == 0:
        return np.nan
    return float(-np.mean(tail))


def return_skewness(returns: np.ndarray) -> float:
    """Fisher's sample skewness of return distribution."""
    mask = ~np.isnan(returns)
    x = returns[mask]
    n = len(x)
    if n < 3:
        return np.nan
    m = np.mean(x)
    s = float(np.std(x, ddof=1))
    if s < 1e-15:
        return np.nan
    return float((n / ((n - 1) * (n - 2))) * np.sum(((x - m) / s) ** 3))


def return_kurtosis(returns: np.ndarray) -> float:
    """Excess kurtosis (Fisher's definition, normal=0)."""
    mask = ~np.isnan(returns)
    x = returns[mask]
    n = len(x)
    if n < 4:
        return np.nan
    m = np.mean(x)
    s = float(np.std(x, ddof=1))
    if s < 1e-15:
        return np.nan
    m4 = float(np.mean(((x - m) / s) ** 4))
    excess = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * m4 - 3 * (n - 1)) + 3
    return excess - 3.0


def max_drawdown(prices: np.ndarray) -> Dict[str, float]:
    """
    Maximum drawdown from peak to trough.

    Returns dict with drawdown magnitude, peak index, and trough index.
    """
    mask = ~np.isnan(prices)
    if mask.sum() < 2:
        return {"max_drawdown": np.nan, "peak_idx": 0, "trough_idx": 0}
    p = prices[mask]
    running_max = np.maximum.accumulate(p)
    drawdowns = (p - running_max) / running_max
    trough_idx = int(np.argmin(drawdowns))
    peak_idx = int(np.argmax(p[:trough_idx + 1]))
    return {
        "max_drawdown": float(-drawdowns[trough_idx]),
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
    }


def compute_risk_report(
    results: pd.DataFrame,
    horizons: List[int],
) -> Dict[int, Dict[str, Any]]:
    """
    Compute risk analytics per horizon from backtest results.

    Returns nested dict with VaR, CVaR, skewness, kurtosis, drawdown
    for each horizon's realized returns.
    """
    report: Dict[int, Dict[str, Any]] = {}
    for H in horizons:
        col = f"realized_return_{H}"
        if col not in results.columns:
            continue
        rets = results[col].to_numpy(dtype=float)
        mask = ~np.isnan(rets)
        if mask.sum() < 10:
            continue

        rets_clean = rets[mask]
        report[H] = {
            "var_95": value_at_risk(rets_clean, 0.95),
            "var_99": value_at_risk(rets_clean, 0.99),
            "cvar_95": conditional_var(rets_clean, 0.95),
            "cvar_99": conditional_var(rets_clean, 0.99),
            "skewness": return_skewness(rets_clean),
            "kurtosis": return_kurtosis(rets_clean),
            "mean_return": float(np.mean(rets_clean)),
            "std_return": float(np.std(rets_clean)),
            "n": int(mask.sum()),
        }

    return report


def compute_metrics(
    results: pd.DataFrame,
    horizons: List[int],
) -> Dict[str, Any]:
    """
    Compute evaluation metrics per horizon, both overlapping and non-overlapping.

    Returns a nested dict:
        {
            "overlapping": { H: {brier_raw, brier_cal, logloss_raw, logloss_cal,
                                 auc_raw, auc_cal, separation_raw, separation_cal,
                                 event_rate, n} },
            "non_overlapping": { H: { ... same fields ... } },
        }
    """
    metrics = {"overlapping": {}, "non_overlapping": {}}

    for H in horizons:
        y_col = f"y_{H}"
        p_raw_col = f"p_raw_{H}"
        p_cal_col = f"p_cal_{H}"

        if y_col not in results.columns:
            continue

        y = results[y_col].values
        p_raw = results[p_raw_col].values
        p_cal = results[p_cal_col].values

        # A) Overlapping metrics (all rows with resolved labels)
        mask_all = ~np.isnan(y)
        metrics["overlapping"][H] = _compute_horizon_metrics(
            p_raw[mask_all], p_cal[mask_all], y[mask_all], H=H,
        )

        # B) Non-overlapping metrics (every H-th row)
        non_overlap_idx = np.arange(0, len(results), H)
        mask_no = ~np.isnan(y[non_overlap_idx])
        if mask_no.sum() > 0:
            metrics["non_overlapping"][H] = _compute_horizon_metrics(
                p_raw[non_overlap_idx][mask_no],
                p_cal[non_overlap_idx][mask_no],
                y[non_overlap_idx][mask_no],
                H=1,  # non-overlapping: no autocorrelation adjustment needed
            )
        else:
            metrics["non_overlapping"][H] = {
                "brier_raw": np.nan, "brier_cal": np.nan,
                "bss_raw": np.nan, "bss_cal": np.nan,
                "logloss_raw": np.nan, "logloss_cal": np.nan,
                "auc_raw": np.nan, "auc_cal": np.nan,
                "separation_raw": np.nan, "separation_cal": np.nan,
                "event_rate": np.nan, "n": 0, "n_eff": 0.0,
            }

    return metrics


def _compute_horizon_metrics(
    p_raw: np.ndarray, p_cal: np.ndarray, y: np.ndarray,
    H: int = 1,
) -> Dict[str, Any]:
    """Compute all evaluation metrics for raw and calibrated probabilities."""
    mask = ~np.isnan(y)
    event_rate = float(y[mask].mean()) if mask.sum() > 0 else np.nan
    return {
        "brier_raw": brier_score(p_raw, y),
        "brier_cal": brier_score(p_cal, y),
        "bss_raw": brier_skill_score(p_raw, y),
        "bss_cal": brier_skill_score(p_cal, y),
        "logloss_raw": log_loss(p_raw, y),
        "logloss_cal": log_loss(p_cal, y),
        "auc_raw": auc_roc(p_raw, y),
        "auc_cal": auc_roc(p_cal, y),
        "separation_raw": separation(p_raw, y),
        "separation_cal": separation(p_cal, y),
        "event_rate": event_rate,
        "n": int(mask.sum()),
        "n_eff": effective_sample_size(y, H),
    }


def compute_reliability(
    results: pd.DataFrame,
    horizons: List[int],
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute reliability diagram data (binned calibration curve).

    Returns DataFrame with columns:
        horizon, bin_mid, bin_count, mean_predicted, mean_observed
    """
    rows = []
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for H in horizons:
        y_col = f"y_{H}"
        for p_col, label in [(f"p_raw_{H}", "raw"), (f"p_cal_{H}", "calibrated")]:
            if y_col not in results.columns or p_col not in results.columns:
                continue

            y = results[y_col].values
            p = results[p_col].values
            mask = ~np.isnan(y) & ~np.isnan(p)

            for i in range(n_bins):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                in_bin = mask & (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)

                if in_bin.sum() > 0:
                    n_bin = int(in_bin.sum())
                    p_hat = float(y[in_bin].mean())
                    # Wilson score interval (95% CI, better coverage than Wald for small n)
                    z = 1.96
                    denom = 1.0 + z ** 2 / n_bin
                    center = (p_hat + z ** 2 / (2.0 * n_bin)) / denom
                    margin = z * np.sqrt((p_hat * (1.0 - p_hat) + z ** 2 / (4.0 * n_bin)) / n_bin) / denom
                    rows.append({
                        "horizon": H,
                        "type": label,
                        "bin_mid": (lo + hi) / 2,
                        "bin_count": n_bin,
                        "mean_predicted": float(p[in_bin].mean()),
                        "mean_observed": p_hat,
                        "ci_low": max(0.0, center - margin),
                        "ci_high": min(1.0, center + margin),
                    })

    return pd.DataFrame(rows) if rows else pd.DataFrame()
