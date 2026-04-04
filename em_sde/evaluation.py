"""
Evaluation metrics: calibration, discrimination, and density diagnostics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable

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


def effective_sample_size(
    y: np.ndarray,
    H: int,
    p_cal: Optional[np.ndarray] = None,
) -> float:
    """
    Estimate effective sample size accounting for H-step overlap autocorrelation.

    Uses Bartlett-type correction: N_eff = N / (1 + 2 * sum of weighted ACF).
    For non-overlapping (H=1), returns N.

    When p_cal is provided, ACF is computed on prediction residuals (p_cal - y)
    rather than on binary labels. Residuals have a continuous distribution that
    produces more accurate ACF estimates and avoids the upward N_eff bias
    inherent in computing ACF on binary (0/1) outcomes.
    """
    n = len(y)
    mask = ~np.isnan(y)
    if p_cal is not None:
        mask = mask & ~np.isnan(p_cal)
        series = (p_cal[mask] - y[mask])
    else:
        series = y[mask]
    n_clean = len(series)
    if n_clean <= H or H <= 1:
        return float(n_clean)
    centered = series - np.mean(series)
    var_s = float(np.var(series))
    if var_s < 1e-15:
        return float(n_clean)
    acf_sum = 0.0
    for lag in range(1, H):
        acf_lag = float(np.mean(centered[lag:] * centered[:-lag])) / var_s
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


def crps_per_sample_from_quantiles(
    quantiles: np.ndarray,
    q_levels: np.ndarray,
    realized: np.ndarray,
) -> np.ndarray:
    """Return one approximate CRPS value per observation."""
    quantiles = np.asarray(quantiles, dtype=np.float64)
    q_levels = np.asarray(q_levels, dtype=np.float64)
    realized = np.asarray(realized, dtype=np.float64).reshape(-1)
    if quantiles.ndim != 2 or quantiles.shape[0] != len(realized):
        raise ValueError("quantiles must have shape (n_samples, n_quantiles)")
    values = np.full(len(realized), np.nan, dtype=np.float64)
    for i, obs in enumerate(realized):
        if np.isnan(obs):
            continue
        q = quantiles[i]
        crps_i = 0.0
        for j in range(len(q_levels) - 1):
            z_lo, z_hi = q[j], q[j + 1]
            cdf_lo, cdf_hi = q_levels[j], q_levels[j + 1]
            width = z_hi - z_lo
            if width <= 0:
                continue
            z_mid = (z_lo + z_hi) / 2.0
            ind = 1.0 if z_mid >= obs else 0.0
            cdf_mid = (cdf_lo + cdf_hi) / 2.0
            crps_i += (cdf_mid - ind) ** 2 * width
        values[i] = crps_i
    return values


def pit_from_quantiles(
    quantiles: np.ndarray,
    q_levels: np.ndarray,
    realized: np.ndarray,
) -> np.ndarray:
    """
    Approximate PIT values from stored quantiles by piecewise-linear inversion.

    Returns one PIT value per observation in [0, 1].
    """
    quantiles = np.asarray(quantiles, dtype=np.float64)
    q_levels = np.asarray(q_levels, dtype=np.float64).reshape(-1)
    realized = np.asarray(realized, dtype=np.float64).reshape(-1)
    if quantiles.ndim != 2 or quantiles.shape[1] != len(q_levels):
        raise ValueError("quantiles must have shape (n_samples, len(q_levels))")
    if quantiles.shape[0] != len(realized):
        raise ValueError("quantiles and realized must have the same number of rows")

    pit = np.full(len(realized), np.nan, dtype=np.float64)
    for i, obs in enumerate(realized):
        if np.isnan(obs):
            continue
        q = quantiles[i]
        finite = np.isfinite(q)
        if finite.sum() < 2:
            continue
        q = q[finite]
        levels = q_levels[finite]
        if obs <= q[0]:
            pit[i] = float(levels[0])
            continue
        if obs >= q[-1]:
            pit[i] = float(levels[-1])
            continue
        idx = int(np.searchsorted(q, obs, side="right") - 1)
        idx = min(max(idx, 0), len(q) - 2)
        z_lo, z_hi = q[idx], q[idx + 1]
        p_lo, p_hi = levels[idx], levels[idx + 1]
        if z_hi <= z_lo:
            pit[i] = float(0.5 * (p_lo + p_hi))
        else:
            w = (obs - z_lo) / (z_hi - z_lo)
            pit[i] = float(p_lo + w * (p_hi - p_lo))
    return pit


def pit_ks_statistic(pit: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance of PIT values from Uniform(0,1)."""
    pit = np.asarray(pit, dtype=np.float64)
    pit = pit[np.isfinite(pit)]
    n = len(pit)
    if n == 0:
        return np.nan
    pit = np.sort(np.clip(pit, 0.0, 1.0))
    empirical_cdf = np.arange(1, n + 1, dtype=np.float64) / n
    empirical_cdf_left = np.arange(0, n, dtype=np.float64) / n
    d_plus = np.max(empirical_cdf - pit)
    d_minus = np.max(pit - empirical_cdf_left)
    return float(max(d_plus, d_minus))


def central_interval_coverage_error(
    quantiles: np.ndarray,
    q_levels: np.ndarray,
    realized: np.ndarray,
    lower_q: float,
    upper_q: float,
) -> float:
    """
    Absolute error between observed and nominal central interval coverage.
    """
    quantiles = np.asarray(quantiles, dtype=np.float64)
    realized = np.asarray(realized, dtype=np.float64).reshape(-1)
    q_levels = np.asarray(q_levels, dtype=np.float64).reshape(-1)
    lower_idx = np.where(np.isclose(q_levels, lower_q))[0]
    upper_idx = np.where(np.isclose(q_levels, upper_q))[0]
    if len(lower_idx) == 0 or len(upper_idx) == 0:
        return np.nan
    lo = quantiles[:, int(lower_idx[0])]
    hi = quantiles[:, int(upper_idx[0])]
    mask = np.isfinite(realized) & np.isfinite(lo) & np.isfinite(hi)
    if mask.sum() == 0:
        return np.nan
    covered = (realized[mask] >= lo[mask]) & (realized[mask] <= hi[mask])
    nominal = float(upper_q - lower_q)
    return float(abs(np.mean(covered) - nominal))


def paired_bootstrap_loss_diff_pvalue(
    model_loss: np.ndarray,
    baseline_loss: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    block_size: int = 1,
) -> float:
    """
    One-sided paired bootstrap p-value for model beating baseline.

    Null: mean(model_loss - baseline_loss) >= 0.
    Alternative: model loss is lower than baseline loss.

    When block_size > 1, uses circular block bootstrap to account for
    serial dependence in overlapping predictions.
    """
    model_loss = np.asarray(model_loss, dtype=np.float64).reshape(-1)
    baseline_loss = np.asarray(baseline_loss, dtype=np.float64).reshape(-1)
    mask = np.isfinite(model_loss) & np.isfinite(baseline_loss)
    if mask.sum() == 0:
        return np.nan
    diff = model_loss[mask] - baseline_loss[mask]
    n = len(diff)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)

    if block_size <= 1:
        # Standard i.i.d. bootstrap
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_means[b] = float(np.mean(diff[idx]))
    else:
        # Circular block bootstrap for overlapping/dependent data
        n_blocks = max(1, (n + block_size - 1) // block_size)
        for b in range(n_boot):
            starts = rng.integers(0, n, size=n_blocks)
            idx = np.concatenate([
                np.arange(s, s + block_size) % n for s in starts
            ])[:n]
            boot_means[b] = float(np.mean(diff[idx]))

    return float(np.mean(boot_means >= 0.0))


# ---------------------------------------------------------------------------
# Statistical Rigor Utilities
# ---------------------------------------------------------------------------


def expected_calibration_error_detailed(
    p: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    adaptive: bool = True,
) -> Dict[str, Any]:
    """
    ECE with per-bin sample counts for publication-quality reporting.

    Returns dict with 'ece', 'bin_counts', 'min_bin_n', 'n_bins_used',
    and 'bins' (list of per-bin dicts with lo, hi, n, mean_pred, mean_obs, error).
    """
    mask = ~np.isnan(y) & ~np.isnan(p)
    if mask.sum() == 0:
        return {"ece": np.nan, "bin_counts": [], "min_bin_n": 0,
                "n_bins_used": 0, "bins": []}
    p_m, y_m = p[mask], y[mask]
    n = len(p_m)

    if adaptive:
        bin_edges = np.unique(np.quantile(p_m, np.linspace(0, 1, n_bins + 1)))
        if len(bin_edges) < 2:
            return {"ece": float(abs(np.mean(p_m) - np.mean(y_m))),
                    "bin_counts": [n], "min_bin_n": n,
                    "n_bins_used": 1, "bins": []}
        n_actual_bins = len(bin_edges) - 1
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)
        n_actual_bins = n_bins

    ece = 0.0
    bin_counts = []
    bins_detail = []
    for i in range(n_actual_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_actual_bins - 1:
            in_bin = (p_m >= lo) & (p_m < hi)
        else:
            in_bin = (p_m >= lo) & (p_m <= hi)
        n_bin = int(in_bin.sum())
        bin_counts.append(n_bin)
        if n_bin > 0:
            mean_pred = float(np.mean(p_m[in_bin]))
            mean_obs = float(np.mean(y_m[in_bin]))
            err = abs(mean_pred - mean_obs)
            ece += (n_bin / n) * err
            bins_detail.append({"lo": float(lo), "hi": float(hi), "n": n_bin,
                                "mean_pred": mean_pred, "mean_obs": mean_obs,
                                "error": err})

    non_empty = [c for c in bin_counts if c > 0]
    return {
        "ece": float(ece),
        "bin_counts": bin_counts,
        "min_bin_n": min(non_empty) if non_empty else 0,
        "n_bins_used": len(non_empty),
        "bins": bins_detail,
    }


def prediction_sharpness(
    p: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Measure prediction sharpness — how much the model deviates from base rate.

    A model that always predicts near the base rate has low sharpness and low ECE
    trivially. Sharpness quantifies whether the model is actually saying something.

    Returns dict with: std, iqr, pct_deviate_5pp (fraction of predictions >5pp
    from base rate), base_rate, min, max, range.
    """
    mask = np.isfinite(p) & np.isfinite(y)
    p_m, y_m = p[mask], y[mask]
    if len(p_m) < 10:
        return {"std": np.nan, "iqr": np.nan, "pct_deviate_5pp": np.nan,
                "base_rate": np.nan, "min": np.nan, "max": np.nan, "range": np.nan}

    base_rate = float(np.mean(y_m))
    p_std = float(np.std(p_m))
    q25, q75 = np.percentile(p_m, [25, 75])
    iqr = float(q75 - q25)
    pct_deviate = float(np.mean(np.abs(p_m - base_rate) > 0.05))

    return {
        "std": p_std,
        "iqr": iqr,
        "pct_deviate_5pp": pct_deviate,
        "base_rate": base_rate,
        "min": float(np.min(p_m)),
        "max": float(np.max(p_m)),
        "range": float(np.max(p_m) - np.min(p_m)),
    }


def conditional_ece(
    p: np.ndarray,
    y: np.ndarray,
    condition: np.ndarray,
    condition_bins: Optional[List[Tuple[str, float, float]]] = None,
    n_bins: int = 10,
) -> List[Dict[str, Any]]:
    """
    ECE conditioned on an external variable (e.g., vol regime, era).

    Parameters
    ----------
    p, y : predictions and labels
    condition : array of condition values (same length as p)
    condition_bins : list of (name, lo, hi) tuples defining bins.
        If None, uses terciles of condition values.

    Returns list of dicts: {name, ece, n, base_rate}.
    """
    mask = np.isfinite(p) & np.isfinite(y) & np.isfinite(condition)
    p_m, y_m, c_m = p[mask], y[mask], condition[mask]
    if len(p_m) < 30:
        return []

    if condition_bins is None:
        t33, t67 = np.percentile(c_m, [33, 67])
        condition_bins = [
            ("Low", float('-inf'), t33),
            ("Mid", t33, t67),
            ("High", t67, float('inf')),
        ]

    results = []
    for name, lo, hi in condition_bins:
        in_bin = (c_m >= lo) & (c_m < hi) if hi < float('inf') else (c_m >= lo)
        if lo == float('-inf'):
            in_bin = c_m < hi
        if in_bin.sum() < 30:
            continue
        ece_val = expected_calibration_error(p_m[in_bin], y_m[in_bin], adaptive=False, n_bins=n_bins)
        results.append({
            "name": name,
            "ece": float(ece_val),
            "n": int(in_bin.sum()),
            "base_rate": float(np.mean(y_m[in_bin])),
        })

    return results


def bootstrap_metric_ci(
    y_true: np.ndarray,
    p_cal: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    block_size: int = 1,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for any metric(p, y) -> float.

    Uses BCa (bias-corrected accelerated) percentile method for better
    small-sample coverage. When block_size > 1, uses circular block
    bootstrap for serially dependent (overlapping) predictions.

    Returns (point_estimate, ci_low, ci_high).
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    p_cal = np.asarray(p_cal, dtype=np.float64).ravel()
    mask = np.isfinite(y_true) & np.isfinite(p_cal)
    y_true, p_cal = y_true[mask], p_cal[mask]
    n = len(y_true)
    if n < 10:
        point = metric_fn(p_cal, y_true)
        return (point, np.nan, np.nan)

    point = metric_fn(p_cal, y_true)
    rng = np.random.default_rng(seed)
    boot_vals = np.empty(n_boot, dtype=np.float64)

    if block_size <= 1:
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_vals[b] = metric_fn(p_cal[idx], y_true[idx])
    else:
        # Circular block bootstrap for overlapping predictions
        n_blocks = max(1, (n + block_size - 1) // block_size)
        for b in range(n_boot):
            starts = rng.integers(0, n, size=n_blocks)
            idx = np.concatenate([
                np.arange(s, s + block_size) % n for s in starts
            ])[:n]
            boot_vals[b] = metric_fn(p_cal[idx], y_true[idx])

    # BCa correction
    finite_boots = boot_vals[np.isfinite(boot_vals)]
    if len(finite_boots) < 100:
        lo = float(np.nanpercentile(boot_vals, 100 * alpha / 2))
        hi = float(np.nanpercentile(boot_vals, 100 * (1 - alpha / 2)))
        return (point, lo, hi)

    # Bias correction
    z0 = float(_norm_ppf(np.mean(finite_boots < point)))
    # Acceleration via jackknife
    jack = np.empty(n, dtype=np.float64)
    for i in range(n):
        jack_mask = np.ones(n, dtype=bool)
        jack_mask[i] = False
        jack[i] = metric_fn(p_cal[jack_mask], y_true[jack_mask])
    jack_mean = np.mean(jack)
    num = np.sum((jack_mean - jack) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jack) ** 2)) ** 1.5
    a_hat = float(num / denom) if abs(denom) > 1e-15 else 0.0

    # Adjusted percentiles
    z_lo = _norm_ppf(alpha / 2)
    z_hi = _norm_ppf(1 - alpha / 2)
    alpha_lo = _norm_cdf(z0 + (z0 + z_lo) / (1 - a_hat * (z0 + z_lo)))
    alpha_hi = _norm_cdf(z0 + (z0 + z_hi) / (1 - a_hat * (z0 + z_hi)))

    lo = float(np.percentile(finite_boots, 100 * np.clip(alpha_lo, 0.001, 0.999)))
    hi = float(np.percentile(finite_boots, 100 * np.clip(alpha_hi, 0.001, 0.999)))
    return (point, lo, hi)


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (probit). Uses rational approximation."""
    from math import sqrt, log, copysign
    p = np.clip(p, 1e-10, 1 - 1e-10)
    if p == 0.5:
        return 0.0
    # Rational approximation (Abramowitz & Stegun 26.2.23)
    if p < 0.5:
        t = sqrt(-2.0 * log(p))
    else:
        t = sqrt(-2.0 * log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    val = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t ** 3)
    return -val if p < 0.5 else val


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using error function."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def apply_fdr_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """
    Benjamini-Hochberg FDR correction for multiple hypothesis testing.

    Parameters
    ----------
    p_values : list of float
        Raw p-values from individual tests.
    alpha : float
        Target false discovery rate (default 0.05).

    Returns
    -------
    adjusted_pvals : list of float
        FDR-adjusted p-values (monotonically non-decreasing in rank order).
    reject : list of bool
        Whether each null hypothesis is rejected at the given alpha.
    """
    n = len(p_values)
    if n == 0:
        return [], []

    arr = np.asarray(p_values, dtype=np.float64)
    # Handle NaN: treat as non-significant
    finite_mask = np.isfinite(arr)
    adjusted = np.ones(n, dtype=np.float64)
    reject = [False] * n

    if finite_mask.sum() == 0:
        return adjusted.tolist(), reject

    # BH procedure
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)

    # Adjusted p-value: p_adj = p * n / rank, then enforce monotonicity
    adjusted_raw = arr * n / ranks
    # Step-up enforcement: walk from largest rank down
    sorted_idx = order[::-1]
    running_min = 1.0
    for i in sorted_idx:
        if not finite_mask[i]:
            adjusted[i] = 1.0
            continue
        adjusted[i] = min(running_min, adjusted_raw[i])
        running_min = adjusted[i]

    adjusted = np.clip(adjusted, 0.0, 1.0)
    reject = [bool(adjusted[i] <= alpha) for i in range(n)]
    return adjusted.tolist(), reject


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
        "n_eff": effective_sample_size(y, H, p_cal=p_cal),
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
