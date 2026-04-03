"""
Formal baseline models for academic paper comparison.

Four baselines that answer: "Does the full pipeline beat simpler approaches?"

1. Historical Frequency (Climatology): rolling empirical event rate
2. GARCH-CDF: parametric GARCH forecast distribution, no MC simulation
3. Implied-Vol Black-Scholes: options market pricing of large moves
4. Feature Logistic Regression: same features, no MC generation layer

All baselines produce calibrated probabilities on the same dates as the
full model, enabling paired statistical comparison.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from scipy import stats as sp_stats

from em_sde.garch import fit_garch, GarchResult, ewma_volatility, project_to_stationary
from em_sde.evaluation import (
    brier_score, brier_skill_score, auc_roc, expected_calibration_error,
    paired_bootstrap_loss_diff_pvalue,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline 1: Historical Frequency (Climatology)
# ---------------------------------------------------------------------------

def historical_frequency_baseline(
    prices: np.ndarray,
    horizons: List[int],
    thresholds: Dict[int, float],
    window: int = 252,
) -> pd.DataFrame:
    """
    Rolling empirical event rate baseline.

    For each date t, P(large move) = fraction of past `window` periods
    where |return_H| >= threshold. This is the climatological baseline
    that BSS is measured against.
    """
    n = len(prices)
    rows = []

    for t in range(window, n):
        row = {"idx": t}
        for H in horizons:
            thr = thresholds.get(H, 0.05)
            # Count events in the lookback window
            events_in_window = 0
            valid_in_window = 0
            for j in range(max(0, t - window), t - H):
                if j + H < n:
                    ret = abs(prices[j + H] / prices[j] - 1.0)
                    if ret >= thr:
                        events_in_window += 1
                    valid_in_window += 1

            p_clim = events_in_window / max(valid_in_window, 1)
            # Resolve label
            y = np.nan
            if t + H < n:
                y = 1.0 if abs(prices[t + H] / prices[t] - 1.0) >= thr else 0.0

            row[f"p_baseline_{H}"] = p_clim
            row[f"y_{H}"] = y
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Baseline 2: GARCH-CDF (Parametric, No MC)
# ---------------------------------------------------------------------------

def garch_cdf_baseline(
    prices: np.ndarray,
    horizons: List[int],
    thresholds: Dict[int, float],
    garch_window: int = 756,
    garch_min_window: int = 252,
    model_type: str = "gjr",
    t_df: float = 5.0,
) -> pd.DataFrame:
    """
    GARCH parametric CDF baseline — no Monte Carlo simulation.

    Uses the GARCH conditional variance forecast to compute P(|r_H| >= thr)
    analytically under a Student-t distribution assumption.

    This tests whether the MC simulation layer adds value beyond the
    parametric distribution implied by GARCH.
    """
    n = len(prices)
    returns = np.diff(prices) / prices[:-1]
    rows = []

    last_garch: Optional[GarchResult] = None
    last_fit_idx = -999

    for t in range(garch_min_window + 1, n):
        # Refit GARCH periodically (every 21 days) or on first pass
        if t - last_fit_idx >= 21 or last_garch is None:
            window_start = max(0, t - garch_window)
            window_returns = returns[window_start:t]
            if len(window_returns) >= garch_min_window:
                try:
                    last_garch = fit_garch(window_returns, model_type=model_type)
                    last_fit_idx = t
                except Exception:
                    if last_garch is None:
                        sigma_ewma = float(np.std(window_returns)) * np.sqrt(252)
                        last_garch = GarchResult(
                            sigma_1d=sigma_ewma / np.sqrt(252),
                            omega=0.0, alpha=0.05, beta=0.90, gamma=0.0,
                            persistence=0.95, is_stationary=True,
                        )
                        last_fit_idx = t

        if last_garch is None:
            continue

        sigma_1d = last_garch.sigma_1d
        row = {"idx": t}

        for H in horizons:
            thr = thresholds.get(H, 0.05)
            # Scale daily vol to H-day vol (sqrt-time rule)
            sigma_H = sigma_1d * np.sqrt(H)

            # Under Student-t: P(|r| >= thr) = 2 * P(r <= -thr)
            # Standardize: z = thr / sigma_H, then use t-distribution
            if sigma_H > 1e-10:
                z = thr / sigma_H
                if t_df > 2:
                    # Student-t with scaling for unit variance
                    scale = sigma_H * np.sqrt((t_df - 2) / t_df)
                    p_exceed = 2.0 * sp_stats.t.cdf(-thr, df=t_df, scale=scale)
                else:
                    p_exceed = 2.0 * sp_stats.norm.cdf(-z)
            else:
                p_exceed = 0.0

            p_exceed = float(np.clip(p_exceed, 0.0, 1.0))

            # Resolve label
            y = np.nan
            if t + H < n:
                y = 1.0 if abs(prices[t + H] / prices[t] - 1.0) >= thr else 0.0

            row[f"p_baseline_{H}"] = p_exceed
            row[f"y_{H}"] = y
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Baseline 3: Implied-Vol Black-Scholes
# ---------------------------------------------------------------------------

def implied_vol_baseline(
    prices: np.ndarray,
    dates: pd.DatetimeIndex,
    horizons: List[int],
    thresholds: Dict[int, float],
    iv_csv_path: str = "data/vix_history.csv",
) -> pd.DataFrame:
    """
    Options-implied volatility baseline.

    Uses VIX (or ticker-specific IV) to compute P(|r_H| >= thr) under
    a log-normal assumption with implied vol as the volatility input.

    This tests whether the GARCH + MC stack beats the options market's
    own pricing of tail risk.
    """
    # Load VIX data
    try:
        iv_df = pd.read_csv(iv_csv_path, parse_dates=["date"], index_col="date")
    except Exception as e:
        logger.warning("Could not load implied vol data: %s", e)
        return pd.DataFrame()

    n = len(prices)
    rows = []

    for t in range(252, n):
        date = dates[t]
        row = {"idx": t}

        # Find closest VIX observation (within 5 business days)
        iv_row = None
        for offset in range(6):
            lookup = date - pd.Timedelta(days=offset)
            if lookup in iv_df.index:
                iv_row = iv_df.loc[lookup]
                break

        if iv_row is None:
            continue

        for H in horizons:
            thr = thresholds.get(H, 0.05)

            # Select appropriate VIX tenor
            if H <= 9 and "iv_9d" in iv_row.index:
                vix_val = float(iv_row["iv_9d"])
            elif H >= 30 and "iv_3m" in iv_row.index:
                vix_val = float(iv_row["iv_3m"])
            elif "iv_30d" in iv_row.index:
                vix_val = float(iv_row["iv_30d"])
            else:
                # Fallback to any available column
                for col in ["iv_30d", "iv_9d", "iv_3m"]:
                    if col in iv_row.index and pd.notna(iv_row[col]):
                        vix_val = float(iv_row[col])
                        break
                else:
                    continue

            if not np.isfinite(vix_val) or vix_val <= 0:
                continue

            # VIX is annualized implied vol (percentage) -> convert to decimal
            sigma_annual = vix_val / 100.0
            sigma_H = sigma_annual * np.sqrt(H / 252.0)

            # Log-normal: P(|log(S_T/S_0)| >= log(1+thr))
            # Approximate: P(|r| >= thr) ≈ 2 * Phi(-thr / sigma_H)
            if sigma_H > 1e-10:
                p_exceed = 2.0 * sp_stats.norm.cdf(-thr / sigma_H)
            else:
                p_exceed = 0.0

            p_exceed = float(np.clip(p_exceed, 0.0, 1.0))

            y = np.nan
            if t + H < n:
                y = 1.0 if abs(prices[t + H] / prices[t] - 1.0) >= thr else 0.0

            row[f"p_baseline_{H}"] = p_exceed
            row[f"y_{H}"] = y
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Baseline 4: Feature Logistic Regression (No MC)
# ---------------------------------------------------------------------------

def feature_logistic_baseline(
    prices: np.ndarray,
    horizons: List[int],
    thresholds: Dict[int, float],
    garch_window: int = 756,
    garch_min_window: int = 252,
    lr: float = 0.01,
    l2: float = 1e-4,
    warmup: int = 100,
) -> pd.DataFrame:
    """
    Online logistic regression on volatility features — no MC simulation.

    Uses the same features as the multi-feature calibrator (sigma, delta_sigma,
    vol_ratio, vol_of_vol) but predicts the event label directly, bypassing
    the entire Monte Carlo generation layer.

    This tests whether the calibration layer is doing the heavy lifting
    vs. the MC generation layer.
    """
    n = len(prices)
    returns = np.diff(prices) / prices[:-1]

    # Feature computation helpers
    def _compute_features(t: int, sigma_1d: float) -> np.ndarray:
        """Compute feature vector at time t."""
        # Feature 1: logit(sigma_1d * sqrt(252) / 0.5) — scaled vol level
        sigma_ann = sigma_1d * np.sqrt(252)
        feat_vol = np.log(max(sigma_ann, 1e-6) / (1 - min(sigma_ann, 0.999)))

        # Feature 2: 20-day sigma change
        if t >= 20:
            sigma_20d_ago = float(np.std(returns[max(0, t-40):max(1, t-20)])) if t >= 40 else sigma_1d
            delta_sigma = sigma_1d - sigma_20d_ago
        else:
            delta_sigma = 0.0

        # Feature 3: realized vol ratio
        if t >= 20:
            realized_20d = float(np.std(returns[max(0, t-20):t]))
            vol_ratio = realized_20d / max(sigma_1d, 1e-8)
        else:
            vol_ratio = 1.0

        # Feature 4: vol of vol (rolling std of sigma proxy)
        if t >= 60:
            recent_returns = returns[max(0, t-60):t]
            rolling_vols = []
            for j in range(0, len(recent_returns) - 20, 5):
                rv = float(np.std(recent_returns[j:j+20]))
                rolling_vols.append(rv)
            vol_of_vol = float(np.std(rolling_vols)) if len(rolling_vols) >= 3 else 0.0
        else:
            vol_of_vol = 0.0

        return np.array([1.0, feat_vol, delta_sigma, vol_ratio, vol_of_vol])

    # Per-horizon online logistic regressors
    n_features = 5
    weights = {H: np.zeros(n_features) for H in horizons}
    update_counts = {H: 0 for H in horizons}
    pending = {H: [] for H in horizons}  # (t, features) pairs awaiting resolution

    rows = []
    last_garch = None
    last_fit_idx = -999

    for t in range(garch_min_window + 1, n):
        # Fit GARCH for sigma estimate
        if t - last_fit_idx >= 21 or last_garch is None:
            window_start = max(0, t - garch_window)
            window_returns = returns[window_start:t]
            if len(window_returns) >= garch_min_window:
                try:
                    last_garch = fit_garch(window_returns, model_type="gjr")
                    last_fit_idx = t
                except Exception:
                    pass

        if last_garch is None:
            continue

        sigma_1d = last_garch.sigma_1d
        features = _compute_features(t, sigma_1d)

        row = {"idx": t}

        for H in horizons:
            thr = thresholds.get(H, 0.05)
            w = weights[H]

            # Resolve pending predictions
            resolved = []
            for (pt, pf) in pending[H]:
                if pt + H < n:
                    y = 1.0 if abs(prices[pt + H] / prices[pt] - 1.0) >= thr else 0.0
                    resolved.append((pf, y))
            # Keep only unresolved
            pending[H] = [(pt, pf) for (pt, pf) in pending[H] if pt + H >= t]

            # SGD update with resolved labels
            for (feat, y) in resolved:
                logit_val = float(np.dot(w, feat))
                p_pred = 1.0 / (1.0 + np.exp(-np.clip(logit_val, -20, 20)))
                grad = (p_pred - y) * feat + l2 * w
                effective_lr = lr / np.sqrt(1 + update_counts[H])
                w -= effective_lr * grad
                weights[H] = w
                update_counts[H] += 1

            # Predict
            logit_val = float(np.dot(w, features))
            p_pred = 1.0 / (1.0 + np.exp(-np.clip(logit_val, -20, 20)))

            if update_counts[H] < warmup:
                # Before warmup, use simple empirical rate
                lookback = min(t, 252)
                events = 0
                total = 0
                for j in range(max(0, t - lookback), t - H):
                    if j + H < n:
                        if abs(prices[j + H] / prices[j] - 1.0) >= thr:
                            events += 1
                        total += 1
                p_pred = events / max(total, 1)

            y_label = np.nan
            if t + H < n:
                y_label = 1.0 if abs(prices[t + H] / prices[t] - 1.0) >= thr else 0.0

            row[f"p_baseline_{H}"] = float(np.clip(p_pred, 0.001, 0.999))
            row[f"y_{H}"] = y_label

            # Queue for resolution
            pending[H].append((t, features.copy()))

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unified baseline runner
# ---------------------------------------------------------------------------

def run_all_baselines(
    prices: np.ndarray,
    dates: pd.DatetimeIndex,
    horizons: List[int],
    thresholds: Dict[int, float],
    iv_csv_path: Optional[str] = None,
    garch_window: int = 756,
    t_df: float = 5.0,
) -> Dict[str, pd.DataFrame]:
    """
    Run all four baselines and return results keyed by baseline name.

    Each DataFrame has columns: idx, p_baseline_{H}, y_{H} for each horizon.
    """
    logger.info("Running Historical Frequency baseline...")
    hist_freq = historical_frequency_baseline(prices, horizons, thresholds)

    logger.info("Running GARCH-CDF baseline...")
    garch_cdf = garch_cdf_baseline(
        prices, horizons, thresholds,
        garch_window=garch_window, t_df=t_df,
    )

    logger.info("Running Feature Logistic baseline...")
    feat_log = feature_logistic_baseline(
        prices, horizons, thresholds,
        garch_window=garch_window,
    )

    results = {
        "Historical Frequency": hist_freq,
        "GARCH-CDF": garch_cdf,
        "Feature Logistic": feat_log,
    }

    if iv_csv_path:
        logger.info("Running Implied-Vol baseline...")
        impl_vol = implied_vol_baseline(
            prices, dates, horizons, thresholds,
            iv_csv_path=iv_csv_path,
        )
        if len(impl_vol) > 0:
            results["Implied-Vol BS"] = impl_vol

    return results


def evaluate_baseline(
    baseline_df: pd.DataFrame,
    horizons: List[int],
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Evaluate a single baseline: BSS, AUC, ECE, and p-values vs climatology."""
    rows = []
    for H in horizons:
        p_col = f"p_baseline_{H}"
        y_col = f"y_{H}"
        if p_col not in baseline_df.columns or y_col not in baseline_df.columns:
            continue

        p = baseline_df[p_col].to_numpy(dtype=float)
        y = baseline_df[y_col].to_numpy(dtype=float)
        mask = np.isfinite(p) & np.isfinite(y)
        if mask.sum() < 50:
            continue

        p_m, y_m = p[mask], y[mask]
        bss = brier_skill_score(p_m, y_m)
        auc = auc_roc(p_m, y_m)
        ece = expected_calibration_error(p_m, y_m, adaptive=False)
        brier = brier_score(p_m, y_m)

        # Paired bootstrap vs climatology
        clim_rate = float(np.mean(y_m))
        clim_losses = (np.full_like(y_m, clim_rate) - y_m) ** 2
        model_losses = (p_m - y_m) ** 2
        pval = paired_bootstrap_loss_diff_pvalue(model_losses, clim_losses, n_boot=n_boot)

        rows.append({
            "horizon": H,
            "brier": brier,
            "bss": bss,
            "auc": auc,
            "ece": ece,
            "n": int(mask.sum()),
            "event_rate": clim_rate,
            "p_value_vs_clim": pval,
        })

    return pd.DataFrame(rows)
