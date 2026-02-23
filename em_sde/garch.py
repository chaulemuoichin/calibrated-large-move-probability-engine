"""
GARCH(1,1) volatility forecasting with EWMA fallback.

All inputs and outputs are in DAILY decimal return units.
sigma_1d is the daily volatility forecast (standard deviation of daily returns).

Supports symmetric GARCH(1,1) and GJR-GARCH(1,1) (leverage effect).
Returns GarchResult with extracted parameters for GARCH-in-simulation.

Includes HMM regime detection (Markov-switching variance) for
regime-weighted volatility forecasts.
"""

import warnings
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class GarchResult:
    """Result of GARCH fitting with extracted parameters."""
    sigma_1d: float
    source: str            # "garch", "gjr_garch", or "ewma_fallback"
    omega: Optional[float] = None    # GARCH intercept (daily decimal^2 units)
    alpha: Optional[float] = None    # ARCH coefficient
    beta: Optional[float] = None     # GARCH persistence
    gamma: Optional[float] = None    # GJR asymmetry term (None for symmetric GARCH)
    diagnostics: Optional[dict] = None  # stationarity, half-life, unconditional vol


def _to_float(value: object) -> float:
    """Convert scalars or scalar-like pandas/numpy values to float."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot convert empty value to float")
    return arr.item(0)


def fit_garch(
    returns: NDArray[np.float64],
    window: int = 756,
    min_window: int = 252,
    model_type: str = "garch",
) -> GarchResult:
    """
    Fit GARCH(1,1) or GJR-GARCH(1,1) on the most recent `window` daily returns
    and forecast next-day volatility.

    Parameters
    ----------
    returns : np.ndarray
        Daily simple returns (decimal form, e.g. 0.01 = 1%).
    window : int
        Rolling window size (use most recent N returns).
    min_window : int
        Minimum returns required.
    model_type : str
        "garch" for symmetric GARCH(1,1), "gjr" for GJR-GARCH(1,1) with leverage.

    Returns
    -------
    GarchResult
        Contains sigma_1d, source, and optionally omega/alpha/beta/gamma.
    """
    n = len(returns)
    if n < min_window:
        raise ValueError(f"Need >= {min_window} returns, got {n}")

    # Use rolling window
    data = np.asarray(returns[-window:] if n > window else returns, dtype=np.float64)

    try:
        from arch import arch_model

        # Convert to percentage returns for numerical stability
        data_pct = data * 100.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if model_type == "gjr":
                am = arch_model(
                    data_pct,
                    vol="GARCH",
                    p=1, o=1, q=1,
                    mean="Zero",
                    rescale=False,
                )
            else:
                am = arch_model(
                    data_pct,
                    vol="GARCH",
                    p=1, q=1,
                    mean="Zero",
                    rescale=False,
                )
            res = am.fit(disp="off", show_warning=False)
            fcast = res.forecast(horizon=1)
            var_pct = _to_float(fcast.variance.iloc[-1, 0])

            if var_pct <= 0 or not np.isfinite(var_pct):
                raise ValueError(f"Invalid GARCH variance: {var_pct}")

            # Convert from percentage^2 back to decimal
            sigma_1d = _to_float(np.sqrt(np.float64(var_pct)) / np.float64(100.0))

            if sigma_1d <= 0 or sigma_1d > 1.0:
                raise ValueError(f"Unreasonable sigma_1d: {sigma_1d}")

            # Extract GARCH parameters
            # omega is in pct^2 units, convert to decimal^2 by dividing by 10000
            omega_dec = _to_float(res.params["omega"]) / 10000.0
            alpha_val = _to_float(res.params["alpha[1]"])
            beta_val = _to_float(res.params["beta[1]"])

            gamma_val = None
            source = "garch"
            if model_type == "gjr" and "gamma[1]" in res.params:
                gamma_val = _to_float(res.params["gamma[1]"])
                source = "gjr_garch"

            # Stationarity diagnostics
            diagnostics = garch_diagnostics(
                omega_dec, alpha_val, beta_val, gamma_val, model_type,
            )

            return GarchResult(
                sigma_1d=sigma_1d,
                source=source,
                omega=omega_dec,
                alpha=alpha_val,
                beta=beta_val,
                gamma=gamma_val,
                diagnostics=diagnostics,
            )

    except Exception as e:
        logger.debug("GARCH fit failed (%s), using EWMA fallback", e)
        sigma_1d = ewma_volatility(data)
        return GarchResult(sigma_1d=sigma_1d, source="ewma_fallback")


def ewma_volatility(returns: NDArray[np.float64], span: int = 252) -> float:
    """
    EWMA volatility estimate as fallback.

    Parameters
    ----------
    returns : np.ndarray
        Daily simple returns (decimal).
    span : int
        EWMA span parameter.

    Returns
    -------
    sigma_1d : float
        EWMA volatility estimate (daily, decimal).
    """
    effective_span = min(span, len(returns))
    ewm_var = _to_float(pd.Series(returns ** 2).ewm(span=effective_span).mean().iloc[-1])
    sigma_1d = _to_float(np.sqrt(np.float64(ewm_var)))

    # Safety clamp
    sigma_1d = max(sigma_1d, 1e-8)
    return sigma_1d


def garch_diagnostics(
    omega: float,
    alpha: float,
    beta: float,
    gamma: Optional[float] = None,
    model_type: str = "garch",
) -> dict:
    """
    Compute GARCH stationarity and persistence diagnostics.

    For GARCH(1,1): persistence = alpha + beta.
    For GJR-GARCH(1,1,1): persistence = alpha + beta + gamma/2.

    Stationarity requires persistence < 1.

    Returns
    -------
    diagnostics : dict
        Keys: persistence, is_stationary, half_life, unconditional_var,
              unconditional_vol.
    """
    if gamma is not None and model_type == "gjr":
        persistence = alpha + beta + gamma / 2.0
    else:
        persistence = alpha + beta

    is_stationary = persistence < 1.0

    # Half-life of volatility shocks (in days)
    if 0 < persistence < 1.0:
        half_life = float(np.log(0.5) / np.log(persistence))
    else:
        half_life = np.inf

    # Unconditional variance: omega / (1 - persistence)
    if is_stationary and (1.0 - persistence) > 1e-10:
        uncond_var = omega / (1.0 - persistence)
        uncond_vol = float(np.sqrt(uncond_var))
    else:
        uncond_var = np.nan
        uncond_vol = np.nan

    result = {
        "persistence": round(persistence, 6),
        "is_stationary": is_stationary,
        "half_life_days": round(half_life, 1) if np.isfinite(half_life) else None,
        "unconditional_var": uncond_var,
        "unconditional_vol": round(uncond_vol, 8) if np.isfinite(uncond_vol) else None,
    }

    if not is_stationary:
        logger.warning(
            "GARCH non-stationary: persistence=%.4f >= 1.0", persistence,
        )

    return result


def project_to_stationary(
    omega: float,
    alpha: float,
    beta: float,
    gamma: Optional[float] = None,
    model_type: str = "garch",
    target_persistence: float = 0.98,
    variance_anchor: Optional[float] = None,
) -> tuple:
    """
    Project GARCH parameters to the stationary region when persistence >= 1.0.

    Scales alpha, beta (and gamma for GJR) proportionally so that
    persistence = target_persistence, preserving relative parameter ratios.

    When variance_anchor is provided, omega is recomputed so that the
    stationary variance V_inf = omega / (1 - persistence) equals the anchor:
        omega_new = variance_anchor * (1 - target_persistence)
    Without variance_anchor, omega is returned unchanged (legacy behavior).

    For GJR-GARCH: persistence = alpha + beta + gamma/2.

    Parameters
    ----------
    omega : float
        GARCH intercept.
    alpha : float
        ARCH coefficient.
    beta : float
        GARCH persistence coefficient.
    gamma : float or None
        GJR asymmetry coefficient.
    model_type : str
        "garch" or "gjr".
    target_persistence : float
        Target persistence after projection (default 0.98).
    variance_anchor : float or None
        Target stationary variance (typically sigma_1d**2). When provided,
        omega is recomputed to anchor V_inf to this value.

    Returns
    -------
    omega_new, alpha_new, beta_new, gamma_new : tuple
        Projected parameters. gamma_new is None if gamma input is None.
    """
    if gamma is not None and model_type == "gjr":
        current_persistence = alpha + beta + gamma / 2.0
    else:
        current_persistence = alpha + beta

    if current_persistence < 1.0:
        return omega, alpha, beta, gamma

    if current_persistence <= 0.0:
        logger.warning("GARCH persistence <= 0 (%.4f), skipping projection", current_persistence)
        return omega, alpha, beta, gamma

    scale = target_persistence / current_persistence

    alpha_new = alpha * scale
    beta_new = beta * scale

    if gamma is not None and model_type == "gjr":
        gamma_new = gamma * scale
    else:
        gamma_new = gamma

    # Variance-targeted omega: ensure V_inf = variance_anchor
    if variance_anchor is not None and variance_anchor > 0:
        omega_new = variance_anchor * (1.0 - target_persistence)
        logger.info(
            "GARCH variance-targeted omega: %.4e -> %.4e (anchor=%.6f)",
            omega, omega_new, variance_anchor,
        )
    else:
        omega_new = omega

    logger.info(
        "GARCH projection: persistence %.4f -> %.4f (scale=%.4f), "
        "omega %.4e -> %.4e, alpha %.4f -> %.4f, beta %.4f -> %.4f",
        current_persistence, target_persistence, scale,
        omega, omega_new, alpha, alpha_new, beta, beta_new,
    )

    return omega_new, alpha_new, beta_new, gamma_new


def garch_term_structure_vol(
    sigma_1d: float,
    omega: float,
    alpha: float,
    beta: float,
    gamma: Optional[float],
    H: int,
    model_type: str = "garch",
) -> float:
    """
    Compute average expected daily vol over H steps using GARCH mean-reversion.

    The GARCH conditional variance mean-reverts toward the unconditional variance:
        E[σ²_{t+h}] = σ²_unc + pers^h * (σ²_t - σ²_unc)

    This returns sqrt(mean(E[σ²_{t+1}], ..., E[σ²_{t+H}])), which accounts
    for the vol term structure rather than using the 1-step forecast for all
    horizons. Walk-forward safe (uses only fitted GARCH params).

    Parameters
    ----------
    sigma_1d : float
        One-step-ahead daily vol forecast (decimal).
    omega, alpha, beta, gamma : float
        GARCH parameters (decimal^2 units for omega).
    H : int
        Forecast horizon in trading days.
    model_type : str
        "garch" or "gjr".

    Returns
    -------
    sigma_avg : float
        Average expected daily vol over the horizon.
    """
    if omega is None or alpha is None or beta is None:
        return sigma_1d

    if gamma is not None and model_type == "gjr":
        persistence = alpha + beta + gamma / 2.0
    else:
        persistence = alpha + beta

    # Non-stationary: can't compute unconditional variance, return sigma_1d
    if persistence >= 1.0 or (1.0 - persistence) < 1e-10:
        return sigma_1d

    sigma2_t = sigma_1d ** 2
    sigma2_unc = omega / (1.0 - persistence)

    # Average expected variance over h=1..H
    total_var = 0.0
    for h in range(1, H + 1):
        total_var += sigma2_unc + (persistence ** h) * (sigma2_t - sigma2_unc)

    avg_var = total_var / H
    # Safety: ensure positive
    avg_var = max(avg_var, 1e-16)
    return float(np.sqrt(avg_var))


# ---------------------------------------------------------------------------
# HMM regime detection (Markov-switching variance)
# ---------------------------------------------------------------------------

@dataclass
class HmmRegimeResult:
    """Result from HMM regime detection."""
    regime_prob_high: float        # P(high-vol state | data_1:t), filtered
    sigma_low: float               # Unconditional daily vol in low-vol regime (decimal)
    sigma_high: float              # Unconditional daily vol in high-vol regime (decimal)
    transition_matrix: np.ndarray  # 2x2 transition probabilities
    n_regimes: int = 2


def fit_hmm_regime(
    returns: NDArray[np.float64],
    window: int = 756,
    min_window: int = 252,
    n_regimes: int = 2,
) -> Optional[HmmRegimeResult]:
    """
    Fit 2-state Markov-switching model on returns with switching variance.

    Returns filtered probability of being in high-vol state (walk-forward safe)
    and per-regime unconditional volatility.

    Uses statsmodels MarkovRegression with switching_variance=True.
    Falls back to None if fitting fails (caller uses percentile fallback).

    Parameters
    ----------
    returns : np.ndarray
        Daily simple returns (decimal form).
    window : int
        Rolling window of returns to use.
    min_window : int
        Minimum returns required.
    n_regimes : int
        Number of HMM states (default 2: low/high vol).

    Returns
    -------
    HmmRegimeResult or None
        Regime detection result, or None if fitting fails.
    """
    n = len(returns)
    if n < min_window:
        logger.debug("HMM: insufficient data (%d < %d)", n, min_window)
        return None

    data = np.asarray(returns[-window:] if n > window else returns, dtype=np.float64)

    # Convert to percentage returns for numerical stability (same as GARCH)
    data_pct = data * 100.0

    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MarkovRegression(
                data_pct,
                k_regimes=n_regimes,
                switching_variance=True,
            )
            result = model.fit(
                search_reps=3,
                em_iter=30,
                disp=False,
            )

        # Extract per-regime variance from params array.
        # Layout for 2 regimes: [p[0->0], p[1->0], const_0, const_1, sigma2_0, sigma2_1]
        # For k regimes: last k params are sigma2 values, preceded by k const values,
        # preceded by k*(k-1) transition params.
        params = np.asarray(result.params, dtype=np.float64)
        # sigma2 values are the last n_regimes entries
        regime_variances = [float(params[-(n_regimes - k)]) for k in range(n_regimes)]

        # Sort regimes by variance: index 0 = low-vol, index -1 = high-vol
        sort_idx = np.argsort(regime_variances)
        regime_variances_sorted = [regime_variances[sort_idx[i]] for i in range(n_regimes)]

        # Per-regime sigma in decimal units (sqrt(var_pct) / 100)
        sigmas = [float(np.sqrt(max(v, 1e-16))) / 100.0 for v in regime_variances_sorted]

        # Filtered probabilities: P(S_t = k | y_{1:t}) — walk-forward safe
        filtered_probs = result.filtered_marginal_probabilities
        # filtered_probs shape: (T, n_regimes)
        high_regime_idx = sort_idx[-1]  # original index of highest-variance regime
        prob_high = float(filtered_probs[-1, high_regime_idx])

        # Transition matrix (regime_transition has shape (k, k, 1) — squeeze last dim)
        raw_transition = np.squeeze(result.regime_transition)
        # Reorder to match sorted regime labels
        trans = np.zeros((n_regimes, n_regimes))
        for i in range(n_regimes):
            for j in range(n_regimes):
                trans[np.where(sort_idx == i)[0][0], np.where(sort_idx == j)[0][0]] = raw_transition[i, j]

        # Validate
        if not (0.0 <= prob_high <= 1.0):
            logger.debug("HMM: invalid prob_high=%.4f", prob_high)
            return None
        if any(s <= 0 or not np.isfinite(s) for s in sigmas):
            logger.debug("HMM: invalid regime sigmas: %s", sigmas)
            return None

        logger.info(
            "HMM fit: P(high)=%.3f, sigma_low=%.4f, sigma_high=%.4f",
            prob_high, sigmas[0], sigmas[-1],
        )

        return HmmRegimeResult(
            regime_prob_high=prob_high,
            sigma_low=sigmas[0],
            sigma_high=sigmas[-1],
            transition_matrix=trans,
            n_regimes=n_regimes,
        )

    except Exception as e:
        logger.debug("HMM fit failed: %s", e)
        return None


def hmm_adjusted_sigma(
    garch_sigma: float,
    hmm_result: Optional[HmmRegimeResult],
    blend_weight: float = 0.5,
) -> float:
    """
    Compute regime-weighted sigma forecast blending GARCH with HMM.

    Formula:
        sigma_hmm = P(high) * sigma_high + P(low) * sigma_low
        sigma_adj = (1 - blend_weight) * garch_sigma + blend_weight * sigma_hmm

    The blend preserves GARCH's short-term forecast accuracy while
    incorporating HMM's regime-appropriate unconditional level.

    Parameters
    ----------
    garch_sigma : float
        GARCH 1-step daily vol forecast (decimal).
    hmm_result : HmmRegimeResult or None
        HMM regime detection result. If None, returns garch_sigma unchanged.
    blend_weight : float
        Blend weight: 0 = pure GARCH, 1 = pure HMM sigma. Default 0.5.

    Returns
    -------
    sigma_adj : float
        Blended daily vol forecast (decimal).
    """
    if hmm_result is None:
        return garch_sigma

    p_high = hmm_result.regime_prob_high
    p_low = 1.0 - p_high
    sigma_hmm = p_high * hmm_result.sigma_high + p_low * hmm_result.sigma_low

    sigma_adj = (1.0 - blend_weight) * garch_sigma + blend_weight * sigma_hmm

    # Safety: ensure positive and reasonable
    sigma_adj = max(sigma_adj, 1e-8)
    sigma_adj = min(sigma_adj, 1.0)

    return sigma_adj
