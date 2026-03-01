"""
GARCH(1,1) volatility forecasting with EWMA fallback.

All inputs and outputs are in DAILY decimal return units.
sigma_1d is the daily volatility forecast (standard deviation of daily returns).

Supports symmetric GARCH(1,1) and GJR-GARCH(1,1) (leverage effect).
Returns GarchResult with extracted parameters for GARCH-in-simulation.

Includes HAR-RV (Heterogeneous AutoRegressive Realized Variance) model
for horizon-specific volatility forecasting (Corsi, 2009).
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
# HAR-RV: Heterogeneous AutoRegressive Realized Variance (Corsi 2009)
# ---------------------------------------------------------------------------

@dataclass
class HarRvResult:
    """Result from HAR-RV model fitting."""
    sigma_1d: float          # 1-day-ahead vol forecast (decimal)
    sigma_5d: float          # 5-day-ahead average daily vol forecast (decimal)
    sigma_22d: float         # 22-day-ahead average daily vol forecast (decimal)
    coefficients_1d: np.ndarray   # [beta0, beta1, beta2, beta3] for 1d target
    coefficients_5d: np.ndarray   # [beta0, beta1, beta2, beta3] for 5d target
    coefficients_22d: np.ndarray  # [beta0, beta1, beta2, beta3] for 22d target
    r_squared_1d: float      # in-sample R² for 1d regression


def compute_realized_variance(returns: NDArray[np.float64], window: int) -> float:
    """
    Compute realized variance as mean of squared returns over a window.

    Parameters
    ----------
    returns : np.ndarray
        Daily simple returns (decimal form).
    window : int
        Number of recent returns to average.

    Returns
    -------
    rv : float
        Realized variance (decimal^2 units).
    """
    if len(returns) < window:
        return float(np.mean(returns ** 2))
    return float(np.mean(returns[-window:] ** 2))


def _build_har_features(returns: NDArray[np.float64]) -> tuple:
    """
    Build HAR-RV feature matrix and target vectors from returns.

    Computes rolling RV at 1d, 5d, 22d windows, then constructs
    regression targets for 1-day, 5-day, and 22-day ahead forecasts.

    Returns (X, y_1d, y_5d, y_22d) where X has columns [1, RV_1d, RV_5d, RV_22d].
    All arrays are aligned so row t predicts t+1 (or t+1..t+k for multi-day).
    """
    n = len(returns)
    r2 = returns ** 2  # squared returns

    # Rolling RV at different windows
    rv_1d = r2  # daily RV = r_t^2
    rv_5d = np.convolve(r2, np.ones(5) / 5, mode='full')[:n]
    rv_22d = np.convolve(r2, np.ones(22) / 22, mode='full')[:n]

    # Need at least 22 days of lookback for rv_22d to be valid
    start = 21  # first valid index where all RV components are properly averaged

    # Forward targets
    # y_1d[t] = rv_1d[t+1] = r2[t+1]
    # y_5d[t] = mean(r2[t+1]..r2[t+5])
    # y_22d[t] = mean(r2[t+1]..r2[t+22])
    max_t_1d = n - 2      # need at least 1 future day
    max_t_5d = n - 6      # need at least 5 future days
    max_t_22d = n - 23     # need at least 22 future days

    if max_t_22d <= start:
        return None, None, None, None  # insufficient data

    # Build aligned arrays for the longest common range (22d target)
    idx = np.arange(start, max_t_22d + 1)

    X = np.column_stack([
        np.ones(len(idx)),
        rv_1d[idx],
        rv_5d[idx],
        rv_22d[idx],
    ])

    y_1d = r2[idx + 1]
    y_5d = np.array([np.mean(r2[t + 1:t + 6]) for t in idx])
    y_22d = np.array([np.mean(r2[t + 1:t + 23]) for t in idx])

    return X, y_1d, y_5d, y_22d


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Ridge regression: beta = (X'X + alpha*I)^{-1} X'y.

    No external dependencies (pure numpy).
    """
    n_features = X.shape[1]
    reg = alpha * np.eye(n_features)
    reg[0, 0] = 0.0  # don't regularize intercept
    beta = np.linalg.solve(X.T @ X + reg, X.T @ y)
    return beta


def fit_har_rv(
    returns: NDArray[np.float64],
    min_window: int = 252,
    ridge_alpha: float = 0.01,
) -> Optional[HarRvResult]:
    """
    Fit HAR-RV model on available returns and forecast volatility.

    Three separate regressions for three forecast horizons (Corsi 2009):
        RV_target = beta0 + beta1*RV_1d + beta2*RV_5d + beta3*RV_22d

    Where targets are:
        1d:  RV_{t+1}
        5d:  mean(RV_{t+1}...RV_{t+5})
        22d: mean(RV_{t+1}...RV_{t+22})

    Parameters
    ----------
    returns : np.ndarray
        Daily simple returns (decimal form, e.g. 0.01 = 1%).
    min_window : int
        Minimum returns required for fitting.
    ridge_alpha : float
        Ridge regularization strength.

    Returns
    -------
    HarRvResult or None
        HAR-RV forecasts, or None if insufficient data.
    """
    n = len(returns)
    if n < min_window:
        logger.debug("HAR-RV: insufficient data (%d < %d)", n, min_window)
        return None

    data = np.asarray(returns, dtype=np.float64)

    X, y_1d, y_5d, y_22d = _build_har_features(data)
    if X is None:
        logger.debug("HAR-RV: insufficient data for feature construction")
        return None

    if len(X) < 44:  # need reasonable sample for regression
        logger.debug("HAR-RV: too few training samples (%d)", len(X))
        return None

    try:
        # Fit three regressions
        beta_1d = _ridge_fit(X, y_1d, alpha=ridge_alpha)
        beta_5d = _ridge_fit(X, y_5d, alpha=ridge_alpha)
        beta_22d = _ridge_fit(X, y_22d, alpha=ridge_alpha)

        # Current features (last available observation)
        r2 = data ** 2
        rv_1d_now = r2[-1]
        rv_5d_now = float(np.mean(r2[-5:]))
        rv_22d_now = float(np.mean(r2[-22:]))
        x_now = np.array([1.0, rv_1d_now, rv_5d_now, rv_22d_now])

        # Forecasts (variance, then take sqrt for sigma)
        var_1d = float(x_now @ beta_1d)
        var_5d = float(x_now @ beta_5d)
        var_22d = float(x_now @ beta_22d)

        # Floor at small positive value
        var_1d = max(var_1d, 1e-16)
        var_5d = max(var_5d, 1e-16)
        var_22d = max(var_22d, 1e-16)

        sigma_1d = float(np.sqrt(var_1d))
        sigma_5d = float(np.sqrt(var_5d))
        sigma_22d = float(np.sqrt(var_22d))

        # Safety clamp
        sigma_1d = min(max(sigma_1d, 1e-8), 1.0)
        sigma_5d = min(max(sigma_5d, 1e-8), 1.0)
        sigma_22d = min(max(sigma_22d, 1e-8), 1.0)

        # R² for 1d regression
        y_pred = X @ beta_1d
        ss_res = np.sum((y_1d - y_pred) ** 2)
        ss_tot = np.sum((y_1d - np.mean(y_1d)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        logger.info(
            "HAR-RV fit: sigma_1d=%.4f, sigma_5d=%.4f, sigma_22d=%.4f, R²=%.3f",
            sigma_1d, sigma_5d, sigma_22d, r_squared,
        )

        return HarRvResult(
            sigma_1d=sigma_1d,
            sigma_5d=sigma_5d,
            sigma_22d=sigma_22d,
            coefficients_1d=beta_1d,
            coefficients_5d=beta_5d,
            coefficients_22d=beta_22d,
            r_squared_1d=r_squared,
        )

    except Exception as e:
        logger.debug("HAR-RV fit failed: %s", e)
        return None
