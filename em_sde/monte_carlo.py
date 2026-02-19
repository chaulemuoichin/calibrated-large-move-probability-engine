"""
Monte Carlo simulation via Euler-Maruyama discretization in log-space.

Two simulation modes:
    1) Constant-vol GBM: simulate_gbm_terminal() — classic model
    2) GARCH-in-sim:     simulate_garch_terminal() — per-step vol dynamics
       with optional GJR leverage effect and Merton jump-diffusion

Unit convention (MANDATORY):
    - sigma_1d: daily volatility (decimal)
    - sigma_year = sigma_1d * sqrt(252)
    - dt = 1/252
    - steps = H (horizon in trading days)

Constant-vol Euler-Maruyama step (log-price):
    X_{n+1} = X_n + (mu_year - 0.5 * sigma_year^2) * dt
              + sigma_year * sqrt(dt) * Z_n

GARCH-in-sim vol update:
    sigma2[n+1] = omega + alpha * eps[n]^2 + gamma * eps[n]^2 * I(eps[n]<0) + beta * sigma2[n]

Jump-diffusion (Merton):
    dS/S += J * dN,  N ~ Poisson(lambda*dt),  J ~ N(mu_J, sigma_J^2)
"""

import numpy as np
from typing import Optional, Tuple, Union
from numpy.random import SeedSequence, default_rng

# Standard quantile levels for CRPS evaluation
QUANTILE_LEVELS = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])


def simulate_gbm_terminal(
    S0: float,
    sigma_1d: float,
    H: int,
    n_paths: int,
    mu_year: float = 0.0,
    seed: Optional[int] = None,
    t_df: float = 0.0,
) -> np.ndarray:
    """
    Simulate terminal prices via Euler-Maruyama GBM in log-space.

    Parameters
    ----------
    S0 : float
        Current price.
    sigma_1d : float
        Daily volatility (decimal).
    H : int
        Horizon in trading days (number of EM steps).
    n_paths : int
        Number of Monte Carlo paths.
    mu_year : float
        Annualized drift (default 0).
    seed : int, optional
        Random seed for reproducibility.
    t_df : float
        Student-t degrees of freedom for fat-tailed innovations.
        0 or inf = standard Gaussian. Typical equity value: 5.

    Returns
    -------
    terminal_prices : np.ndarray
        Array of shape (n_paths,) with simulated terminal prices.
    """
    rng = default_rng(SeedSequence(seed)) if seed is not None else default_rng()

    sigma_year = sigma_1d * np.sqrt(252.0)
    dt = 1.0 / 252.0
    drift = (mu_year - 0.5 * sigma_year ** 2) * dt
    vol = sigma_year * np.sqrt(dt)

    use_t = t_df > 0 and np.isfinite(t_df) and t_df > 2.0

    # Simulate in log-space, only track current state (memory efficient)
    log_price = np.full(n_paths, np.log(S0))

    for _ in range(H):
        if use_t:
            # Student-t scaled to unit variance: Var(t/sqrt((nu-2)/nu)) = 1
            Z = rng.standard_t(df=t_df, size=n_paths) * np.sqrt((t_df - 2.0) / t_df)
        else:
            Z = rng.standard_normal(n_paths)
        log_price += drift + vol * Z

    return np.exp(log_price)


def simulate_garch_terminal(
    S0: float,
    sigma_1d: float,
    H: int,
    n_paths: int,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float = 0.0,
    mu_year: float = 0.0,
    seed: Optional[int] = None,
    t_df: float = 0.0,
    jump_intensity: float = 0.0,
    jump_mean: float = 0.0,
    jump_vol: float = 0.0,
) -> np.ndarray:
    """
    Simulate terminal prices with GARCH(1,1) vol dynamics within each path.

    Unlike simulate_gbm_terminal() which uses constant vol, this evolves
    variance per-step using GARCH dynamics. Supports GJR leverage (gamma>0)
    and Merton jump-diffusion.

    Parameters
    ----------
    S0 : float
        Current price.
    sigma_1d : float
        Initial daily volatility (decimal), from GARCH forecast.
    H : int
        Horizon in trading days (number of steps).
    n_paths : int
        Number of Monte Carlo paths.
    omega : float
        GARCH intercept (daily decimal^2 units).
    alpha : float
        ARCH coefficient (weight on lagged squared shock).
    beta : float
        GARCH persistence (weight on lagged variance).
    gamma : float
        GJR asymmetry coefficient. gamma>0 means negative shocks
        amplify vol more. Set 0 for symmetric GARCH.
    mu_year : float
        Annualized drift (default 0).
    seed : int, optional
        Random seed for reproducibility.
    t_df : float
        Student-t degrees of freedom. 0 or inf = Gaussian.
    jump_intensity : float
        Merton jump arrival rate (jumps per year). 0 = no jumps.
    jump_mean : float
        Mean jump size in log-space (negative = crash bias).
    jump_vol : float
        Jump size volatility in log-space.

    Returns
    -------
    terminal_prices : np.ndarray
        Array of shape (n_paths,) with simulated terminal prices.
    """
    rng = default_rng(SeedSequence(seed)) if seed is not None else default_rng()

    dt = 1.0 / 252.0
    log_price = np.full(n_paths, np.log(S0))

    # Initialize per-path variance at GARCH forecast level
    sigma2 = np.full(n_paths, sigma_1d ** 2)

    use_t = t_df > 0 and np.isfinite(t_df) and t_df > 2.0
    use_jumps = jump_intensity > 0.0
    lambda_dt = jump_intensity * dt  # jump probability per step

    # Jump drift compensator to keep E[S_T] correct
    if use_jumps:
        jump_compensator = jump_intensity * (np.exp(jump_mean + 0.5 * jump_vol ** 2) - 1.0)
    else:
        jump_compensator = 0.0

    for _ in range(H):
        sigma_step = np.sqrt(sigma2)  # daily vol for this step

        # Annualized vol for this step
        sigma_year_step = sigma_step * np.sqrt(252.0)

        # Diffusion drift and vol
        drift = (mu_year - jump_compensator - 0.5 * sigma_year_step ** 2) * dt
        vol = sigma_year_step * np.sqrt(dt)  # = sigma_step

        # Innovation
        if use_t:
            Z = rng.standard_t(df=t_df, size=n_paths) * np.sqrt((t_df - 2.0) / t_df)
        else:
            Z = rng.standard_normal(n_paths)

        # Daily return shock (for GARCH update)
        eps = sigma_step * Z

        # Diffusion step
        log_price += drift + vol * Z

        # Jump component (Merton)
        if use_jumps:
            N_jump = rng.poisson(lambda_dt, size=n_paths)
            jump_mask = N_jump > 0
            if jump_mask.any():
                # Vectorized: for paths with jumps, sum N_jump[i] normal draws
                for idx in np.where(jump_mask)[0]:
                    n_j = int(N_jump[idx])
                    J_total = rng.normal(jump_mean, jump_vol, size=n_j).sum()
                    log_price[idx] += J_total

        # GARCH(1,1) / GJR-GARCH(1,1) variance update
        # sigma2[n+1] = omega + alpha * eps^2 + gamma * eps^2 * I(eps<0) + beta * sigma2[n]
        eps2 = eps ** 2
        sigma2 = omega + alpha * eps2 + gamma * eps2 * (eps < 0).astype(np.float64) + beta * sigma2

        # Floor variance to prevent collapse
        np.maximum(sigma2, 1e-12, out=sigma2)

    return np.exp(log_price)


def compute_state_dependent_jumps(
    sigma_1d: float,
    vol_history: np.ndarray,
    low_params: tuple,
    high_params: tuple,
    pctile_low: float = 0.25,
    pctile_high: float = 0.75,
) -> tuple:
    """
    Compute state-dependent jump parameters via linear interpolation
    between low-vol and high-vol regime parameter sets.

    Parameters
    ----------
    sigma_1d : float
        Current daily volatility forecast.
    vol_history : np.ndarray
        Historical daily volatility values (backward-looking only).
    low_params : tuple
        (intensity, mean, vol) for low-vol regime.
    high_params : tuple
        (intensity, mean, vol) for high-vol regime.
    pctile_low : float
        Percentile threshold for low vol regime (default 25th).
    pctile_high : float
        Percentile threshold for high vol regime (default 75th).

    Returns
    -------
    intensity, mean, vol : tuple of float
        Interpolated jump parameters.
    """
    if len(vol_history) < 50:
        return tuple(
            (lo + hi) / 2.0 for lo, hi in zip(low_params, high_params)
        )

    sigma_lo = float(np.percentile(vol_history, pctile_low * 100))
    sigma_hi = float(np.percentile(vol_history, pctile_high * 100))

    if sigma_hi <= sigma_lo:
        return low_params

    t = float(np.clip((sigma_1d - sigma_lo) / (sigma_hi - sigma_lo), 0.0, 1.0))

    return tuple(
        lo + t * (hi - lo) for lo, hi in zip(low_params, high_params)
    )


def compute_move_probability(
    terminal_prices: np.ndarray,
    S0: float,
    threshold: float,
    return_quantiles: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, np.ndarray]]:
    """
    Compute probability of a large two-sided move.

    Parameters
    ----------
    terminal_prices : np.ndarray
        Simulated terminal prices.
    S0 : float
        Current price.
    threshold : float
        Absolute return threshold for the event.
    return_quantiles : bool
        If True, also return quantiles of the simulated return distribution
        at QUANTILE_LEVELS for CRPS evaluation.

    Returns
    -------
    p_raw : float
        Raw Monte Carlo probability estimate.
    se : float
        Standard error of the estimate: sqrt(p*(1-p)/M).
    quantiles : np.ndarray (only if return_quantiles=True)
        Return distribution quantiles at QUANTILE_LEVELS.
    """
    sim_returns = terminal_prices / S0 - 1.0
    events = np.abs(sim_returns) >= threshold
    M = len(terminal_prices)
    p_raw = float(np.mean(events))
    se = float(np.sqrt(p_raw * (1.0 - p_raw) / M)) if M > 0 else 0.0
    if return_quantiles:
        quantiles = np.quantile(sim_returns, QUANTILE_LEVELS)
        return p_raw, se, quantiles
    return p_raw, se
