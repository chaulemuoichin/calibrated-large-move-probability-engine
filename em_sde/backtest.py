"""
Walk-forward backtest engine with strict no-lookahead guarantee.

For each date t:
    1) Resolve any pending predictions whose horizon has elapsed.
    2) Update calibrators with resolved labels.
    3) Estimate sigma using only data <= t (GARCH or EWMA fallback).
    4) Compute vol-scaled thresholds for each horizon.
    5) Simulate MC probabilities for H=5,10,20 in parallel.
    6) Apply online calibrators to get calibrated probabilities.
    7) Store prediction row and queue for future resolution.

Resolution queues guarantee that calibrator updates only use
realized outcomes, never future information.
"""

import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.random import SeedSequence

from .calibration import OnlineCalibrator, MultiFeatureCalibrator, RegimeMultiFeatureCalibrator
from .garch import fit_garch, fit_garch_ensemble, GarchResult, project_to_stationary, garch_diagnostics as _garch_diag, ewma_volatility, garch_term_structure_vol, fit_har_rv, fit_har_ohlc, HarRvResult
from .monte_carlo import (
    simulate_gbm_terminal, simulate_garch_terminal, compute_move_probability,
    compute_state_dependent_jumps, QUANTILE_LEVELS,
)

logger = logging.getLogger(__name__)


@dataclass
class PendingPrediction:
    """A prediction awaiting label resolution."""
    row_idx: int        # index into results list
    price_idx: int      # index into price array (prediction date)
    p_raw: float        # raw MC probability
    threshold: float    # threshold used
    sigma_1d: float = 0.0      # daily vol at prediction time
    delta_sigma: float = 0.0   # 20d sigma change
    vol_ratio: float = 1.0     # realized_vol / forecast_vol
    vol_of_vol: float = 0.0    # rolling std of sigma
    earnings_proximity: float = 0.0  # proximity to nearest earnings date
    implied_vol_ratio: float = 1.0   # sigma_implied / sigma_hist (1.0 = no implied data)
    range_vol_ratio: float = 1.0     # Parkinson-style range vol / forecast vol
    overnight_gap: float = 0.0       # abs(log(open_t / close_{t-1}))
    intraday_range: float = 0.0      # log(high_t / low_t)


@dataclass
class MetaPrediction:
    """Ensemble prediction awaiting y_20 resolution."""
    row_idx: int
    price_idx: int
    risk_combo: float


class RegimeRouter:
    """Walk-forward safe vol regime classifier for threshold routing."""

    def __init__(self, warmup: int = 252, vol_window: int = 252,
                 low_mode: str = "fixed_pct", mid_mode: str = "fixed_pct",
                 high_mode: str = "anchored_vol"):
        self._vol_history: Deque[float] = deque(maxlen=vol_window)
        self._warmup = warmup
        self._low_mode = low_mode
        self._mid_mode = mid_mode
        self._high_mode = high_mode

    def observe(self, sigma_1d: float) -> None:
        """Record vol observation for percentile computation."""
        self._vol_history.append(sigma_1d)

    def get_threshold_mode(self, sigma_1d: float) -> str:
        """Return threshold mode for current vol level. Falls back to mid during warmup."""
        if len(self._vol_history) < self._warmup:
            return self._mid_mode
        arr = np.sort(np.array(self._vol_history))
        pctile = np.searchsorted(arr, sigma_1d) / len(arr)
        if pctile < 0.25:
            return self._low_mode
        elif pctile > 0.75:
            return self._high_mode
        else:
            return self._mid_mode

    @property
    def is_warmed_up(self) -> bool:
        return len(self._vol_history) >= self._warmup


def _blend_sigma_variance(sigma_base: float, sigma_alt: Optional[float], weight: float) -> float:
    """Blend two daily volatility estimates in variance space."""
    w = float(min(max(weight, 0.0), 1.0))
    if w <= 0.0 or sigma_alt is None or not np.isfinite(sigma_alt) or sigma_alt <= 0.0:
        return float(sigma_base)
    if not np.isfinite(sigma_base) or sigma_base <= 0.0:
        return float(sigma_alt)
    return float(np.sqrt((1.0 - w) * sigma_base ** 2 + w * sigma_alt ** 2))


def _count_scheduled_events_in_horizon(
    trading_dates: pd.Index,
    current_idx: int,
    horizon: int,
    event_dates: Optional[np.ndarray],
) -> int:
    """Count scheduled events from today through the horizon end date."""
    if event_dates is None or len(event_dates) == 0:
        return 0
    current_day = np.datetime64(trading_dates[current_idx], "D")
    end_idx = min(current_idx + horizon, len(trading_dates) - 1)
    end_day = np.datetime64(trading_dates[end_idx], "D")
    mask = (event_dates >= current_day) & (event_dates <= end_day)
    return int(np.sum(mask))


def _estimate_scheduled_event_variance(
    trading_dates: pd.Index,
    returns: npt.NDArray[np.float64],
    event_dates: Optional[np.ndarray],
    current_idx: int,
    sigma_daily: float,
    lookback_events: int,
    min_events: int,
    scale: float,
) -> float:
    """
    Estimate residual daily variance contributed by scheduled event jumps.

    Uses the largest absolute close-to-close move around each past event date,
    then subtracts the current diffusive daily variance anchor.
    """
    if event_dates is None or len(event_dates) == 0 or current_idx <= 1:
        return 0.0

    current_day = np.datetime64(trading_dates[current_idx], "D")
    past_events = np.unique(event_dates[event_dates < current_day])
    if len(past_events) == 0:
        return 0.0

    trade_days = np.asarray(trading_dates.values, dtype="datetime64[D]")
    event_moves: List[float] = []
    used_return_idx = set()

    for event_day in past_events[::-1]:
        pos = int(np.searchsorted(trade_days, event_day))
        candidate_idx: List[int] = []
        if 0 <= pos - 1 < current_idx:
            candidate_idx.append(pos - 1)
        if 0 <= pos < current_idx:
            candidate_idx.append(pos)
        if not candidate_idx:
            continue
        best_idx = max(candidate_idx, key=lambda j: abs(float(returns[j])))
        if best_idx in used_return_idx:
            continue
        used_return_idx.add(best_idx)
        event_moves.append(float(returns[best_idx]))
        if len(event_moves) >= lookback_events:
            break

    if len(event_moves) < min_events:
        return 0.0

    event_var = float(np.mean(np.square(event_moves)))
    residual_var = max(event_var - max(sigma_daily, 0.0) ** 2, 0.0)
    return float(scale * residual_var)


def run_walkforward(
    df: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """
    Execute the full walk-forward backtest.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with DatetimeIndex and column 'price'.
    cfg : PipelineConfig
        Full pipeline configuration.

    Returns
    -------
    results : pd.DataFrame
        Complete results with all columns per the spec.
    """
    prices: npt.NDArray[np.float64] = df["price"].to_numpy(dtype=np.float64, copy=True)
    open_prices: Optional[npt.NDArray[np.float64]] = None
    high_prices: Optional[npt.NDArray[np.float64]] = None
    low_prices: Optional[npt.NDArray[np.float64]] = None
    has_ohlc = all(col in df.columns for col in ("open", "high", "low"))
    if has_ohlc:
        open_prices = df["open"].to_numpy(dtype=np.float64, copy=True)
        high_prices = df["high"].to_numpy(dtype=np.float64, copy=True)
        low_prices = df["low"].to_numpy(dtype=np.float64, copy=True)
    dates = df.index
    n = len(prices)

    # Compute all daily returns (returns[i] = prices[i+1]/prices[i] - 1)
    all_returns: npt.NDArray[np.float64] = np.diff(prices) / prices[:-1]

    horizons: List[int] = list(cfg.model.horizons)
    k = cfg.model.k
    mu_year = cfg.model.mu_year
    mc_base = cfg.model.mc_base_paths
    mc_boost = cfg.model.mc_boost_paths
    mc_boost_thr = cfg.model.mc_boost_threshold
    base_seed = cfg.model.seed
    garch_window = cfg.model.garch_window
    garch_min = cfg.model.garch_min_window
    t_df = cfg.model.t_df
    mc_vol_term_structure = cfg.model.mc_vol_term_structure
    mc_regime_t_df = cfg.model.mc_regime_t_df
    mc_regime_t_df_low = cfg.model.mc_regime_t_df_low
    mc_regime_t_df_mid = cfg.model.mc_regime_t_df_mid
    mc_regime_t_df_high = cfg.model.mc_regime_t_df_high
    garch_in_sim = cfg.model.garch_in_sim
    garch_model_type = cfg.model.garch_model_type
    jump_enabled = cfg.model.jump_enabled
    jump_intensity = cfg.model.jump_intensity if jump_enabled else 0.0
    jump_mean = cfg.model.jump_mean if jump_enabled else 0.0
    jump_vol = cfg.model.jump_vol if jump_enabled else 0.0
    jump_state_dependent = cfg.model.jump_state_dependent if jump_enabled else False
    jump_low_params = (cfg.model.jump_low_intensity, cfg.model.jump_low_mean, cfg.model.jump_low_vol)
    jump_high_params = (cfg.model.jump_high_intensity, cfg.model.jump_high_mean, cfg.model.jump_high_vol)
    lr = cfg.calibration.lr
    adaptive_lr = cfg.calibration.adaptive_lr
    min_updates = cfg.calibration.min_updates
    safety_gate = cfg.calibration.safety_gate
    gate_window = cfg.calibration.gate_window
    gate_on_disc = cfg.calibration.gate_on_discrimination
    gate_auc_thr = cfg.calibration.gate_auc_threshold
    gate_sep_thr = cfg.calibration.gate_separation_threshold
    gate_disc_win = cfg.calibration.gate_discrimination_window
    ensemble_enabled = cfg.calibration.ensemble_enabled
    ensemble_weights = cfg.calibration.ensemble_weights
    regime_n_bins = cfg.calibration.regime_n_bins
    multi_feature = cfg.calibration.multi_feature
    multi_feature_lr = cfg.calibration.multi_feature_lr
    multi_feature_l2 = cfg.calibration.multi_feature_l2
    multi_feature_min_updates = cfg.calibration.multi_feature_min_updates
    histogram_post_cal = cfg.calibration.histogram_post_calibration
    post_cal_method = cfg.calibration.post_cal_method
    histogram_n_bins = cfg.calibration.histogram_n_bins
    histogram_min_samples = cfg.calibration.histogram_min_samples
    histogram_prior_strength = cfg.calibration.histogram_prior_strength
    histogram_monotonic = cfg.calibration.histogram_monotonic
    histogram_interpolate = cfg.calibration.histogram_interpolate
    histogram_n_bins_by_horizon = cfg.calibration.histogram_n_bins_by_horizon or {}
    histogram_prior_by_horizon = cfg.calibration.histogram_prior_strength_by_horizon or {}
    beta_calibration = cfg.calibration.beta_calibration
    multi_feature_regime_conditional = cfg.calibration.multi_feature_regime_conditional
    threshold_mode = cfg.model.threshold_mode
    fixed_threshold_pct = cfg.model.fixed_threshold_pct
    store_quantiles = cfg.model.store_quantiles
    garch_stationarity = cfg.model.garch_stationarity_constraint
    garch_target_persistence = cfg.model.garch_target_persistence
    garch_fallback_ewma = cfg.model.garch_fallback_to_ewma
    har_rv_enabled = cfg.model.har_rv
    har_rv_min_window = cfg.model.har_rv_min_window
    har_rv_refit_interval = cfg.model.har_rv_refit_interval
    har_rv_ridge_alpha = cfg.model.har_rv_ridge_alpha
    har_rv_variant = getattr(cfg.model, "har_rv_variant", "rv")
    hybrid_variance_enabled = cfg.model.hybrid_variance_enabled
    hybrid_range_blend = cfg.model.hybrid_range_blend
    fixed_pct_by_horizon = cfg.model.regime_gated_fixed_pct_by_horizon or {}
    fhs_enabled = cfg.model.fhs_enabled
    garch_ensemble_enabled = cfg.model.garch_ensemble
    earnings_calendar_enabled = cfg.model.earnings_calendar
    implied_vol_enabled = cfg.model.implied_vol_enabled
    implied_vol_blend = cfg.model.implied_vol_blend
    implied_vol_as_feature = cfg.model.implied_vol_as_feature
    scheduled_jump_variance_enabled = cfg.model.scheduled_jump_variance
    scheduled_jump_lookback_events = cfg.model.scheduled_jump_lookback_events
    scheduled_jump_min_events = cfg.model.scheduled_jump_min_events
    scheduled_jump_scale = cfg.model.scheduled_jump_scale
    ohlc_features_enabled = cfg.model.ohlc_features_enabled and has_ohlc

    # Regime-gated threshold router
    regime_router: Optional[RegimeRouter] = None
    if threshold_mode == "regime_gated":
        regime_router = RegimeRouter(
            warmup=cfg.model.regime_gated_warmup,
            vol_window=cfg.model.regime_gated_vol_window,
            low_mode=cfg.model.regime_gated_low_mode,
            mid_mode=cfg.model.regime_gated_mid_mode,
            high_mode=cfg.model.regime_gated_high_mode,
        )

    # Seed sequence for reproducible parallel MC
    ss = SeedSequence(base_seed)

    # Load earnings dates if enabled
    earnings_dates = None
    if earnings_calendar_enabled or scheduled_jump_variance_enabled:
        from .data_layer import load_earnings_dates, compute_earnings_proximity
        earnings_dates = load_earnings_dates(cfg.data.ticker)
        if earnings_dates is None:
            logger.warning("Earnings dates unavailable for %s; disabling earnings-linked features", cfg.data.ticker)
            earnings_calendar_enabled = False
            scheduled_jump_variance_enabled = False

    # Load implied vol data if enabled
    iv_df = None
    if implied_vol_enabled:
        from .data_layer import load_implied_vol, get_implied_vol_for_horizon
        iv_df = load_implied_vol(cfg.model.implied_vol_csv_path)
        logger.info("Implied vol loaded: %d rows, blend=%.2f, as_feature=%s",
                     len(iv_df), implied_vol_blend, implied_vol_as_feature)

    # Calibrators: one per horizon (multi-feature or standard online)
    # Per-horizon histogram overrides allow coarser bins / stronger priors
    # at long horizons where events are sparse (e.g., H=20: 7 bins, prior=25).
    calibrators_online: Optional[Dict[int, OnlineCalibrator]] = None
    calibrators_mf: Optional[Dict[int, MultiFeatureCalibrator]] = None

    def _hist_kwargs(H: int) -> dict:
        """Per-horizon histogram settings with fallback to global defaults."""
        return dict(
            histogram_n_bins=histogram_n_bins_by_horizon.get(H, histogram_n_bins),
            histogram_prior_strength=histogram_prior_by_horizon.get(H, histogram_prior_strength),
        )

    _mf_kwargs_base = dict(
        lr=multi_feature_lr,
        l2_reg=multi_feature_l2,
        min_updates=multi_feature_min_updates,
        safety_gate=safety_gate,
        gate_window=gate_window,
        gate_on_discrimination=gate_on_disc,
        gate_auc_threshold=gate_auc_thr,
        gate_separation_threshold=gate_sep_thr,
        gate_discrimination_window=gate_disc_win,
        histogram_post_cal=histogram_post_cal,
        histogram_min_samples=histogram_min_samples,
        histogram_monotonic=histogram_monotonic,
        histogram_interpolate=histogram_interpolate,
        post_cal_method=post_cal_method,
        beta_calibration=beta_calibration,
        implied_vol_aware=implied_vol_enabled and implied_vol_as_feature,
        ohlc_aware=ohlc_features_enabled,
    )
    if multi_feature and multi_feature_regime_conditional:
        calibrators_mf = {
            H: RegimeMultiFeatureCalibrator(
                n_bins=regime_n_bins,
                earnings_aware=earnings_calendar_enabled and H <= 5,
                **_mf_kwargs_base,
                **_hist_kwargs(H),
            )
            for H in horizons
        }
    elif multi_feature:
        calibrators_mf = {
            H: MultiFeatureCalibrator(
                earnings_aware=earnings_calendar_enabled and H <= 5,
                **_mf_kwargs_base,
                **_hist_kwargs(H),
            )
            for H in horizons
        }
    else:
        calibrators_online = {
            H: OnlineCalibrator(
                lr=lr,
                adaptive_lr=adaptive_lr,
                min_updates=min_updates,
                safety_gate=safety_gate,
                gate_window=gate_window,
                gate_on_discrimination=gate_on_disc,
                gate_auc_threshold=gate_auc_thr,
                gate_separation_threshold=gate_sep_thr,
                gate_discrimination_window=gate_disc_win,
                histogram_post_cal=histogram_post_cal,
                histogram_min_samples=histogram_min_samples,
                histogram_monotonic=histogram_monotonic,
                histogram_interpolate=histogram_interpolate,
                post_cal_method=post_cal_method,
                **_hist_kwargs(H),
            )
            for H in horizons
        }

    # Meta-calibrator for ensemble
    meta_calibrator = (OnlineCalibrator(lr=lr, adaptive_lr=adaptive_lr, min_updates=min_updates,
                                        safety_gate=safety_gate, gate_window=gate_window)
                       if ensemble_enabled else None)

    # Resolution queues: pending predictions per horizon
    queues: Dict[int, Deque[PendingPrediction]] = {H: deque() for H in horizons}
    meta_queue: Deque[MetaPrediction] = deque()

    # Track resolved labels for adaptive MC path count
    resolved_labels: Dict[int, Deque[float]] = {
        H: deque(maxlen=252) for H in horizons
    }

    # Warmup: need garch_min returns before first prediction
    # At price index idx, we have returns[0..idx-1], so we need idx >= garch_min
    warmup = garch_min
    max_H = max(horizons)

    results_list: List[Dict[str, Any]] = []

    # Auxiliary feature tracking for multi-feature calibration
    sigma_history: Deque[float] = deque(maxlen=300)  # rolling sigma history

    # HAR-RV volatility model state
    har_rv_result: Optional[HarRvResult] = None
    last_har_rv_fit: int = -999  # last index where HAR-RV was fitted

    logger.info(
        "Starting walk-forward: %d dates, warmup=%d, horizons=%s",
        n, warmup, horizons,
    )

    last_garch = None
    for idx in range(warmup, n):
        row = {"date": dates[idx]}

        # === Step 1: Resolve pending predictions ===
        if calibrators_mf is not None:
            _resolve_predictions_mf(
                idx, prices, horizons, queues, calibrators_mf,
                resolved_labels, results_list,
            )
        else:
            assert calibrators_online is not None
            _resolve_predictions_online(
                idx, prices, horizons, queues, calibrators_online,
                resolved_labels, results_list,
            )

        if meta_calibrator is not None:
            _resolve_meta(
                idx, prices, k, horizons[-1],  # H=20 for meta
                meta_queue, meta_calibrator, results_list,
            )

        # === Step 2: Fit GARCH on returns up to idx ===
        # returns[0..idx-1] are known at date idx
        available_returns = all_returns[:idx]
        if garch_ensemble_enabled:
            garch_result = fit_garch_ensemble(
                available_returns, window=garch_window, min_window=garch_min,
            )
        else:
            garch_result = fit_garch(
                available_returns, window=garch_window, min_window=garch_min,
                model_type=garch_model_type if garch_in_sim else "garch",
            )
        # Apply stationarity projection if needed
        projected = False
        if (garch_stationarity and garch_in_sim
                and garch_result.diagnostics is not None
                and not garch_result.diagnostics.get("is_stationary", True)):
            effective_garch_model_type = garch_result.model_type
            if garch_fallback_ewma:
                sigma_ewma = ewma_volatility(available_returns, span=252)
                garch_result = GarchResult(
                    sigma_1d=sigma_ewma,
                    source="ewma_fallback_nonstationary",
                    standardized_residuals=garch_result.standardized_residuals,
                    model_type=effective_garch_model_type,
                )
                projected = True
            elif garch_result.omega is not None:
                omega_p, alpha_p, beta_p, gamma_p = project_to_stationary(
                    garch_result.omega, garch_result.alpha, garch_result.beta,
                    garch_result.gamma, effective_garch_model_type,
                    target_persistence=garch_target_persistence,
                    variance_anchor=garch_result.sigma_1d ** 2,
                )
                garch_result = GarchResult(
                    sigma_1d=garch_result.sigma_1d,
                    source=garch_result.source + "_projected",
                    omega=omega_p, alpha=alpha_p, beta=beta_p, gamma=gamma_p,
                    diagnostics=_garch_diag(omega_p, alpha_p, beta_p, gamma_p, effective_garch_model_type),
                    standardized_residuals=garch_result.standardized_residuals,
                    model_type=effective_garch_model_type,
                )
                projected = True

        sigma_1d = garch_result.sigma_1d
        last_garch = garch_result
        row["sigma_garch_1d"] = sigma_1d
        row["sigma_source"] = garch_result.source
        row["garch_projected"] = projected

        # === HAR-RV sigma override (refit every har_rv_refit_interval days) ===
        if har_rv_enabled:
            if har_rv_result is None or (idx - last_har_rv_fit) >= har_rv_refit_interval:
                if har_rv_variant in ("range", "rvx") and has_ohlc and open_prices is not None and high_prices is not None and low_prices is not None:
                    har_rv_result = fit_har_ohlc(
                        available_returns,
                        close_prices=prices[:idx + 1],
                        high_prices=high_prices[:idx + 1],
                        low_prices=low_prices[:idx + 1],
                        open_prices=open_prices[:idx + 1],
                        min_window=har_rv_min_window,
                        ridge_alpha=har_rv_ridge_alpha,
                        variant=har_rv_variant,
                    )
                    if har_rv_result is None:
                        har_rv_result = fit_har_rv(
                            available_returns,
                            min_window=har_rv_min_window,
                            ridge_alpha=har_rv_ridge_alpha,
                        )
                else:
                    har_rv_result = fit_har_rv(
                        available_returns,
                        min_window=har_rv_min_window,
                        ridge_alpha=har_rv_ridge_alpha,
                    )
                last_har_rv_fit = idx

            if har_rv_result is not None:
                sigma_1d = har_rv_result.sigma_1d
                row["sigma_har_rv_1d"] = har_rv_result.sigma_1d
                row["sigma_har_rv_5d"] = har_rv_result.sigma_5d
                row["sigma_har_rv_22d"] = har_rv_result.sigma_22d
                row["har_rv_r2"] = har_rv_result.r_squared_1d
                row["har_rv_variant"] = har_rv_variant

        sigma_range_20d: Optional[float] = None
        range_vol_ratio = 1.0
        overnight_gap = 0.0
        intraday_range = 0.0
        if ohlc_features_enabled and open_prices is not None and high_prices is not None and low_prices is not None:
            if idx > 0 and open_prices[idx] > 0 and prices[idx - 1] > 0:
                overnight_gap = float(abs(np.log(open_prices[idx] / prices[idx - 1])))
            if high_prices[idx] > 0 and low_prices[idx] > 0:
                intraday_range = float(np.log(high_prices[idx] / low_prices[idx]))

            win_start = max(0, idx - 19)
            highs = high_prices[win_start:idx + 1]
            lows = low_prices[win_start:idx + 1]
            valid_range = (highs > 0.0) & (lows > 0.0) & (highs >= lows)
            if valid_range.any():
                park_var = np.log(highs[valid_range] / lows[valid_range]) ** 2 / (4.0 * np.log(2.0))
                sigma_range_20d = float(np.sqrt(np.mean(park_var))) if len(park_var) > 0 else None
                if sigma_range_20d is not None:
                    row["sigma_range_20d"] = sigma_range_20d

        # Build the physical horizon sigma curve before implied-vol blending.
        sigma_per_h: Dict[int, float] = {}
        if har_rv_enabled and har_rv_result is not None:
            for H in horizons:
                if H <= 1:
                    sigma_per_h[H] = har_rv_result.sigma_1d
                elif H < 5:
                    w = (H - 1) / (5 - 1)
                    sigma_per_h[H] = (1 - w) * har_rv_result.sigma_1d + w * har_rv_result.sigma_5d
                elif H < 22:
                    w = (H - 5) / (22 - 5)
                    sigma_per_h[H] = (1 - w) * har_rv_result.sigma_5d + w * har_rv_result.sigma_22d
                else:
                    sigma_per_h[H] = har_rv_result.sigma_22d
        else:
            for H in horizons:
                if mc_vol_term_structure and garch_result.omega is not None:
                    sigma_per_h[H] = garch_term_structure_vol(
                        sigma_1d, garch_result.omega, garch_result.alpha,
                        garch_result.beta, garch_result.gamma, H,
                        model_type=garch_result.model_type,
                    )
                else:
                    sigma_per_h[H] = sigma_1d

        if hybrid_variance_enabled and sigma_range_20d is not None and sigma_range_20d > 0.0:
            sigma_1d_pre_hybrid = sigma_1d
            sigma_1d = _blend_sigma_variance(sigma_1d_pre_hybrid, sigma_range_20d, hybrid_range_blend)
            row["sigma_hybrid_1d"] = sigma_1d
            for H in horizons:
                range_sigma_h = sigma_range_20d
                if sigma_1d_pre_hybrid > 0.0:
                    range_sigma_h = sigma_range_20d * (sigma_per_h[H] / sigma_1d_pre_hybrid)
                sigma_per_h[H] = _blend_sigma_variance(sigma_per_h[H], range_sigma_h, hybrid_range_blend)
                row[f"sigma_physical_{H}"] = sigma_per_h[H]
        else:
            row["sigma_hybrid_1d"] = sigma_1d
            for H in horizons:
                row[f"sigma_physical_{H}"] = sigma_per_h[H]

        # Compute auxiliary features for multi-feature calibration (walk-forward safe)
        sigma_history.append(sigma_1d)
        if regime_router is not None:
            regime_router.observe(sigma_1d)
        delta_sigma = (sigma_1d - sigma_history[-21]) if len(sigma_history) > 20 else 0.0
        if idx >= 20:
            realized_vol = float(np.std(all_returns[idx - 20:idx]))
            vol_ratio = realized_vol / sigma_1d if sigma_1d > 0 else 1.0
        else:
            realized_vol = sigma_1d
            vol_ratio = 1.0
        if len(sigma_history) >= 60:
            vol_of_vol = float(np.std(list(sigma_history)[-60:]))
        else:
            vol_of_vol = 0.0
        if sigma_range_20d is not None and sigma_1d > 0:
            range_vol_ratio = sigma_range_20d / sigma_1d

        row["delta_sigma"] = delta_sigma
        row["vol_ratio"] = vol_ratio
        row["vol_of_vol"] = vol_of_vol
        row["range_vol_ratio"] = range_vol_ratio
        row["overnight_gap"] = overnight_gap
        row["intraday_range"] = intraday_range

        # Earnings proximity feature
        if earnings_calendar_enabled and earnings_dates is not None:
            current_date = np.datetime64(dates[idx], "D")
            earnings_prox = compute_earnings_proximity(current_date, earnings_dates)
            row["earnings_proximity"] = earnings_prox
        else:
            earnings_prox = 0.0

        # Feed vol to regime-MF calibrators
        if calibrators_mf is not None:
            for H in horizons:
                cal = calibrators_mf[H]
                if hasattr(cal, 'observe_vol'):
                    cal.observe_vol(sigma_1d)

        # === Step 3: Compute thresholds ===
        thresholds = {}
        if regime_router is not None:
            effective_mode = regime_router.get_threshold_mode(sigma_1d)
            row["threshold_regime"] = effective_mode
        else:
            effective_mode = threshold_mode
        for H in horizons:
            # Per-horizon fixed_pct override (for regime_gated configs)
            h_fixed_pct = fixed_pct_by_horizon.get(H, fixed_threshold_pct) if fixed_pct_by_horizon else fixed_threshold_pct
            if effective_mode == "fixed_pct":
                thr = h_fixed_pct
            elif effective_mode == "anchored_vol":
                # Use expanding-window unconditional vol (changes slowly)
                sigma_uncond = float(np.std(all_returns[:idx])) if idx > garch_min else sigma_1d
                thr = k * sigma_uncond * np.sqrt(H)
            else:
                raise ValueError(f"Unknown threshold mode: {effective_mode!r}")
            thresholds[H] = thr
            row[f"thr_{H}"] = thr

        # === Step 4: Determine MC path counts ===
        path_counts = {}
        for H in horizons:
            if (len(resolved_labels[H]) >= 252
                    and sum(resolved_labels[H]) / len(resolved_labels[H]) < mc_boost_thr):
                path_counts[H] = mc_boost
            else:
                path_counts[H] = mc_base

        # === Step 5: Simulate MC for all horizons in parallel ===
        S0 = prices[idx]
        child_seeds = ss.spawn(len(horizons))

        # Compute per-step jump parameters (state-dependent or static)
        if jump_state_dependent and len(sigma_history) >= 50:
            vol_hist_arr = np.array(list(sigma_history))
            step_jump_intensity, step_jump_mean, step_jump_vol = compute_state_dependent_jumps(
                sigma_1d, vol_hist_arr, jump_low_params, jump_high_params,
            )
            row["jump_intensity_step"] = step_jump_intensity
            row["jump_mean_step"] = step_jump_mean
            row["jump_vol_step"] = step_jump_vol
        else:
            step_jump_intensity = jump_intensity
            step_jump_mean = jump_mean
            step_jump_vol = jump_vol

        # === Step 5c: Implied vol blending ===
        implied_vol_ratios: Dict[int, float] = {H: 1.0 for H in horizons}
        if implied_vol_enabled and iv_df is not None:
            for H in horizons:
                iv_val = get_implied_vol_for_horizon(iv_df, dates[idx], H)
                if iv_val is not None:
                    # Convert annualized implied vol to daily
                    sigma_implied_daily = iv_val / np.sqrt(252.0)
                    sigma_hist = sigma_per_h[H]
                    implied_vol_ratios[H] = sigma_implied_daily / sigma_hist if sigma_hist > 0 else 1.0
                    sigma_per_h[H] = _blend_sigma_variance(sigma_hist, sigma_implied_daily, implied_vol_blend)
                    row[f"iv_implied_{H}"] = iv_val
                    row[f"iv_ratio_{H}"] = implied_vol_ratios[H]

        scheduled_event_var = 0.0
        if scheduled_jump_variance_enabled and earnings_dates is not None:
            scheduled_event_var = _estimate_scheduled_event_variance(
                dates,
                all_returns,
                earnings_dates,
                idx,
                sigma_1d,
                lookback_events=scheduled_jump_lookback_events,
                min_events=scheduled_jump_min_events,
                scale=scheduled_jump_scale,
            )
            row["scheduled_jump_var_daily"] = scheduled_event_var

        for H in horizons:
            event_count = 0
            if scheduled_jump_variance_enabled and earnings_dates is not None:
                event_count = _count_scheduled_events_in_horizon(dates, idx, H, earnings_dates)
            if event_count > 0 and scheduled_event_var > 0.0:
                iv_shrink = 1.0
                if implied_vol_enabled and implied_vol_ratios.get(H, 1.0) != 1.0:
                    iv_shrink = max(0.0, 1.0 - implied_vol_blend)
                total_event_var = scheduled_event_var * event_count * iv_shrink
                sigma_per_h[H] = float(np.sqrt(sigma_per_h[H] ** 2 + total_event_var / max(H, 1)))
                row[f"scheduled_jump_events_{H}"] = event_count
                row[f"scheduled_jump_var_{H}"] = total_event_var
            else:
                row[f"scheduled_jump_events_{H}"] = event_count
                row[f"scheduled_jump_var_{H}"] = 0.0

        # Persist the effective horizon sigma actually fed into MC so
        # downstream OOF gating can bucket on the true forecast state.
        for H in horizons:
            row[f"sigma_forecast_{H}"] = sigma_per_h[H]

        # WS3: Regime-conditional t_df
        if mc_regime_t_df and len(sigma_history) >= 50:
            vol_arr = np.sort(np.array(list(sigma_history)))
            vol_pctile = np.searchsorted(vol_arr, sigma_1d) / len(vol_arr)
            if vol_pctile < 0.25:
                step_t_df = mc_regime_t_df_low
            elif vol_pctile > 0.75:
                step_t_df = mc_regime_t_df_high
            else:
                step_t_df = mc_regime_t_df_mid
        else:
            step_t_df = t_df

        # FHS: pass standardized residuals to MC simulation
        fhs_residuals = garch_result.standardized_residuals if fhs_enabled else None

        mc_results = {}
        with ThreadPoolExecutor(max_workers=len(horizons)) as executor:
            futures = {}
            for i, H in enumerate(horizons):
                # generate_state incorporates spawn_key, so each horizon
                # gets a truly distinct seed (unlike .entropy which is shared)
                seed_val = int(child_seeds[i].generate_state(1)[0]) + idx
                futures[H] = executor.submit(
                    _simulate_horizon,
                    S0, sigma_per_h[H], H, path_counts[H], mu_year, thresholds[H],
                    int(seed_val % (2**31)), step_t_df,
                    garch_in_sim=garch_in_sim,
                    omega=garch_result.omega,
                    alpha=garch_result.alpha,
                    beta=garch_result.beta,
                    gamma=garch_result.gamma if garch_result.gamma is not None else 0.0,
                    jump_intensity=step_jump_intensity,
                    jump_mean=step_jump_mean,
                    jump_vol=step_jump_vol,
                    return_quantiles=store_quantiles,
                    standardized_residuals=fhs_residuals,
                )
            for H in horizons:
                mc_results[H] = futures[H].result()

        # === Step 6: Store predictions and apply calibration ===
        result_row_idx = len(results_list)
        p_cals: Dict[int, float] = {}

        for H in horizons:
            mc_result = mc_results[H]
            if store_quantiles:
                p_raw, se, quantiles = mc_result
                for i, q in enumerate(QUANTILE_LEVELS):
                    row[f"q{int(q*100):02d}_{H}"] = quantiles[i]
            else:
                p_raw, se = mc_result
            if calibrators_mf is not None:
                p_cal = calibrators_mf[H].calibrate(
                    p_raw, sigma_1d, delta_sigma, vol_ratio, vol_of_vol,
                    earnings_prox, implied_vol_ratios.get(H, 1.0),
                    range_vol_ratio, overnight_gap, intraday_range)
                state = calibrators_mf[H].state()
            else:
                assert calibrators_online is not None
                p_cal = calibrators_online[H].calibrate(p_raw)
                state = calibrators_online[H].state()
            p_cals[H] = p_cal
            ci_low = max(0.0, p_raw - 1.96 * se)
            ci_high = min(1.0, p_raw + 1.96 * se)

            row[f"p_raw_{H}"] = p_raw
            row[f"p_cal_{H}"] = p_cal
            row[f"mc_se_{H}"] = se
            row[f"ci_low_{H}"] = ci_low
            row[f"ci_high_{H}"] = ci_high
            row[f"paths_used_{H}"] = path_counts[H]

            # Initialize labels as NaN (filled on resolution)
            row[f"y_{H}"] = np.nan
            row[f"realized_return_{H}"] = np.nan

            # Store calibrator state
            row[f"calib_a_{H}"] = state["a"]
            row[f"calib_b_{H}"] = state["b"]

            # Queue for resolution
            queues[H].append(PendingPrediction(
                row_idx=result_row_idx,
                price_idx=idx,
                p_raw=p_raw,
                threshold=thresholds[H],
                sigma_1d=sigma_1d,
                delta_sigma=delta_sigma,
                vol_ratio=vol_ratio,
                vol_of_vol=vol_of_vol,
                earnings_proximity=earnings_prox,
                implied_vol_ratio=implied_vol_ratios.get(H, 1.0),
                range_vol_ratio=range_vol_ratio,
                overnight_gap=overnight_gap,
                intraday_range=intraday_range,
            ))

        # === Step 7: Ensemble (optional) ===
        if meta_calibrator is not None:
            risk_combo = float(sum(
                w * p_cals[H]
                for w, H in zip(ensemble_weights, horizons)
            ))
            p_meta20 = meta_calibrator.calibrate(risk_combo)
            row["risk_combo"] = risk_combo
            row["p_meta20"] = p_meta20
            row["meta_a"] = meta_calibrator.a
            row["meta_b"] = meta_calibrator.b

            meta_queue.append(MetaPrediction(
                row_idx=result_row_idx,
                price_idx=idx,
                risk_combo=risk_combo,
            ))

        results_list.append(row)

        # Progress logging
        if (idx - warmup) % 100 == 0:
            pct = (idx - warmup) / (n - warmup) * 100
            logger.info("Walk-forward progress: %d/%d (%.1f%%)", idx - warmup, n - warmup, pct)

    # Final resolution pass: resolve anything remaining
    for idx_final in range(n, n + max_H + 1):
        if calibrators_mf is not None:
            _resolve_predictions_mf(
                idx_final, prices, horizons, queues, calibrators_mf,
                resolved_labels, results_list,
            )
        else:
            assert calibrators_online is not None
            _resolve_predictions_online(
                idx_final, prices, horizons, queues, calibrators_online,
                resolved_labels, results_list,
            )

        if meta_calibrator is not None:
            _resolve_meta(
                idx_final, prices, k, horizons[-1],
                meta_queue, meta_calibrator, results_list,
            )

    logger.info("Walk-forward complete: %d prediction rows", len(results_list))
    result_df = pd.DataFrame(results_list)

    # Attach final calibrator states for checkpoint export
    cal_states = {}
    if calibrators_mf is not None:
        for h, cal in calibrators_mf.items():
            cal_states[h] = cal.export_state()
    elif calibrators_online is not None:
        for h, cal in calibrators_online.items():
            cal_states[h] = cal.export_state()
    result_df.attrs["calibrator_states"] = cal_states
    if last_garch is not None:
        result_df.attrs["garch_state"] = last_garch.export_state()

    return result_df


def _simulate_horizon(
    S0: float,
    sigma_1d: float,
    H: int,
    n_paths: int,
    mu_year: float,
    threshold: float,
    seed: int,
    t_df: float = 0.0,
    garch_in_sim: bool = False,
    omega: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: float = 0.0,
    jump_intensity: float = 0.0,
    jump_mean: float = 0.0,
    jump_vol: float = 0.0,
    return_quantiles: bool = False,
    standardized_residuals: Optional[np.ndarray] = None,
):
    """Simulate and compute move probability for one horizon.

    Dispatches to GARCH-in-sim when enabled and parameters are available,
    otherwise falls back to constant-vol GBM.
    """
    if garch_in_sim and omega is not None and alpha is not None and beta is not None:
        terminal = simulate_garch_terminal(
            S0, sigma_1d, H, n_paths,
            omega=float(omega), alpha=float(alpha), beta=float(beta), gamma=gamma,
            mu_year=mu_year, seed=seed, t_df=t_df,
            jump_intensity=jump_intensity, jump_mean=jump_mean, jump_vol=jump_vol,
            standardized_residuals=standardized_residuals,
        )
    else:
        terminal = simulate_gbm_terminal(S0, sigma_1d, H, n_paths, mu_year, seed, t_df=t_df,
                                          standardized_residuals=standardized_residuals)
    return compute_move_probability(terminal, S0, threshold, return_quantiles=return_quantiles)


def _resolve_predictions_online(
    current_idx: int,
    prices: npt.NDArray[np.float64],
    horizons: List[int],
    queues: Dict[int, Deque[PendingPrediction]],
    calibrators: Dict[int, OnlineCalibrator],
    resolved_labels: Dict[int, Deque[float]],
    results_list: List[Dict[str, Any]],
) -> None:
    """Resolve pending predictions for standard online calibrators."""
    n = len(prices)
    for H in horizons:
        while queues[H] and (current_idx >= queues[H][0].price_idx + H):
            pred = queues[H].popleft()
            resolve_idx = pred.price_idx + H

            if resolve_idx < n:
                realized_ret = float(prices[resolve_idx] / prices[pred.price_idx] - 1.0)
                y = 1.0 if abs(realized_ret) >= pred.threshold else 0.0

                calibrators[H].update(pred.p_raw, y)
                resolved_labels[H].append(y)

                if pred.row_idx < len(results_list):
                    results_list[pred.row_idx][f"y_{H}"] = y
                    results_list[pred.row_idx][f"realized_return_{H}"] = realized_ret


def _resolve_predictions_mf(
    current_idx: int,
    prices: npt.NDArray[np.float64],
    horizons: List[int],
    queues: Dict[int, Deque[PendingPrediction]],
    calibrators: Dict[int, 'MultiFeatureCalibrator'],
    resolved_labels: Dict[int, Deque[float]],
    results_list: List[Dict[str, Any]],
) -> None:
    """Resolve pending predictions for multi-feature calibrators."""
    n = len(prices)
    for H in horizons:
        while queues[H] and (current_idx >= queues[H][0].price_idx + H):
            pred = queues[H].popleft()
            resolve_idx = pred.price_idx + H

            if resolve_idx < n:
                realized_ret = float(prices[resolve_idx] / prices[pred.price_idx] - 1.0)
                y = 1.0 if abs(realized_ret) >= pred.threshold else 0.0

                calibrators[H].update(
                    pred.p_raw, y, pred.sigma_1d,
                    pred.delta_sigma, pred.vol_ratio, pred.vol_of_vol,
                    pred.earnings_proximity, pred.implied_vol_ratio,
                    pred.range_vol_ratio, pred.overnight_gap, pred.intraday_range,
                )
                resolved_labels[H].append(y)

                if pred.row_idx < len(results_list):
                    results_list[pred.row_idx][f"y_{H}"] = y
                    results_list[pred.row_idx][f"realized_return_{H}"] = realized_ret


def compute_backtest_analytics(
    results: pd.DataFrame,
    horizons: List[int],
    signal_threshold: float = 0.5,
) -> Dict[int, Dict[str, Any]]:
    """
    Post-hoc analytics on backtest results.

    Computes per-horizon:
        - hit_rate: fraction of high-signal predictions where the event occurred
        - signal_turnover: mean |delta(p_cal)| per step (lower = more stable)
        - precision/recall at the given signal_threshold

    Parameters
    ----------
    results : pd.DataFrame
        Walk-forward results from run_walkforward().
    horizons : list of int
        Forecast horizons.
    signal_threshold : float
        Probability threshold for defining "high signal" predictions.

    Returns
    -------
    analytics : dict
        Per-horizon analytics.
    """
    analytics: Dict[int, Dict[str, Any]] = {}
    for H in horizons:
        y_col = f"y_{H}"
        p_col = f"p_cal_{H}"
        if y_col not in results.columns or p_col not in results.columns:
            continue

        y = results[y_col].to_numpy(dtype=float)
        p = results[p_col].to_numpy(dtype=float)
        mask = ~np.isnan(y) & ~np.isnan(p)

        # Hit rate: P(event | signal > threshold)
        high_signal = mask & (p >= signal_threshold)
        n_high = int(high_signal.sum())
        if n_high > 0:
            hit_rate = float(np.mean(y[high_signal]))
        else:
            hit_rate = np.nan

        # Signal turnover: average absolute change in p_cal
        p_clean = p[~np.isnan(p)]
        if len(p_clean) > 1:
            turnover = float(np.mean(np.abs(np.diff(p_clean))))
        else:
            turnover = 0.0

        # Precision / recall at threshold
        if mask.sum() > 0:
            tp = int(((p >= signal_threshold) & (y == 1.0) & mask).sum())
            fp = int(((p >= signal_threshold) & (y == 0.0) & mask).sum())
            fn = int(((p < signal_threshold) & (y == 1.0) & mask).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        else:
            precision = np.nan
            recall = np.nan

        analytics[H] = {
            "hit_rate": hit_rate,
            "n_high_signal": n_high,
            "signal_turnover": turnover,
            "precision": precision,
            "recall": recall,
            "signal_threshold": signal_threshold,
        }

    return analytics


def _resolve_meta(
    current_idx: int,
    prices: npt.NDArray[np.float64],
    k: float,
    H_meta: int,
    meta_queue: Deque[MetaPrediction],
    meta_calibrator: OnlineCalibrator,
    results_list: List[Dict[str, Any]],
) -> None:
    """Resolve ensemble meta-predictions using y_20."""
    n = len(prices)
    while meta_queue and (current_idx >= meta_queue[0].price_idx + H_meta):
        pred = meta_queue.popleft()
        resolve_idx = pred.price_idx + H_meta

        if resolve_idx < n:
            # Use the already-resolved y_20 from the results row
            if pred.row_idx < len(results_list):
                y_20_val = results_list[pred.row_idx].get(f"y_{H_meta}", np.nan)
                if isinstance(y_20_val, (int, float, np.floating)):
                    y_20 = float(y_20_val)
                    if not np.isnan(y_20):
                        meta_calibrator.update(pred.risk_combo, y_20)
