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

from .calibration import OnlineCalibrator, RegimeCalibrator, MultiFeatureCalibrator
from .garch import fit_garch, GarchResult
from .monte_carlo import simulate_gbm_terminal, simulate_garch_terminal, compute_move_probability, QUANTILE_LEVELS

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


@dataclass
class MetaPrediction:
    """Ensemble prediction awaiting y_20 resolution."""
    row_idx: int
    price_idx: int
    risk_combo: float


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
    garch_in_sim = cfg.model.garch_in_sim
    garch_model_type = cfg.model.garch_model_type
    jump_enabled = cfg.model.jump_enabled
    jump_intensity = cfg.model.jump_intensity if jump_enabled else 0.0
    jump_mean = cfg.model.jump_mean if jump_enabled else 0.0
    jump_vol = cfg.model.jump_vol if jump_enabled else 0.0
    lr = cfg.calibration.lr
    adaptive_lr = cfg.calibration.adaptive_lr
    min_updates = cfg.calibration.min_updates
    safety_gate = cfg.calibration.safety_gate
    gate_window = cfg.calibration.gate_window
    ensemble_enabled = cfg.calibration.ensemble_enabled
    ensemble_weights = cfg.calibration.ensemble_weights
    regime_conditional = cfg.calibration.regime_conditional
    regime_n_bins = cfg.calibration.regime_n_bins
    multi_feature = cfg.calibration.multi_feature
    multi_feature_lr = cfg.calibration.multi_feature_lr
    multi_feature_l2 = cfg.calibration.multi_feature_l2
    multi_feature_min_updates = cfg.calibration.multi_feature_min_updates
    threshold_mode = cfg.model.threshold_mode
    fixed_threshold_pct = cfg.model.fixed_threshold_pct
    store_quantiles = cfg.model.store_quantiles

    # Seed sequence for reproducible parallel MC
    ss = SeedSequence(base_seed)

    # Calibrators: one per horizon (multi-feature, regime-conditional, or standard)
    calibrators_online: Optional[Dict[int, OnlineCalibrator]] = None
    calibrators_regime: Optional[Dict[int, RegimeCalibrator]] = None
    calibrators_mf: Optional[Dict[int, MultiFeatureCalibrator]] = None
    if multi_feature:
        calibrators_mf = {
            H: MultiFeatureCalibrator(
                lr=multi_feature_lr,
                l2_reg=multi_feature_l2,
                min_updates=multi_feature_min_updates,
                safety_gate=safety_gate,
                gate_window=gate_window,
            )
            for H in horizons
        }
    elif regime_conditional:
        calibrators_regime = {
            H: RegimeCalibrator(
                n_bins=regime_n_bins,
                lr=lr,
                adaptive_lr=adaptive_lr,
                min_updates=min_updates,
                safety_gate=safety_gate,
                gate_window=gate_window,
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

    logger.info(
        "Starting walk-forward: %d dates, warmup=%d, horizons=%s",
        n, warmup, horizons,
    )

    for idx in range(warmup, n):
        row = {"date": dates[idx]}

        # === Step 1: Resolve pending predictions ===
        if calibrators_mf is not None:
            _resolve_predictions_mf(
                idx, prices, horizons, queues, calibrators_mf,
                resolved_labels, results_list,
            )
        elif calibrators_regime is not None:
            _resolve_predictions_regime(
                idx, prices, horizons, queues, calibrators_regime,
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
        garch_result = fit_garch(
            available_returns, window=garch_window, min_window=garch_min,
            model_type=garch_model_type if garch_in_sim else "garch",
        )
        sigma_1d = garch_result.sigma_1d
        row["sigma_garch_1d"] = sigma_1d
        row["sigma_source"] = garch_result.source

        # Compute auxiliary features for multi-feature calibration (walk-forward safe)
        sigma_history.append(sigma_1d)
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
        row["delta_sigma"] = delta_sigma
        row["vol_ratio"] = vol_ratio
        row["vol_of_vol"] = vol_of_vol

        # Feed vol to regime calibrators
        if calibrators_regime is not None:
            for H in horizons:
                calibrators_regime[H].observe_vol(sigma_1d)

        # === Step 3: Compute thresholds ===
        thresholds = {}
        for H in horizons:
            if threshold_mode == "fixed_pct":
                thr = fixed_threshold_pct
            elif threshold_mode == "anchored_vol":
                # Use expanding-window unconditional vol (changes slowly)
                sigma_uncond = float(np.std(all_returns[:idx])) if idx > garch_min else sigma_1d
                thr = k * sigma_uncond * np.sqrt(H)
            else:  # vol_scaled (legacy)
                thr = k * sigma_1d * np.sqrt(H)
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

        mc_results = {}
        with ThreadPoolExecutor(max_workers=len(horizons)) as executor:
            futures = {}
            for i, H in enumerate(horizons):
                # generate_state incorporates spawn_key, so each horizon
                # gets a truly distinct seed (unlike .entropy which is shared)
                seed_val = int(child_seeds[i].generate_state(1)[0]) + idx
                futures[H] = executor.submit(
                    _simulate_horizon,
                    S0, sigma_1d, H, path_counts[H], mu_year, thresholds[H],
                    int(seed_val % (2**31)), t_df,
                    garch_in_sim=garch_in_sim,
                    omega=garch_result.omega,
                    alpha=garch_result.alpha,
                    beta=garch_result.beta,
                    gamma=garch_result.gamma if garch_result.gamma is not None else 0.0,
                    jump_intensity=jump_intensity,
                    jump_mean=jump_mean,
                    jump_vol=jump_vol,
                    return_quantiles=store_quantiles,
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
                    p_raw, sigma_1d, delta_sigma, vol_ratio, vol_of_vol)
                state = calibrators_mf[H].state()
            elif calibrators_regime is not None:
                p_cal = calibrators_regime[H].calibrate(p_raw, sigma_1d)
                state = calibrators_regime[H].state()
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
        elif calibrators_regime is not None:
            _resolve_predictions_regime(
                idx_final, prices, horizons, queues, calibrators_regime,
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
    return pd.DataFrame(results_list)


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
        )
    else:
        terminal = simulate_gbm_terminal(S0, sigma_1d, H, n_paths, mu_year, seed, t_df=t_df)
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


def _resolve_predictions_regime(
    current_idx: int,
    prices: npt.NDArray[np.float64],
    horizons: List[int],
    queues: Dict[int, Deque[PendingPrediction]],
    calibrators: Dict[int, RegimeCalibrator],
    resolved_labels: Dict[int, Deque[float]],
    results_list: List[Dict[str, Any]],
) -> None:
    """Resolve pending predictions for regime-conditional calibrators."""
    n = len(prices)
    for H in horizons:
        while queues[H] and (current_idx >= queues[H][0].price_idx + H):
            pred = queues[H].popleft()
            resolve_idx = pred.price_idx + H

            if resolve_idx < n:
                realized_ret = float(prices[resolve_idx] / prices[pred.price_idx] - 1.0)
                y = 1.0 if abs(realized_ret) >= pred.threshold else 0.0

                calibrators[H].update(pred.p_raw, y, pred.sigma_1d)
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
