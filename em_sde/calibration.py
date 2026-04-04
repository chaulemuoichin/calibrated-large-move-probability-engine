"""
Online probability calibration via logistic mapping.

Mapping:
    p_cal = sigmoid(a + b * logit(p_raw))

Update on resolution (online SGD):
    a += lr * (y - p_cal)
    b += lr * (y - p_cal) * logit(p_raw)

Default: a=0, b=1 (identity mapping initially), lr=0.05.

Safety gate: when enabled, tracks rolling Brier of raw vs calibrated.
If calibration is degrading performance, automatically falls back to raw.
"""

import json
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# Clipping bounds to avoid log(0) and overflow
_CLIP_LO = 1e-7
_CLIP_HI = 1.0 - 1e-7


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        return ex / (1.0 + ex)


def logit(p: float) -> float:
    """Logit (inverse sigmoid) with clipping."""
    p = np.clip(p, _CLIP_LO, _CLIP_HI)
    return float(np.log(p / (1.0 - p)))


class HistogramCalibrator:
    """
    Online histogram binning recalibrator with Bayesian shrinkage and
    monotonic enforcement via Pool Adjacent Violators (PAV).

    Tracks running per-bin statistics (mean predicted vs mean observed)
    and applies shrinkage-damped additive bias correction to reduce
    calibration error without noisy overcorrection.

    Designed as a post-pass on top of Platt/logistic calibration:
        raw_correction = mean_pred_in_bin - mean_obs_in_bin
        shrinkage = count / (count + prior_strength)
        p_corrected = p_cal - raw_correction * shrinkage

    When monotonic=True (default), corrections are adjusted via PAV so that
    corrected bin-center values are non-decreasing. This guarantees the
    mapping preserves probability rank ordering (AUC invariant).

    Uses equal-width bins aligned with the ECE evaluation (default 10).
    Optional exponential decay (off by default) can track Platt drift.

    Parameters
    ----------
    n_bins : int
        Number of equal-width bins over [0, 1].
    min_samples_per_bin : int
        Minimum samples in a bin before correction is applied.
    decay : float
        Per-update multiplicative decay for bin statistics.
        Default 1.0 (no decay). Set < 1.0 to down-weight old observations.
    prior_strength : float
        Bayesian shrinkage strength. Correction is scaled by
        count / (count + prior_strength). Higher values = more conservative.
        At count=prior_strength, correction is halved.
    monotonic : bool
        Enforce monotone non-decreasing corrected values via PAV.
        Preserves discrimination (AUC) while reducing calibration error.
    """

    def __init__(self, n_bins: int = 10, min_samples_per_bin: int = 15,
                 decay: float = 1.0, prior_strength: float = 15.0,
                 monotonic: bool = True, interpolate: bool = False):
        self.n_bins = n_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.decay = decay
        self.prior_strength = prior_strength
        self.monotonic = monotonic
        self.interpolate = interpolate
        self.bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        self._bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        self._sum_pred = np.zeros(n_bins)
        self._sum_obs = np.zeros(n_bins)
        self._count = np.zeros(n_bins, dtype=np.float64)
        self._corrections = np.zeros(n_bins)

    def _get_bin(self, p: float) -> int:
        """Get bin index for a prediction."""
        p = np.clip(p, 0.0, 1.0)
        idx = int(np.searchsorted(self.bin_edges[1:], p, side='right'))
        return min(idx, self.n_bins - 1)

    @staticmethod
    def _pav(values: np.ndarray) -> np.ndarray:
        """Pool Adjacent Violators: enforce monotone non-decreasing."""
        result = values.copy()
        n = len(result)
        for _ in range(n):
            changed = False
            for i in range(n - 1):
                if result[i] > result[i + 1]:
                    avg = (result[i] + result[i + 1]) / 2.0
                    result[i] = avg
                    result[i + 1] = avg
                    changed = True
            if not changed:
                break
        return result

    def _recompute_corrections(self):
        """Recompute cached corrections with optional monotonic enforcement."""
        corrections = np.zeros(self.n_bins)
        for i in range(self.n_bins):
            if self._count[i] >= self.min_samples_per_bin:
                mean_pred = self._sum_pred[i] / self._count[i]
                mean_obs = self._sum_obs[i] / self._count[i]
                raw = mean_pred - mean_obs
                shrinkage = self._count[i] / (self._count[i] + self.prior_strength)
                corrections[i] = raw * shrinkage
        if self.monotonic:
            centers = (np.arange(self.n_bins) + 0.5) / self.n_bins
            corrected = centers - corrections
            corrected = self._pav(corrected)
            corrections = centers - corrected
        self._corrections = corrections

    def calibrate(self, p_cal: float) -> float:
        """Apply shrinkage-damped histogram bias correction.

        When interpolate=True, linearly interpolates between adjacent bin
        corrections to produce a smooth calibration map (avoids staircase
        artifacts that inflate ECE at bin boundaries).
        """
        idx = self._get_bin(p_cal)
        if self._count[idx] < self.min_samples_per_bin:
            return p_cal

        if not self.interpolate:
            return float(np.clip(p_cal - self._corrections[idx], 0.0, 1.0))

        # Linear interpolation between adjacent bin corrections
        center = self._bin_centers[idx]
        if p_cal <= center and idx > 0 and self._count[idx - 1] >= self.min_samples_per_bin:
            # Interpolate with left neighbor
            left_center = self._bin_centers[idx - 1]
            t = (p_cal - left_center) / (center - left_center)
            corr = (1 - t) * self._corrections[idx - 1] + t * self._corrections[idx]
        elif p_cal > center and idx < self.n_bins - 1 and self._count[idx + 1] >= self.min_samples_per_bin:
            # Interpolate with right neighbor
            right_center = self._bin_centers[idx + 1]
            t = (p_cal - center) / (right_center - center)
            corr = (1 - t) * self._corrections[idx] + t * self._corrections[idx + 1]
        else:
            corr = self._corrections[idx]

        return float(np.clip(p_cal - corr, 0.0, 1.0))

    def update(self, p_cal: float, y: float):
        """Update bin statistics with a resolved outcome."""
        idx = self._get_bin(p_cal)
        if self.decay < 1.0:
            self._sum_pred *= self.decay
            self._sum_obs *= self.decay
            self._count *= self.decay
        self._sum_pred[idx] += p_cal
        self._sum_obs[idx] += y
        self._count[idx] += 1.0
        self._recompute_corrections()

    def export_state(self) -> dict:
        """Export full state as a JSON-serializable dict."""
        return {
            "type": "HistogramCalibrator",
            "n_bins": self.n_bins,
            "min_samples_per_bin": self.min_samples_per_bin,
            "decay": self.decay,
            "prior_strength": self.prior_strength,
            "monotonic": self.monotonic,
            "interpolate": self.interpolate,
            "sum_pred": self._sum_pred.tolist(),
            "sum_obs": self._sum_obs.tolist(),
            "count": self._count.tolist(),
            "corrections": self._corrections.tolist(),
        }

    @classmethod
    def from_state(cls, state: dict) -> "HistogramCalibrator":
        """Restore a HistogramCalibrator from exported state."""
        cal = cls(
            n_bins=state["n_bins"],
            min_samples_per_bin=state["min_samples_per_bin"],
            decay=state["decay"],
            prior_strength=state["prior_strength"],
            monotonic=state["monotonic"],
            interpolate=state.get("interpolate", False),
        )
        cal._sum_pred = np.array(state["sum_pred"], dtype=np.float64)
        cal._sum_obs = np.array(state["sum_obs"], dtype=np.float64)
        cal._count = np.array(state["count"], dtype=np.float64)
        cal._corrections = np.array(state["corrections"], dtype=np.float64)
        return cal


def _make_post_calibrator(method: str, n_bins: int, min_samples: int,
                          prior_strength: float, monotonic: bool,
                          interpolate: bool = False):
    """Factory for post-calibration method selection."""
    if method == "histogram":
        return HistogramCalibrator(n_bins=n_bins, min_samples_per_bin=min_samples,
                                   prior_strength=prior_strength, monotonic=monotonic,
                                   interpolate=interpolate)
    elif method == "none":
        return None
    else:
        raise ValueError(f"Unknown post_cal_method: {method!r}")


def _sigmoid_array(x: np.ndarray) -> np.ndarray:
    """Vectorized sigmoid with clipping for numerical stability."""
    x_clip = np.clip(np.asarray(x, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _identity_weights(
    multi_feature: bool,
    beta_calibration: bool,
    earnings_aware: bool,
    implied_vol_aware: bool,
    ohlc_aware: bool,
) -> np.ndarray:
    """Return identity mapping weights for the requested feature layout."""
    if multi_feature:
        n_base = 7 if beta_calibration else 6
        n_optional = (
            (1 if earnings_aware else 0)
            + (1 if implied_vol_aware else 0)
            + (3 if ohlc_aware else 0)
        )
        n_features = n_base + n_optional
    else:
        n_features = 3 if beta_calibration else 2

    w = np.zeros(n_features, dtype=np.float64)
    if beta_calibration:
        w[1] = 1.0
        w[2] = -1.0
    else:
        w[1] = 1.0
    return w


def _build_feature_vector(
    p_raw: float,
    sigma_1d: float = 0.0,
    delta_sigma: float = 0.0,
    vol_ratio: float = 1.0,
    vol_of_vol: float = 0.0,
    earnings_proximity: float = 0.0,
    implied_vol_ratio: float = 1.0,
    range_vol_ratio: float = 1.0,
    overnight_gap: float = 0.0,
    intraday_range: float = 0.0,
    *,
    multi_feature: bool,
    beta_calibration: bool,
    earnings_aware: bool,
    implied_vol_aware: bool,
    ohlc_aware: bool,
) -> np.ndarray:
    """Build a feature vector for online or offline calibration."""
    p_clipped = np.clip(p_raw, _CLIP_LO, _CLIP_HI)

    if multi_feature:
        if beta_calibration:
            features = [
                1.0,
                float(np.log(p_clipped)),
                float(np.log1p(-p_clipped)),
                sigma_1d * 100.0,
                delta_sigma * 100.0,
                vol_ratio,
                vol_of_vol * 100.0,
            ]
        else:
            features = [
                1.0,
                logit(p_raw),
                sigma_1d * 100.0,
                delta_sigma * 100.0,
                vol_ratio,
                vol_of_vol * 100.0,
            ]
        if earnings_aware:
            features.append(earnings_proximity)
        if implied_vol_aware:
            features.append(implied_vol_ratio)
        if ohlc_aware:
            features.extend([
                range_vol_ratio,
                overnight_gap * 100.0,
                intraday_range * 100.0,
            ])
    else:
        if beta_calibration:
            features = [
                1.0,
                float(np.log(p_clipped)),
                float(np.log1p(-p_clipped)),
            ]
        else:
            features = [1.0, logit(p_raw)]

    return np.asarray(features, dtype=np.float64)


def _results_sigma_column(results: pd.DataFrame, horizon: int) -> str | None:
    """Return the best available sigma column for a horizon in backtest results."""
    candidates = (
        f"sigma_forecast_{horizon}",
        "sigma_har_rv_1d",
        "sigma_garch_1d",
    )
    for col in candidates:
        if col in results.columns:
            return col
    return None


def _results_column(
    results: pd.DataFrame,
    column: str,
    default: float,
) -> np.ndarray:
    """Fetch a float column from backtest results with a scalar default."""
    if column in results.columns:
        values = results[column].to_numpy(dtype=np.float64, copy=False)
        if np.isfinite(values).all():
            return values
        filled = values.copy()
        filled[~np.isfinite(filled)] = default
        return filled
    return np.full(len(results), default, dtype=np.float64)


def _build_feature_matrix_from_results(
    results: pd.DataFrame,
    horizon: int,
    *,
    multi_feature: bool,
    beta_calibration: bool,
    earnings_aware: bool,
    implied_vol_aware: bool,
    ohlc_aware: bool,
) -> np.ndarray:
    """Construct a calibration design matrix from walk-forward results."""
    p_raw = results[f"p_raw_{horizon}"].to_numpy(dtype=np.float64, copy=False)
    sigma_col = _results_sigma_column(results, horizon)
    sigma = (
        results[sigma_col].to_numpy(dtype=np.float64, copy=False)
        if sigma_col is not None
        else np.zeros(len(results), dtype=np.float64)
    )
    delta_sigma = _results_column(results, "delta_sigma", 0.0)
    vol_ratio = _results_column(results, "vol_ratio", 1.0)
    vol_of_vol = _results_column(results, "vol_of_vol", 0.0)
    earnings = _results_column(results, "earnings_proximity", 0.0)
    implied = _results_column(results, f"iv_ratio_{horizon}", 1.0)
    range_ratio = _results_column(results, "range_vol_ratio", 1.0)
    overnight = _results_column(results, "overnight_gap", 0.0)
    intraday = _results_column(results, "intraday_range", 0.0)

    rows = [
        _build_feature_vector(
            float(p),
            float(sig),
            float(ds),
            float(vr),
            float(vov),
            float(ep),
            float(ivr),
            float(rr),
            float(og),
            float(ir),
            multi_feature=multi_feature,
            beta_calibration=beta_calibration,
            earnings_aware=earnings_aware,
            implied_vol_aware=implied_vol_aware,
            ohlc_aware=ohlc_aware,
        )
        for p, sig, ds, vr, vov, ep, ivr, rr, og, ir in zip(
            p_raw, sigma, delta_sigma, vol_ratio, vol_of_vol, earnings,
            implied, range_ratio, overnight, intraday,
        )
    ]
    return np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float64)


def _fit_logistic_irls(
    X: np.ndarray,
    y: np.ndarray,
    w_init: np.ndarray,
    l2_reg: float,
    max_iter: int,
    tol: float = 1e-8,
) -> np.ndarray:
    """Penalized logistic regression via IRLS."""
    w = np.asarray(w_init, dtype=np.float64).copy()
    penalty = np.ones(X.shape[1], dtype=np.float64)
    penalty[0] = 0.0

    for _ in range(max_iter):
        eta = X @ w
        p = _sigmoid_array(eta)
        w_diag = np.clip(p * (1.0 - p), 1e-6, None)
        z = eta + (y - p) / w_diag
        xtw = X.T * w_diag
        lhs = xtw @ X + l2_reg * np.diag(penalty)
        rhs = xtw @ z
        try:
            w_new = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            w_new = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        if np.linalg.norm(w_new - w) <= tol * (1.0 + np.linalg.norm(w)):
            w = w_new
            break
        w = w_new

    return w


@dataclass
class OfflinePooledCalibrator:
    """Batch calibrator fit on train rows and applied to held-out fold rows."""

    weights: np.ndarray
    multi_feature: bool
    beta_calibration: bool
    earnings_aware: bool
    implied_vol_aware: bool
    ohlc_aware: bool
    histogram_calibrator: Optional[HistogramCalibrator] = None
    n_train: int = 0
    event_rate: float = np.nan

    def calibrate_frame(self, results: pd.DataFrame, horizon: int) -> np.ndarray:
        """Apply the fitted offline calibrator to a results slice."""
        if len(results) == 0:
            return np.empty(0, dtype=np.float64)
        X = _build_feature_matrix_from_results(
            results,
            horizon,
            multi_feature=self.multi_feature,
            beta_calibration=self.beta_calibration,
            earnings_aware=self.earnings_aware,
            implied_vol_aware=self.implied_vol_aware,
            ohlc_aware=self.ohlc_aware,
        )
        p = _sigmoid_array(X @ self.weights)
        if self.histogram_calibrator is not None:
            p = np.asarray(
                [self.histogram_calibrator.calibrate(float(v)) for v in p],
                dtype=np.float64,
            )
        return np.clip(p, 0.0, 1.0)


def fit_offline_pooled_calibrator(
    train_results: pd.DataFrame,
    horizon: int,
    *,
    multi_feature: bool,
    l2_reg: float,
    max_iter: int,
    beta_calibration: bool,
    earnings_aware: bool,
    implied_vol_aware: bool,
    ohlc_aware: bool,
    post_cal_method: str,
    histogram_n_bins: int,
    histogram_min_samples: int,
    histogram_prior_strength: float,
    histogram_monotonic: bool,
    histogram_interpolate: bool,
) -> OfflinePooledCalibrator:
    """Fit a batch calibrator on walk-forward-safe train rows."""
    identity = _identity_weights(
        multi_feature,
        beta_calibration,
        earnings_aware,
        implied_vol_aware,
        ohlc_aware,
    )
    y_col = f"y_{horizon}"
    p_col = f"p_raw_{horizon}"
    if y_col not in train_results.columns or p_col not in train_results.columns:
        return OfflinePooledCalibrator(
            weights=identity,
            multi_feature=multi_feature,
            beta_calibration=beta_calibration,
            earnings_aware=earnings_aware,
            implied_vol_aware=implied_vol_aware,
            ohlc_aware=ohlc_aware,
        )

    mask = (
        train_results[y_col].notna()
        & train_results[p_col].notna()
    )
    train = train_results.loc[mask].copy()
    if len(train) == 0:
        return OfflinePooledCalibrator(
            weights=identity,
            multi_feature=multi_feature,
            beta_calibration=beta_calibration,
            earnings_aware=earnings_aware,
            implied_vol_aware=implied_vol_aware,
            ohlc_aware=ohlc_aware,
        )

    X = _build_feature_matrix_from_results(
        train,
        horizon,
        multi_feature=multi_feature,
        beta_calibration=beta_calibration,
        earnings_aware=earnings_aware,
        implied_vol_aware=implied_vol_aware,
        ohlc_aware=ohlc_aware,
    )
    y = train[y_col].to_numpy(dtype=np.float64, copy=False)
    if len(np.unique(y)) < 2 or len(train) < max(X.shape[1] * 5, 30):
        return OfflinePooledCalibrator(
            weights=identity,
            multi_feature=multi_feature,
            beta_calibration=beta_calibration,
            earnings_aware=earnings_aware,
            implied_vol_aware=implied_vol_aware,
            ohlc_aware=ohlc_aware,
            n_train=int(len(train)),
            event_rate=float(np.mean(y)) if len(y) else np.nan,
        )

    weights = _fit_logistic_irls(X, y, identity, l2_reg=l2_reg, max_iter=max_iter)

    histogram_calibrator = _make_post_calibrator(
        post_cal_method,
        histogram_n_bins,
        histogram_min_samples,
        histogram_prior_strength,
        histogram_monotonic,
        interpolate=histogram_interpolate,
    )
    if histogram_calibrator is not None:
        train_p = _sigmoid_array(X @ weights)
        for p_i, y_i in zip(train_p, y):
            histogram_calibrator.update(float(p_i), float(y_i))

    return OfflinePooledCalibrator(
        weights=weights,
        multi_feature=multi_feature,
        beta_calibration=beta_calibration,
        earnings_aware=earnings_aware,
        implied_vol_aware=implied_vol_aware,
        ohlc_aware=ohlc_aware,
        histogram_calibrator=histogram_calibrator,
        n_train=int(len(train)),
        event_rate=float(np.mean(y)),
    )


class OnlineCalibrator:
    """
    Online logistic calibration for probability estimates.

    The mapping p_cal = sigmoid(a + b * logit(p_raw)) starts as identity
    (a=0, b=1) and is updated incrementally as labels resolve.

    Features:
        - Warm-up: returns p_raw unchanged until min_updates labels resolve
        - Adaptive lr: scales lr by 1/sqrt(1 + n_updates) to stabilize over time
        - Safety gate: tracks rolling Brier of raw vs cal; falls back to raw
          when calibration is degrading performance
        - Convergence diagnostics: tracks parameter velocity for monitoring
    """

    def __init__(self, lr: float = 0.05, adaptive_lr: bool = True,
                 min_updates: int = 50, safety_gate: bool = False,
                 gate_window: int = 200,
                 gate_on_discrimination: bool = False,
                 gate_auc_threshold: float = 0.50,
                 gate_separation_threshold: float = 0.0,
                 gate_discrimination_window: int = 200,
                 histogram_post_cal: bool = False,
                 histogram_n_bins: int = 10,
                 histogram_min_samples: int = 15,
                 histogram_prior_strength: float = 15.0,
                 histogram_monotonic: bool = True,
                 histogram_interpolate: bool = False,
                 post_cal_method: str = ""):
        self.a: float = 0.0
        self.b: float = 1.0
        self.lr: float = lr
        self.adaptive_lr: bool = adaptive_lr
        self.min_updates: int = min_updates
        self.n_updates: int = 0

        # Safety gate: track rolling Brier to decide raw vs cal
        self.safety_gate: bool = safety_gate
        self.gate_window: int = gate_window
        self._gate_raw_brier: deque = deque(maxlen=gate_window)
        self._gate_cal_brier: deque = deque(maxlen=gate_window)
        self._gate_active: bool = False  # True when gate decides to use raw

        # Discrimination guardrail: gate on rolling AUC and separation
        self.gate_on_discrimination: bool = gate_on_discrimination
        self.gate_auc_threshold: float = gate_auc_threshold
        self.gate_separation_threshold: float = gate_separation_threshold
        self._gate_discrimination_window: int = gate_discrimination_window
        self._gate_pry_triples: deque = deque(maxlen=gate_discrimination_window)
        self._discrimination_gate_active: bool = False

        # Post-calibration: resolve method from post_cal_method or legacy histogram_post_cal
        if post_cal_method:
            effective_method = post_cal_method
        elif histogram_post_cal:
            effective_method = "histogram"
        else:
            effective_method = "none"
        self._histogram_cal = _make_post_calibrator(
            effective_method, histogram_n_bins, histogram_min_samples,
            histogram_prior_strength, histogram_monotonic,
            interpolate=histogram_interpolate,
        )

        # Convergence tracking
        self._param_history_a: deque = deque(maxlen=100)
        self._param_history_b: deque = deque(maxlen=100)

    @property
    def gated(self) -> bool:
        """Whether any safety gate is currently blocking calibration."""
        return self._gate_active or self._discrimination_gate_active

    def _compute_cal(self, p_raw: float) -> float:
        """Apply the logistic mapping (internal, ignoring gate)."""
        if self.n_updates < self.min_updates:
            return p_raw
        lp = logit(p_raw)
        return sigmoid(self.a + self.b * lp)

    def calibrate(self, p_raw: float) -> float:
        """Apply calibration mapping. Falls back to raw if safety gate triggers."""
        p_cal = self._compute_cal(p_raw)
        if self.safety_gate and self._gate_active:
            return p_raw
        if self.gate_on_discrimination and self._discrimination_gate_active:
            return p_raw
        if self._histogram_cal is not None:
            p_cal = self._histogram_cal.calibrate(p_cal)
        return p_cal

    def update(self, p_raw: float, y: float):
        """
        Update calibration parameters on label resolution.

        Parameters
        ----------
        p_raw : float
            Raw probability that was predicted.
        y : float
            Realized binary label (0 or 1).
        """
        p_cal = self._compute_cal(p_raw)
        error = y - p_cal
        lp = logit(p_raw)

        effective_lr = self.lr
        if self.adaptive_lr:
            effective_lr = self.lr / np.sqrt(1.0 + self.n_updates)

        self.a += effective_lr * error
        self.b += effective_lr * error * lp
        self.n_updates += 1

        # Update histogram post-calibrator with Platt-scaled output
        if self._histogram_cal is not None:
            self._histogram_cal.update(p_cal, y)

        # Track parameter history for convergence diagnostics
        self._param_history_a.append(self.a)
        self._param_history_b.append(self.b)

        # Update safety gate rolling Brier tracking
        if self.safety_gate:
            self._gate_raw_brier.append((p_raw - y) ** 2)
            self._gate_cal_brier.append((p_cal - y) ** 2)

            # Only evaluate gate after sufficient samples
            if len(self._gate_raw_brier) >= self.gate_window:
                mean_raw = sum(self._gate_raw_brier) / len(self._gate_raw_brier)
                mean_cal = sum(self._gate_cal_brier) / len(self._gate_cal_brier)
                self._gate_active = bool(mean_cal > mean_raw)

        # Update discrimination gate
        if self.gate_on_discrimination:
            self._update_discrimination_gate(p_raw, p_cal, y)

    def _update_discrimination_gate(self, p_raw: float, p_cal: float, y: float) -> None:
        """Update rolling AUC and separation tracking for discrimination gate."""
        self._gate_pry_triples.append((p_raw, p_cal, y))

        if len(self._gate_pry_triples) < self._gate_discrimination_window:
            return

        from .evaluation import auc_roc, separation as sep_fn
        triples = list(self._gate_pry_triples)
        p_cal_arr = np.array([t[1] for t in triples])
        y_arr = np.array([t[2] for t in triples])

        rolling_auc = auc_roc(p_cal_arr, y_arr)
        rolling_sep = sep_fn(p_cal_arr, y_arr)

        auc_bad = (np.isfinite(rolling_auc) and rolling_auc < self.gate_auc_threshold)
        sep_bad = (np.isfinite(rolling_sep) and rolling_sep < self.gate_separation_threshold)

        self._discrimination_gate_active = bool(auc_bad or sep_bad)

    def convergence_diagnostics(self) -> dict:
        """
        Return convergence diagnostics for monitoring calibrator stability.

        Metrics:
            - param_velocity_a/b: mean |delta| over last 50 updates (lower = more stable)
            - param_range_a/b: max - min over last 100 updates
            - is_converged: True if velocity < 0.001 and > 50 updates
        """
        result: dict[str, float | bool | int] = {"n_updates": self.n_updates}
        for name, history in [("a", self._param_history_a), ("b", self._param_history_b)]:
            if len(history) >= 2:
                arr = np.array(list(history))
                deltas = np.abs(np.diff(arr))
                # Velocity: mean absolute change over recent window
                recent = deltas[-min(50, len(deltas)):]
                velocity = float(np.mean(recent))
                param_range = float(np.max(arr) - np.min(arr))
            else:
                velocity = np.nan
                param_range = np.nan
            result[f"velocity_{name}"] = velocity
            result[f"range_{name}"] = param_range

        # Convergence criterion: low velocity after warmup
        va = result.get("velocity_a", np.nan)
        vb = result.get("velocity_b", np.nan)
        if self.n_updates >= 50 and np.isfinite(va) and np.isfinite(vb):
            result["is_converged"] = va < 0.001 and vb < 0.001
        else:
            result["is_converged"] = False

        return result

    def state(self) -> dict:
        """Return current calibrator state."""
        return {
            "a": self.a, "b": self.b, "n_updates": self.n_updates,
            "gated": self._gate_active,
        }

    def export_state(self) -> dict:
        """Export full state as a JSON-serializable dict."""
        state = {
            "type": "OnlineCalibrator",
            "a": self.a,
            "b": self.b,
            "lr": self.lr,
            "adaptive_lr": self.adaptive_lr,
            "min_updates": self.min_updates,
            "n_updates": self.n_updates,
            "safety_gate": self.safety_gate,
            "gate_active": self._gate_active,
        }
        if self._histogram_cal is not None:
            state["histogram_cal"] = self._histogram_cal.export_state()
        return state

    @classmethod
    def from_state(cls, state: dict) -> "OnlineCalibrator":
        """Restore an OnlineCalibrator from exported state."""
        cal = cls(
            lr=state["lr"],
            adaptive_lr=state["adaptive_lr"],
            min_updates=state["min_updates"],
            safety_gate=state["safety_gate"],
        )
        cal.a = state["a"]
        cal.b = state["b"]
        cal.n_updates = state["n_updates"]
        cal._gate_active = state.get("gate_active", False)
        if "histogram_cal" in state:
            cal._histogram_cal = HistogramCalibrator.from_state(state["histogram_cal"])
        return cal


class RegimeMultiFeatureCalibrator:
    """
    Regime-conditional multi-feature calibration.

    Maintains separate MultiFeatureCalibrator instances for each vol regime
    (low/mid/high). Regime assignment uses rolling sigma_1d history
    (walk-forward safe, identical to RegimeCalibrator).

    When multi_feature_regime_conditional=true in config, this is used
    instead of a single global MultiFeatureCalibrator.
    """

    def __init__(self, n_bins: int = 3, beta_calibration: bool = False, **mf_kwargs):
        self.n_bins = n_bins
        self.calibrators: list = []  # populated after class def
        self._mf_kwargs = mf_kwargs
        self._mf_kwargs['beta_calibration'] = beta_calibration
        self._vol_history: deque = deque(maxlen=252)
        self._warmup: int = 50

    def _init_calibrators(self):
        """Lazy init after MultiFeatureCalibrator class is defined."""
        if not self.calibrators:
            self.calibrators = [
                MultiFeatureCalibrator(**self._mf_kwargs)
                for _ in range(self.n_bins)
            ]

    def _get_regime(self, sigma_1d: float) -> int:
        """Assign regime bin based on rolling vol percentile."""
        if len(self._vol_history) < self._warmup:
            return 0
        arr = np.sort(np.array(self._vol_history))
        pctile = np.searchsorted(arr, sigma_1d) / len(arr)
        return min(int(pctile * self.n_bins), self.n_bins - 1)

    def observe_vol(self, sigma_1d: float):
        """Record vol observation for percentile computation."""
        self._vol_history.append(sigma_1d)

    def calibrate(self, p_raw: float, sigma_1d: float,
                  delta_sigma: float, vol_ratio: float,
                  vol_of_vol: float,
                  earnings_proximity: float = 0.0,
                  implied_vol_ratio: float = 1.0,
                  range_vol_ratio: float = 1.0,
                  overnight_gap: float = 0.0,
                  intraday_range: float = 0.0) -> float:
        """Apply regime-specific multi-feature calibration."""
        self._init_calibrators()
        regime = self._get_regime(sigma_1d)
        return self.calibrators[regime].calibrate(
            p_raw, sigma_1d, delta_sigma, vol_ratio, vol_of_vol,
            earnings_proximity, implied_vol_ratio,
            range_vol_ratio, overnight_gap, intraday_range)

    def update(self, p_raw: float, y: float, sigma_1d: float,
               delta_sigma: float, vol_ratio: float,
               vol_of_vol: float,
               earnings_proximity: float = 0.0,
               implied_vol_ratio: float = 1.0,
               range_vol_ratio: float = 1.0,
               overnight_gap: float = 0.0,
               intraday_range: float = 0.0) -> None:
        """Update the regime-specific multi-feature calibrator."""
        self._init_calibrators()
        regime = self._get_regime(sigma_1d)
        self.calibrators[regime].update(
            p_raw, y, sigma_1d, delta_sigma, vol_ratio, vol_of_vol,
            earnings_proximity, implied_vol_ratio,
            range_vol_ratio, overnight_gap, intraday_range)

    def state(self) -> Dict:
        """Return state of the currently active regime calibrator."""
        self._init_calibrators()
        if self._vol_history:
            regime = self._get_regime(self._vol_history[-1])
        else:
            regime = 0
        active = self.calibrators[regime]
        return {
            "a": float(active.w[0]),
            "b": float(active.w[1]),
            "n_updates": active.n_updates,
            "regime": regime,
            "gated": active._gate_active,
            "weights": active.w.tolist(),
        }


class MultiFeatureCalibrator:
    """
    Online multi-feature logistic calibration.

    Instead of the 2-parameter Platt scaling (a + b*logit(p_raw)),
    this uses a 6-feature logistic model:

        p_cal = sigmoid(w^T @ x)

    where x = [1, logit(p_raw), sigma_1d, delta_sigma, vol_ratio, vol_of_vol]

    Updates via online SGD with L2 regularization:
        w += lr * (y - p_cal) * x - l2 * w

    Features:
        - Gradient clipping (max norm 10) for stability
        - Adaptive lr: 1/sqrt(1 + n_updates)
        - Safety gate: falls back to raw when calibration degrades Brier
        - Identity initialization: w[1]=1 so it starts as sigmoid(logit(p_raw))=p_raw
    """

    def __init__(self, lr: float = 0.01, l2_reg: float = 1e-4,
                 min_updates: int = 100, safety_gate: bool = True,
                 gate_window: int = 200,
                 gate_on_discrimination: bool = False,
                 gate_auc_threshold: float = 0.50,
                 gate_separation_threshold: float = 0.0,
                 gate_discrimination_window: int = 200,
                 histogram_post_cal: bool = False,
                 histogram_n_bins: int = 10,
                 histogram_min_samples: int = 15,
                 histogram_prior_strength: float = 15.0,
                 histogram_monotonic: bool = True,
                 histogram_interpolate: bool = False,
                 post_cal_method: str = "",
                 earnings_aware: bool = False,
                 beta_calibration: bool = False,
                 implied_vol_aware: bool = False,
                 ohlc_aware: bool = False):
        self.earnings_aware = earnings_aware
        self.beta_calibration = beta_calibration
        self.implied_vol_aware = implied_vol_aware
        self.ohlc_aware = ohlc_aware
        # Beta cal splits logit(p) into log(p) + log(1-p), adding 1 feature
        n_base = 7 if beta_calibration else 6
        n_optional = (
            (1 if earnings_aware else 0)
            + (1 if implied_vol_aware else 0)
            + (3 if ohlc_aware else 0)
        )
        self.N_FEATURES = n_base + n_optional
        self.w = np.zeros(self.N_FEATURES)
        if beta_calibration:
            # Identity: sigmoid(0 + 1*log(p) + (-1)*log(1-p)) = sigmoid(logit(p)) = p
            self.w[1] = 1.0   # weight on log(p)
            self.w[2] = -1.0  # weight on log(1-p)
        else:
            self.w[1] = 1.0  # identity on logit(p_raw)
        self.lr = lr
        self.l2_reg = l2_reg
        self.min_updates = min_updates
        self.n_updates = 0

        # Safety gate
        self.safety_gate = safety_gate
        self.gate_window = gate_window
        self._gate_raw_brier: deque = deque(maxlen=gate_window)
        self._gate_cal_brier: deque = deque(maxlen=gate_window)
        self._gate_active: bool = False

        # Discrimination guardrail
        self.gate_on_discrimination: bool = gate_on_discrimination
        self.gate_auc_threshold: float = gate_auc_threshold
        self.gate_separation_threshold: float = gate_separation_threshold
        self._gate_discrimination_window: int = gate_discrimination_window
        self._gate_pry_triples: deque = deque(maxlen=gate_discrimination_window)
        self._discrimination_gate_active: bool = False

        # Post-calibration: resolve method
        if post_cal_method:
            effective_method = post_cal_method
        elif histogram_post_cal:
            effective_method = "histogram"
        else:
            effective_method = "none"
        self._histogram_cal = _make_post_calibrator(
            effective_method, histogram_n_bins, histogram_min_samples,
            histogram_prior_strength, histogram_monotonic,
            interpolate=histogram_interpolate,
        )

        # Convergence tracking
        self._weight_history: deque = deque(maxlen=100)

    def _build_features(self, p_raw: float, sigma_1d: float,
                        delta_sigma: float, vol_ratio: float,
                        vol_of_vol: float,
                        earnings_proximity: float = 0.0,
                        implied_vol_ratio: float = 1.0,
                        range_vol_ratio: float = 1.0,
                        overnight_gap: float = 0.0,
                        intraday_range: float = 0.0) -> np.ndarray:
        """Build feature vector from raw inputs.

        Standard mode (6 features): [1, logit(p), sigma, delta_sigma, vol_ratio, vol_of_vol]
        Beta calibration (7 features): [1, log(p), log(1-p), sigma, delta_sigma, vol_ratio, vol_of_vol]
        + optional: earnings_proximity, implied_vol_ratio, OHLC realized-state features
        """
        p_clipped = np.clip(p_raw, _CLIP_LO, _CLIP_HI)
        if self.beta_calibration:
            features = [
                1.0,                         # intercept
                float(np.log(p_clipped)),    # log(p) — beta calibration
                float(np.log(1 - p_clipped)),  # log(1-p) — beta calibration
                sigma_1d * 100.0,
                delta_sigma * 100.0,
                vol_ratio,
                vol_of_vol * 100.0,
            ]
        else:
            features = [
                1.0,                     # intercept
                logit(p_raw),            # logit of MC probability
                sigma_1d * 100.0,        # daily vol (scaled for gradient stability)
                delta_sigma * 100.0,     # 20d vol change (scaled)
                vol_ratio,               # realized_vol / forecast_vol
                vol_of_vol * 100.0,      # rolling std of sigma (scaled)
            ]
        if self.earnings_aware:
            features.append(earnings_proximity)  # 0=far from earnings, 1=earnings day
        if self.implied_vol_aware:
            features.append(implied_vol_ratio)   # sigma_implied / sigma_hist (centered ~1.0)
        if self.ohlc_aware:
            features.extend([
                range_vol_ratio,
                overnight_gap * 100.0,
                intraday_range * 100.0,
            ])
        return np.array(features)

    def _compute_cal(self, p_raw: float, sigma_1d: float,
                     delta_sigma: float, vol_ratio: float,
                     vol_of_vol: float,
                     earnings_proximity: float = 0.0,
                     implied_vol_ratio: float = 1.0,
                     range_vol_ratio: float = 1.0,
                     overnight_gap: float = 0.0,
                     intraday_range: float = 0.0) -> float:
        """Apply the logistic mapping (internal, ignoring gate)."""
        if self.n_updates < self.min_updates:
            return p_raw
        x = self._build_features(p_raw, sigma_1d, delta_sigma, vol_ratio, vol_of_vol,
                                 earnings_proximity, implied_vol_ratio,
                                 range_vol_ratio, overnight_gap, intraday_range)
        return sigmoid(float(self.w @ x))

    def calibrate(self, p_raw: float, sigma_1d: float,
                  delta_sigma: float, vol_ratio: float,
                  vol_of_vol: float,
                  earnings_proximity: float = 0.0,
                  implied_vol_ratio: float = 1.0,
                  range_vol_ratio: float = 1.0,
                  overnight_gap: float = 0.0,
                  intraday_range: float = 0.0) -> float:
        """Apply calibration. Falls back to raw if safety gate triggers."""
        p_cal = self._compute_cal(p_raw, sigma_1d, delta_sigma, vol_ratio, vol_of_vol,
                                  earnings_proximity, implied_vol_ratio,
                                  range_vol_ratio, overnight_gap, intraday_range)
        if self.safety_gate and self._gate_active:
            return p_raw
        if self.gate_on_discrimination and self._discrimination_gate_active:
            return p_raw
        if self._histogram_cal is not None:
            p_cal = self._histogram_cal.calibrate(p_cal)
        return p_cal

    def update(self, p_raw: float, y: float, sigma_1d: float,
               delta_sigma: float, vol_ratio: float,
               vol_of_vol: float,
               earnings_proximity: float = 0.0,
               implied_vol_ratio: float = 1.0,
               range_vol_ratio: float = 1.0,
               overnight_gap: float = 0.0,
               intraday_range: float = 0.0) -> None:
        """Update weights via SGD on label resolution."""
        x = self._build_features(p_raw, sigma_1d, delta_sigma, vol_ratio, vol_of_vol,
                                 earnings_proximity, implied_vol_ratio,
                                 range_vol_ratio, overnight_gap, intraday_range)
        p_cal = sigmoid(float(self.w @ x))
        error = y - p_cal

        effective_lr = self.lr / np.sqrt(1.0 + self.n_updates)

        # SGD with L2 regularization
        grad = error * x - self.l2_reg * self.w
        # Gradient clipping for numerical stability
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 10.0:
            grad = grad * (10.0 / grad_norm)
        self.w += effective_lr * grad
        self.n_updates += 1

        # Update histogram post-calibrator with logistic output
        if self._histogram_cal is not None:
            self._histogram_cal.update(p_cal, y)

        # Track weight history for convergence
        self._weight_history.append(self.w.copy())

        # Safety gate
        if self.safety_gate:
            self._gate_raw_brier.append((p_raw - y) ** 2)
            self._gate_cal_brier.append((p_cal - y) ** 2)
            if len(self._gate_raw_brier) >= self.gate_window:
                mean_raw = sum(self._gate_raw_brier) / len(self._gate_raw_brier)
                mean_cal = sum(self._gate_cal_brier) / len(self._gate_cal_brier)
                self._gate_active = bool(mean_cal > mean_raw)

        # Discrimination guardrail
        if self.gate_on_discrimination:
            self._update_discrimination_gate(p_raw, p_cal, y)

    def _update_discrimination_gate(self, p_raw: float, p_cal: float, y: float) -> None:
        """Update rolling AUC and separation tracking for discrimination gate."""
        self._gate_pry_triples.append((p_raw, p_cal, y))

        if len(self._gate_pry_triples) < self._gate_discrimination_window:
            return

        from .evaluation import auc_roc, separation as sep_fn
        triples = list(self._gate_pry_triples)
        p_cal_arr = np.array([t[1] for t in triples])
        y_arr = np.array([t[2] for t in triples])

        rolling_auc = auc_roc(p_cal_arr, y_arr)
        rolling_sep = sep_fn(p_cal_arr, y_arr)

        auc_bad = (np.isfinite(rolling_auc) and rolling_auc < self.gate_auc_threshold)
        sep_bad = (np.isfinite(rolling_sep) and rolling_sep < self.gate_separation_threshold)

        self._discrimination_gate_active = bool(auc_bad or sep_bad)

    def convergence_diagnostics(self) -> dict:
        """
        Return convergence diagnostics for the multi-feature calibrator.

        Metrics:
            - weight_velocity: mean |delta_w| norm over last 50 updates
            - max_weight: max absolute weight value
            - is_converged: True if velocity < 0.01 and > 100 updates
        """
        result = {"n_updates": self.n_updates, "max_weight": float(np.max(np.abs(self.w)))}
        if len(self._weight_history) >= 2:
            history = np.array(list(self._weight_history))
            deltas = np.linalg.norm(np.diff(history, axis=0), axis=1)
            recent = deltas[-min(50, len(deltas)):]
            velocity = float(np.mean(recent))
        else:
            velocity = np.nan
        result["weight_velocity"] = velocity
        if self.n_updates >= 100 and np.isfinite(velocity):
            result["is_converged"] = velocity < 0.01
        else:
            result["is_converged"] = False
        return result

    def state(self) -> Dict:
        """Return current calibrator state."""
        return {
            "a": float(self.w[0]),
            "b": float(self.w[1]),
            "n_updates": self.n_updates,
            "gated": self._gate_active,
            "weights": self.w.tolist(),
        }

    def export_state(self) -> dict:
        """Export full state as a JSON-serializable dict."""
        state = {
            "type": "MultiFeatureCalibrator",
            "weights": self.w.tolist(),
            "lr": self.lr,
            "l2_reg": self.l2_reg,
            "min_updates": self.min_updates,
            "n_updates": self.n_updates,
            "safety_gate": self.safety_gate,
            "gate_active": self._gate_active,
            "earnings_aware": self.earnings_aware,
            "beta_calibration": self.beta_calibration,
            "implied_vol_aware": self.implied_vol_aware,
            "ohlc_aware": self.ohlc_aware,
            "N_FEATURES": self.N_FEATURES,
        }
        if self._histogram_cal is not None:
            state["histogram_cal"] = self._histogram_cal.export_state()
        return state

    @classmethod
    def from_state(cls, state: dict) -> "MultiFeatureCalibrator":
        """Restore a MultiFeatureCalibrator from exported state."""
        cal = cls(
            lr=state["lr"],
            l2_reg=state["l2_reg"],
            min_updates=state["min_updates"],
            safety_gate=state["safety_gate"],
            earnings_aware=state["earnings_aware"],
            beta_calibration=state.get("beta_calibration", False),
            implied_vol_aware=state.get("implied_vol_aware", False),
            ohlc_aware=state.get("ohlc_aware", False),
        )
        cal.w = np.array(state["weights"], dtype=np.float64)
        cal.n_updates = state["n_updates"]
        cal._gate_active = state.get("gate_active", False)
        if "histogram_cal" in state:
            cal._histogram_cal = HistogramCalibrator.from_state(state["histogram_cal"])
        return cal


