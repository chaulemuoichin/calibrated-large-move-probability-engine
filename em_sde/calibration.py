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

import numpy as np
from collections import deque
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
    Online histogram binning recalibrator.

    Tracks running per-bin statistics (mean predicted vs mean observed)
    and applies additive bias correction to reduce calibration error.

    Designed as a post-pass on top of Platt/logistic calibration:
        p_corrected = p_cal - (mean_pred_in_bin - mean_obs_in_bin)

    Uses equal-width bins with fine granularity (default 20) so the
    correction can operate at the resolution where predictions cluster.

    Parameters
    ----------
    n_bins : int
        Number of equal-width bins over [0, 1].
    min_samples_per_bin : int
        Minimum samples in a bin before correction is applied.
    """

    def __init__(self, n_bins: int = 20, min_samples_per_bin: int = 30):
        self.n_bins = n_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        self._sum_pred = np.zeros(n_bins)
        self._sum_obs = np.zeros(n_bins)
        self._count = np.zeros(n_bins, dtype=int)

    def _get_bin(self, p: float) -> int:
        """Get bin index for a prediction."""
        p = np.clip(p, 0.0, 1.0)
        idx = int(np.searchsorted(self.bin_edges[1:], p, side='right'))
        return min(idx, self.n_bins - 1)

    def calibrate(self, p_cal: float) -> float:
        """Apply histogram bias correction."""
        idx = self._get_bin(p_cal)
        if self._count[idx] < self.min_samples_per_bin:
            return p_cal
        mean_pred = self._sum_pred[idx] / self._count[idx]
        mean_obs = self._sum_obs[idx] / self._count[idx]
        correction = mean_pred - mean_obs
        return float(np.clip(p_cal - correction, 0.0, 1.0))

    def update(self, p_cal: float, y: float):
        """Update bin statistics with a resolved outcome."""
        idx = self._get_bin(p_cal)
        self._sum_pred[idx] += p_cal
        self._sum_obs[idx] += y
        self._count[idx] += 1


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
                 histogram_post_cal: bool = False):
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

        # Histogram post-calibration: bin-level bias correction after Platt scaling
        self._histogram_cal: Optional[HistogramCalibrator] = (
            HistogramCalibrator() if histogram_post_cal else None
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


class RegimeCalibrator:
    """
    Regime-conditional calibration using volatility percentile bins.

    Maintains separate OnlineCalibrator instances for each vol regime
    (e.g., low/mid/high vol). Regime assignment is based on rolling
    percentile of sigma_1d — fully walk-forward safe.
    """

    def __init__(self, n_bins: int = 3, lr: float = 0.05,
                 adaptive_lr: bool = True, min_updates: int = 50,
                 safety_gate: bool = False, gate_window: int = 200,
                 gate_on_discrimination: bool = False,
                 gate_auc_threshold: float = 0.50,
                 gate_separation_threshold: float = 0.0,
                 gate_discrimination_window: int = 200,
                 histogram_post_cal: bool = False):
        self.n_bins = n_bins
        self.calibrators = [
            OnlineCalibrator(lr=lr, adaptive_lr=adaptive_lr,
                             min_updates=min_updates, safety_gate=safety_gate,
                             gate_window=gate_window,
                             gate_on_discrimination=gate_on_discrimination,
                             gate_auc_threshold=gate_auc_threshold,
                             gate_separation_threshold=gate_separation_threshold,
                             gate_discrimination_window=gate_discrimination_window,
                             histogram_post_cal=histogram_post_cal)
            for _ in range(n_bins)
        ]
        self._vol_history: deque = deque(maxlen=252)
        self._warmup: int = 50  # minimum vol observations before regime assignment

    def _get_regime(self, sigma_1d: float) -> int:
        """Assign regime bin based on rolling vol percentile."""
        if len(self._vol_history) < self._warmup:
            return 0  # default to bin 0 during warmup
        arr = np.sort(np.array(self._vol_history))
        pctile = np.searchsorted(arr, sigma_1d) / len(arr)
        return min(int(pctile * self.n_bins), self.n_bins - 1)

    def observe_vol(self, sigma_1d: float):
        """Record vol observation for percentile computation."""
        self._vol_history.append(sigma_1d)

    def calibrate(self, p_raw: float, sigma_1d: float) -> float:
        """Apply regime-specific calibration."""
        regime = self._get_regime(sigma_1d)
        return self.calibrators[regime].calibrate(p_raw)

    def update(self, p_raw: float, y: float, sigma_1d: float):
        """Update the regime-specific calibrator."""
        regime = self._get_regime(sigma_1d)
        self.calibrators[regime].update(p_raw, y)

    def state(self) -> dict:
        """Return state of the currently active regime calibrator."""
        if self._vol_history:
            regime = self._get_regime(self._vol_history[-1])
        else:
            regime = 0
        active = self.calibrators[regime]
        return {
            "a": active.a, "b": active.b, "n_updates": active.n_updates,
            "regime": regime, "gated": active._gate_active,
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

    N_FEATURES = 6

    def __init__(self, lr: float = 0.01, l2_reg: float = 1e-4,
                 min_updates: int = 100, safety_gate: bool = True,
                 gate_window: int = 200,
                 gate_on_discrimination: bool = False,
                 gate_auc_threshold: float = 0.50,
                 gate_separation_threshold: float = 0.0,
                 gate_discrimination_window: int = 200,
                 histogram_post_cal: bool = False):
        self.w = np.zeros(self.N_FEATURES)
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

        # Histogram post-calibration: bin-level bias correction
        self._histogram_cal: Optional[HistogramCalibrator] = (
            HistogramCalibrator() if histogram_post_cal else None
        )

        # Convergence tracking
        self._weight_history: deque = deque(maxlen=100)

    def _build_features(self, p_raw: float, sigma_1d: float,
                        delta_sigma: float, vol_ratio: float,
                        vol_of_vol: float) -> np.ndarray:
        """Build feature vector from raw inputs."""
        return np.array([
            1.0,                     # intercept
            logit(p_raw),            # logit of MC probability
            sigma_1d * 100.0,        # daily vol (scaled for gradient stability)
            delta_sigma * 100.0,     # 20d vol change (scaled)
            vol_ratio,               # realized_vol / forecast_vol
            vol_of_vol * 100.0,      # rolling std of sigma (scaled)
        ])

    def _compute_cal(self, p_raw: float, sigma_1d: float,
                     delta_sigma: float, vol_ratio: float,
                     vol_of_vol: float) -> float:
        """Apply the logistic mapping (internal, ignoring gate)."""
        if self.n_updates < self.min_updates:
            return p_raw
        x = self._build_features(p_raw, sigma_1d, delta_sigma, vol_ratio, vol_of_vol)
        return sigmoid(float(self.w @ x))

    def calibrate(self, p_raw: float, sigma_1d: float,
                  delta_sigma: float, vol_ratio: float,
                  vol_of_vol: float) -> float:
        """Apply calibration. Falls back to raw if safety gate triggers."""
        p_cal = self._compute_cal(p_raw, sigma_1d, delta_sigma, vol_ratio, vol_of_vol)
        if self.safety_gate and self._gate_active:
            return p_raw
        if self.gate_on_discrimination and self._discrimination_gate_active:
            return p_raw
        if self._histogram_cal is not None:
            p_cal = self._histogram_cal.calibrate(p_cal)
        return p_cal

    def update(self, p_raw: float, y: float, sigma_1d: float,
               delta_sigma: float, vol_ratio: float,
               vol_of_vol: float) -> None:
        """Update weights via SGD on label resolution."""
        x = self._build_features(p_raw, sigma_1d, delta_sigma, vol_ratio, vol_of_vol)
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
