"""
Live prediction engine for calibrated large-move probabilities.

Loads a state checkpoint (calibrators + GARCH params) saved after a
walk-forward backtest and generates forward predictions without requiring
future data.

Usage:
    engine = PredictionEngine.from_checkpoint("outputs/state/spy_abc123")
    result = engine.predict(latest_prices_df)
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .calibration import (
    MultiFeatureCalibrator,
    OnlineCalibrator,
    HistogramCalibrator,
    sigmoid,
    logit,
)
from .garch import GarchResult, fit_garch, ewma_volatility
from .monte_carlo import simulate_garch_terminal, simulate_gbm_terminal

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """A single-horizon prediction with metadata."""
    horizon: int
    p_cal: float
    p_raw: float
    sigma_1d: float
    threshold: float
    event_rate_historical: float
    calibrator_n_updates: int
    timestamp: str


class PredictionEngine:
    """
    Stateful prediction engine that generates forward-looking probabilities.

    Loaded from a checkpoint directory containing:
        - calibrators.json: serialized calibrator state per horizon
        - garch_state.json: last GARCH fit parameters
        - metadata.json: config, thresholds, last training date
    """

    def __init__(
        self,
        calibrators: Dict[int, dict],
        garch_state: dict,
        metadata: dict,
    ):
        self._cal_states = calibrators
        self._garch_state = garch_state
        self._metadata = metadata
        self._calibrators: Dict[int, object] = {}

        # Restore calibrators from state
        for h_str, state in self._cal_states.items():
            h = int(h_str)
            cal_type = state.get("type", "OnlineCalibrator")
            if cal_type == "MultiFeatureCalibrator":
                self._calibrators[h] = MultiFeatureCalibrator.from_state(state)
            elif cal_type == "OnlineCalibrator":
                self._calibrators[h] = OnlineCalibrator.from_state(state)
            else:
                logger.warning(f"Unknown calibrator type {cal_type} for H={h}")

        # Restore GARCH
        self._garch = GarchResult.from_state(garch_state) if garch_state else None

    @classmethod
    def from_checkpoint(cls, state_dir: str) -> "PredictionEngine":
        """Load a PredictionEngine from a checkpoint directory."""
        state_path = Path(state_dir)
        if not state_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {state_dir}")

        with open(state_path / "calibrators.json", "r") as f:
            calibrators = json.load(f)
        with open(state_path / "garch_state.json", "r") as f:
            garch_state = json.load(f)
        with open(state_path / "metadata.json", "r") as f:
            metadata = json.load(f)

        return cls(calibrators, garch_state, metadata)

    def save_checkpoint(self, state_dir: str) -> None:
        """Save the current engine state to a checkpoint directory."""
        state_path = Path(state_dir)
        state_path.mkdir(parents=True, exist_ok=True)

        cal_export = {}
        for h, cal in self._calibrators.items():
            cal_export[str(h)] = cal.export_state()
        with open(state_path / "calibrators.json", "w") as f:
            json.dump(cal_export, f, indent=2)

        garch_export = self._garch.export_state() if self._garch else {}
        with open(state_path / "garch_state.json", "w") as f:
            json.dump(garch_export, f, indent=2)

        with open(state_path / "metadata.json", "w") as f:
            json.dump(self._metadata, f, indent=2)

    def predict(
        self,
        prices: np.ndarray,
        horizons: Optional[List[int]] = None,
        n_paths: int = 30000,
        seed: int = 42,
    ) -> Dict[int, PredictionResult]:
        """
        Generate calibrated probabilities for each horizon.

        Parameters
        ----------
        prices : np.ndarray
            Historical price series ending at the current day.
        horizons : list of int, optional
            Horizons to predict. Defaults to all horizons in checkpoint.
        n_paths : int
            MC simulation paths.
        seed : int
            RNG seed.

        Returns
        -------
        dict mapping horizon -> PredictionResult
        """
        if horizons is None:
            horizons = [int(h) for h in self._calibrators.keys()]

        thresholds = self._metadata.get("thresholds", {})
        returns = np.diff(prices) / prices[:-1]

        # Fit fresh GARCH on full history (warm-started from checkpoint)
        garch_window = self._metadata.get("garch_window", 756)
        garch_min = self._metadata.get("garch_min_window", 252)
        model_type = self._metadata.get("garch_model_type", "gjr")

        try:
            garch_result = fit_garch(returns, window=garch_window,
                                     min_window=garch_min, model_type=model_type)
        except Exception:
            if self._garch is not None:
                garch_result = self._garch
            else:
                sigma_ewma = ewma_volatility(returns)
                garch_result = GarchResult(sigma_1d=sigma_ewma, source="ewma_fallback")

        sigma_1d = garch_result.sigma_1d
        S0 = float(prices[-1])

        # Compute vol features for MF calibrator
        if len(returns) >= 20:
            sigma_20d = float(np.std(returns[-20:]))
            sigma_prev = float(np.std(returns[-40:-20])) if len(returns) >= 40 else sigma_20d
            delta_sigma = sigma_1d - sigma_prev
            vol_ratio = sigma_20d / max(sigma_1d, 1e-8)
        else:
            delta_sigma = 0.0
            vol_ratio = 1.0

        if len(returns) >= 60:
            rolling_vols = [float(np.std(returns[-60+j:-60+j+20]))
                           for j in range(0, 40, 5) if -60+j+20 <= 0]
            vol_of_vol = float(np.std(rolling_vols)) if len(rolling_vols) >= 3 else 0.0
        else:
            vol_of_vol = 0.0

        results = {}
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for H in horizons:
            thr = float(thresholds.get(str(H), thresholds.get(H, 0.05)))

            # MC simulation
            if garch_result.omega is not None and garch_result.alpha is not None:
                terminals = simulate_garch_terminal(
                    S0=S0, sigma_1d=sigma_1d, H=H, n_paths=n_paths,
                    omega=garch_result.omega, alpha=garch_result.alpha,
                    beta=garch_result.beta or 0.85,
                    gamma=garch_result.gamma or 0.0,
                    seed=seed + H,
                )
            else:
                terminals = simulate_gbm_terminal(
                    S0=S0, sigma_1d=sigma_1d, H=H, n_paths=n_paths,
                    seed=seed + H,
                )

            sim_returns = terminals / S0 - 1.0
            p_raw = float(np.mean(np.abs(sim_returns) >= thr))
            p_raw = np.clip(p_raw, 0.001, 0.999)

            # Apply calibrator
            cal = self._calibrators.get(H)
            if cal is None:
                p_cal = p_raw
                n_updates = 0
            elif isinstance(cal, MultiFeatureCalibrator):
                p_cal = cal.calibrate(p_raw, sigma_1d, delta_sigma, vol_ratio, vol_of_vol)
                n_updates = cal.n_updates
            elif isinstance(cal, OnlineCalibrator):
                p_cal = cal.calibrate(p_raw)
                n_updates = cal.n_updates
            else:
                p_cal = p_raw
                n_updates = 0

            # Historical event rate (last 252 days)
            lookback = min(len(returns), 252)
            events = sum(1 for j in range(max(0, len(prices) - lookback - H), len(prices) - H)
                        if abs(prices[j + H] / prices[j] - 1.0) >= thr)
            total = max(lookback - H, 1)
            hist_rate = events / total

            results[H] = PredictionResult(
                horizon=H,
                p_cal=float(np.clip(p_cal, 0.0, 1.0)),
                p_raw=float(p_raw),
                sigma_1d=sigma_1d,
                threshold=thr,
                event_rate_historical=hist_rate,
                calibrator_n_updates=n_updates,
                timestamp=now_str,
            )

        return results
