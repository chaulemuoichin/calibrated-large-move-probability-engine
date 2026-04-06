"""
Generate a demo fixture ledger for previewing the live verification site.

Creates realistic-looking forecast and resolution records using actual
backtest results, clearly labeled as DEMO DATA.

Usage:
    python scripts/generate_demo_ledger.py
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from em_sde.config import load_config
from em_sde.data_layer import load_data
from em_sde.ledger import (
    make_forecast_id,
    FORECAST_COLS,
    RESOLUTION_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEMO_DIR = Path("outputs/live_verification/demo")

TICKER_CONFIGS = {
    "SPY": "configs/exp_suite/exp_spy_regime_gated.yaml",
    "GOOGL": "configs/exp_suite/exp_googl_regime_gated.yaml",
}


def generate_demo():
    """Generate demo ledger from the last 120 trading days of backtest data."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    forecast_path = DEMO_DIR / "forecasts.jsonl"
    resolution_path = DEMO_DIR / "resolutions.jsonl"

    # Clear existing demo data
    for p in [forecast_path, resolution_path]:
        if p.exists():
            p.unlink()

    n_forecasts = 0
    n_resolved = 0

    for ticker, config_path in TICKER_CONFIGS.items():
        logger.info("Generating demo data for %s...", ticker)
        cfg = load_config(config_path)
        df, _ = load_data(cfg)
        prices = df["price"].to_numpy(dtype=float)
        dates = df.index

        n = len(prices)
        horizons = cfg.model.horizons

        # Get thresholds
        thresholds = {}
        fixed_pct = cfg.model.regime_gated_fixed_pct_by_horizon or {}
        for H in horizons:
            thresholds[H] = fixed_pct.get(H, cfg.model.fixed_threshold_pct)

        # Use last 120 trading days as "live" forecasts
        start_idx = max(n - 120, 252)
        returns = np.diff(prices) / prices[:-1]

        for t in range(start_idx, n):
            date_str = str(dates[t].date())
            # Simulate GARCH sigma from recent returns
            if t >= 20:
                sigma = float(np.std(returns[t-20:t])) * np.sqrt(252)
                sigma_1d = sigma / np.sqrt(252)
            else:
                sigma_1d = 0.01

            delta_sigma = 0.0
            vol_ratio = 1.0
            vol_of_vol = 0.0
            if t >= 40:
                s20 = float(np.std(returns[t-20:t]))
                s_prev = float(np.std(returns[t-40:t-20]))
                delta_sigma = sigma_1d - s_prev
                vol_ratio = s20 / max(sigma_1d, 1e-8)
            if t >= 60:
                rolling = [float(np.std(returns[t-60+j:t-60+j+20]))
                          for j in range(0, 40, 5) if t-60+j+20 <= t]
                vol_of_vol = float(np.std(rolling)) if len(rolling) >= 3 else 0.0

            for H in horizons:
                thr = thresholds[H]
                # Simple MC-like probability estimate
                sigma_H = sigma_1d * np.sqrt(H)
                if sigma_H > 1e-8:
                    from scipy import stats
                    p_raw = float(2.0 * stats.norm.cdf(-thr / sigma_H))
                else:
                    p_raw = 0.01
                p_raw = np.clip(p_raw, 0.001, 0.999)

                # Simulated calibration (slight adjustment)
                p_cal = p_raw * 0.9 + 0.01 * np.random.default_rng(t + H).standard_normal()
                p_cal = float(np.clip(p_cal, 0.001, 0.999))

                ts = f"2025-01-01T16:30:00Z"  # Fixed demo timestamp
                fid = make_forecast_id(ticker, date_str, H, ts)

                # Expected resolution
                res_idx = t + H
                if res_idx < n:
                    exp_res_date = str(dates[res_idx].date())
                else:
                    exp_res_date = str((dates[t] + pd.Timedelta(days=int(H * 1.5))).date())

                # Historical event rate
                lookback = min(t, 252)
                events = sum(1 for j in range(max(0, t - lookback), t - H)
                            if j + H < n and abs(prices[j+H]/prices[j] - 1) >= thr)
                total = max(lookback - H, 1)
                hist_rate = events / total

                forecast = {
                    "forecast_id": fid,
                    "forecast_timestamp_utc": ts,
                    "forecast_date_market": date_str,
                    "ticker": ticker,
                    "horizon": H,
                    "threshold": round(thr, 6),
                    "p_raw": round(p_raw, 6),
                    "p_cal": round(p_cal, 6),
                    "sigma_1d": round(sigma_1d, 8),
                    "delta_sigma": round(delta_sigma, 8),
                    "vol_ratio": round(vol_ratio, 6),
                    "vol_of_vol": round(vol_of_vol, 8),
                    "earnings_proximity": 0.0,
                    "implied_vol_ratio": 1.0,
                    "event_rate_historical": round(hist_rate, 6),
                    "calibrator_n_updates": max(0, t - start_idx),
                    "model_version": "demo-v1.0",
                    "git_commit": "demo",
                    "config_path": config_path,
                    "config_hash": "demo",
                    "checkpoint_hash": "demo",
                    "expected_resolution_date": exp_res_date,
                    "status": "pending",
                }

                with open(forecast_path, "a") as f:
                    f.write(json.dumps(forecast, separators=(",", ":")) + "\n")
                n_forecasts += 1

                # Resolve if we have the data
                if res_idx < n:
                    price_f = float(prices[t])
                    price_r = float(prices[res_idx])
                    realized = price_r / price_f - 1.0
                    event = 1 if abs(realized) >= thr else 0

                    resolution = {
                        "forecast_id": fid,
                        "resolution_timestamp_utc": ts,
                        "resolution_date_market": str(dates[res_idx].date()),
                        "realized_return": round(realized, 8),
                        "event_occurred": event,
                        "price_at_forecast": round(price_f, 4),
                        "price_at_resolution": round(price_r, 4),
                    }

                    with open(resolution_path, "a") as f:
                        f.write(json.dumps(resolution, separators=(",", ":")) + "\n")
                    n_resolved += 1

    print(f"\nDemo ledger generated:")
    print(f"  Forecasts:   {n_forecasts} -> {forecast_path}")
    print(f"  Resolutions: {n_resolved} -> {resolution_path}")
    print(f"\nThis is DEMO DATA generated from historical backtest prices.")
    print(f"It is NOT real out-of-sample live verification data.")


if __name__ == "__main__":
    generate_demo()
