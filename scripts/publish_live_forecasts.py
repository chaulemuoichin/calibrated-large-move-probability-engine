"""
Publish live forecasts to the append-only verification ledger.

Generates calibrated large-move probabilities for all configured tickers
and appends them to outputs/live_verification/ledger/forecasts.jsonl.

This script wraps the existing PredictionEngine and daily_predict logic
but writes to the immutable JSONL ledger instead of the mutable CSV log.

Usage:
    python scripts/publish_live_forecasts.py
    python scripts/publish_live_forecasts.py --tickers SPY GOOGL
    python scripts/publish_live_forecasts.py --model-version v1.1
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from em_sde.config import load_config
from em_sde.data_layer import load_data
from em_sde.predict import PredictionEngine
from em_sde.backtest import run_walkforward
from em_sde.garch import fit_garch
from em_sde.ledger import append_forecast, LEDGER_DIR
from scripts.baselines import compute_live_baseline_forecasts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

TICKER_CONFIGS = {
    "SPY": "configs/exp_suite/exp_spy_regime_gated.yaml",
    "GOOGL": "configs/exp_suite/exp_googl_regime_gated.yaml",
    "AMZN": "configs/exp_suite/exp_amzn_regime_gated.yaml",
    "NVDA": "configs/exp_suite/exp_nvda_regime_gated.yaml",
}

STATE_DIR = Path("outputs/state")


def _get_or_build_engine(ticker: str, config_path: str):
    """Load checkpoint or build from backtest."""
    cfg = load_config(config_path)
    df, _ = load_data(cfg)
    prices = df["price"].to_numpy(dtype=float)

    state_dir = STATE_DIR / ticker.lower()

    if state_dir.exists():
        engine = PredictionEngine.from_checkpoint(str(state_dir))
        logger.info("Loaded checkpoint for %s from %s", ticker, state_dir)
    else:
        logger.info("No checkpoint for %s. Building from backtest...", ticker)
        results = run_walkforward(df, cfg)
        returns = np.diff(prices) / prices[:-1]
        garch = fit_garch(
            returns, window=cfg.model.garch_window,
            min_window=cfg.model.garch_min_window,
            model_type=cfg.model.garch_model_type,
        )

        thresholds = {}
        for H in cfg.model.horizons:
            thr_col = f"thr_{H}"
            if thr_col in results.columns:
                thresholds[str(H)] = float(results[thr_col].dropna().iloc[-1])

        cal_states = results.attrs.get("calibrator_states", {})
        engine = PredictionEngine(
            calibrators={str(h): s for h, s in cal_states.items()},
            garch_state=garch.export_state(),
            metadata={
                "ticker": ticker, "thresholds": thresholds,
                "garch_window": cfg.model.garch_window,
                "garch_min_window": cfg.model.garch_min_window,
                "garch_model_type": cfg.model.garch_model_type,
            },
        )
        engine.save_checkpoint(str(state_dir))

    return engine, cfg, df, prices


def publish(tickers=None, model_version="v1.0"):
    """Publish forecasts for all specified tickers."""
    if tickers is None:
        tickers = list(TICKER_CONFIGS.keys())

    LEDGER_DIR.mkdir(parents=True, exist_ok=True)

    all_forecasts = {}

    for ticker in tickers:
        ticker = ticker.upper()
        config_path = TICKER_CONFIGS.get(ticker)
        if config_path is None:
            logger.warning("No config for ticker %s, skipping", ticker)
            continue

        try:
            engine, cfg, df, prices = _get_or_build_engine(ticker, config_path)
            preds = engine.predict(prices, n_paths=cfg.model.mc_base_paths)

            ticker_forecasts = []
            forecast_date = str(df.index[-1].date())
            trading_calendar = df.index

            # Stale-data gate: reject if data is not from a recent trading day
            last_data_date = pd.Timestamp(forecast_date)
            today = pd.Timestamp.now().normalize()
            # Allow up to 3 calendar days stale (covers weekends)
            staleness = (today - last_data_date).days
            if staleness > 3:
                logger.error(
                    "STALE DATA for %s: last data date is %s (%d days old). "
                    "Refusing to publish. Update price data first.",
                    ticker, forecast_date, staleness,
                )
                continue

            for H, pred in sorted(preds.items()):
                record = append_forecast(
                    ticker=ticker,
                    forecast_date_market=forecast_date,
                    horizon=H,
                    threshold=pred.threshold,
                    p_raw=pred.p_raw,
                    p_cal=pred.p_cal,
                    sigma_1d=pred.sigma_1d,
                    config_path=config_path,
                    model_version=model_version,
                    delta_sigma=pred.delta_sigma,
                    vol_ratio=pred.vol_ratio,
                    vol_of_vol=pred.vol_of_vol,
                    earnings_proximity=pred.earnings_proximity,
                    implied_vol_ratio=pred.implied_vol_ratio,
                    event_rate_historical=pred.event_rate_historical,
                    calibrator_n_updates=pred.calibrator_n_updates,
                    checkpoint_dir=str(STATE_DIR / ticker.lower()),
                    trading_calendar=trading_calendar,
                )
                if record is None:
                    logger.info("Skipped duplicate: %s H=%d on %s", ticker, H, forecast_date)
                    continue
                ticker_forecasts.append(record)

            # --- Baseline forecasts ---
            thresholds = {}
            for H in cfg.model.horizons:
                thr_col = f"thr_{H}"
                if hasattr(engine, "metadata") and "thresholds" in engine.metadata:
                    thr_val = engine.metadata["thresholds"].get(str(H))
                    if thr_val is not None:
                        thresholds[H] = float(thr_val)
                # Fallback: use the threshold from the main model prediction
                if H not in thresholds:
                    pred = preds.get(H)
                    if pred is not None:
                        thresholds[H] = pred.threshold

            iv_path = getattr(cfg.model, "implied_vol_csv_path", None)
            if iv_path and not Path(iv_path).exists():
                iv_path = None

            try:
                baseline_preds = compute_live_baseline_forecasts(
                    prices=prices,
                    dates=df.index,
                    horizons=cfg.model.horizons,
                    thresholds=thresholds,
                    iv_csv_path=iv_path or "data/vix_history.csv",
                    garch_window=cfg.model.garch_window,
                )
                for bl_name, bl_horizons in baseline_preds.items():
                    for H, p_bl in bl_horizons.items():
                        thr = thresholds.get(H, 0.05)
                        rec = append_forecast(
                            ticker=ticker,
                            forecast_date_market=forecast_date,
                            horizon=H,
                            threshold=thr,
                            p_raw=p_bl,
                            p_cal=p_bl,  # baselines are not further calibrated
                            sigma_1d=0.0,
                            config_path=config_path,
                            model_version=bl_name,
                            trading_calendar=trading_calendar,
                        )
                        if rec is not None:
                            ticker_forecasts.append(rec)
            except Exception as e:
                logger.warning("Baseline publish failed for %s: %s", ticker, e)

            all_forecasts[ticker] = ticker_forecasts
            logger.info("Published %d forecasts for %s", len(ticker_forecasts), ticker)

        except Exception as e:
            logger.error("Failed for %s: %s", ticker, e, exc_info=True)

    # Print summary
    total = sum(len(v) for v in all_forecasts.values())
    print(f"\n{'='*60}")
    print(f"LIVE FORECASTS PUBLISHED: {total} forecasts")
    print(f"Ledger: {LEDGER_DIR / 'forecasts.jsonl'}")
    print(f"{'='*60}")
    for ticker, forecasts in all_forecasts.items():
        for f in forecasts:
            print(f"  {ticker} H={f['horizon']:2d}  p_cal={f['p_cal']:.4f}  "
                  f"thr={f['threshold']:.4f}  resolve~{f['expected_resolution_date']}")

    return all_forecasts


def main():
    parser = argparse.ArgumentParser(description="Publish live forecasts to verification ledger")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Tickers to forecast (default: all configured)")
    parser.add_argument("--model-version", default="v1.0",
                        help="Model version string (default: v1.0)")
    args = parser.parse_args()

    publish(tickers=args.tickers, model_version=args.model_version)


if __name__ == "__main__":
    main()
