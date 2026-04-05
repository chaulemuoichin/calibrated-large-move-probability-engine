"""
Daily prediction runner with scheduling support.

Generates calibrated large-move probabilities for configured tickers,
logs predictions, and resolves past outcomes.

Usage:
    # Single run
    python scripts/daily_predict.py

    # Schedule via Windows Task Scheduler:
    #   Action: .venv/Scripts/python.exe scripts/daily_predict.py
    #   Trigger: Daily at 4:30 PM ET (after market close)

    # Schedule via cron (Linux):
    #   30 16 * * 1-5 cd /path/to/repo && .venv/bin/python scripts/daily_predict.py
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from em_sde.config import load_config
from em_sde.data_layer import load_data
from em_sde.predict import PredictionEngine
from em_sde.resolve import append_prediction, resolve_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

TICKER_CONFIGS = {
    "spy": "configs/exp_suite/exp_spy_regime_gated.yaml",
    "googl": "configs/exp_suite/exp_googl_regime_gated.yaml",
    "amzn": "configs/exp_suite/exp_amzn_regime_gated.yaml",
    "nvda": "configs/exp_suite/exp_nvda_regime_gated.yaml",
}

PREDICTION_LOG = "outputs/predictions/prediction_log.csv"
STATE_DIR = "outputs/state"


def run_daily():
    """Run daily prediction for all configured tickers."""
    os.makedirs("outputs/predictions", exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    all_predictions = {}

    for ticker, config_path in TICKER_CONFIGS.items():
        logger.info("Processing %s...", ticker.upper())
        try:
            cfg = load_config(config_path)
            df, _ = load_data(cfg)
            prices = df["price"].to_numpy(dtype=float)

            state_dir = f"{STATE_DIR}/{ticker}"

            # Load or build engine
            if Path(state_dir).exists():
                engine = PredictionEngine.from_checkpoint(state_dir)
            else:
                logger.info("No checkpoint for %s. Building from backtest...", ticker)
                from em_sde.backtest import run_walkforward
                from em_sde.garch import fit_garch
                import numpy as np

                results = run_walkforward(df, cfg)
                returns = np.diff(prices) / prices[:-1]
                garch = fit_garch(returns, window=cfg.model.garch_window,
                                  min_window=cfg.model.garch_min_window,
                                  model_type=cfg.model.garch_model_type)

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
                engine.save_checkpoint(state_dir)

            # RESOLVE FIRST: update calibrators before predicting (causal ordering)
            if Path(PREDICTION_LOG).exists():
                try:
                    resolved = resolve_predictions(
                        PREDICTION_LOG,
                        prices,
                        df.index,
                        ticker=ticker.upper(),
                    )
                    if len(resolved) > 0:
                        n_resolved = len(resolved[resolved["y"].notna()])
                        logger.info("Resolved %d predictions for %s", n_resolved, ticker.upper())
                        n_updated = engine.update_calibrators(resolved)
                        if n_updated > 0:
                            engine.save_checkpoint(state_dir)
                            logger.info("Updated calibrators with %d labels for %s",
                                       n_updated, ticker.upper())
                except Exception as re:
                    logger.warning("Resolution/update failed for %s: %s", ticker, re)

            # THEN predict with updated calibrators
            preds = engine.predict(prices, n_paths=cfg.model.mc_base_paths)
            ticker_preds = {}

            for H, pred in sorted(preds.items()):
                ticker_preds[f"H={H}"] = {
                    "p_cal": round(pred.p_cal, 4),
                    "p_raw": round(pred.p_raw, 4),
                    "sigma_1d": round(pred.sigma_1d, 6),
                    "threshold": round(pred.threshold, 4),
                }
                # Log prediction with full MF feature vector
                append_prediction(
                    PREDICTION_LOG, ticker.upper(), today, H,
                    pred.p_cal, pred.p_raw, pred.threshold, pred.sigma_1d,
                    delta_sigma=pred.delta_sigma,
                    vol_ratio=pred.vol_ratio,
                    vol_of_vol=pred.vol_of_vol,
                    earnings_proximity=pred.earnings_proximity,
                    implied_vol_ratio=pred.implied_vol_ratio,
                )

            all_predictions[ticker.upper()] = ticker_preds

        except Exception as e:
            logger.error("Failed for %s: %s", ticker, e, exc_info=True)

    # Print summary
    print(f"\n{'='*60}")
    print(f"DAILY PREDICTIONS - {today}")
    print(f"{'='*60}")
    print(json.dumps(all_predictions, indent=2))


if __name__ == "__main__":
    run_daily()
