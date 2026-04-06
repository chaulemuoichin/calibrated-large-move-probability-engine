"""
Resolve pending live forecasts using realized price data.

Scans the forecast ledger for pending forecasts where enough trading days
have elapsed, computes realized returns, and appends resolution records
to the separate resolutions ledger.

The forecast ledger is NEVER modified. Resolutions are append-only in
their own file.

Usage:
    python scripts/resolve_live_forecasts.py
    python scripts/resolve_live_forecasts.py --tickers SPY GOOGL
    python scripts/resolve_live_forecasts.py --dry-run
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from em_sde.config import load_config
from em_sde.data_layer import load_data
from em_sde.ledger import (
    load_forecasts,
    get_resolved_forecast_ids,
    append_resolution,
    FORECAST_FILE,
    RESOLUTION_FILE,
)

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


def resolve(tickers=None, dry_run=False):
    """Resolve pending forecasts against realized price data."""
    forecasts = load_forecasts()
    if len(forecasts) == 0:
        print("No forecasts in ledger.")
        return []

    already_resolved = get_resolved_forecast_ids()

    # Filter to pending only
    pending = forecasts[~forecasts["forecast_id"].isin(already_resolved)]
    if len(pending) == 0:
        print("No pending forecasts to resolve.")
        return []

    if tickers is not None:
        tickers_upper = [t.upper() for t in tickers]
        pending = pending[pending["ticker"].isin(tickers_upper)]

    # Group by ticker for efficient data loading
    resolutions = []
    for ticker, group in pending.groupby("ticker"):
        config_path = TICKER_CONFIGS.get(ticker)
        if config_path is None:
            logger.warning("No config for %s, skipping resolution", ticker)
            continue

        try:
            cfg = load_config(config_path)
            df, _ = load_data(cfg)
            prices = df["price"].to_numpy(dtype=float)
            dates = df.index
            date_to_idx = {d.date(): i for i, d in enumerate(dates)}

            for _, row in group.iterrows():
                forecast_date = pd.Timestamp(row["forecast_date_market"]).date()
                H = int(row["horizon"])
                thr = float(row["threshold"])
                forecast_id = row["forecast_id"]

                if forecast_date not in date_to_idx:
                    logger.debug("Forecast date %s not in data for %s", forecast_date, ticker)
                    continue

                forecast_idx = date_to_idx[forecast_date]
                resolve_idx = forecast_idx + H

                if resolve_idx >= len(prices):
                    logger.debug("Not enough data yet for %s H=%d (need idx %d, have %d)",
                                ticker, H, resolve_idx, len(prices))
                    continue

                price_at_forecast = float(prices[forecast_idx])
                price_at_resolution = float(prices[resolve_idx])
                realized_return = price_at_resolution / price_at_forecast - 1.0
                event_occurred = 1 if abs(realized_return) >= thr else 0
                resolution_date = str(dates[resolve_idx].date())

                if dry_run:
                    print(f"  [DRY RUN] {forecast_id[:8]} {ticker} H={H}: "
                          f"return={realized_return:+.4f} event={event_occurred} "
                          f"on {resolution_date}")
                else:
                    rec = append_resolution(
                        forecast_id=forecast_id,
                        resolution_date_market=resolution_date,
                        realized_return=realized_return,
                        event_occurred=event_occurred,
                        price_at_forecast=price_at_forecast,
                        price_at_resolution=price_at_resolution,
                    )
                    resolutions.append(rec)

        except Exception as e:
            logger.error("Failed resolving %s: %s", ticker, e, exc_info=True)

    # Summary
    n = len(resolutions)
    action = "would resolve" if dry_run else "resolved"
    print(f"\n{'='*60}")
    print(f"RESOLUTION: {action} {n} forecasts")
    if not dry_run:
        print(f"Resolutions ledger: {RESOLUTION_FILE}")
    print(f"{'='*60}")

    if not dry_run and n > 0:
        events = sum(1 for r in resolutions if r["event_occurred"] == 1)
        print(f"  Events: {events}/{n} ({100*events/n:.1f}%)")

    return resolutions


def main():
    parser = argparse.ArgumentParser(description="Resolve pending live forecasts")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Tickers to resolve (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be resolved without writing")
    args = parser.parse_args()

    resolve(tickers=args.tickers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
