"""
Async label resolution for live predictions.

Checks which past predictions can now be resolved (enough time has passed)
and records actual outcomes. Optionally updates calibrators with new labels.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def resolve_predictions(
    prediction_log: str,
    prices: np.ndarray,
    dates: pd.DatetimeIndex,
    ticker: Optional[str] = None,
    mark_resolved: bool = True,
) -> pd.DataFrame:
    """
    Resolve outstanding predictions using current price data.

    Parameters
    ----------
    prediction_log : str
        Path to CSV/JSON prediction log with columns:
        date, ticker, horizon, p_cal, threshold
    prices : np.ndarray
        Full price history (must extend beyond prediction dates by H days).
    dates : pd.DatetimeIndex
        Date index corresponding to prices.
    ticker : str, optional
        Filter to only resolve predictions for this ticker. Required when
        the log contains multiple tickers to prevent cross-ticker contamination.
    mark_resolved : bool
        If True, rewrite the log marking resolved rows so they aren't
        re-resolved on subsequent calls.

    Returns
    -------
    resolved : pd.DataFrame
        Predictions with actual outcomes attached (y, realized_return).
    """
    log_path = Path(prediction_log)
    if not log_path.exists():
        logger.warning(f"Prediction log not found: {prediction_log}")
        return pd.DataFrame()

    if log_path.suffix == ".json":
        with open(log_path) as f:
            records = json.load(f)
        preds = pd.DataFrame(records)
    else:
        preds = pd.read_csv(log_path, parse_dates=["date"])

    if len(preds) == 0:
        return pd.DataFrame()

    # Ensure 'resolved' column exists for tracking
    if "resolved" not in preds.columns:
        preds["resolved"] = False

    # Filter by ticker if specified
    if ticker is not None and "ticker" in preds.columns:
        ticker_mask = preds["ticker"].str.upper() == ticker.upper()
    else:
        ticker_mask = pd.Series(True, index=preds.index)

    date_to_idx = {d: i for i, d in enumerate(dates)}
    resolved_rows = []
    resolved_indices = []

    for idx, row in preds[ticker_mask].iterrows():
        # Skip already-resolved rows
        if row.get("resolved", False):
            continue

        pred_date = pd.Timestamp(row["date"])
        H = int(row["horizon"])
        thr = float(row["threshold"])

        if pred_date not in date_to_idx:
            continue

        pred_idx = date_to_idx[pred_date]
        resolve_idx = pred_idx + H

        if resolve_idx >= len(prices):
            # Not enough data yet -- still pending
            continue

        realized_return = prices[resolve_idx] / prices[pred_idx] - 1.0
        y = 1.0 if abs(realized_return) >= thr else 0.0

        resolved_rows.append({
            **row.to_dict(),
            "y": y,
            "realized_return": float(realized_return),
            "resolved_date": str(dates[resolve_idx].date()),
            "resolved": True,
        })
        resolved_indices.append(idx)

    # Mark resolved rows in the log file
    if mark_resolved and resolved_indices and log_path.suffix == ".csv":
        preds.loc[resolved_indices, "resolved"] = True
        preds.to_csv(log_path, index=False)
        logger.info("Marked %d rows as resolved in %s", len(resolved_indices), log_path)

    return pd.DataFrame(resolved_rows) if resolved_rows else pd.DataFrame()


def append_prediction(
    prediction_log: str,
    ticker: str,
    date: str,
    horizon: int,
    p_cal: float,
    p_raw: float,
    threshold: float,
    sigma_1d: float,
    delta_sigma: float = 0.0,
    vol_ratio: float = 1.0,
    vol_of_vol: float = 0.0,
    earnings_proximity: float = 0.0,
    implied_vol_ratio: float = 1.0,
) -> None:
    """Append a prediction with full MF feature vector to the log (CSV).

    Features are persisted so that resolved labels can update MF calibrators
    with the same inputs the backtest uses.
    """
    log_path = Path(prediction_log)
    row = pd.DataFrame([{
        "date": date,
        "ticker": ticker,
        "horizon": horizon,
        "p_cal": p_cal,
        "p_raw": p_raw,
        "threshold": threshold,
        "sigma_1d": sigma_1d,
        "delta_sigma": delta_sigma,
        "vol_ratio": vol_ratio,
        "vol_of_vol": vol_of_vol,
        "earnings_proximity": earnings_proximity,
        "implied_vol_ratio": implied_vol_ratio,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }])

    if log_path.exists():
        row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(log_path, index=False)
