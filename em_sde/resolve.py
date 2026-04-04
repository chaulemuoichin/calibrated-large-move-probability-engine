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

    date_to_idx = {d: i for i, d in enumerate(dates)}
    resolved_rows = []

    for _, row in preds.iterrows():
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
        })

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
) -> None:
    """Append a prediction to the prediction log (CSV)."""
    log_path = Path(prediction_log)
    row = pd.DataFrame([{
        "date": date,
        "ticker": ticker,
        "horizon": horizon,
        "p_cal": p_cal,
        "p_raw": p_raw,
        "threshold": threshold,
        "sigma_1d": sigma_1d,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }])

    if log_path.exists():
        row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(log_path, index=False)
