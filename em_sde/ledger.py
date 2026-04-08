"""
Append-only forecast ledger for live verification.

The ledger is the immutable source of truth. Forecasts are appended once
and never edited. Resolutions are stored in a separate file keyed by
forecast_id.
"""

import hashlib
import json
import logging
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LEDGER_DIR = Path("outputs/live_verification/ledger")
FORECAST_FILE = LEDGER_DIR / "forecasts.jsonl"
RESOLUTION_FILE = LEDGER_DIR / "resolutions.jsonl"

# Column order for CSV exports
FORECAST_COLS = [
    "forecast_id", "forecast_timestamp_utc", "forecast_date_market",
    "ticker", "horizon", "threshold", "p_raw", "p_cal", "sigma_1d",
    "delta_sigma", "vol_ratio", "vol_of_vol", "earnings_proximity",
    "implied_vol_ratio", "event_rate_historical", "calibrator_n_updates",
    "model_version", "git_commit", "config_path", "config_hash",
    "checkpoint_hash", "expected_resolution_date", "status",
]

RESOLUTION_COLS = [
    "forecast_id", "resolution_timestamp_utc", "resolution_date_market",
    "realized_return", "event_occurred", "price_at_forecast",
    "price_at_resolution",
]


def _get_git_commit() -> str:
    """Get current git short commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _hash_file(path: str) -> str:
    """SHA-256 hash of a file's contents (first 12 hex chars)."""
    try:
        h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return h[:12]
    except Exception:
        return "unknown"


def _hash_dict(d: dict) -> str:
    """SHA-256 hash of a dict serialized as sorted JSON."""
    raw = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def make_forecast_id(ticker: str, date: str, horizon: int, model_version: str = "") -> str:
    """
    Deterministic forecast ID from ticker + date + horizon + model_version.

    The same ticker/date/horizon/version always produces the same ID, making
    same-session republishes impossible (idempotent). Different model_versions
    (e.g., main model vs baselines) get distinct IDs for the same date.
    """
    raw = f"{ticker}:{date}:H{horizon}:{model_version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def compute_expected_resolution_date(
    forecast_date: str,
    horizon: int,
    trading_calendar: Optional[pd.DatetimeIndex] = None,
) -> str:
    """
    Compute expected resolution date using trading-day arithmetic.

    Requires a trading calendar. For a proof ledger, calendar-day
    approximations are not acceptable — raises ValueError if no
    calendar is provided or if it has insufficient future dates.
    """
    fd = pd.Timestamp(forecast_date)

    if trading_calendar is None:
        raise ValueError(
            f"Trading calendar required for resolution date computation. "
            f"Cannot approximate for proof ledger (forecast_date={forecast_date}, H={horizon})."
        )

    future = trading_calendar[trading_calendar > fd]
    if len(future) >= horizon:
        return str(future[horizon - 1].date())

    raise ValueError(
        f"Trading calendar has only {len(future)} dates after {forecast_date}, "
        f"but horizon requires {horizon}. Extend the calendar or use more recent data."
    )


def _load_forecast_ids(ledger_path: Path) -> set:
    """Load just the forecast_id values from a JSONL ledger (fast scan)."""
    ids = set()
    try:
        with open(ledger_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    fid = rec.get("forecast_id")
                    if fid:
                        ids.add(fid)
    except FileNotFoundError:
        pass
    return ids


def append_forecast(
    ticker: str,
    forecast_date_market: str,
    horizon: int,
    threshold: float,
    p_raw: float,
    p_cal: float,
    sigma_1d: float,
    config_path: str,
    model_version: str = "v1.0",
    delta_sigma: float = 0.0,
    vol_ratio: float = 1.0,
    vol_of_vol: float = 0.0,
    earnings_proximity: float = 0.0,
    implied_vol_ratio: float = 1.0,
    event_rate_historical: float = 0.0,
    calibrator_n_updates: int = 0,
    checkpoint_dir: Optional[str] = None,
    trading_calendar: Optional[pd.DatetimeIndex] = None,
    ledger_path: Optional[Path] = None,
) -> dict:
    """
    Append a single forecast record to the JSONL ledger.

    Returns the forecast record dict.
    """
    if ledger_path is None:
        ledger_path = FORECAST_FILE

    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_commit = _get_git_commit()

    forecast_id = make_forecast_id(ticker, forecast_date_market, horizon, model_version)

    # Idempotency: reject if this ticker+date+horizon already exists
    if ledger_path.exists():
        existing_ids = _load_forecast_ids(ledger_path)
        if forecast_id in existing_ids:
            logger.warning(
                "Duplicate forecast rejected: %s %s H=%d (id=%s) — "
                "already published for this market session",
                ticker, forecast_date_market, horizon, forecast_id[:8],
            )
            return None

    config_hash = _hash_file(config_path) if Path(config_path).exists() else "unknown"

    checkpoint_hash = "none"
    if checkpoint_dir:
        cp = Path(checkpoint_dir)
        if (cp / "metadata.json").exists():
            checkpoint_hash = _hash_file(str(cp / "metadata.json"))

    expected_res = compute_expected_resolution_date(
        forecast_date_market, horizon, trading_calendar,
    )

    record = {
        "forecast_id": forecast_id,
        "forecast_timestamp_utc": now_utc,
        "forecast_date_market": forecast_date_market,
        "ticker": ticker.upper(),
        "horizon": horizon,
        "threshold": round(float(threshold), 6),
        "p_raw": round(float(p_raw), 6),
        "p_cal": round(float(p_cal), 6),
        "sigma_1d": round(float(sigma_1d), 8),
        "delta_sigma": round(float(delta_sigma), 8),
        "vol_ratio": round(float(vol_ratio), 6),
        "vol_of_vol": round(float(vol_of_vol), 8),
        "earnings_proximity": round(float(earnings_proximity), 4),
        "implied_vol_ratio": round(float(implied_vol_ratio), 6),
        "event_rate_historical": round(float(event_rate_historical), 6),
        "calibrator_n_updates": int(calibrator_n_updates),
        "model_version": model_version,
        "git_commit": git_commit,
        "config_path": str(config_path),
        "config_hash": config_hash,
        "checkpoint_hash": checkpoint_hash,
        "expected_resolution_date": expected_res,
        "status": "pending",
    }

    with open(ledger_path, "a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")

    logger.info(
        "Forecast %s: %s H=%d p_cal=%.4f (pending, resolve ~%s)",
        forecast_id[:8], ticker, horizon, p_cal, expected_res,
    )
    return record


def append_resolution(
    forecast_id: str,
    resolution_date_market: str,
    realized_return: float,
    event_occurred: int,
    price_at_forecast: float,
    price_at_resolution: float,
    ledger_path: Optional[Path] = None,
) -> dict:
    """
    Append a resolution record to the resolutions JSONL ledger.

    Does NOT modify the forecast file. The forecast remains immutable.
    """
    if ledger_path is None:
        ledger_path = RESOLUTION_FILE

    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    # Reject duplicate resolutions for the same forecast
    if ledger_path.exists():
        existing_ids = set()
        try:
            with open(ledger_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        fid = rec.get("forecast_id")
                        if fid:
                            existing_ids.add(fid)
        except FileNotFoundError:
            pass
        if forecast_id in existing_ids:
            logger.warning(
                "Duplicate resolution rejected: forecast_id=%s already resolved",
                forecast_id[:8],
            )
            return None

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    record = {
        "forecast_id": forecast_id,
        "resolution_timestamp_utc": now_utc,
        "resolution_date_market": resolution_date_market,
        "realized_return": round(float(realized_return), 8),
        "event_occurred": int(event_occurred),
        "price_at_forecast": round(float(price_at_forecast), 4),
        "price_at_resolution": round(float(price_at_resolution), 4),
    }

    with open(ledger_path, "a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")

    logger.info(
        "Resolved %s: return=%.4f event=%d on %s",
        forecast_id[:8], realized_return, event_occurred, resolution_date_market,
    )
    return record


def load_forecasts(ledger_path: Optional[Path] = None) -> pd.DataFrame:
    """Load all forecast records from the JSONL ledger."""
    if ledger_path is None:
        ledger_path = FORECAST_FILE
    if not ledger_path.exists():
        return pd.DataFrame(columns=FORECAST_COLS)
    records = []
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return pd.DataFrame(columns=FORECAST_COLS)
    return pd.DataFrame(records)


def load_resolutions(ledger_path: Optional[Path] = None) -> pd.DataFrame:
    """Load all resolution records from the JSONL ledger."""
    if ledger_path is None:
        ledger_path = RESOLUTION_FILE
    if not ledger_path.exists():
        return pd.DataFrame(columns=RESOLUTION_COLS)
    records = []
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return pd.DataFrame(columns=RESOLUTION_COLS)
    return pd.DataFrame(records)


def load_joined(
    forecast_path: Optional[Path] = None,
    resolution_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Join forecasts and resolutions into a single DataFrame.

    Resolved forecasts will have event_occurred, realized_return, etc.
    Pending forecasts will have NaN in resolution columns.
    """
    forecasts = load_forecasts(forecast_path)
    resolutions = load_resolutions(resolution_path)

    if len(forecasts) == 0:
        return forecasts

    if len(resolutions) == 0:
        for col in RESOLUTION_COLS:
            if col != "forecast_id":
                forecasts[col] = np.nan
        return forecasts

    # Deduplicate resolutions: keep first per forecast_id (belt-and-suspenders)
    resolutions = resolutions.drop_duplicates(subset=["forecast_id"], keep="first")
    merged = forecasts.merge(resolutions, on="forecast_id", how="left")
    return merged


def get_resolved_forecast_ids(ledger_path: Optional[Path] = None) -> set:
    """Return the set of forecast_ids that already have resolutions."""
    resolutions = load_resolutions(ledger_path)
    if len(resolutions) == 0:
        return set()
    return set(resolutions["forecast_id"].tolist())
