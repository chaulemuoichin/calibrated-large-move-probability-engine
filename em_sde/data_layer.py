"""
Data ingestion layer with robust yfinance fetching, CSV loading,
synthetic generation, parquet caching, and validation.

All downstream modules consume only df["price"].
"""

import os
import time
import warnings
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")


def _date_str(value: object) -> str:
    """Return ISO date string for index/timestamp-like values."""
    try:
        return pd.to_datetime(str(value)).date().isoformat()
    except Exception:
        text = str(value)
        return text[:10] if len(text) >= 10 else text


def load_data(cfg) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load price data according to config. Returns (df, metadata).
    df has DatetimeIndex and column 'price'.
    """
    meta = {
        "ticker": cfg.data.ticker,
        "source_requested": cfg.data.source,
        "start": cfg.data.start,
        "end": cfg.data.end,
        "attempts": 0,
        "cache_hit": False,
        "field_used": None,
        "warnings": [],
    }

    if cfg.data.source == "yfinance":
        df, meta = _load_yfinance(cfg, meta)
    elif cfg.data.source == "csv":
        df, meta = _load_csv(cfg, meta)
    elif cfg.data.source == "synthetic":
        df, meta = _generate_synthetic(cfg, meta)
    else:
        raise ValueError(f"Unknown data source: {cfg.data.source}")

    df, meta = _clean_and_validate(df, cfg.data.min_rows, meta)

    meta["rows"] = len(df)
    meta["actual_start"] = _date_str(df.index[0])
    meta["actual_end"] = _date_str(df.index[-1])

    return df, meta


def _load_yfinance(cfg, meta: dict) -> Tuple[pd.DataFrame, dict]:
    """Fetch data from yfinance with retry, dual methods, and caching."""
    try:
        import yfinance as yf
    except ImportError:
        if cfg.data.fallback_to_synthetic:
            meta["warnings"].append("yfinance not installed, falling back to synthetic")
            return _generate_synthetic(cfg, meta)
        raise ImportError("yfinance is required but not installed")

    # Check cache first
    cached = _cache_load(cfg.data.ticker, cfg.data.cache_max_age_days)
    if cached is not None:
        meta["cache_hit"] = True
        meta["field_used"] = "cached"
        logger.info("Loaded %s from cache (%d rows)", cfg.data.ticker, len(cached))
        return cached, meta

    df = None
    max_retry = cfg.data.max_retry

    # Method 1: yf.download()
    for attempt in range(1, max_retry + 1):
        meta["attempts"] = attempt
        try:
            logger.info("yf.download attempt %d/%d for %s",
                        attempt, max_retry, cfg.data.ticker)
            raw = yf.download(
                cfg.data.ticker,
                start=cfg.data.start,
                end=cfg.data.end,
                auto_adjust=True,
                progress=False,
            )
            if raw is not None and len(raw) > 0:
                # Handle MultiIndex columns from yf.download
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                if "Close" in raw.columns:
                    df = pd.DataFrame({"price": raw["Close"].values}, index=raw.index)
                    meta["field_used"] = "Close (auto_adjust=True)"
                    break
                elif "Adj Close" in raw.columns:
                    df = pd.DataFrame({"price": raw["Adj Close"].values}, index=raw.index)
                    meta["field_used"] = "Adj Close"
                    break
        except Exception as e:
            logger.warning("yf.download attempt %d failed: %s", attempt, e)
            time.sleep(min(2 ** attempt, 30))

    # Method 2: yf.Ticker().history()
    if df is None:
        for attempt in range(1, max_retry + 1):
            meta["attempts"] += 1
            try:
                logger.info("yf.Ticker.history attempt %d/%d for %s",
                            attempt, max_retry, cfg.data.ticker)
                ticker = yf.Ticker(cfg.data.ticker)
                raw = ticker.history(
                    start=cfg.data.start,
                    end=cfg.data.end,
                    auto_adjust=True,
                )
                if raw is not None and len(raw) > 0:
                    if "Close" in raw.columns:
                        df = pd.DataFrame({"price": raw["Close"].values}, index=raw.index)
                        meta["field_used"] = "Close (Ticker.history, auto_adjust=True)"
                        break
            except Exception as e:
                logger.warning("Ticker.history attempt %d failed: %s", attempt, e)
                time.sleep(min(2 ** attempt, 30))

    if df is None:
        if cfg.data.fallback_to_synthetic:
            meta["warnings"].append("All yfinance methods failed, falling back to synthetic")
            logger.warning("yfinance failed for %s, using synthetic fallback", cfg.data.ticker)
            return _generate_synthetic(cfg, meta)
        raise RuntimeError(
            f"Failed to fetch {cfg.data.ticker} after {meta['attempts']} attempts"
        )

    # Cache the result
    _cache_save(cfg.data.ticker, df)
    return df, meta


def _load_csv(cfg, meta: dict) -> Tuple[pd.DataFrame, dict]:
    """Load price data from CSV file."""
    path = cfg.data.csv_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    raw = pd.read_csv(path, parse_dates=True, index_col=0)

    # Look for price column
    for col in ["price", "Price", "Close", "close", "Adj Close", "adj_close"]:
        if col in raw.columns:
            df = pd.DataFrame({"price": raw[col].values}, index=raw.index)
            meta["field_used"] = col
            return df, meta

    # If single column, use it
    if len(raw.columns) == 1:
        df = pd.DataFrame({"price": raw.iloc[:, 0].values}, index=raw.index)
        meta["field_used"] = raw.columns[0]
        return df, meta

    raise ValueError(f"Cannot identify price column in CSV. Columns: {list(raw.columns)}")


def _generate_synthetic(cfg, meta: dict) -> Tuple[pd.DataFrame, dict]:
    """Generate synthetic GBM price data for testing."""
    rng = np.random.default_rng(cfg.data.synthetic_seed)
    n_days = cfg.data.synthetic_days

    # GBM parameters
    mu = 0.08       # 8% annual drift
    sigma = 0.20    # 20% annual vol
    S0 = 100.0
    dt = 1 / 252

    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_days)
    log_prices = np.concatenate([[np.log(S0)], np.cumsum(log_returns) + np.log(S0)])
    prices = np.exp(log_prices)

    dates = pd.bdate_range(start=cfg.data.start, periods=n_days + 1, freq="B")
    df = pd.DataFrame({"price": prices}, index=dates)

    meta["field_used"] = "synthetic_gbm"
    meta["source_requested"] = "synthetic"
    return df, meta


def _clean_and_validate(
    df: pd.DataFrame, min_rows: int, meta: dict
) -> Tuple[pd.DataFrame, dict]:
    """Clean and validate price data."""
    # Drop NaN
    before = len(df)
    df = df.dropna(subset=["price"])
    if len(df) < before:
        meta["warnings"].append(f"Dropped {before - len(df)} NaN rows")

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Remove timezone info if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Sort by date, remove duplicates
    df = df.sort_index()
    dup_mask = np.asarray(df.index.duplicated(keep="first"), dtype=bool)
    if dup_mask.any():
        n_dups = int(dup_mask.sum())
        meta["warnings"].append(f"Removed {n_dups} duplicate dates")
        df = df.loc[~dup_mask].copy()

    # Price > 0
    neg_mask = np.asarray(df["price"].to_numpy(dtype=float) <= 0.0, dtype=bool)
    if neg_mask.any():
        n_neg = int(neg_mask.sum())
        meta["warnings"].append(f"Removed {n_neg} non-positive prices")
        df = df.loc[~neg_mask].copy()

    # Validate min rows
    if len(df) < min_rows:
        raise ValueError(
            f"Insufficient data: {len(df)} rows, need >= {min_rows}"
        )

    # Data quality checks (non-destructive — warnings only)
    price_series = pd.Series(df["price"].to_numpy(dtype=float), index=df.index, name="price")
    returns = price_series.pct_change().dropna()
    dq = run_data_quality_checks(df, returns)
    meta["data_quality"] = dq
    for w in dq.get("warnings", []):
        meta["warnings"].append(w)

    return df, meta


# ---------------------------------------------------------------------------
# Data Quality Pipeline
# ---------------------------------------------------------------------------

def detect_outliers(returns: pd.Series, iqr_multiplier: float = 5.0) -> Dict[str, Any]:
    """
    Detect return outliers using IQR method.

    Flags returns beyond Q1 - iqr_multiplier*IQR or Q3 + iqr_multiplier*IQR.
    Uses a wide multiplier (5x) to only flag truly extreme moves, not
    legitimate volatility events.

    Returns dict with outlier count, dates, and bounds.
    """
    q1 = float(returns.quantile(0.25))
    q3 = float(returns.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    outlier_mask = np.asarray((returns < lower) | (returns > upper), dtype=bool)
    outlier_pos = np.where(outlier_mask)[0]
    return {
        "n_outliers": int(outlier_mask.sum()),
        "outlier_dates": [_date_str(returns.index[int(i)]) for i in outlier_pos[:10]],
        "iqr_lower_bound": round(lower, 6),
        "iqr_upper_bound": round(upper, 6),
    }


def detect_stale_prices(prices: pd.Series, max_consecutive: int = 5) -> Dict[str, Any]:
    """
    Detect stale (unchanged) prices — consecutive identical closes.

    Stale prices may indicate data feed issues, halted trading, or
    illiquid instruments. Flags runs of `max_consecutive` or more
    identical prices.

    Returns dict with stale period count and date ranges.
    """
    prices = pd.Series(np.asarray(prices, dtype=float), index=prices.index, name=prices.name)
    is_unchanged = prices.diff().abs() < 1e-10
    stale_runs = []
    run_start = None
    run_len = 0

    for i in range(len(is_unchanged)):
        if is_unchanged.iloc[i]:
            if run_start is None:
                run_start = i - 1  # include the first price
            run_len += 1
        else:
            if run_len >= max_consecutive and run_start is not None:
                stale_runs.append({
                    "start": _date_str(prices.index[run_start]),
                    "end": _date_str(prices.index[min(i, len(prices) - 1)]),
                    "days": run_len,
                })
            run_start = None
            run_len = 0
    # Handle trailing run
    if run_len >= max_consecutive and run_start is not None:
        stale_runs.append({
            "start": _date_str(prices.index[run_start]),
            "end": _date_str(prices.index[-1]),
            "days": run_len,
        })

    return {
        "n_stale_periods": len(stale_runs),
        "stale_periods": stale_runs[:10],
    }


def detect_data_gaps(
    df: pd.DataFrame, max_gap_days: int = 5,
) -> Dict[str, Any]:
    """
    Detect gaps in the date index exceeding `max_gap_days` business days.

    Normal weekends and market holidays produce 1-3 day gaps. Gaps
    beyond `max_gap_days` may indicate missing data.

    Returns dict with gap count and date ranges.
    """
    gaps = []
    dates = pd.DatetimeIndex(pd.to_datetime(df.index))
    date_vals = dates.to_numpy(dtype="datetime64[D]")
    for i in range(1, len(date_vals)):
        prev_day = date_vals[i - 1]
        curr_day = date_vals[i]
        bdays = int(np.busday_count(prev_day, curr_day))
        if bdays > max_gap_days:
            gaps.append({
                "from": _date_str(date_vals[i - 1]),
                "to": _date_str(date_vals[i]),
                "business_days": bdays,
            })
    return {
        "n_gaps": len(gaps),
        "gaps": gaps[:10],
    }


def run_data_quality_checks(
    df: pd.DataFrame,
    returns: pd.Series,
) -> Dict[str, Any]:
    """
    Run full data quality pipeline and return structured report.

    Checks:
        1. Return outlier detection (IQR-based)
        2. Stale price detection (consecutive unchanged)
        3. Data gap detection (missing business days)
        4. Basic return statistics

    All checks are non-destructive (warning-only).
    """
    warnings_list = []

    # 1. Outlier detection
    outliers = detect_outliers(returns)
    if outliers["n_outliers"] > 0:
        warnings_list.append(
            f"{outliers['n_outliers']} return outliers detected (IQR method)"
        )
        logger.warning(
            "%d return outliers detected beyond [%.4f, %.4f]",
            outliers["n_outliers"], outliers["iqr_lower_bound"], outliers["iqr_upper_bound"],
        )

    # 2. Stale price detection
    price_series = pd.Series(df["price"].to_numpy(dtype=float), index=df.index, name="price")
    stale = detect_stale_prices(price_series)
    if stale["n_stale_periods"] > 0:
        warnings_list.append(
            f"{stale['n_stale_periods']} stale price periods detected (5+ unchanged)"
        )
        logger.warning(
            "%d stale price periods detected", stale["n_stale_periods"]
        )

    # 3. Data gap detection
    gaps = detect_data_gaps(df)
    if gaps["n_gaps"] > 0:
        warnings_list.append(
            f"{gaps['n_gaps']} data gaps detected (>5 business days)"
        )
        logger.warning("%d data gaps detected", gaps["n_gaps"])

    # 4. Basic return statistics
    ret_vals = returns.to_numpy(dtype=float)
    stats = {
        "mean_daily_return": round(float(np.mean(ret_vals)), 8),
        "std_daily_return": round(float(np.std(ret_vals)), 8),
        "skewness": round(float(_skewness(ret_vals)), 4),
        "kurtosis": round(float(_kurtosis(ret_vals)), 4),
        "min_return": round(float(np.min(ret_vals)), 6),
        "max_return": round(float(np.max(ret_vals)), 6),
        "n_trading_days": len(df),
    }

    return {
        "outliers": outliers,
        "stale_prices": stale,
        "data_gaps": gaps,
        "return_statistics": stats,
        "warnings": warnings_list,
    }


def _skewness(x: np.ndarray) -> float:
    """Compute sample skewness (Fisher's definition)."""
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-15:
        return 0.0
    return float((n / ((n - 1) * (n - 2))) * np.sum(((x - m) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis (Fisher's definition, normal=0)."""
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-15:
        return 0.0
    m4 = float(np.mean(((x - m) / s) ** 4))
    # Bias correction for sample excess kurtosis
    excess = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * m4 - 3 * (n - 1)) + 3
    return excess - 3.0  # excess kurtosis


def _cache_load(ticker: str, max_age_days: int):
    """Load cached data if fresh enough."""
    cache_path = CACHE_DIR / f"{ticker}.parquet"
    if not cache_path.exists():
        return None

    age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
    if age > timedelta(days=max_age_days):
        logger.info("Cache expired for %s (age: %s)", ticker, age)
        return None

    try:
        df = pd.read_parquet(cache_path)
        logger.info("Cache hit for %s (%d rows)", ticker, len(df))
        return df
    except Exception as e:
        logger.warning("Failed to read cache for %s: %s", ticker, e)
        return None


def _cache_save(ticker: str, df: pd.DataFrame):
    """Save data to parquet cache."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"{ticker}.parquet"
        df.to_parquet(cache_path)
        logger.info("Cached %s (%d rows)", ticker, len(df))
    except Exception as e:
        logger.warning("Failed to cache %s: %s", ticker, e)
