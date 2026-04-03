"""
Data ingestion layer with robust yfinance fetching, CSV loading,
synthetic generation, parquet caching, validation, and provenance.

Downstream consumers require df["price"] and may optionally use canonical
OHLCV columns when available: open, high, low, volume.
"""

import os
import time
import warnings
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, Optional

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
    df has DatetimeIndex and required column 'price'. Optional columns:
    open, high, low, volume.
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
        "strict_validation": cfg.data.strict_validation,
    }

    if cfg.data.source == "yfinance":
        df, meta = _load_yfinance(cfg, meta)
    elif cfg.data.source == "csv":
        df, meta = _load_csv(cfg, meta)
    elif cfg.data.source == "synthetic":
        df, meta = _generate_synthetic(cfg, meta)
    else:
        raise ValueError(f"Unknown data source: {cfg.data.source}")

    df, meta = _clean_and_validate(
        df, cfg.data.min_rows, meta, strict=cfg.data.strict_validation,
    )

    meta["rows"] = len(df)
    meta["actual_start"] = _date_str(df.index[0])
    meta["actual_end"] = _date_str(df.index[-1])
    meta["columns"] = list(df.columns)
    meta["has_ohlc"] = all(col in df.columns for col in ("open", "high", "low"))
    meta["has_volume"] = "volume" in df.columns
    meta["dataset_hash"] = _dataset_hash(df)

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
                try:
                    df, field_used = _canonicalize_market_frame(raw)
                    meta["field_used"] = f"{field_used} (auto_adjust=True)"
                    break
                except ValueError:
                    pass
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
                    try:
                        df, field_used = _canonicalize_market_frame(raw)
                        meta["field_used"] = f"{field_used} (Ticker.history, auto_adjust=True)"
                        break
                    except ValueError:
                        pass
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
    df, field_used = _canonicalize_market_frame(raw)
    meta["field_used"] = field_used
    meta["csv_path"] = str(path)
    return df, meta


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


def _canonicalize_market_frame(raw: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Normalize market data columns to price/open/high/low/volume."""
    col_map = {
        "price": ["price", "Price", "Close", "close", "Adj Close", "adj_close"],
        "open": ["Open", "open"],
        "high": ["High", "high"],
        "low": ["Low", "low"],
        "volume": ["Volume", "volume"],
    }

    def _first_match(candidates):
        for candidate in candidates:
            if candidate in raw.columns:
                return candidate
        return None

    close_col = _first_match(col_map["price"])
    if close_col is None:
        if len(raw.columns) == 1:
            col = raw.columns[0]
            return pd.DataFrame({"price": raw.iloc[:, 0].astype(float).values}, index=raw.index), str(col)
        raise ValueError(f"Cannot identify price column in data. Columns: {list(raw.columns)}")

    data = {"price": raw[close_col].astype(float).values}
    for canon in ("open", "high", "low", "volume"):
        matched = _first_match(col_map[canon])
        if matched is not None:
            data[canon] = raw[matched].astype(float).values
    return pd.DataFrame(data, index=raw.index), str(close_col)


def _dataset_hash(df: pd.DataFrame) -> str:
    """Stable SHA256 hash of the canonical dataset contents."""
    hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _clean_and_validate(
    df: pd.DataFrame, min_rows: int, meta: dict, strict: bool = False
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
        msg = f"Removed {n_dups} duplicate dates"
        meta["warnings"].append(msg)
        if strict:
            raise ValueError(f"Strict data validation failed: {msg}")
        df = df.loc[~dup_mask].copy()

    # Price > 0
    neg_mask = np.asarray(df["price"].to_numpy(dtype=float) <= 0.0, dtype=bool)
    if neg_mask.any():
        n_neg = int(neg_mask.sum())
        msg = f"Removed {n_neg} non-positive prices"
        meta["warnings"].append(msg)
        if strict:
            raise ValueError(f"Strict data validation failed: {msg}")
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
    if strict:
        _raise_for_strict_quality_failures(dq)

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


def detect_split_like_moves(returns: pd.Series, tolerance: float = 0.08) -> Dict[str, Any]:
    """
    Detect split-like close-to-close jumps that often indicate unadjusted data.

    Flags ratios near common split factors (2, 3, 4, 5, 10 and their inverses).
    """
    ratios = (1.0 + returns.to_numpy(dtype=float))
    common = np.array([2.0, 3.0, 4.0, 5.0, 10.0, 0.5, 1 / 3, 0.25, 0.2, 0.1])
    matches = []
    for i, ratio in enumerate(ratios):
        if ratio <= 0:
            continue
        rel_err = np.min(np.abs(ratio - common) / common)
        if rel_err <= tolerance:
            matches.append({
                "date": _date_str(returns.index[i]),
                "return": round(float(returns.iloc[i]), 6),
                "ratio": round(float(ratio), 4),
            })
    return {
        "n_split_like_moves": len(matches),
        "split_like_moves": matches[:10],
    }


def detect_ohlc_inconsistencies(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect impossible OHLC relationships when open/high/low are present."""
    required = {"price", "open", "high", "low"}
    if not required.issubset(df.columns):
        return {"n_ohlc_issues": 0, "ohlc_issues": []}

    issues = []
    vals = df[["price", "open", "high", "low"]].astype(float)
    tol = 1e-10 * vals.abs().max(axis=1).clip(lower=1.0)
    bad = (
        ((vals["high"] + tol) < vals["low"])
        | ((vals["high"] + tol) < vals[["price", "open"]].max(axis=1))
        | ((vals["low"] - tol) > vals[["price", "open"]].min(axis=1))
        | (vals[["open", "high", "low"]].min(axis=1) <= 0.0)
    )
    for ts in vals.index[bad][:10]:
        row = vals.loc[ts]
        issues.append({
            "date": _date_str(ts),
            "open": round(float(row["open"]), 6),
            "high": round(float(row["high"]), 6),
            "low": round(float(row["low"]), 6),
            "close": round(float(row["price"]), 6),
        })
    return {"n_ohlc_issues": int(bad.sum()), "ohlc_issues": issues}


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

    # 4. Split-like move detection
    split_like = detect_split_like_moves(returns)
    if split_like["n_split_like_moves"] > 0:
        warnings_list.append(
            f"{split_like['n_split_like_moves']} split-like close jumps detected"
        )
        logger.warning("%d split-like moves detected", split_like["n_split_like_moves"])

    # 5. OHLC consistency checks
    ohlc_issues = detect_ohlc_inconsistencies(df)
    if ohlc_issues["n_ohlc_issues"] > 0:
        warnings_list.append(
            f"{ohlc_issues['n_ohlc_issues']} OHLC consistency issues detected"
        )
        logger.warning("%d OHLC consistency issues detected", ohlc_issues["n_ohlc_issues"])

    # 6. Basic return statistics
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
        "split_like_moves": split_like,
        "ohlc_inconsistencies": ohlc_issues,
        "return_statistics": stats,
        "warnings": warnings_list,
    }


def _raise_for_strict_quality_failures(dq: Dict[str, Any]) -> None:
    """Raise on high-confidence data integrity failures."""
    reasons = []
    if dq.get("split_like_moves", {}).get("n_split_like_moves", 0) > 0:
        reasons.append("split_like_moves")
    if dq.get("ohlc_inconsistencies", {}).get("n_ohlc_issues", 0) > 0:
        reasons.append("ohlc_inconsistencies")
    if reasons:
        joined = ", ".join(reasons)
        raise ValueError(f"Strict data validation failed: {joined}")


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

def _earnings_cache_path(ticker: str) -> Path:
    """Return local cache path for normalized earnings dates."""
    return CACHE_DIR / "earnings" / f"{ticker.upper()}_dates.csv"


_earnings_mem_cache: Dict[str, Optional[np.ndarray]] = {}

_EARNINGS_CACHE_MAX_AGE_DAYS = 30


def load_earnings_dates(ticker: str) -> Optional[np.ndarray]:
    """
    Load historical earnings announcement dates for a ticker.

    Returns an array of datetime64 dates sorted chronologically,
    or None if unavailable. Walk-forward safe: earnings dates are
    publicly announced weeks before the event.

    Caches successful results in memory. Does NOT cache None (failures)
    so transient errors can be retried.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "AAPL", "GOOGL").

    Returns
    -------
    earnings_dates : np.ndarray of datetime64 or None
    """
    key = ticker.upper()
    if key in _earnings_mem_cache:
        return _earnings_mem_cache[key]

    cache_path = _earnings_cache_path(ticker)
    if cache_path.exists():
        # Check cache age
        import time as _time
        age_days = (_time.time() - cache_path.stat().st_mtime) / 86400
        if age_days <= _EARNINGS_CACHE_MAX_AGE_DAYS:
            try:
                raw = pd.read_csv(cache_path)
                if "date" in raw.columns and len(raw) > 0:
                    dates = pd.to_datetime(raw["date"], errors="coerce").dropna()
                    if len(dates) > 0:
                        result = np.sort(dates.values.astype("datetime64[D]"))
                        _earnings_mem_cache[key] = result
                        return result
            except Exception as e:
                logger.debug("Failed to read cached earnings dates for %s: %s", ticker, e)
        else:
            logger.debug("Earnings cache for %s is %d days old, refreshing", ticker, int(age_days))

    try:
        import yfinance as yf
        try:
            yf.set_tz_cache_location(str((CACHE_DIR / "yfinance").resolve()))
        except Exception:
            pass
        tk = yf.Ticker(ticker)
        # get_earnings_dates returns future and past dates
        ed = tk.get_earnings_dates(limit=100)
        if ed is None or len(ed) == 0:
            logger.debug("No earnings dates found for %s", ticker)
            return None  # Don't cache failures
        dates = pd.to_datetime(ed.index)
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        dates = np.sort(dates.values.astype("datetime64[D]"))
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"date": pd.to_datetime(dates).strftime("%Y-%m-%d")}).to_csv(
                cache_path, index=False,
            )
        except Exception as e:
            logger.debug("Failed to cache earnings dates for %s: %s", ticker, e)
        logger.info("Loaded %d earnings dates for %s", len(dates), ticker)
        _earnings_mem_cache[key] = dates
        return dates
    except Exception as e:
        logger.debug("Failed to load earnings dates for %s: %s", ticker, e)
        return None  # Don't cache failures


def compute_earnings_proximity(
    current_date: np.datetime64,
    earnings_dates: np.ndarray,
    max_days: int = 20,
) -> float:
    """
    Compute proximity to nearest earnings date.

    Returns a value in [0, 1] where 1.0 = on earnings day,
    0.0 = 20+ trading days from nearest earnings.
    Walk-forward safe: uses all known earnings dates (past and scheduled).

    Parameters
    ----------
    current_date : np.datetime64
        The current date.
    earnings_dates : np.ndarray
        Array of earnings dates (datetime64[D]).
    max_days : int
        Number of calendar days at which proximity drops to 0.

    Returns
    -------
    proximity : float
        Value in [0, 1].
    """
    current = np.datetime64(current_date, "D")
    diffs = np.abs((earnings_dates - current).astype(int))
    min_diff = int(np.min(diffs))
    return max(0.0, 1.0 - min_diff / max_days)


def load_implied_vol(csv_path: str) -> pd.DataFrame:
    """
    Load implied volatility data from CSV.

    Expected format: date index with one or more of these columns:
        - iv_9d:  9-day implied vol (annualized decimal, e.g. 0.18)
        - iv_30d: 30-day implied vol (or 'vix' column, auto-scaled from pct)
        - iv_3m:  3-month implied vol
        - vix, vix9d, vix3m: CBOE VIX indices (in percentage points, e.g. 17.5)

    Returns DataFrame with DatetimeIndex and columns normalized to annualized
    decimal (e.g. 0.175 = 17.5% annualized).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Implied vol CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path, parse_dates=True, index_col=0)

    if not isinstance(raw.index, pd.DatetimeIndex):
        raw.index = pd.to_datetime(raw.index)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    raw = raw.sort_index()

    result = pd.DataFrame(index=raw.index)

    # Map VIX columns (percentage points) to decimal annualized
    vix_map = {"vix": "iv_30d", "vix9d": "iv_9d", "vix3m": "iv_3m"}
    for vix_col, iv_col in vix_map.items():
        matches = [c for c in raw.columns if c.lower() == vix_col]
        if matches:
            result[iv_col] = raw[matches[0]].astype(float) / 100.0

    # Direct IV columns (already in decimal annualized)
    for col in ["iv_9d", "iv_30d", "iv_3m"]:
        if col in raw.columns and col not in result.columns:
            result[col] = raw[col].astype(float)

    if result.empty:
        raise ValueError(
            f"No recognized implied vol columns in {csv_path}. "
            f"Expected: vix/vix9d/vix3m (pct) or iv_9d/iv_30d/iv_3m (decimal). "
            f"Got: {list(raw.columns)}"
        )

    result = result.dropna(how="all")
    logger.info("Loaded implied vol from %s: %d rows, columns=%s",
                csv_path, len(result), list(result.columns))
    return result


def get_implied_vol_for_horizon(
    iv_df: pd.DataFrame,
    current_date,
    H: int,
) -> Optional[float]:
    """
    Look up horizon-matched implied vol for a given date.

    Mapping:
        H <= 9      -> iv_9d  (or iv_30d fallback)
        9 < H < 30  -> interpolate iv_9d and iv_30d, or single-tenor fallback
        H >= 30     -> iv_3m  (or iv_30d fallback)

    Returns annualized decimal implied vol, or None if no data available.
    Walk-forward safe: uses most recent available date <= current_date.
    """
    current_dt = pd.Timestamp(current_date)

    # Get most recent row on or before current_date
    mask = iv_df.index <= current_dt
    if not mask.any():
        return None
    row = iv_df.loc[mask].iloc[-1]

    # Check staleness: skip if data is more than 5 business days old
    last_date = iv_df.index[mask][-1]
    gap = np.busday_count(
        last_date.to_numpy().astype("datetime64[D]"),
        current_dt.to_numpy().astype("datetime64[D]"),
    )
    if gap > 5:
        return None

    has_9d = "iv_9d" in row.index and pd.notna(row.get("iv_9d"))
    has_30d = "iv_30d" in row.index and pd.notna(row.get("iv_30d"))
    has_3m = "iv_3m" in row.index and pd.notna(row.get("iv_3m"))

    if H <= 9:
        if has_9d:
            return float(row["iv_9d"])
        elif has_30d:
            return float(row["iv_30d"])
    elif H < 30:
        # 9 < H < 30: interpolate iv_9d → iv_30d
        if has_9d and has_30d:
            w = max(0.0, min(1.0, (H - 9) / (30 - 9)))
            return float((1 - w) * row["iv_9d"] + w * row["iv_30d"])
        elif has_30d:
            return float(row["iv_30d"])
        elif has_9d:
            return float(row["iv_9d"])
    else:
        # H >= 30
        if has_3m:
            return float(row["iv_3m"])
        elif has_30d:
            return float(row["iv_30d"])

    return None


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
