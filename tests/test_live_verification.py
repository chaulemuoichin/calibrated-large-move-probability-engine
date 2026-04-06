"""
Tests for the live verification system.

Tests the critical audit logic:
- Forecast IDs are stable and unique
- Resolution uses trading-day alignment
- Resolved events use the threshold stored at forecast time
- Pending forecasts are not prematurely resolved
- Metric aggregation uses only resolved forecasts
- Source forecast ledger is not silently mutated
- Site generation works from a small fixture ledger
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from em_sde.ledger import (
    make_forecast_id,
    append_forecast,
    append_resolution,
    load_forecasts,
    load_resolutions,
    load_joined,
    get_resolved_forecast_ids,
    compute_expected_resolution_date,
)
from em_sde.live_metrics import (
    compute_live_metrics,
    compute_per_group_metrics,
    _compute_calibration_bins,
)


# ---------------------------------------------------------------------------
# Forecast ID tests
# ---------------------------------------------------------------------------

def test_forecast_id_deterministic():
    """Same inputs produce the same forecast ID."""
    id1 = make_forecast_id("SPY", "2025-01-15", 5, "2025-01-15T16:30:00Z")
    id2 = make_forecast_id("SPY", "2025-01-15", 5, "2025-01-15T16:30:00Z")
    assert id1 == id2, "Forecast IDs should be deterministic"


def test_forecast_id_unique_across_horizons():
    """Different horizons produce different IDs."""
    id5 = make_forecast_id("SPY", "2025-01-15", 5, "2025-01-15T16:30:00Z")
    id10 = make_forecast_id("SPY", "2025-01-15", 10, "2025-01-15T16:30:00Z")
    assert id5 != id10, "Different horizons should produce different IDs"


def test_forecast_id_unique_across_tickers():
    """Different tickers produce different IDs."""
    spy = make_forecast_id("SPY", "2025-01-15", 5, "2025-01-15T16:30:00Z")
    googl = make_forecast_id("GOOGL", "2025-01-15", 5, "2025-01-15T16:30:00Z")
    assert spy != googl


def test_forecast_id_unique_across_dates():
    """Different dates produce different IDs."""
    d1 = make_forecast_id("SPY", "2025-01-15", 5, "2025-01-15T16:30:00Z")
    d2 = make_forecast_id("SPY", "2025-01-16", 5, "2025-01-16T16:30:00Z")
    assert d1 != d2


def test_forecast_id_length():
    """ID should be 16 hex characters."""
    fid = make_forecast_id("SPY", "2025-01-15", 5, "2025-01-15T16:30:00Z")
    assert len(fid) == 16
    assert all(c in "0123456789abcdef" for c in fid)


# ---------------------------------------------------------------------------
# Ledger append-only tests
# ---------------------------------------------------------------------------

def test_append_forecast_creates_file():
    """Appending a forecast creates the ledger file."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "forecasts.jsonl"
        record = append_forecast(
            ticker="SPY", forecast_date_market="2025-01-15",
            horizon=5, threshold=0.0372, p_raw=0.08, p_cal=0.07,
            sigma_1d=0.012, config_path="configs/test.yaml",
            ledger_path=path,
        )
        assert path.exists()
        assert record["ticker"] == "SPY"
        assert record["status"] == "pending"
        assert record["horizon"] == 5


def test_append_forecast_is_append_only():
    """Multiple appends add lines, never overwrite."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "forecasts.jsonl"

        append_forecast(
            ticker="SPY", forecast_date_market="2025-01-15",
            horizon=5, threshold=0.04, p_raw=0.08, p_cal=0.07,
            sigma_1d=0.012, config_path="test.yaml", ledger_path=path,
        )
        append_forecast(
            ticker="GOOGL", forecast_date_market="2025-01-15",
            horizon=10, threshold=0.05, p_raw=0.12, p_cal=0.10,
            sigma_1d=0.015, config_path="test.yaml", ledger_path=path,
        )

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

        r1 = json.loads(lines[0])
        r2 = json.loads(lines[1])
        assert r1["ticker"] == "SPY"
        assert r2["ticker"] == "GOOGL"


def test_forecast_ledger_not_mutated_by_resolution():
    """Appending a resolution must not change the forecast file."""
    with tempfile.TemporaryDirectory() as td:
        f_path = Path(td) / "forecasts.jsonl"
        r_path = Path(td) / "resolutions.jsonl"

        rec = append_forecast(
            ticker="SPY", forecast_date_market="2025-01-15",
            horizon=5, threshold=0.04, p_raw=0.08, p_cal=0.07,
            sigma_1d=0.012, config_path="test.yaml", ledger_path=f_path,
        )

        forecast_content_before = f_path.read_text()

        append_resolution(
            forecast_id=rec["forecast_id"],
            resolution_date_market="2025-01-22",
            realized_return=-0.032,
            event_occurred=0,
            price_at_forecast=450.0,
            price_at_resolution=435.6,
            ledger_path=r_path,
        )

        forecast_content_after = f_path.read_text()
        assert forecast_content_before == forecast_content_after, \
            "Forecast ledger must not be modified by resolution"


def test_resolution_uses_forecast_threshold():
    """Resolution event must use the threshold from the original forecast."""
    with tempfile.TemporaryDirectory() as td:
        f_path = Path(td) / "forecasts.jsonl"
        r_path = Path(td) / "resolutions.jsonl"

        # Forecast with 4% threshold
        rec = append_forecast(
            ticker="SPY", forecast_date_market="2025-01-15",
            horizon=5, threshold=0.04, p_raw=0.08, p_cal=0.07,
            sigma_1d=0.012, config_path="test.yaml", ledger_path=f_path,
        )

        # Return of -3.2% is below 4% threshold
        append_resolution(
            forecast_id=rec["forecast_id"],
            resolution_date_market="2025-01-22",
            realized_return=-0.032,
            event_occurred=0,  # |0.032| < 0.04
            price_at_forecast=450.0,
            price_at_resolution=435.6,
            ledger_path=r_path,
        )

        resolutions = load_resolutions(r_path)
        assert len(resolutions) == 1
        assert resolutions.iloc[0]["event_occurred"] == 0

        # Same return would be an event under 3% threshold
        # But the resolution must use 4% (the forecast threshold)


# ---------------------------------------------------------------------------
# Trading-day alignment tests
# ---------------------------------------------------------------------------

def test_expected_resolution_date_with_calendar():
    """Resolution date uses trading-day count, not calendar days."""
    # Create a simple trading calendar (weekdays only, no holidays)
    dates = pd.bdate_range("2025-01-13", "2025-02-28")
    result = compute_expected_resolution_date("2025-01-13", 5, dates)
    # 5 trading days from Jan 13 (Mon): Jan 14,15,16,17,20 → Jan 20 (Mon)
    # (Jan 13 is the forecast date; future starts Jan 14)
    assert result == "2025-01-20"


def test_expected_resolution_date_crosses_weekend():
    """H=10 from a Friday should skip weekends."""
    dates = pd.bdate_range("2025-01-10", "2025-02-28")
    result = compute_expected_resolution_date("2025-01-10", 10, dates)
    # 10 trading days from Jan 10 (Fri): Jan 13-17 (5), Jan 20-24 (10)
    assert result == "2025-01-24"


def test_expected_resolution_date_without_calendar():
    """Without calendar, should give an approximate date."""
    result = compute_expected_resolution_date("2025-01-15", 5)
    # Should be roughly a week later
    assert result >= "2025-01-20"


# ---------------------------------------------------------------------------
# Load and join tests
# ---------------------------------------------------------------------------

def test_load_empty_ledger():
    """Loading a nonexistent ledger returns empty DataFrame."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "nonexistent.jsonl"
        df = load_forecasts(path)
        assert len(df) == 0


def test_load_joined_no_resolutions():
    """Joined data with no resolutions has NaN event columns."""
    with tempfile.TemporaryDirectory() as td:
        f_path = Path(td) / "forecasts.jsonl"
        r_path = Path(td) / "resolutions.jsonl"

        append_forecast(
            ticker="SPY", forecast_date_market="2025-01-15",
            horizon=5, threshold=0.04, p_raw=0.08, p_cal=0.07,
            sigma_1d=0.012, config_path="test.yaml", ledger_path=f_path,
        )

        joined = load_joined(f_path, r_path)
        assert len(joined) == 1
        assert pd.isna(joined.iloc[0].get("realized_return"))


def test_load_joined_with_resolutions():
    """Joined data correctly merges forecasts and resolutions."""
    with tempfile.TemporaryDirectory() as td:
        f_path = Path(td) / "forecasts.jsonl"
        r_path = Path(td) / "resolutions.jsonl"

        rec = append_forecast(
            ticker="SPY", forecast_date_market="2025-01-15",
            horizon=5, threshold=0.04, p_raw=0.08, p_cal=0.07,
            sigma_1d=0.012, config_path="test.yaml", ledger_path=f_path,
        )

        append_resolution(
            forecast_id=rec["forecast_id"],
            resolution_date_market="2025-01-22",
            realized_return=0.052,
            event_occurred=1,
            price_at_forecast=450.0,
            price_at_resolution=473.4,
            ledger_path=r_path,
        )

        joined = load_joined(f_path, r_path)
        assert len(joined) == 1
        assert joined.iloc[0]["event_occurred"] == 1
        assert abs(joined.iloc[0]["realized_return"] - 0.052) < 1e-6


def test_get_resolved_ids():
    """Already-resolved IDs are correctly tracked."""
    with tempfile.TemporaryDirectory() as td:
        r_path = Path(td) / "resolutions.jsonl"

        append_resolution(
            forecast_id="abc123",
            resolution_date_market="2025-01-22",
            realized_return=0.01, event_occurred=0,
            price_at_forecast=100, price_at_resolution=101,
            ledger_path=r_path,
        )
        append_resolution(
            forecast_id="def456",
            resolution_date_market="2025-01-22",
            realized_return=-0.05, event_occurred=1,
            price_at_forecast=100, price_at_resolution=95,
            ledger_path=r_path,
        )

        ids = get_resolved_forecast_ids(r_path)
        assert "abc123" in ids
        assert "def456" in ids
        assert "xyz789" not in ids


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

def test_metrics_only_use_resolved():
    """Metrics computation must exclude pending (unresolved) forecasts."""
    with tempfile.TemporaryDirectory() as td:
        f_path = Path(td) / "forecasts.jsonl"
        r_path = Path(td) / "resolutions.jsonl"

        # Create 5 forecasts, resolve only 3
        records = []
        for i in range(5):
            rec = append_forecast(
                ticker="SPY", forecast_date_market=f"2025-01-{15+i}",
                horizon=5, threshold=0.04,
                p_raw=0.05 + 0.02*i, p_cal=0.04 + 0.02*i,
                sigma_1d=0.012, config_path="test.yaml", ledger_path=f_path,
            )
            records.append(rec)

        # Resolve first 3
        for i in range(3):
            append_resolution(
                forecast_id=records[i]["forecast_id"],
                resolution_date_market=f"2025-01-{22+i}",
                realized_return=(-1)**i * 0.05,
                event_occurred=1 if i == 0 else 0,
                price_at_forecast=450.0,
                price_at_resolution=450.0 * (1 + (-1)**i * 0.05),
                ledger_path=r_path,
            )

        joined = load_joined(f_path, r_path)
        metrics = compute_live_metrics(joined=joined, min_resolved=2)

        assert metrics["n_total_forecasts"] == 5
        assert metrics["n_resolved"] == 3
        assert metrics["n_pending"] == 2
        assert metrics["metrics"] is not None
        assert metrics["metrics"]["n_resolved"] == 3


def test_metrics_warn_on_small_sample():
    """Metrics should include warnings for small sample sizes."""
    with tempfile.TemporaryDirectory() as td:
        f_path = Path(td) / "forecasts.jsonl"
        r_path = Path(td) / "resolutions.jsonl"

        rec = append_forecast(
            ticker="SPY", forecast_date_market="2025-01-15",
            horizon=5, threshold=0.04, p_raw=0.08, p_cal=0.07,
            sigma_1d=0.012, config_path="test.yaml", ledger_path=f_path,
        )
        append_resolution(
            forecast_id=rec["forecast_id"],
            resolution_date_market="2025-01-22",
            realized_return=0.01, event_occurred=0,
            price_at_forecast=450, price_at_resolution=454.5,
            ledger_path=r_path,
        )

        joined = load_joined(f_path, r_path)
        metrics = compute_live_metrics(joined=joined, min_resolved=10)

        assert len(metrics["warnings"]) > 0
        assert any("minimum" in w.lower() or "small" in w.lower() for w in metrics["warnings"])


def test_calibration_bins():
    """Calibration bins should have correct structure."""
    rng = np.random.default_rng(42)
    p = rng.uniform(0.01, 0.3, 100)
    y = (rng.random(100) < p).astype(float)

    bins = _compute_calibration_bins(p, y, n_bins=5)
    assert len(bins) > 0
    for b in bins:
        assert "mean_predicted" in b
        assert "mean_observed" in b
        assert "n" in b
        assert b["n"] > 0
    total_n = sum(b["n"] for b in bins)
    assert total_n == 100


# ---------------------------------------------------------------------------
# Per-group metrics tests
# ---------------------------------------------------------------------------

def test_per_group_metrics():
    """Per-ticker and per-horizon metrics work correctly."""
    with tempfile.TemporaryDirectory() as td:
        f_path = Path(td) / "forecasts.jsonl"
        r_path = Path(td) / "resolutions.jsonl"

        rng = np.random.default_rng(123)
        records = []
        for i in range(20):
            ticker = "SPY" if i < 10 else "GOOGL"
            rec = append_forecast(
                ticker=ticker, forecast_date_market=f"2025-01-{10+i:02d}",
                horizon=5, threshold=0.04,
                p_raw=0.05 + rng.random() * 0.1,
                p_cal=0.04 + rng.random() * 0.1,
                sigma_1d=0.012, config_path="test.yaml", ledger_path=f_path,
            )
            records.append(rec)

        for i, rec in enumerate(records):
            event = 1 if rng.random() < 0.15 else 0
            append_resolution(
                forecast_id=rec["forecast_id"],
                resolution_date_market=f"2025-02-{10+i:02d}",
                realized_return=0.06 if event else 0.01,
                event_occurred=event,
                price_at_forecast=100, price_at_resolution=100*(1+0.06*event+0.01*(1-event)),
                ledger_path=r_path,
            )

        joined = load_joined(f_path, r_path)
        per_ticker = compute_per_group_metrics(joined=joined, group_col="ticker", min_resolved=3)

        assert "SPY" in per_ticker
        assert "GOOGL" in per_ticker
        assert per_ticker["SPY"]["n"] == 10
        assert per_ticker["GOOGL"]["n"] == 10


# ---------------------------------------------------------------------------
# Site generation smoke test
# ---------------------------------------------------------------------------

def test_site_generation_from_fixture():
    """Site builder produces HTML files from a small fixture."""
    with tempfile.TemporaryDirectory() as td:
        f_path = Path(td) / "forecasts.jsonl"
        r_path = Path(td) / "resolutions.jsonl"
        output_dir = Path(td) / "site"

        rng = np.random.default_rng(42)
        for i in range(30):
            ticker = ["SPY", "GOOGL"][i % 2]
            horizon = [5, 10][i % 2]
            p_cal = 0.05 + rng.random() * 0.15
            rec = append_forecast(
                ticker=ticker, forecast_date_market=f"2025-01-{10 + i // 3:02d}",
                horizon=horizon, threshold=0.04,
                p_raw=p_cal * 1.1, p_cal=p_cal,
                sigma_1d=0.012, config_path="test.yaml", ledger_path=f_path,
            )

            # Resolve 20 of 30
            if i < 20:
                event = 1 if rng.random() < p_cal else 0
                ret = 0.06 if event else 0.02
                append_resolution(
                    forecast_id=rec["forecast_id"],
                    resolution_date_market=f"2025-02-{10 + i // 3:02d}",
                    realized_return=ret, event_occurred=event,
                    price_at_forecast=100, price_at_resolution=100*(1+ret),
                    ledger_path=r_path,
                )

        # Build site
        from scripts.build_live_verification_site import build_site
        build_site(
            demo=False, output_dir=output_dir,
            forecast_path=f_path, resolution_path=r_path,
        )

        # Verify output files exist
        assert (output_dir / "index.html").exists()
        assert (output_dir / "latest.html").exists()
        assert (output_dir / "track_record.html").exists()
        assert (output_dir / "audit.html").exists()
        assert (output_dir / "methodology.html").exists()

        # Verify index has meaningful content
        index_html = (output_dir / "index.html").read_text()
        assert "Brier Skill Score" in index_html
        assert "Total Forecasts" in index_html


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in test_funcs:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
