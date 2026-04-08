"""
Build static live-verification site from the forecast ledger.

Generates a set of HTML pages from the ledger data that can be served
as a static site. No backend required.

Usage:
    python scripts/build_live_verification_site.py
    python scripts/build_live_verification_site.py --demo   # use demo data
    python scripts/build_live_verification_site.py --output-dir docs/live
"""

import argparse
import csv
import html
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from em_sde.ledger import (
    load_forecasts, load_resolutions, load_joined,
    FORECAST_FILE, RESOLUTION_FILE, LEDGER_DIR,
)
from em_sde.live_metrics import (
    compute_live_metrics, compute_per_group_metrics, compute_rolling_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEMO_DIR = Path("outputs/live_verification/demo")
DEFAULT_OUTPUT = Path("outputs/live_verification/site")


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

def _css() -> str:
    return """
    :root { --bg: #fafafa; --fg: #1a1a1a; --border: #ddd; --accent: #2563eb;
            --warn: #b45309; --good: #047857; --bad: #b91c1c; --muted: #6b7280; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'IBM Plex Mono', 'Consolas', monospace; font-size: 14px;
           color: var(--fg); background: var(--bg); line-height: 1.6;
           max-width: 1100px; margin: 0 auto; padding: 20px; }
    h1 { font-size: 20px; margin-bottom: 4px; }
    h2 { font-size: 16px; margin: 28px 0 12px 0; border-bottom: 1px solid var(--border); padding-bottom: 4px; }
    h3 { font-size: 14px; margin: 16px 0 8px 0; }
    p, li { margin-bottom: 8px; }
    a { color: var(--accent); }
    nav { margin: 12px 0 24px 0; padding: 8px 0; border-bottom: 1px solid var(--border); }
    nav a { margin-right: 16px; text-decoration: none; font-weight: 600; }
    nav a:hover { text-decoration: underline; }
    .warn { background: #fef3cd; border: 1px solid #ffc107; padding: 10px 14px; margin: 12px 0; font-size: 13px; }
    .demo-banner { background: #fce4ec; border: 2px solid #e53935; padding: 12px 16px;
                   margin: 12px 0; font-weight: bold; text-align: center; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
    th, td { border: 1px solid var(--border); padding: 6px 10px; text-align: right; }
    th { background: #f0f0f0; font-weight: 600; text-align: left; }
    td:first-child, th:first-child { text-align: left; }
    .good { color: var(--good); font-weight: 600; }
    .bad { color: var(--bad); font-weight: 600; }
    .muted { color: var(--muted); }
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                   gap: 12px; margin: 12px 0; }
    .metric-card { border: 1px solid var(--border); padding: 12px; background: white; }
    .metric-card .label { font-size: 11px; color: var(--muted); text-transform: uppercase; }
    .metric-card .value { font-size: 22px; font-weight: 700; margin-top: 2px; }
    .metric-card .detail { font-size: 11px; color: var(--muted); margin-top: 4px; }
    .chart-container { margin: 16px 0; background: white; border: 1px solid var(--border); padding: 16px; }
    pre { background: #f5f5f5; padding: 12px; overflow-x: auto; font-size: 12px; border: 1px solid var(--border); }
    footer { margin-top: 40px; padding-top: 12px; border-top: 1px solid var(--border);
             font-size: 11px; color: var(--muted); }
    """


def _nav(current: str, is_demo: bool) -> str:
    pages = [
        ("index.html", "Overview"),
        ("latest.html", "Latest Forecasts"),
        ("track_record.html", "Track Record"),
        ("audit.html", "Audit Trail"),
        ("methodology.html", "Methodology"),
    ]
    links = []
    for href, label in pages:
        if href == current:
            links.append(f'<strong>{label}</strong>')
        else:
            links.append(f'<a href="{href}">{label}</a>')

    demo_banner = ""
    if is_demo:
        demo_banner = '<div class="demo-banner">DEMO DATA — Generated from historical backtest, not real live forecasts</div>'

    return f"""<nav>{'  '.join(links)}</nav>{demo_banner}"""


def _page(title: str, body: str, current: str, is_demo: bool) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} — Live Verification</title>
  <style>{_css()}</style>
</head>
<body>
  <h1>Calibrated Large-Move Probability Engine</h1>
  <div class="muted">Live Verification Dashboard</div>
  {_nav(current, is_demo)}
  {body}
  <footer>
    Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} &middot;
    calibrated-large-move-probability-engine
  </footer>
</body>
</html>"""


def _metric_card(label: str, value: str, detail: str = "", css_class: str = "") -> str:
    cls = f' class="{css_class}"' if css_class else ""
    return f"""<div class="metric-card">
  <div class="label">{label}</div>
  <div class="value"{cls}>{value}</div>
  <div class="detail">{detail}</div>
</div>"""


def _table_html(headers: List[str], rows: List[List[str]], max_rows: int = 500) -> str:
    hdr = "".join(f"<th>{h}</th>" for h in headers)
    body_rows = []
    for row in rows[:max_rows]:
        cells = "".join(f"<td>{c}</td>" for c in row)
        body_rows.append(f"<tr>{cells}</tr>")
    truncated = ""
    if len(rows) > max_rows:
        truncated = f'<p class="muted">Showing {max_rows} of {len(rows)} rows. Download full data from Audit Trail.</p>'
    return f"<table><thead><tr>{hdr}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>{truncated}"


def _reliability_chart_svg(bins: List[dict], width: int = 500, height: int = 400) -> str:
    """Generate an inline SVG reliability diagram."""
    if not bins:
        return '<p class="muted">No calibration data available yet.</p>'

    margin = 50
    pw = width - 2 * margin
    ph = height - 2 * margin

    elements = []
    # Perfect calibration line
    elements.append(
        f'<line x1="{margin}" y1="{margin + ph}" x2="{margin + pw}" y2="{margin}" '
        f'stroke="#ccc" stroke-width="1" stroke-dasharray="4,4"/>'
    )
    # Axes
    elements.append(
        f'<line x1="{margin}" y1="{margin + ph}" x2="{margin + pw}" y2="{margin + ph}" stroke="#333" stroke-width="1"/>'
    )
    elements.append(
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin + ph}" stroke="#333" stroke-width="1"/>'
    )

    max_val = max(max(b["mean_predicted"] for b in bins), max(b["mean_observed"] for b in bins), 0.3)
    scale = min(1.0, 1.0 / max(max_val, 0.01))

    for b in bins:
        x = margin + b["mean_predicted"] * scale * pw
        y = margin + ph - b["mean_observed"] * scale * ph
        r = max(4, min(12, b["n"] / 5))
        elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="#2563eb" opacity="0.7"/>'
        )
        elements.append(
            f'<text x="{x:.1f}" y="{y - r - 3:.1f}" text-anchor="middle" '
            f'font-size="10" fill="#666">n={b["n"]}</text>'
        )

    # Axis labels
    elements.append(
        f'<text x="{margin + pw/2}" y="{height - 5}" text-anchor="middle" font-size="12">Predicted Probability</text>'
    )
    elements.append(
        f'<text x="12" y="{margin + ph/2}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90, 12, {margin + ph/2})">Observed Frequency</text>'
    )

    # Tick labels
    for v in [0, 0.1, 0.2, 0.3, 0.5]:
        if v * scale <= 1.0:
            xp = margin + v * scale * pw
            yp = margin + ph - v * scale * ph
            elements.append(f'<text x="{xp}" y="{margin + ph + 14}" text-anchor="middle" font-size="10">{v}</text>')
            elements.append(f'<text x="{margin - 5}" y="{yp + 4}" text-anchor="end" font-size="10">{v}</text>')

    inner = "\n".join(elements)
    return f"""<div class="chart-container">
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{width}" height="{height}" fill="white"/>
  {inner}
</svg>
<p class="muted" style="margin-top:8px">Reliability diagram. Each point is a calibration bin.
Diameter proportional to sample count. Perfect calibration falls on the diagonal.</p>
</div>"""


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def _build_overview(metrics: dict, per_ticker: dict, per_horizon: dict, is_demo: bool, per_version: dict = None) -> str:
    parts = []

    parts.append("<h2>Overview</h2>")
    parts.append("<p>This dashboard tracks the out-of-sample performance of the calibrated "
                 "large-move probability engine on live forecasts. All metrics are computed "
                 "from resolved forecasts only — never from pending predictions.</p>")

    m = metrics.get("metrics")

    # Warnings
    for w in metrics.get("warnings", []):
        parts.append(f'<div class="warn">{html.escape(w)}</div>')

    # Summary cards
    cards = []
    cards.append(_metric_card("Total Forecasts", str(metrics["n_total_forecasts"])))
    cards.append(_metric_card("Resolved", str(metrics["n_resolved"]),
                              f"{metrics['n_pending']} pending"))

    if m:
        bss_class = "good" if m["brier_skill_score"] > 0 else "bad"
        bss_ci = ""
        if "bss_ci_95" in m:
            bss_ci = f"95% CI: [{m['bss_ci_95'][0]}, {m['bss_ci_95'][1]}]"
        cards.append(_metric_card("Brier Skill Score", f"{m['brier_skill_score']:+.4f}",
                                  bss_ci, bss_class))

        auc_str = f"{m['auc']:.3f}" if m["auc"] is not None else "N/A"
        auc_class = "good" if m.get("auc") and m["auc"] >= 0.55 else ""
        cards.append(_metric_card("AUC", auc_str, "", auc_class))

        ece_class = "good" if m["ece"] <= 0.02 else "bad"
        cards.append(_metric_card("ECE", f"{m['ece']:.4f}", "Target: < 0.02", ece_class))

        cards.append(_metric_card("Event Rate", f"{m['event_rate']:.1%}",
                                  f"{m['n_events']} events / {m['n_resolved']} total"))
        cards.append(_metric_card("Sharpness (Std)", f"{m['p_cal_std']:.4f}",
                                  f"Range: [{m['p_cal_min']:.3f}, {m['p_cal_max']:.3f}]"))

    parts.append('<div class="metric-grid">' + "\n".join(cards) + '</div>')

    # Reliability diagram
    if m and m.get("calibration_bins"):
        parts.append("<h2>Calibration (Reliability Diagram)</h2>")
        parts.append(_reliability_chart_svg(m["calibration_bins"]))

    # Per-ticker breakdown
    if per_ticker:
        parts.append("<h2>Per-Ticker Metrics</h2>")
        headers = ["Ticker", "N", "Events", "Event Rate", "BSS", "AUC", "ECE"]
        rows = []
        for tk, tm in sorted(per_ticker.items()):
            if "warning" in tm:
                rows.append([tk, str(tm["n"]), str(tm["n_events"]), "—", "—", "—", f"<i>{tm['warning']}</i>"])
            else:
                bss_fmt = f'<span class="{"good" if tm["bss"] > 0 else "bad"}">{tm["bss"]:+.4f}</span>'
                auc_fmt = f'{tm["auc"]:.3f}' if tm["auc"] is not None else "N/A"
                ece_fmt = f'<span class="{"good" if tm["ece"] <= 0.02 else "bad"}">{tm["ece"]:.4f}</span>'
                rows.append([tk, str(tm["n"]), str(tm["n_events"]), f"{tm['event_rate']:.1%}",
                            bss_fmt, auc_fmt, ece_fmt])
        parts.append(_table_html(headers, rows))

    # Per-horizon breakdown
    if per_horizon:
        parts.append("<h2>Per-Horizon Metrics</h2>")
        headers = ["Horizon", "N", "Events", "Event Rate", "BSS", "AUC", "ECE"]
        rows = []
        for hz, hm in sorted(per_horizon.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
            if "warning" in hm:
                rows.append([f"H={hz}", str(hm["n"]), str(hm["n_events"]), "—", "—", "—", f"<i>{hm['warning']}</i>"])
            else:
                bss_fmt = f'<span class="{"good" if hm["bss"] > 0 else "bad"}">{hm["bss"]:+.4f}</span>'
                auc_fmt = f'{hm["auc"]:.3f}' if hm["auc"] is not None else "N/A"
                ece_fmt = f'<span class="{"good" if hm["ece"] <= 0.02 else "bad"}">{hm["ece"]:.4f}</span>'
                rows.append([f"H={hz}", str(hm["n"]), str(hm["n_events"]), f"{hm['event_rate']:.1%}",
                            bss_fmt, auc_fmt, ece_fmt])
        parts.append(_table_html(headers, rows))

    # Per-version breakdown
    if per_version and len(per_version) > 0:
        parts.append("<h2>Per-Version Metrics</h2>")
        headers = ["Version", "N", "Events", "Event Rate", "BSS", "AUC", "ECE"]
        rows = []
        for ver, vm in sorted(per_version.items()):
            if "warning" in vm:
                rows.append([ver, str(vm["n"]), str(vm["n_events"]), "\u2014", "\u2014", "\u2014", f"<i>{vm['warning']}</i>"])
            else:
                bss_fmt = f'<span class="{"good" if vm["bss"] > 0 else "bad"}">{vm["bss"]:+.4f}</span>'
                auc_fmt = f'{vm["auc"]:.3f}' if vm["auc"] is not None else "N/A"
                ece_fmt = f'<span class="{"good" if vm["ece"] <= 0.02 else "bad"}">{vm["ece"]:.4f}</span>'
                rows.append([ver, str(vm["n"]), str(vm["n_events"]), f"{vm['event_rate']:.1%}",
                            bss_fmt, auc_fmt, ece_fmt])
        parts.append(_table_html(headers, rows))

    return "\n".join(parts)


def _build_latest(joined: pd.DataFrame) -> str:
    parts = []
    parts.append("<h2>Latest Forecasts</h2>")
    parts.append("<p>Most recent forecasts published to the ledger.</p>")

    pending = joined[joined["event_occurred"].isna()].copy() if "event_occurred" in joined.columns else joined.copy()
    if len(pending) == 0:
        parts.append('<p class="muted">No pending forecasts.</p>')
        return "\n".join(parts)

    pending = pending.sort_values("forecast_date_market", ascending=False)

    headers = ["Date", "Ticker", "H", "Threshold", "p_cal", "p_raw", "Sigma", "Resolve By", "Version"]
    rows = []
    for _, r in pending.head(100).iterrows():
        rows.append([
            str(r.get("forecast_date_market", "")),
            str(r.get("ticker", "")),
            str(int(r.get("horizon", 0))),
            f"{r.get('threshold', 0):.4f}",
            f"{r.get('p_cal', 0):.4f}",
            f"{r.get('p_raw', 0):.4f}",
            f"{r.get('sigma_1d', 0):.6f}",
            str(r.get("expected_resolution_date", "")),
            str(r.get("model_version", "")),
        ])

    parts.append(_table_html(headers, rows))
    return "\n".join(parts)


def _build_track_record(joined: pd.DataFrame, rolling: pd.DataFrame) -> str:
    parts = []
    parts.append("<h2>Track Record</h2>")
    parts.append("<p>All resolved forecasts and cumulative performance.</p>")

    resolved = joined.dropna(subset=["event_occurred"]).copy() if "event_occurred" in joined.columns else pd.DataFrame()
    if len(resolved) == 0:
        parts.append('<p class="muted">No resolved forecasts yet.</p>')
        return "\n".join(parts)

    resolved = resolved.sort_values("forecast_date_market", ascending=False)

    # Resolved forecasts table
    parts.append("<h3>Resolved Forecasts</h3>")
    headers = ["Date", "Ticker", "H", "p_cal", "Threshold", "Return", "Event", "Resolve Date", "Version"]
    rows = []
    for _, r in resolved.head(200).iterrows():
        event_str = "Yes" if r.get("event_occurred") == 1 else "No"
        event_class = "good" if r.get("event_occurred") == 1 else ""
        ret = r.get("realized_return", 0)
        ret_str = f'<span class="{"bad" if abs(ret) > 0.03 else ""}">{ret:+.4f}</span>'
        rows.append([
            str(r.get("forecast_date_market", "")),
            str(r.get("ticker", "")),
            str(int(r.get("horizon", 0))),
            f"{r.get('p_cal', 0):.4f}",
            f"{r.get('threshold', 0):.4f}",
            ret_str,
            f'<span class="{event_class}">{event_str}</span>',
            str(r.get("resolution_date_market", "")),
            str(r.get("model_version", "")),
        ])
    parts.append(_table_html(headers, rows))

    # Rolling metrics table
    if len(rolling) > 0:
        parts.append("<h3>Rolling Metrics</h3>")
        parts.append("<p>Metrics computed over sliding windows of resolved forecasts.</p>")
        headers = ["Window Start", "Window End", "N", "Events", "BSS", "AUC", "ECE"]
        rows = []
        for _, r in rolling.iterrows():
            bss_val = r["bss"]
            bss_fmt = f'{bss_val:+.4f}' if not (pd.isna(bss_val) or np.isinf(bss_val)) else "N/A"
            auc_fmt = f'{r["auc"]:.3f}' if r.get("auc") is not None and not pd.isna(r.get("auc")) else "N/A"
            ece_val = r["ece"]
            ece_fmt = f'{ece_val:.4f}' if not (pd.isna(ece_val) or np.isinf(ece_val)) else "N/A"
            rows.append([
                str(r["window_start"]), str(r["window_end"]), str(r["n"]),
                str(r["n_events"]), bss_fmt, auc_fmt, ece_fmt,
            ])
        parts.append(_table_html(headers, rows))

    return "\n".join(parts)


def _build_audit(forecast_path: Path, resolution_path: Path, output_dir: Path) -> str:
    parts = []
    parts.append("<h2>Audit Trail &amp; Data Download</h2>")
    parts.append("<p>Raw ledger files for independent verification. "
                 "The forecast ledger is append-only; forecasts are never edited after publication. "
                 "Resolutions are stored separately, keyed by forecast_id.</p>")

    # Copy raw ledger files to site for download
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    for src, name, desc in [
        (forecast_path, "forecasts.jsonl", "Forecast Ledger (JSONL)"),
        (resolution_path, "resolutions.jsonl", "Resolution Ledger (JSONL)"),
    ]:
        if src.exists():
            import shutil
            shutil.copy2(src, data_dir / name)
            size = src.stat().st_size
            parts.append(f'<p><a href="data/{name}" download>{desc}</a> ({size:,} bytes)</p>')
        else:
            parts.append(f'<p class="muted">{desc}: not yet created</p>')

    # Also export derived joined CSV
    try:
        joined = load_joined(forecast_path, resolution_path)
        if len(joined) > 0:
            csv_path = data_dir / "joined_forecasts.csv"
            joined.to_csv(csv_path, index=False)
            parts.append(f'<p><a href="data/joined_forecasts.csv" download>Joined Forecasts + Resolutions (CSV)</a></p>')
    except Exception:
        pass

    # Metrics JSON
    try:
        from em_sde.live_metrics import compute_live_metrics
        metrics = compute_live_metrics(joined=joined)
        metrics_path = data_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        parts.append(f'<p><a href="data/metrics.json" download>Computed Metrics (JSON)</a></p>')
    except Exception:
        pass

    # Field definitions
    parts.append("<h3>Forecast Fields</h3>")
    parts.append("<pre>" + "\n".join([
        "forecast_id              — SHA-256-derived unique ID (ticker:date:horizon)",
        "forecast_timestamp_utc   — Wall-clock time the forecast was written (UTC)",
        "forecast_date_market     — Market date the forecast is made as-of",
        "ticker                   — Stock ticker symbol",
        "horizon                  — Trading days until resolution",
        "threshold                — Return threshold for event definition (decimal)",
        "p_raw                    — Raw Monte Carlo probability (uncalibrated)",
        "p_cal                    — Calibrated probability (final output)",
        "sigma_1d                 — GARCH 1-day conditional volatility",
        "delta_sigma              — Change in sigma vs prior window",
        "vol_ratio                — Ratio of realized vol to GARCH sigma",
        "vol_of_vol               — Volatility of volatility",
        "event_rate_historical    — Trailing 252-day empirical event rate",
        "calibrator_n_updates     — Number of online calibrator updates",
        "model_version            — Model version string",
        "git_commit               — Git commit hash at forecast time",
        "config_path              — YAML config file used",
        "config_hash              — SHA-256 hash of config file",
        "checkpoint_hash          — Hash of model checkpoint metadata",
        "expected_resolution_date — Estimated resolution date",
        "status                   — Always 'pending' at write time",
    ]) + "</pre>")

    parts.append("<h3>Resolution Fields</h3>")
    parts.append("<pre>" + "\n".join([
        "forecast_id              — Links to the original forecast",
        "resolution_timestamp_utc — Wall-clock time resolution was computed",
        "resolution_date_market   — Actual market date of resolution",
        "realized_return          — Actual H-day return (decimal)",
        "event_occurred           — 1 if |return| >= threshold, else 0",
        "price_at_forecast        — Closing price on forecast date",
        "price_at_resolution      — Closing price H trading days later",
    ]) + "</pre>")

    return "\n".join(parts)


def _build_methodology() -> str:
    parts = []
    parts.append("<h2>Methodology</h2>")

    parts.append("""
    <h3>What the system predicts</h3>
    <p>For each ticker and horizon H, the model estimates:
    <em>"What is the probability that |price return| >= threshold over the next H trading days?"</em>
    This is a two-sided (direction-agnostic) large-move probability.</p>

    <h3>Event definition</h3>
    <p>An event occurs when <code>|P(t+H)/P(t) - 1| >= threshold</code>,
    where P(t) is the closing price on the forecast date and P(t+H) is the
    closing price exactly H <strong>trading days</strong> later. The threshold
    used for resolution is always the threshold stored at forecast time, not any
    later threshold.</p>

    <h3>Trading-day resolution</h3>
    <p>Resolution uses the actual trading calendar, not naive calendar-day
    arithmetic. H=5 means 5 trading days (one week), H=10 means 10 trading
    days (two weeks). Weekends and market holidays are skipped.</p>

    <h3>No-lookahead guarantee</h3>
    <p>The forecast ledger is append-only. Each forecast is written with a
    UTC timestamp before the resolution date. The resolution ledger is a
    separate file — forecast records are never modified after publication.
    Anyone can verify the temporal ordering by inspecting the raw JSONL files.</p>

    <h3>Model versioning</h3>
    <p>Each forecast records its model_version, git_commit, config_hash, and
    checkpoint_hash. The overview page groups metrics by model_version, showing
    the main model and baselines side by side. The site can also be built
    filtered to a single version with <code>--version</code>. Pooled metrics
    across all versions are shown at the top; per-version breakdowns below.</p>

    <h3>Calibration methodology</h3>
    <p>The underlying model uses GJR-GARCH volatility estimation, Monte Carlo
    simulation with Student-t fat tails and Merton jump-diffusion, followed
    by multi-feature online recalibration with histogram post-calibration.
    Full details are in the paper and METHODOLOGY.md.</p>

    <h3>Limitations</h3>
    <ul>
    <li><strong>Small sample sizes:</strong> Live verification starts with zero
    resolved forecasts. Statistical conclusions require at minimum 50–100
    resolved forecasts per group. Do not treat early metrics as definitive.</li>
    <li><strong>Overlapping predictions:</strong> Forecasts at different horizons
    from the same date share price information. Effective sample size (N_eff)
    is lower than raw count.</li>
    <li><strong>Simplified live engine:</strong> The live prediction engine is a
    research prototype. It does not include all features of the backtest
    (e.g., implied-vol blending, earnings proximity, regime routing).</li>
    <li><strong>No guarantee of stationarity:</strong> The model's backtest
    performance does not guarantee future performance. Market regimes change.</li>
    <li><strong>Data dependency:</strong> Forecasts depend on price data quality
    and availability. Data delays or errors affect both forecasts and resolutions.</li>
    </ul>

    <h3>What the live record can prove</h3>
    <ul>
    <li>Whether forecasts were published before outcomes were known (timestamp audit)</li>
    <li>Whether predicted probabilities are calibrated over time (ECE, reliability curve)</li>
    <li>Whether the model has skill vs. climatology (BSS)</li>
    <li>Whether the model discriminates events from non-events (AUC)</li>
    </ul>

    <h3>What the live record cannot prove</h3>
    <ul>
    <li>That the model will continue to work in future market regimes</li>
    <li>That the model is optimal or best-in-class</li>
    <li>That the economic value of the forecasts exceeds trading costs</li>
    <li>Superiority over baselines in real time (live baseline tracking is not yet
    implemented; baseline comparisons are available only in the backtest paper results)</li>
    </ul>
    """)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_site(
    demo: bool = False,
    output_dir: Optional[Path] = None,
    forecast_path: Optional[Path] = None,
    resolution_path: Optional[Path] = None,
    version_filter: Optional[str] = None,
):
    """Build the full static site, optionally filtered to a single model_version."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    if forecast_path is None or resolution_path is None:
        if demo:
            forecast_path = DEMO_DIR / "forecasts.jsonl"
            resolution_path = DEMO_DIR / "resolutions.jsonl"
            if not forecast_path.exists():
                print("Demo ledger not found. Run: python scripts/generate_demo_ledger.py")
                return
        else:
            forecast_path = FORECAST_FILE
            resolution_path = RESOLUTION_FILE

    is_demo = demo
    logger.info("Building site from %s (demo=%s)", forecast_path, demo)

    # Load data
    joined = load_joined(forecast_path, resolution_path)

    # Version filtering: restrict all data to a single model_version
    if version_filter and "model_version" in joined.columns:
        before = len(joined)
        joined = joined[joined["model_version"] == version_filter].copy()
        logger.info("Version filter '%s': %d -> %d records", version_filter, before, len(joined))
        if len(joined) == 0:
            print(f"No records match version '{version_filter}'. Available versions:")
            all_joined = load_joined(forecast_path, resolution_path)
            if "model_version" in all_joined.columns:
                for v in sorted(all_joined["model_version"].unique()):
                    print(f"  {v}")
            return

    metrics = compute_live_metrics(joined=joined)
    per_ticker = compute_per_group_metrics(joined=joined, group_col="ticker")
    per_horizon = compute_per_group_metrics(joined=joined, group_col="horizon")
    per_version = compute_per_group_metrics(joined=joined, group_col="model_version")
    rolling = compute_rolling_metrics(joined=joined)

    # Build pages
    pages = {
        "index.html": ("Overview", _build_overview(metrics, per_ticker, per_horizon, is_demo, per_version)),
        "latest.html": ("Latest Forecasts", _build_latest(joined)),
        "track_record.html": ("Track Record", _build_track_record(joined, rolling)),
        "audit.html": ("Audit Trail", _build_audit(forecast_path, resolution_path, output_dir)),
        "methodology.html": ("Methodology", _build_methodology()),
    }

    for filename, (title, body) in pages.items():
        page_html = _page(title, body, filename, is_demo)
        (output_dir / filename).write_text(page_html, encoding="utf-8")
        logger.info("Wrote %s", output_dir / filename)

    print(f"\nSite built: {output_dir}/")
    print(f"  Open {output_dir / 'index.html'} in a browser to view.")


def main():
    parser = argparse.ArgumentParser(description="Build live verification static site")
    parser.add_argument("--demo", action="store_true",
                        help="Use demo/fixture data instead of live ledger")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for site files")
    parser.add_argument("--version", type=str, default=None,
                        help="Filter to a single model_version (e.g., v1.0, baseline:hist_freq)")
    args = parser.parse_args()

    build_site(demo=args.demo, output_dir=args.output_dir, version_filter=args.version)


if __name__ == "__main__":
    main()
