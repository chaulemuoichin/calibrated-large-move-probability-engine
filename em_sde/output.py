"""
Output generation: CSV, JSON summary, and matplotlib charts.

Charts (matplotlib only, no seaborn):
    1) probability_timeseries.png  (improved: auto-scaled, sigma overlay)
    2) reliability_diagram.png     (improved: zoomed, sized points)
    3) realized_return_hist.png    (improved: tail shading, % annotation)
    4) rolling_brier.png           (improved: stress period shading)
    5) probability_vs_price.png    (NEW: flagship investor chart)
    6) volatility_regime.png       (NEW: regime coloring)
    7) signal_heatmap.png          (NEW: date x horizon risk heatmap)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Consistent visual style across all charts
# ---------------------------------------------------------------------------
_STYLE = {
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.color": "#CCCCCC",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
}

_COLOR_PRICE = "#2D2D2D"    # near-black for price
_COLOR_PROB = "#1B6CA8"     # blue for probability
_COLOR_RAW = "#7FBBDB"      # light blue for raw probability
_COLOR_EVENT = "#D62728"    # red for events / danger
_COLOR_SIGMA = "#FF8C00"    # dark orange for volatility


def write_outputs(
    results: pd.DataFrame,
    reliability: pd.DataFrame,
    metrics: Dict[str, Any],
    metadata: Dict[str, Any],
    cfg,
    run_id: str,
    prices: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Write all output files to outputs/<run_id>/.

    Returns the output directory path.
    """
    out_dir = Path(cfg.output.base_dir) / run_id
    charts_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    # results.csv
    results_path = out_dir / "results.csv"
    results.to_csv(results_path, index=False)
    logger.info("Wrote %s (%d rows)", results_path, len(results))

    # reliability.csv
    rel_path = out_dir / "reliability.csv"
    reliability.to_csv(rel_path, index=False)
    logger.info("Wrote %s", rel_path)

    # summary.json
    summary = _build_summary(results, metrics, metadata, cfg, run_id)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_serializer)
    logger.info("Wrote %s", summary_path)

    # Charts
    if cfg.output.charts:
        _generate_charts(results, reliability, cfg.model.horizons, charts_dir, prices=prices)

    return out_dir


def _build_summary(
    results: pd.DataFrame,
    metrics: Dict[str, Any],
    metadata: Dict[str, Any],
    cfg,
    run_id: str,
) -> dict:
    """Build the summary.json content."""
    return {
        "run_id": run_id,
        "config": {
            "data_source": cfg.data.source,
            "ticker": cfg.data.ticker,
            "horizons": cfg.model.horizons,
            "k": cfg.model.k,
            "mc_base_paths": cfg.model.mc_base_paths,
            "garch_window": cfg.model.garch_window,
            "calibration_lr": cfg.calibration.lr,
            "safety_gate": cfg.calibration.safety_gate,
            "gate_window": cfg.calibration.gate_window,
            "ensemble_enabled": cfg.calibration.ensemble_enabled,
            "seed": cfg.model.seed,
            "t_df": cfg.model.t_df,
            "garch_in_sim": cfg.model.garch_in_sim,
            "garch_model_type": cfg.model.garch_model_type,
            "jump_enabled": cfg.model.jump_enabled,
            "threshold_mode": cfg.model.threshold_mode,
            "fixed_threshold_pct": cfg.model.fixed_threshold_pct,
            "multi_feature": cfg.calibration.multi_feature,
        },
        "unit_convention": {
            "returns": "daily simple returns (decimal, e.g. 0.01 = 1%)",
            "sigma_1d": "daily volatility (decimal std dev of daily returns)",
            "sigma_year": "sigma_1d * sqrt(252)",
            "simulation_dt": "1/252",
            "simulation_steps": "H (horizon in trading days)",
            "threshold": f"mode={cfg.model.threshold_mode}",
        },
        "data_metadata": metadata,
        "prediction_rows": len(results),
        "metrics": metrics,
    }


def _json_serializer(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return str(obj)


def _generate_charts(
    results: pd.DataFrame,
    reliability: pd.DataFrame,
    horizons: List[int],
    charts_dir: Path,
    prices: Optional[pd.DataFrame] = None,
):
    """Generate all matplotlib charts. No seaborn. No hardcoded colors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_STYLE)

    cmap = plt.get_cmap("tab10")
    colors = {H: cmap(i) for i, H in enumerate(horizons)}

    # Improved existing charts
    _chart_probability_timeseries(results, horizons, colors, charts_dir)
    _chart_reliability_diagram(reliability, horizons, colors, charts_dir)
    _chart_realized_return_hist(results, horizons, colors, charts_dir)
    _chart_rolling_brier(results, horizons, colors, charts_dir)

    # New investor-facing charts
    if prices is not None:
        _chart_probability_price_overlay(results, prices, horizons, charts_dir)
    _chart_volatility_regime(results, charts_dir)
    _chart_signal_heatmap(results, horizons, charts_dir)

    logger.info("Charts saved to %s (7 charts)", charts_dir)


def _chart_probability_timeseries(results, horizons, colors, charts_dir):
    """Chart 1: Probability time series (raw + calibrated) per horizon."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(horizons), 1, figsize=(16, 4.5 * len(horizons)), sharex=True)
    if len(horizons) == 1:
        axes = [axes]

    dates = pd.to_datetime(results["date"])
    sigma = results["sigma_garch_1d"].to_numpy(dtype=float)

    for ax, H in zip(axes, horizons):
        p_raw = results[f"p_raw_{H}"].to_numpy(dtype=float)
        p_cal = results[f"p_cal_{H}"].to_numpy(dtype=float)
        ci_lo = results[f"ci_low_{H}"].to_numpy(dtype=float)
        ci_hi = results[f"ci_high_{H}"].to_numpy(dtype=float)

        ax.plot(dates, p_raw, color=_COLOR_RAW, alpha=0.6, linewidth=0.8, label=f"p_raw")
        ax.plot(dates, p_cal, color=colors[H], linewidth=1.2, label=f"p_cal")
        ax.fill_between(dates, ci_lo, ci_hi, color=colors[H], alpha=0.1, label="95% CI")

        # Mark events — larger triangle-down markers at the top of data range
        y = results[f"y_{H}"].to_numpy(dtype=float)
        event_mask = y == 1.0

        # Auto-scale y-axis to data range
        all_vals = np.concatenate([p_raw, p_cal, ci_hi])
        y_max = float(np.nanmax(all_vals))
        padding = max(0.02, y_max * 0.15)
        ax.set_ylim(-0.005, y_max + padding)

        if event_mask.any():
            ax.scatter(
                dates[event_mask],
                np.full(event_mask.sum(), y_max + padding * 0.6),
                color=_COLOR_EVENT, s=40, marker="v", zorder=5,
                edgecolors="darkred", linewidths=0.5,
                label="event occurred", clip_on=False,
            )

        # Secondary axis: GARCH sigma for regime context
        ax2 = ax.twinx()
        ax2.plot(dates, sigma, color=_COLOR_SIGMA, alpha=0.3, linewidth=0.6)
        ax2.set_ylabel("\u03c3\u2081\u2084", color=_COLOR_SIGMA, fontsize=8)
        ax2.tick_params(axis="y", labelcolor=_COLOR_SIGMA, labelsize=7)

        ax.set_ylabel(f"Probability (H={H})")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Large-Move Probability: H={H} days")

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(charts_dir / "probability_timeseries.png", dpi=150)
    plt.close(fig)


def _chart_reliability_diagram(reliability, horizons, colors, charts_dir):
    """Chart 2: Reliability (calibration) diagram — zoomed to data range."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, ptype in zip(axes, ["raw", "calibrated"]):
        subset = reliability[reliability["type"] == ptype] if len(reliability) > 0 else pd.DataFrame()

        # Compute data-driven axis limit
        if len(subset) > 0:
            max_val = max(
                subset["mean_predicted"].max(),
                subset["mean_observed"].max(),
                0.05,
            )
            lim = max_val * 1.3
        else:
            lim = 1.0

        ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5, label="perfect")

        for H in horizons:
            h_data = subset[subset["horizon"] == H] if len(subset) > 0 else pd.DataFrame()
            if len(h_data) > 0:
                # Size points by bin_count
                counts = np.asarray(h_data["bin_count"], dtype=float)
                max_count = float(np.max(counts)) if float(np.max(counts)) > 0 else 1.0
                sizes = 30 + 170 * (counts / max_count)

                ax.scatter(
                    h_data["mean_predicted"], h_data["mean_observed"],
                    s=sizes, color=colors[H], alpha=0.8, zorder=5,
                    edgecolors="white", linewidths=0.5, label=f"H={H}",
                )
                ax.plot(
                    h_data["mean_predicted"], h_data["mean_observed"],
                    "-", color=colors[H], alpha=0.4, linewidth=0.8,
                )

                # Wilson confidence bands (if available)
                if "ci_low" in h_data.columns and "ci_high" in h_data.columns:
                    ci_lo = np.asarray(h_data["ci_low"], dtype=float)
                    ci_hi = np.asarray(h_data["ci_high"], dtype=float)
                    x_pred = np.asarray(h_data["mean_predicted"], dtype=float)
                    ax.fill_between(
                        x_pred, ci_lo, ci_hi,
                        color=colors[H], alpha=0.12, zorder=3,
                    )

                # Annotate bin counts
                x_vals = np.asarray(h_data["mean_predicted"], dtype=float)
                y_vals = np.asarray(h_data["mean_observed"], dtype=float)
                bin_counts = np.asarray(h_data["bin_count"], dtype=int)
                for x_val, y_val, bin_count_val in zip(x_vals, y_vals, bin_counts):
                    ax.annotate(
                        f"n={bin_count_val}",
                        (float(x_val), float(y_val)),
                        fontsize=6, alpha=0.6, xytext=(5, 5),
                        textcoords="offset points",
                    )

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Mean Observed Frequency")
        ax.set_title(f"Reliability Diagram ({ptype})")
        ax.set_xlim(-0.005, lim)
        ax.set_ylim(-0.005, lim)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(charts_dir / "reliability_diagram.png", dpi=150)
    plt.close(fig)


def _chart_realized_return_hist(results, horizons, colors, charts_dir):
    """Chart 3: Histogram of realized returns with tail shading."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(horizons), figsize=(5.5 * len(horizons), 4.5))
    if len(horizons) == 1:
        axes = [axes]

    for ax, H in zip(axes, horizons):
        col = f"realized_return_{H}"
        data = results[col].dropna().to_numpy(dtype=float)

        if len(data) > 0:
            ax.hist(data, bins=50, color=colors[H], alpha=0.7, edgecolor="black", linewidth=0.3)

            # Mark threshold zones
            thr_mean = results[f"thr_{H}"].dropna().mean()
            ax.axvline(-thr_mean, color=_COLOR_EVENT, linestyle="--", linewidth=1.2)
            ax.axvline(thr_mean, color=_COLOR_EVENT, linestyle="--", linewidth=1.2)

            # Shade tail regions beyond threshold
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.axvspan(xlim[0], -thr_mean, alpha=0.10, color=_COLOR_EVENT, zorder=0)
            ax.axvspan(thr_mean, xlim[1], alpha=0.10, color=_COLOR_EVENT, zorder=0)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            # Annotate tail percentage
            n_beyond = np.sum(np.abs(data) >= thr_mean)
            pct_beyond = n_beyond / len(data) * 100
            ax.text(
                0.97, 0.95,
                f"{pct_beyond:.1f}% beyond\nthreshold",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85, edgecolor="#CCCCCC"),
            )

        ax.set_xlabel(f"Realized Return (H={H})")
        ax.set_ylabel("Count")
        ax.set_title(f"Return Distribution: H={H}")

    fig.tight_layout()
    fig.savefig(charts_dir / "realized_return_hist.png", dpi=150)
    plt.close(fig)


def _chart_rolling_brier(results, horizons, colors, charts_dir):
    """Chart 4: Rolling Brier score with stress period shading."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 5))
    window = 60  # rolling window for Brier
    dates = pd.to_datetime(results["date"])

    # Shade stress periods (sigma > 75th percentile) first as background
    sigma = results["sigma_garch_1d"].values
    sigma_75 = np.nanpercentile(sigma, 75)
    stress_mask = sigma > sigma_75
    ax.fill_between(
        dates, 0, 1, where=stress_mask,
        alpha=0.06, color=_COLOR_EVENT, label="elevated vol (>75th pctile)",
    )

    all_brier = []
    for H in horizons:
        y = results[f"y_{H}"].to_numpy(dtype=float)
        p_raw = results[f"p_raw_{H}"].to_numpy(dtype=float)
        p_cal = results[f"p_cal_{H}"].to_numpy(dtype=float)

        # Compute rolling Brier for calibrated
        brier_series = (p_cal - y) ** 2
        mask = ~np.isnan(brier_series)

        if mask.sum() > window:
            rolling = pd.Series(brier_series).rolling(window, min_periods=window // 2).mean()
            ax.plot(dates, rolling, color=colors[H], linewidth=1.0, label=f"H={H} (cal)")
            all_brier.extend(brier_series[mask].tolist())

            # Raw for comparison (lighter)
            brier_raw = (p_raw - y) ** 2
            rolling_raw = pd.Series(brier_raw).rolling(window, min_periods=window // 2).mean()
            ax.plot(dates, rolling_raw, color=colors[H], linewidth=0.6, alpha=0.4,
                    linestyle="--", label=f"H={H} (raw)")

    # Mean Brier reference line
    if all_brier:
        mean_brier = float(np.nanmean(all_brier))
        ax.axhline(mean_brier, color="#888888", linestyle=":", linewidth=0.8,
                    alpha=0.6, label=f"mean={mean_brier:.4f}")

    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Brier Score")
    ax.set_title(f"Rolling Brier Score (window={window} days)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(charts_dir / "rolling_brier.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# NEW investor-facing charts (5, 6, 7)
# ---------------------------------------------------------------------------


def _chart_probability_price_overlay(results, prices, horizons, charts_dir):
    """Chart 5: Price with probability overlay — the flagship investor chart."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    H = max(horizons)  # longest horizon is most strategically relevant
    dates = pd.to_datetime(results["date"])
    p_cal = results[f"p_cal_{H}"].to_numpy(dtype=float)

    # Align prices to results dates
    price_aligned = prices.reindex(dates, method="ffill")["price"].to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(16, 6))

    # Left axis: price
    ax1.plot(dates, price_aligned, color=_COLOR_PRICE, linewidth=1.0, label="Price")
    ax1.set_ylabel("Price", color=_COLOR_PRICE, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=_COLOR_PRICE)

    # Right axis: probability
    ax2 = ax1.twinx()
    ax2.plot(dates, p_cal, color=_COLOR_PROB, linewidth=0.9, alpha=0.8,
             label=f"P(large move, H={H})")

    # Fill high-risk zones (above 75th percentile of probability)
    p_75 = float(np.nanpercentile(p_cal, 75))
    ax2.fill_between(
        dates, 0, p_cal,
        where=p_cal >= p_75,
        alpha=0.15, color=_COLOR_EVENT,
        label=f"High risk (>{p_75:.1%})",
    )

    # Auto-scale probability axis to data range
    p_max = float(np.nanmax(p_cal))
    ax2.set_ylim(0, p_max * 1.3)
    ax2.set_ylabel(f"P(|ret| > thr, H={H})", color=_COLOR_PROB, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=_COLOR_PROB)

    # Mark actual events on the price line
    y = results[f"y_{H}"].to_numpy(dtype=float)
    event_mask = y == 1.0
    if event_mask.any():
        ax1.scatter(
            dates[event_mask], price_aligned[event_mask],
            color=_COLOR_EVENT, s=25, zorder=5, marker="o",
            edgecolors="darkred", linewidths=0.5, label="Event occurred",
        )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    ax1.set_title(f"Price vs. Large-Move Probability (H={H} days)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    fig.savefig(charts_dir / "probability_vs_price.png", dpi=150)
    plt.close(fig)


def _chart_volatility_regime(results, charts_dir):
    """Chart 6: GARCH sigma with three-zone regime coloring."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    dates = pd.to_datetime(results["date"])
    sigma = results["sigma_garch_1d"].to_numpy(dtype=float)

    # Regime thresholds from percentiles
    p25 = float(np.nanpercentile(sigma, 25))
    p75 = float(np.nanpercentile(sigma, 75))

    fig, ax = plt.subplots(figsize=(16, 5))

    # Plot sigma line
    ax.plot(dates, sigma, color="#333333", linewidth=0.8)

    # Three-zone regime coloring
    ax.fill_between(dates, 0, sigma,
                    where=sigma <= p25,
                    alpha=0.25, color="#2ECC71", label=f"Low vol (<{p25:.4f})")
    ax.fill_between(dates, 0, sigma,
                    where=(sigma > p25) & (sigma <= p75),
                    alpha=0.12, color="#F39C12", label="Normal vol")
    ax.fill_between(dates, 0, sigma,
                    where=sigma > p75,
                    alpha=0.25, color="#E74C3C", label=f"High vol (>{p75:.4f})")

    # Reference lines at percentiles
    ax.axhline(p25, color="#2ECC71", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.axhline(p75, color="#E74C3C", linestyle=":", linewidth=0.7, alpha=0.6)

    # Secondary axis: annualized vol for intuition
    ax2 = ax.twinx()
    y_lo, y_hi = ax.get_ylim()
    ax2.set_ylim(y_lo * np.sqrt(252), y_hi * np.sqrt(252))
    ax2.set_ylabel("Annualized Vol", fontsize=9, color="#666666")
    ax2.tick_params(axis="y", labelsize=8, colors="#666666")

    ax.set_ylabel("Daily \u03c3 (GARCH)")
    ax.set_title("Volatility Regime Over Time")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    fig.savefig(charts_dir / "volatility_regime.png", dpi=150)
    plt.close(fig)


def _chart_signal_heatmap(results, horizons, charts_dir):
    """Chart 7: Date x Horizon probability heatmap — compact risk scanner."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import LinearSegmentedColormap

    dates = pd.to_datetime(results["date"])

    # Build matrix: rows = horizons, columns = dates
    matrix = np.column_stack([results[f"p_cal_{H}"].to_numpy(dtype=float) for H in horizons]).T

    # Custom white -> yellow -> orange -> red colormap
    cmap = LinearSegmentedColormap.from_list("risk", [
        "#FFFFFF", "#FFF9C4", "#FFB74D", "#E53935",
    ])

    fig, ax = plt.subplots(figsize=(16, 2.5 + 0.6 * len(horizons)))

    # Use imshow for the heatmap
    date_nums = mdates.date2num(dates)
    vmax = float(np.nanpercentile(matrix, 97))  # clip at 97th percentile for contrast

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=max(vmax, 0.01),  # avoid vmax=0
        extent=(float(date_nums[0]), float(date_nums[-1]), float(len(horizons) - 0.5), -0.5),
        interpolation="nearest",
    )

    ax.set_yticks(range(len(horizons)))
    ax.set_yticklabels([f"H={H}" for H in horizons])
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=30)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Calibrated Probability", fontsize=9)

    ax.set_title("Risk Signal Heatmap: All Horizons")

    fig.tight_layout()
    fig.savefig(charts_dir / "signal_heatmap.png", dpi=150)
    plt.close(fig)
