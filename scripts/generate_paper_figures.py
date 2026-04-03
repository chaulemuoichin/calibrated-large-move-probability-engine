"""
Publication-quality figure generation for academic paper.

Generates:
  1. Pipeline architecture diagram (text-based, for LaTeX tikz)
  2. Reliability diagrams (predicted vs observed) — multi-panel
  3. Ablation performance heatmap
  4. Baseline comparison bar chart
  5. Rolling calibration error over time
  6. Probability time series with regime shading

Usage:
    python scripts/generate_paper_figures.py
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Publication style settings
STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

COLORS = {
    "Full Model": "#2c3e50",
    "Historical Frequency": "#95a5a6",
    "GARCH-CDF": "#e67e22",
    "Implied-Vol BS": "#27ae60",
    "Feature Logistic": "#8e44ad",
}

OUTDIR = "outputs/paper/figures"


def setup_matplotlib():
    """Import and configure matplotlib for publication figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)
    return plt


def generate_reliability_diagrams(plt):
    """
    Generate multi-panel reliability diagrams showing predicted vs observed
    probability for each ticker and horizon.
    """
    from em_sde.config import load_config
    from em_sde.data_layer import load_data
    from em_sde.backtest import run_walkforward

    tickers = {
        "SPY": "configs/exp_suite/exp_spy_regime_gated.yaml",
        "GOOGL": "configs/exp_suite/exp_googl_regime_gated.yaml",
        "AMZN": "configs/exp_suite/exp_amzn_regime_gated.yaml",
        "NVDA": "configs/exp_suite/exp_nvda_regime_gated.yaml",
    }

    fig, axes = plt.subplots(len(tickers), 3, figsize=(12, 3.2 * len(tickers)),
                              squeeze=False, sharex=True, sharey=True)

    for row, (ticker, config_path) in enumerate(tickers.items()):
        try:
            cfg = load_config(config_path)
            df = load_data(cfg.data)
            results = run_walkforward(df, cfg)

            for col, H in enumerate(cfg.model.horizons):
                ax = axes[row, col]
                p_col = f"p_cal_{H}"
                y_col = f"y_{H}"

                if p_col not in results.columns:
                    ax.set_visible(False)
                    continue

                p = results[p_col].to_numpy(dtype=float)
                y = results[y_col].to_numpy(dtype=float)
                mask = np.isfinite(p) & np.isfinite(y)
                p_m, y_m = p[mask], y[mask]

                if len(p_m) < 50:
                    ax.set_visible(False)
                    continue

                # Compute reliability curve with equal-width bins
                n_bins = 10
                bin_edges = np.linspace(0, 1, n_bins + 1)
                bin_means_pred = []
                bin_means_obs = []
                bin_counts = []

                for i in range(n_bins):
                    lo, hi = bin_edges[i], bin_edges[i + 1]
                    in_bin = (p_m >= lo) & (p_m < hi) if i < n_bins - 1 else (p_m >= lo) & (p_m <= hi)
                    if in_bin.sum() > 0:
                        bin_means_pred.append(float(np.mean(p_m[in_bin])))
                        bin_means_obs.append(float(np.mean(y_m[in_bin])))
                        bin_counts.append(int(in_bin.sum()))

                # Plot
                ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=0.8, label="Perfect")
                ax.scatter(bin_means_pred, bin_means_obs, s=np.array(bin_counts) * 0.15,
                          c=COLORS["Full Model"], alpha=0.7, edgecolors="white", linewidths=0.5)
                ax.plot(bin_means_pred, bin_means_obs, c=COLORS["Full Model"], alpha=0.8, linewidth=1.2)

                # ECE annotation
                from em_sde.evaluation import expected_calibration_error
                ece = expected_calibration_error(p_m, y_m, adaptive=False)
                ax.text(0.05, 0.90, f"ECE={ece:.4f}", transform=ax.transAxes, fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

                if row == 0:
                    ax.set_title(f"H={H}")
                if col == 0:
                    ax.set_ylabel(f"{ticker}\nObserved Frequency")
                if row == len(tickers) - 1:
                    ax.set_xlabel("Predicted Probability")

                ax.set_xlim(-0.02, 0.62)
                ax.set_ylim(-0.02, 0.62)
                ax.set_aspect("equal")

        except Exception as e:
            logger.warning("Could not generate reliability diagram for %s: %s", ticker, e)

    fig.suptitle("Reliability Diagrams: Predicted vs. Observed Event Frequency", y=1.02, fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "reliability_diagrams.pdf"))
    fig.savefig(os.path.join(OUTDIR, "reliability_diagrams.png"))
    plt.close(fig)
    logger.info("Reliability diagrams saved")


def generate_ablation_heatmap(plt):
    """Generate ablation study heatmap from pre-computed results."""
    ablation_path = "outputs/paper/ablation_all.csv"
    if not os.path.exists(ablation_path):
        logger.warning("Ablation results not found at %s. Run run_ablation_study.py first.", ablation_path)
        return

    df = pd.read_csv(ablation_path)
    if len(df) == 0:
        return

    # Pivot: variant x (ticker, horizon) -> ECE
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    metrics = [("ece", "ECE (lower is better)"), ("bss", "BSS (higher is better)"), ("auc", "AUC (higher is better)")]

    for ax, (metric, title) in zip(axes, metrics):
        pivot = df.pivot_table(index="label", columns=["ticker", "horizon"], values=metric, aggfunc="mean")
        if len(pivot) == 0:
            continue

        # Sort by variant order
        variant_order = ["Base GBM", "+GARCH-in-Sim", "+Student-t",
                        "+MF Calibration", "+Histogram Post-Cal", "+Implied Vol", "Full Model"]
        present = [v for v in variant_order if v in pivot.index]
        pivot = pivot.loc[present]

        import matplotlib.colors as mcolors
        if metric == "ece":
            cmap = "RdYlGn_r"  # Lower is better
            vmin, vmax = 0, 0.05
        else:
            cmap = "RdYlGn"    # Higher is better
            vmin = pivot.values.min() if len(pivot) > 0 else 0
            vmax = pivot.values.max() if len(pivot) > 0 else 1

        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{t}\nH={h}" for t, h in pivot.columns], fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(title, fontsize=10)

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=6,
                           color="white" if abs(val - vmin) > 0.6 * (vmax - vmin) else "black")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Ablation Study: Component Contributions", y=1.02, fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "ablation_heatmap.pdf"))
    fig.savefig(os.path.join(OUTDIR, "ablation_heatmap.png"))
    plt.close(fig)
    logger.info("Ablation heatmap saved")


def generate_baseline_comparison_chart(plt):
    """Generate grouped bar chart comparing full model vs baselines."""
    baseline_path = "outputs/paper/tables/baseline_comparison.csv"
    if not os.path.exists(baseline_path):
        logger.warning("Baseline comparison not found. Run run_paper_results.py first.")
        return

    df = pd.read_csv(baseline_path)
    if len(df) == 0:
        return

    tickers = df["Ticker"].unique()
    fig, axes = plt.subplots(1, len(tickers), figsize=(4.5 * len(tickers), 4), sharey=True)
    if len(tickers) == 1:
        axes = [axes]

    for ax, ticker in zip(axes, tickers):
        t_df = df[df["Ticker"] == ticker]
        methods = t_df["Method"].unique()
        n_methods = len(methods)
        horizons = sorted(t_df["H"].unique())
        n_horizons = len(horizons)

        x = np.arange(n_horizons)
        width = 0.8 / n_methods

        for i, method in enumerate(methods):
            m_df = t_df[t_df["Method"] == method]
            bss_vals = [float(m_df[m_df["H"] == h]["BSS"].values[0])
                       if len(m_df[m_df["H"] == h]) > 0 else 0.0 for h in horizons]
            color = COLORS.get(method, f"C{i}")
            ax.bar(x + i * width - 0.4 + width / 2, bss_vals, width,
                  label=method, color=color, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f"H={h}" for h in horizons])
        ax.set_title(ticker)
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
        ax.set_ylabel("Brier Skill Score" if ax == axes[0] else "")

    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.suptitle("BSS: Full Model vs. Baselines", y=1.02, fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "baseline_comparison.pdf"))
    fig.savefig(os.path.join(OUTDIR, "baseline_comparison.png"))
    plt.close(fig)
    logger.info("Baseline comparison chart saved")


def generate_rolling_ece_plot(plt):
    """Generate rolling ECE over time to show calibration stability."""
    from em_sde.config import load_config
    from em_sde.data_layer import load_data
    from em_sde.backtest import run_walkforward
    from em_sde.evaluation import expected_calibration_error

    tickers = {
        "SPY": "configs/exp_suite/exp_spy_regime_gated.yaml",
        "GOOGL": "configs/exp_suite/exp_googl_regime_gated.yaml",
        "AMZN": "configs/exp_suite/exp_amzn_regime_gated.yaml",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    horizons = [5, 10, 20]

    for ax, H in zip(axes, horizons):
        for ticker, config_path in tickers.items():
            try:
                cfg = load_config(config_path)
                df = load_data(cfg.data)
                results = run_walkforward(df, cfg)

                p_col = f"p_cal_{H}"
                y_col = f"y_{H}"
                if p_col not in results.columns:
                    continue

                p = results[p_col].to_numpy(dtype=float)
                y = results[y_col].to_numpy(dtype=float)
                dates_arr = pd.to_datetime(results["date"])

                # Rolling ECE with 500-day window
                window = 500
                rolling_ece = []
                rolling_dates = []
                for i in range(window, len(p)):
                    p_win = p[i - window:i]
                    y_win = y[i - window:i]
                    mask = np.isfinite(p_win) & np.isfinite(y_win)
                    if mask.sum() >= 50:
                        ece = expected_calibration_error(p_win[mask], y_win[mask], adaptive=False)
                        rolling_ece.append(ece)
                        rolling_dates.append(dates_arr.iloc[i])

                ax.plot(rolling_dates, rolling_ece, label=ticker, alpha=0.8, linewidth=1)

            except Exception as e:
                logger.warning("Rolling ECE failed for %s H=%d: %s", ticker, H, e)

        ax.axhline(y=0.02, color="red", linestyle="--", alpha=0.5, linewidth=0.8, label="ECE Gate (0.02)")
        ax.set_title(f"H={H}")
        ax.set_ylabel("Rolling ECE (500d)" if ax == axes[0] else "")
        ax.set_xlabel("Date")
        ax.legend(fontsize=7)

    fig.suptitle("Rolling Calibration Error Over Time", y=1.02, fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "rolling_ece.pdf"))
    fig.savefig(os.path.join(OUTDIR, "rolling_ece.png"))
    plt.close(fig)
    logger.info("Rolling ECE plot saved")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plt = setup_matplotlib()

    logger.info("Generating publication figures...")

    # These don't require pre-computed data
    generate_reliability_diagrams(plt)
    generate_rolling_ece_plot(plt)

    # These require pre-computed results
    generate_ablation_heatmap(plt)
    generate_baseline_comparison_chart(plt)

    logger.info("All figures saved to %s", OUTDIR)


if __name__ == "__main__":
    main()
