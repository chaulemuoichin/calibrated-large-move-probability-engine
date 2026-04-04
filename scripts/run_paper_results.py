"""
Paper results collection: unified script to generate all tables for the paper.

Runs:
  1. Main results (5-fold CV) with significance testing
  2. Baseline comparison with paired bootstrap p-values
  3. Ablation study summary
  4. Temporal hold-out results
  5. Economic significance analysis

Outputs LaTeX-ready tables to outputs/paper/tables/

Usage:
    python scripts/run_paper_results.py spy
    python scripts/run_paper_results.py all
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from em_sde.config import load_config
from em_sde.data_layer import load_data
from em_sde.model_selection import (
    expanding_window_cv, apply_promotion_gates_oof,
    compute_benchmark_report, compute_pairwise_significance_report,
)
from em_sde.evaluation import (
    brier_score, brier_skill_score, auc_roc, expected_calibration_error,
    effective_sample_size, paired_bootstrap_loss_diff_pvalue,
    bootstrap_metric_ci, apply_fdr_correction,
    expected_calibration_error_detailed,
)
from scripts.baselines import run_all_baselines, evaluate_baseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TICKER_CONFIGS = {
    "spy": "configs/exp_suite/exp_spy_regime_gated.yaml",
    "googl": "configs/exp_suite/exp_googl_regime_gated.yaml",
    "amzn": "configs/exp_suite/exp_amzn_regime_gated.yaml",
    "nvda": "configs/exp_suite/exp_nvda_regime_gated.yaml",
}


def _sig_stars(p: float) -> str:
    """Convert p-value to significance stars."""
    if np.isnan(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _format_metric(val: float, fmt: str = ".4f") -> str:
    """Format metric with sign for BSS."""
    if np.isnan(val):
        return "---"
    return f"{val:{fmt}}"


def generate_main_results_table(ticker: str, n_folds: int = 5) -> pd.DataFrame:
    """Generate Table 1: Main CV results with gate pass/fail and significance."""
    config_path = TICKER_CONFIGS[ticker.lower()]
    cfg = load_config(config_path)
    df, _ = load_data(cfg)

    logger.info("Running %d-fold CV for %s...", n_folds, ticker.upper())
    cv_results, oof_df = expanding_window_cv(df, [cfg], [ticker.upper()], n_folds=n_folds)

    rows = []
    for H in cfg.model.horizons:
        h_oof = oof_df[(oof_df["config_name"] == ticker.upper()) & (oof_df["horizon"] == H)]
        if len(h_oof) < 50:
            continue

        p = h_oof["p_cal"].to_numpy(dtype=float)
        y = h_oof["y"].to_numpy(dtype=float)
        mask = np.isfinite(p) & np.isfinite(y)
        p_m, y_m = p[mask], y[mask]

        n_eff = effective_sample_size(y_m, H, p_cal=p_m)
        bss = brier_skill_score(p_m, y_m)
        auc = auc_roc(p_m, y_m)
        ece_detail = expected_calibration_error_detailed(p_m, y_m, adaptive=False)
        ece = ece_detail["ece"]

        # Bootstrap CIs for BSS, AUC, and ECE
        bss_pt, bss_lo, bss_hi = bootstrap_metric_ci(y_m, p_m, brier_skill_score, n_boot=2000)
        auc_pt, auc_lo, auc_hi = bootstrap_metric_ci(y_m, p_m, auc_roc, n_boot=2000)
        ece_fn = lambda p, y: expected_calibration_error(p, y, adaptive=False)
        ece_pt, ece_lo, ece_hi = bootstrap_metric_ci(y_m, p_m, ece_fn, n_boot=2000)

        # Significance vs climatology
        clim = float(np.mean(y_m))
        model_losses = (p_m - y_m) ** 2
        clim_losses = (np.full_like(y_m, clim) - y_m) ** 2
        pval = paired_bootstrap_loss_diff_pvalue(model_losses, clim_losses, n_boot=2000)

        # Threshold
        thresholds = cfg.model.regime_gated_fixed_pct_by_horizon or {}
        thr = thresholds.get(H, cfg.model.fixed_threshold_pct)

        rows.append({
            "Ticker": ticker.upper(),
            "H": H,
            "Threshold": f"{thr*100:.2f}%",
            "N": int(mask.sum()),
            "N_eff": round(n_eff, 0),
            "Event Rate": f"{clim*100:.1f}%",
            "BSS": bss,
            "BSS 95% CI": f"[{bss_lo:.4f}, {bss_hi:.4f}]",
            "AUC": auc,
            "AUC 95% CI": f"[{auc_lo:.3f}, {auc_hi:.3f}]",
            "ECE": ece,
            "ECE 95% CI": f"[{ece_lo:.4f}, {ece_hi:.4f}]",
            "Min Bin N": ece_detail["min_bin_n"],
            "p-value": pval,
            "Sig": _sig_stars(pval),
            "Gates": "PASS" if (bss > 0 and auc >= 0.55 and ece <= 0.02) else "FAIL",
        })

    return pd.DataFrame(rows)


def generate_baseline_comparison_table(ticker: str) -> pd.DataFrame:
    """Generate Table 2: Full model vs all baselines with paired significance."""
    config_path = TICKER_CONFIGS[ticker.lower()]
    cfg = load_config(config_path)
    df, _ = load_data(cfg)

    prices = df["price"].to_numpy(dtype=float)
    dates_idx = df.index
    horizons = cfg.model.horizons

    thresholds = {}
    fixed_pct = cfg.model.regime_gated_fixed_pct_by_horizon or {}
    for H in horizons:
        thresholds[H] = fixed_pct.get(H, cfg.model.fixed_threshold_pct)

    # Run baselines
    baseline_results = run_all_baselines(
        prices, dates_idx, horizons, thresholds,
        iv_csv_path=cfg.model.implied_vol_csv_path if cfg.model.implied_vol_enabled else None,
        garch_window=cfg.model.garch_window,
        t_df=cfg.model.t_df,
    )

    # Run full model
    logger.info("Running full model for baseline comparison...")
    from em_sde.backtest import run_walkforward
    full_results = run_walkforward(df, cfg)

    rows = []
    for H in horizons:
        # Full model test
        p_col = f"p_cal_{H}"
        y_col = f"y_{H}"
        if p_col not in full_results.columns:
            continue

        p_full = full_results[p_col].to_numpy(dtype=float)
        y_full = full_results[y_col].to_numpy(dtype=float)
        full_mask = np.isfinite(p_full) & np.isfinite(y_full)
        p_full_m = p_full[full_mask]
        y_full_m = y_full[full_mask]

        if len(p_full_m) < 50:
            continue

        rows.append({
            "Ticker": ticker.upper(),
            "Method": "Full Model",
            "H": H,
            "BSS": brier_skill_score(p_full_m, y_full_m),
            "AUC": auc_roc(p_full_m, y_full_m),
            "ECE": expected_calibration_error(p_full_m, y_full_m, adaptive=False),
            "p-value vs Full": "---",
        })

        # Build date-indexed full model series for paired alignment
        full_dates = pd.to_datetime(full_results["date"]) if "date" in full_results.columns else None

        for bl_name, bl_df in baseline_results.items():
            if len(bl_df) == 0:
                continue
            p_bl_col = f"p_baseline_{H}"
            y_bl_col = f"y_{H}"
            if p_bl_col not in bl_df.columns:
                continue

            p_bl = bl_df[p_bl_col].to_numpy(dtype=float)
            y_bl = bl_df[y_bl_col].to_numpy(dtype=float)
            bl_mask = np.isfinite(p_bl) & np.isfinite(y_bl)
            p_bl_m = p_bl[bl_mask]
            y_bl_m = y_bl[bl_mask]

            if len(p_bl_m) < 50:
                continue

            # Date-aligned paired comparison
            pval = np.nan
            if full_dates is not None and "idx" in bl_df.columns:
                # Convert baseline idx to dates for alignment
                bl_dates_all = dates_idx[bl_df["idx"].to_numpy(dtype=int).clip(0, len(dates_idx) - 1)]
                bl_dates = bl_dates_all[bl_mask]
                full_date_set = set(full_dates[full_mask])
                # Find common dates
                common_mask_bl = np.array([d in full_date_set for d in bl_dates])
                bl_date_to_idx = {d: i for i, d in enumerate(full_dates[full_mask])}
                common_bl_dates = bl_dates[common_mask_bl]
                common_fm_indices = np.array([bl_date_to_idx[d] for d in common_bl_dates
                                              if d in bl_date_to_idx])
                common_bl_indices = np.where(common_mask_bl)[0]
                n_common = min(len(common_fm_indices), len(common_bl_indices))
                if n_common >= 50:
                    fm_losses = (p_full_m[common_fm_indices[:n_common]] - y_full_m[common_fm_indices[:n_common]]) ** 2
                    bl_losses = (p_bl_m[common_bl_indices[:n_common]] - y_bl_m[common_bl_indices[:n_common]]) ** 2
                    pval = paired_bootstrap_loss_diff_pvalue(fm_losses, bl_losses, n_boot=2000)
            else:
                # Fallback: truncate to common length (less reliable)
                n_common = min(len(p_full_m), len(p_bl_m))
                if n_common >= 50:
                    fm_losses = (p_full_m[:n_common] - y_full_m[:n_common]) ** 2
                    bl_losses = (p_bl_m[:n_common] - y_bl_m[:n_common]) ** 2
                    pval = paired_bootstrap_loss_diff_pvalue(fm_losses, bl_losses, n_boot=2000)

            pval_str = f"{pval:.4f}{_sig_stars(pval)}" if np.isfinite(pval) else "n/a"
            rows.append({
                "Ticker": ticker.upper(),
                "Method": bl_name,
                "H": H,
                "BSS": brier_skill_score(p_bl_m, y_bl_m),
                "AUC": auc_roc(p_bl_m, y_bl_m),
                "ECE": expected_calibration_error(p_bl_m, y_bl_m, adaptive=False),
                "p-value vs Full": pval_str,
            })

    return pd.DataFrame(rows)


def to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Convert DataFrame to LaTeX table."""
    n_cols = len(df.columns)
    col_fmt = "l" + "c" * (n_cols - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
    ]

    # Header
    header = " & ".join(str(c) for c in df.columns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Body
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                if "ECE" in str(col) or "BSS" in str(col) or "AUC" in str(col):
                    cells.append(f"{val:.4f}")
                elif "p-value" in str(col) or "p_value" in str(col):
                    cells.append(f"{val:.4f}" if np.isfinite(val) else "---")
                else:
                    cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate paper results tables")
    parser.add_argument("ticker", help="Ticker or 'all'")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    os.makedirs("outputs/paper/tables", exist_ok=True)

    tickers = list(TICKER_CONFIGS.keys()) if args.ticker.lower() == "all" else [args.ticker.lower()]

    all_main = []
    all_baseline = []

    for ticker in tickers:
        try:
            logger.info("=== Generating results for %s ===", ticker.upper())

            # Table 1: Main results
            main_df = generate_main_results_table(ticker, n_folds=args.n_folds)
            all_main.append(main_df)

            # Table 2: Baseline comparison
            bl_df = generate_baseline_comparison_table(ticker)
            all_baseline.append(bl_df)

        except Exception as e:
            logger.error("Failed for %s: %s", ticker, e, exc_info=True)

    # Combine and save
    if all_main:
        combined_main = pd.concat(all_main, ignore_index=True)

        # Apply FDR correction across all p-values
        raw_pvals = combined_main["p-value"].tolist()
        adj_pvals, reject = apply_fdr_correction(raw_pvals, alpha=0.05)
        combined_main["p-value (FDR)"] = adj_pvals
        combined_main["FDR Sig"] = [_sig_stars(p) for p in adj_pvals]

        combined_main.to_csv("outputs/paper/tables/main_results.csv", index=False)

        latex = to_latex_table(
            combined_main[["Ticker", "H", "BSS", "BSS 95% CI", "AUC", "AUC 95% CI",
                           "ECE", "ECE 95% CI", "N_eff", "p-value (FDR)", "Gates"]],
            "Cross-validated calibration results (5-fold expanding window). "
            "BSS: Brier Skill Score vs.~climatology. 95\\% bootstrap CIs reported. "
            "p-values: Benjamini-Hochberg FDR-corrected across all tests. "
            "$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
            "tab:main_results",
        )
        with open("outputs/paper/tables/main_results.tex", "w") as f:
            f.write(latex)
        logger.info("Main results table saved")

    if all_baseline:
        combined_bl = pd.concat(all_baseline, ignore_index=True)
        combined_bl.to_csv("outputs/paper/tables/baseline_comparison.csv", index=False)

        latex = to_latex_table(
            combined_bl[["Ticker", "Method", "H", "BSS", "AUC", "ECE", "p-value vs Full"]],
            "Baseline comparison. Full model vs.~four baselines. "
            "$p$-values from paired bootstrap test of Brier score differences.",
            "tab:baseline_comparison",
        )
        with open("outputs/paper/tables/baseline_comparison.tex", "w") as f:
            f.write(latex)
        logger.info("Baseline comparison table saved")

    print("\n=== Paper tables generated in outputs/paper/tables/ ===")


if __name__ == "__main__":
    main()
