"""
Ablation study: systematically measure the contribution of each pipeline component.

Variants tested (cumulative):
  A) Base GBM (constant-vol, no calibration beyond online Platt)
  B) + GARCH-in-sim (vol dynamics within MC paths)
  C) + Student-t fat tails (regime-conditional df)
  D) + Multi-feature calibration (6 features + L2)
  E) + Histogram post-calibration (Bayesian shrinkage + PAV)
  F) + Implied vol blending (when available)
  G) Full model (all features from production config)

Each variant runs through 5-fold expanding-window CV.
Output: ablation_results.csv with per-variant, per-horizon metrics.

Usage:
    python scripts/run_ablation_study.py spy
    python scripts/run_ablation_study.py all
"""

import argparse
import copy
import logging
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from em_sde.config import load_config, PipelineConfig
from em_sde.data_layer import load_data
from em_sde.model_selection import expanding_window_cv, apply_promotion_gates_oof
from em_sde.evaluation import (
    brier_score, brier_skill_score, auc_roc, expected_calibration_error,
    paired_bootstrap_loss_diff_pvalue, apply_fdr_correction,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TICKER_CONFIGS = {
    "spy": "configs/exp_suite/exp_spy_regime_gated.yaml",
    "googl": "configs/exp_suite/exp_googl_regime_gated.yaml",
    "amzn": "configs/exp_suite/exp_amzn_regime_gated.yaml",
    "nvda": "configs/exp_suite/exp_nvda_regime_gated.yaml",
}


def _make_ablation_variant(base_cfg: PipelineConfig, variant: str) -> PipelineConfig:
    """Create a config variant for ablation by disabling features cumulatively."""
    cfg = copy.deepcopy(base_cfg)

    # Reduce MC paths for faster ablation runs
    cfg.model.mc_base_paths = 10000
    cfg.model.mc_boost_paths = 20000
    cfg.output.charts = False

    if variant == "A_base_gbm":
        # Constant-vol GBM, simple online Platt calibration
        cfg.model.garch_in_sim = False
        cfg.model.mc_regime_t_df = False
        cfg.model.t_df = 0.0  # Gaussian innovations
        cfg.calibration.multi_feature = False
        cfg.calibration.multi_feature_regime_conditional = False
        cfg.calibration.histogram_post_calibration = False
        cfg.calibration.post_cal_method = "none"
        cfg.model.implied_vol_enabled = False
        cfg.model.earnings_calendar = False
        cfg.model.jump_enabled = False
        cfg.model.mc_vol_term_structure = False

    elif variant == "B_garch_in_sim":
        # + GARCH-in-sim, still Gaussian, no MF calibration
        cfg.model.garch_in_sim = True
        cfg.model.mc_regime_t_df = False
        cfg.model.t_df = 0.0
        cfg.calibration.multi_feature = False
        cfg.calibration.multi_feature_regime_conditional = False
        cfg.calibration.histogram_post_calibration = False
        cfg.calibration.post_cal_method = "none"
        cfg.model.implied_vol_enabled = False
        cfg.model.earnings_calendar = False
        cfg.model.jump_enabled = True  # Keep jumps since GARCH-in-sim includes them
        cfg.model.mc_vol_term_structure = False

    elif variant == "C_student_t":
        # + Student-t fat tails with regime-conditional df
        cfg.model.garch_in_sim = True
        cfg.model.mc_regime_t_df = True
        cfg.model.t_df = 5.0
        cfg.calibration.multi_feature = False
        cfg.calibration.multi_feature_regime_conditional = False
        cfg.calibration.histogram_post_calibration = False
        cfg.calibration.post_cal_method = "none"
        cfg.model.implied_vol_enabled = False
        cfg.model.earnings_calendar = False
        cfg.model.jump_enabled = True
        cfg.model.mc_vol_term_structure = True

    elif variant == "D_multi_feature_cal":
        # + Multi-feature online calibration
        cfg.model.garch_in_sim = True
        cfg.model.mc_regime_t_df = True
        cfg.model.t_df = 5.0
        cfg.calibration.multi_feature = True
        cfg.calibration.multi_feature_regime_conditional = True
        cfg.calibration.histogram_post_calibration = False
        cfg.calibration.post_cal_method = "none"
        cfg.model.implied_vol_enabled = False
        cfg.model.earnings_calendar = False
        cfg.model.jump_enabled = True
        cfg.model.mc_vol_term_structure = True

    elif variant == "E_histogram_postcal":
        # + Histogram post-calibration
        cfg.model.garch_in_sim = True
        cfg.model.mc_regime_t_df = True
        cfg.model.t_df = 5.0
        cfg.calibration.multi_feature = True
        cfg.calibration.multi_feature_regime_conditional = True
        cfg.calibration.histogram_post_calibration = True
        cfg.calibration.histogram_interpolate = True
        cfg.model.implied_vol_enabled = False
        cfg.model.earnings_calendar = False
        cfg.model.jump_enabled = True
        cfg.model.mc_vol_term_structure = True

    elif variant == "F_implied_vol":
        # + Implied vol blending (full model minus earnings)
        cfg.model.garch_in_sim = True
        cfg.model.mc_regime_t_df = True
        cfg.model.t_df = 5.0
        cfg.calibration.multi_feature = True
        cfg.calibration.multi_feature_regime_conditional = True
        cfg.calibration.histogram_post_calibration = True
        cfg.calibration.histogram_interpolate = True
        cfg.model.implied_vol_enabled = True
        cfg.model.earnings_calendar = False
        cfg.model.jump_enabled = True
        cfg.model.mc_vol_term_structure = True

    elif variant == "G_full_model":
        # Full production config (no changes)
        pass

    return cfg


ABLATION_VARIANTS = [
    "A_base_gbm",
    "B_garch_in_sim",
    "C_student_t",
    "D_multi_feature_cal",
    "E_histogram_postcal",
    "F_implied_vol",
    "G_full_model",
]

VARIANT_LABELS = {
    "A_base_gbm": "Base GBM",
    "B_garch_in_sim": "+GARCH-in-Sim",
    "C_student_t": "+Student-t",
    "D_multi_feature_cal": "+MF Calibration",
    "E_histogram_postcal": "+Histogram Post-Cal",
    "F_implied_vol": "+Implied Vol",
    "G_full_model": "Full Model",
}


def run_ablation(ticker: str, n_folds: int = 5) -> pd.DataFrame:
    """Run full ablation study for a ticker."""
    config_path = TICKER_CONFIGS.get(ticker.lower())
    if not config_path:
        raise ValueError(f"Unknown ticker: {ticker}. Available: {list(TICKER_CONFIGS.keys())}")

    base_cfg = load_config(config_path)
    df, _ = load_data(base_cfg)
    horizons = base_cfg.model.horizons

    logger.info("=== Ablation Study: %s (%d rows, %d folds) ===", ticker.upper(), len(df), n_folds)

    # Skip implied vol variant if not available in base config
    variants = ABLATION_VARIANTS.copy()
    if not base_cfg.model.implied_vol_enabled:
        variants = [v for v in variants if v != "F_implied_vol"]

    configs = []
    names = []
    for variant in variants:
        cfg = _make_ablation_variant(base_cfg, variant)
        configs.append(cfg)
        names.append(variant)

    # Run CV for all variants
    cv_results, oof_df = expanding_window_cv(df, configs, names, n_folds=n_folds)

    # Compute metrics from OOF predictions
    results_rows = []
    for variant in names:
        variant_oof = oof_df[oof_df["config_name"] == variant]
        for H in horizons:
            h_oof = variant_oof[variant_oof["horizon"] == H]
            if len(h_oof) < 50:
                continue

            p = h_oof["p_cal"].to_numpy(dtype=float)
            y = h_oof["y"].to_numpy(dtype=float)
            mask = np.isfinite(p) & np.isfinite(y)
            p_m, y_m = p[mask], y[mask]

            if len(p_m) < 50:
                continue

            bss = brier_skill_score(p_m, y_m)
            auc = auc_roc(p_m, y_m)
            ece = expected_calibration_error(p_m, y_m, adaptive=False)

            # Significance vs base GBM
            base_oof = oof_df[(oof_df["config_name"] == "A_base_gbm") & (oof_df["horizon"] == H)]
            pval = np.nan
            if len(base_oof) > 0 and variant != "A_base_gbm":
                # Align by date
                if "date" in h_oof.columns and "date" in base_oof.columns:
                    merged = h_oof.merge(base_oof[["date", "p_cal", "y"]], on="date", suffixes=("", "_base"))
                    if len(merged) > 50:
                        y_merged = merged["y"].to_numpy(dtype=float)
                        model_loss = (merged["p_cal"].to_numpy(dtype=float) - y_merged) ** 2
                        base_loss = (merged["p_cal_base"].to_numpy(dtype=float) - y_merged) ** 2
                        pval = paired_bootstrap_loss_diff_pvalue(model_loss, base_loss, n_boot=2000, block_size=H)

            results_rows.append({
                "ticker": ticker.upper(),
                "variant": variant,
                "label": VARIANT_LABELS.get(variant, variant),
                "horizon": H,
                "bss": bss,
                "auc": auc,
                "ece": ece,
                "n": int(mask.sum()),
                "event_rate": float(np.mean(y_m)),
                "p_value_vs_base": pval,
                "ece_pass": ece <= 0.02,
                "bss_pass": bss > 0.0,
                "auc_pass": auc >= 0.55,
                "all_gates_pass": ece <= 0.02 and bss > 0.0 and auc >= 0.55,
            })

    results_df = pd.DataFrame(results_rows)

    # Apply FDR correction across all ablation p-values
    raw_pvals = results_df["p_value_vs_base"].tolist()
    adj_pvals, reject = apply_fdr_correction(raw_pvals, alpha=0.05)
    results_df["p_value_fdr"] = adj_pvals
    results_df["fdr_reject"] = reject

    # Save
    os.makedirs("outputs/paper", exist_ok=True)
    outpath = f"outputs/paper/ablation_{ticker.lower()}.csv"
    results_df.to_csv(outpath, index=False)
    logger.info("Ablation results saved to %s", outpath)

    # Print summary
    print(f"\n{'='*80}")
    print(f"ABLATION STUDY: {ticker.upper()}")
    print(f"{'='*80}")
    for H in horizons:
        print(f"\n--- Horizon H={H} ---")
        h_results = results_df[results_df["horizon"] == H]
        print(h_results[["label", "bss", "auc", "ece", "p_value_vs_base", "all_gates_pass"]].to_string(index=False))

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Ablation study for academic paper")
    parser.add_argument("ticker", help="Ticker symbol or 'all'")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    if args.ticker.lower() == "all":
        all_results = []
        for ticker in TICKER_CONFIGS:
            try:
                result = run_ablation(ticker, n_folds=args.n_folds)
                all_results.append(result)
            except Exception as e:
                logger.error("Ablation failed for %s: %s", ticker, e)
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            combined.to_csv("outputs/paper/ablation_all.csv", index=False)
            logger.info("Combined ablation saved to outputs/paper/ablation_all.csv")
    else:
        run_ablation(args.ticker, n_folds=args.n_folds)


if __name__ == "__main__":
    main()
