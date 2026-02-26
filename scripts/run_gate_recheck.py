"""
Re-run CV gates for the two best configs after P0 fixes:
  - ECE gate now uses equal-width binning (threshold 0.02 was calibrated for it)
  - Histogram post-calibration is active
  - Adaptive ECE reported as diagnostic column

Usage:
    python -u scripts/run_gate_recheck.py [jump|cluster]
"""

from __future__ import annotations

import sys
import time
import warnings
import logging

import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from em_sde.config import load_config
from em_sde.data_layer import load_data
from em_sde.model_selection import (
    expanding_window_cv, compare_models,
    apply_promotion_gates, apply_promotion_gates_oof,
)

OUT_DIR = Path("outputs/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_single_config(name: str, config_path: str) -> None:
    t0 = time.perf_counter()
    print(f"\n=== CV {name} (equal-width ECE for gates, histogram post-cal ON) ===")

    cfg = load_config(config_path)
    cfg_name = Path(config_path).stem
    df, _ = load_data(cfg)

    cv_results, oof_df = expanding_window_cv(df, [cfg], [cfg_name], n_folds=5)
    summary = compare_models(cv_results)

    # Primary: row-level OOF gates (statistically valid, pooled across regimes)
    gates_oof = apply_promotion_gates_oof(
        oof_df,
        gates={"bss_cal": 0.0, "auc_cal": 0.55, "ece_cal": 0.02},
        pooled_gate=True,
    )
    # Legacy: fold-level gates (for comparison)
    gates_legacy = apply_promotion_gates(
        cv_results,
        gates={"bss_cal": 0.0, "auc_cal": 0.55, "ece_cal": 0.02},
    )

    # Save results
    cv_results.to_csv(OUT_DIR / f"cv_{name}_folds_recheck.csv", index=False)
    summary.to_csv(OUT_DIR / f"cv_{name}_summary_recheck.csv", index=False)
    gates_oof.to_csv(OUT_DIR / f"cv_{name}_gates_recheck.csv", index=False)
    gates_legacy.to_csv(OUT_DIR / f"cv_{name}_gates_legacy.csv", index=False)

    elapsed = (time.perf_counter() - t0) / 60.0

    # Print OOF gate report (primary)
    print(f"\nElapsed: {elapsed:.1f} min\n")
    print("=== OOF Gates (pooled=primary, per-regime=diagnostic) ===\n")
    for (cfg_n, H), grp in gates_oof.groupby(["config_name", "horizon"]):
        promo = grp["promotion_status"].iloc[0] if "promotion_status" in grp.columns else "?"
        print(f"{cfg_n} H={H}: {promo}")
        # Show pooled (primary) first, then per-regime (diagnostic)
        for _, row in grp.sort_values("regime", key=lambda s: s.map(lambda x: "0" if x == "pooled" else x)).iterrows():
            metric = row["metric"]
            value = row["value"]
            threshold = row["threshold"]
            passed = row["passed"]
            margin = row["margin"]
            n_s = row.get("n_samples", "?")
            n_e = row.get("n_events", "?")
            status = row.get("status", "?")
            prefix = "*" if status == "evaluated" else " "
            if status == "insufficient_data":
                reason = row.get("insufficient_reason", "?")
                n_ne = row.get("n_nonevents", "?")
                tag = f"[SKIP: {reason}]"
                print(f" {prefix}{row['regime']:>8s}/{metric} = n/a  (n={n_s}, events={n_e}, nonevents={n_ne}) {tag}")
            else:
                tag = "[OK]" if passed else "[FAIL]"
                diag = " (diag)" if status == "diagnostic" else ""
                ci_str = ""
                if metric == "ece_cal" and not np.isnan(row.get("ece_ci_low", np.nan)):
                    ci_str = f"  CI=[{row['ece_ci_low']:.4f}, {row['ece_ci_high']:.4f}]"
                print(f" {prefix}{row['regime']:>8s}/{metric} = {value:+.6f}  (thr {threshold}, margin {margin:+.6f}, n={n_s}, events={n_e}){ci_str} {tag}{diag}")

    n_pass = gates_oof.groupby(["config_name", "horizon"])["all_gates_passed"].all().sum()
    n_undecided = (gates_oof.groupby(["config_name", "horizon"])["promotion_status"].first() == "UNDECIDED").sum() if "promotion_status" in gates_oof.columns else 0
    n_total = gates_oof.groupby(["config_name", "horizon"]).ngroups
    print(f"\nOOF gate: {n_pass}/{n_total} PASS, {n_undecided}/{n_total} UNDECIDED")

    # Shadow gate at ECE<=0.04 (reporting only, no policy change)
    shadow_gates = apply_promotion_gates_oof(
        oof_df,
        gates={"bss_cal": 0.0, "auc_cal": 0.55, "ece_cal": 0.04},
    )
    shadow_pass = shadow_gates.groupby(["config_name", "horizon"])["all_gates_passed"].all().sum()
    shadow_total = shadow_gates.groupby(["config_name", "horizon"]).ngroups
    if shadow_pass > n_pass:
        print(f"\n--- Shadow gate (ECE<=0.04): {shadow_pass}/{shadow_total} would pass ---")
        for (cfg_n, H), grp in shadow_gates.groupby(["config_name", "horizon"]):
            if grp["all_gates_passed"].all():
                print(f"  {cfg_n} H={H}: SHADOW PASS")


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "both"

    all_configs = {
        "jump": "configs/exp_suite/exp_jump_regime_gated.yaml",
        "cluster": "configs/exp_suite/exp_cluster_regime_gated.yaml",
        "spy": "configs/exp_suite/exp_spy_regime_gated.yaml",
        "aapl": "configs/exp_suite/exp_aapl_regime_gated.yaml",
        "googl": "configs/exp_suite/exp_googl_regime_gated.yaml",
    }

    configs = {}
    if target == "both":
        configs = {k: v for k, v in all_configs.items() if k in ("jump", "cluster")}
    elif target == "all":
        configs = all_configs
    elif target in all_configs:
        configs[target] = all_configs[target]

    t_all = time.perf_counter()
    for name, path in configs.items():
        run_single_config(name, path)

    print(f"\n=== DONE ({(time.perf_counter() - t_all) / 60.0:.1f} min total) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
