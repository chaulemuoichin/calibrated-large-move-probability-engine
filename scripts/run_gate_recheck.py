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

    # Primary: row-level OOF gates (statistically valid)
    gates_oof = apply_promotion_gates_oof(
        oof_df,
        gates={"bss_cal": 0.0, "auc_cal": 0.55, "ece_cal": 0.02},
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
    print("=== OOF Row-Level Gates (primary) ===\n")
    for (cfg_n, H), grp in gates_oof.groupby(["config_name", "horizon"]):
        all_pass = grp["all_gates_passed"].all()
        status = "PASS" if all_pass else "FAIL"
        print(f"{cfg_n} H={H}: {status}")
        for _, row in grp.iterrows():
            metric = row["metric"]
            value = row["value"]
            threshold = row["threshold"]
            passed = row["passed"]
            margin = row["margin"]
            n_s = row.get("n_samples", "?")
            n_e = row.get("n_events", "?")
            if row.get("status") == "insufficient_data":
                tag = "[SKIP: insufficient data]"
                print(f"  {row['regime']:>8s}/{metric} = n/a  (n={n_s}, events={n_e}) {tag}")
            else:
                tag = "[OK]" if passed else "[FAIL]"
                ci_str = ""
                if metric == "ece_cal" and not np.isnan(row.get("ece_ci_low", np.nan)):
                    ci_str = f"  CI=[{row['ece_ci_low']:.4f}, {row['ece_ci_high']:.4f}]"
                print(f"  {row['regime']:>8s}/{metric} = {value:+.6f}  (thr {threshold}, margin {margin:+.6f}, n={n_s}, events={n_e}){ci_str} {tag}")

    n_pass = gates_oof.groupby(["config_name", "horizon"])["all_gates_passed"].all().sum()
    n_total = gates_oof.groupby(["config_name", "horizon"]).ngroups
    print(f"\nOOF gate pass: {n_pass}/{n_total} config-horizon combos")

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

    configs = {}
    if target in ("jump", "both"):
        configs["jump"] = "configs/exp_suite/exp_jump_regime_gated.yaml"
    if target in ("cluster", "both"):
        configs["cluster"] = "configs/exp_suite/exp_cluster_regime_gated.yaml"

    t_all = time.perf_counter()
    for name, path in configs.items():
        run_single_config(name, path)

    print(f"\n=== DONE ({(time.perf_counter() - t_all) / 60.0:.1f} min total) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
