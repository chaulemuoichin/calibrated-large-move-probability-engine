"""
Run only the remaining CV families (jump + trend) and write diagnostics CSVs.

Usage:
    python -u run_remaining_cv.py
"""

from __future__ import annotations

import time
from pathlib import Path

from em_sde.config import load_config
from em_sde.data_layer import load_data
from em_sde.model_selection import expanding_window_cv, compare_models, apply_promotion_gates


OUT_DIR = Path("outputs/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_family(name: str, paths: list[str]) -> None:
    t0 = time.perf_counter()
    print(f"\n=== CV {name.upper()} ===")
    configs = [load_config(p) for p in paths]
    names = [Path(p).stem for p in paths]
    df, _ = load_data(configs[0])

    cv_results = expanding_window_cv(df, configs, names, n_folds=5)
    summary = compare_models(cv_results)
    gates = apply_promotion_gates(
        cv_results,
        gates={"bss_cal": 0.0, "auc_cal": 0.55, "ece_cal": 0.02},
    )

    cv_path = OUT_DIR / f"cv_{name}_folds.csv"
    summary_path = OUT_DIR / f"cv_{name}_summary.csv"
    gates_path = OUT_DIR / f"cv_{name}_gates.csv"

    cv_results.to_csv(cv_path, index=False)
    summary.to_csv(summary_path, index=False)
    gates.to_csv(gates_path, index=False)

    print(f"Saved: {cv_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {gates_path}")
    print(f"Elapsed: {(time.perf_counter() - t0) / 60.0:.1f} min")


def main() -> int:
    families = {
        "jump": [
            "configs/exp_suite/exp_jump_legacy.yaml",
            "configs/exp_suite/exp_jump_dyn_volscaled.yaml",
            "configs/exp_suite/exp_jump_inst_fixed_multi.yaml",
            "configs/exp_suite/exp_jump_regime_gated.yaml",
        ],
        "trend": [
            "configs/exp_suite/exp_trend_legacy.yaml",
            "configs/exp_suite/exp_trend_dyn_volscaled.yaml",
            "configs/exp_suite/exp_trend_inst_fixed_multi.yaml",
            "configs/exp_suite/exp_trend_regime_gated.yaml",
        ],
    }

    t_all = time.perf_counter()
    for name, paths in families.items():
        run_family(name, paths)

    print("\n=== DONE ===")
    print(f"Total elapsed: {(time.perf_counter() - t_all) / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
