"""
Full institutional validation battery runner.

Stages:
1) Unit tests (pytest)
2) Quick validation (6 configs)
3) Stress suite (12 configs)
4) Multi-seed stability (30 runs = 6 configs x 5 seeds)
5) CV + promotion gates per pattern family

Usage:
    python -u run_full_institutional.py
"""

from __future__ import annotations

import subprocess
import sys
import time
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from em_sde.config import load_config
from em_sde.data_layer import load_data
from em_sde.backtest import run_walkforward
from em_sde.evaluation import compute_metrics
from em_sde.model_selection import (
    expanding_window_cv,
    compare_models,
    apply_promotion_gates,
)


warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


OUT_DIR = Path("outputs/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _run_subprocess_stage(name: str, cmd: list[str]) -> None:
    """Run a subprocess stage and save stdout/stderr log."""
    log_path = OUT_DIR / f"{name}.log"
    t0 = time.perf_counter()
    print(f"\n=== {name} ===")
    print("cmd:", " ".join(cmd))

    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {proc.returncode}. See {log_path}"
        )
    print(f"{name} completed in {elapsed / 60.0:.1f} min")


def _stage_seed_stability() -> Path:
    """30-run stability test: 6 configs x 5 seeds."""
    configs = [
        "configs/exp_suite/exp_cluster_inst_fixed_multi.yaml",
        "configs/exp_suite/exp_cluster_regime_gated.yaml",
        "configs/exp_suite/exp_jump_inst_fixed_multi.yaml",
        "configs/exp_suite/exp_jump_regime_gated.yaml",
        "configs/exp_suite/exp_trend_inst_fixed_multi.yaml",
        "configs/exp_suite/exp_trend_regime_gated.yaml",
    ]
    seeds = [11, 22, 33, 44, 55]

    rows: list[dict[str, float | int | str]] = []
    t0 = time.perf_counter()
    print("\n=== 03_stage1_seed_stability ===")
    print(f"Runs: {len(configs)} configs x {len(seeds)} seeds = {len(configs) * len(seeds)}")

    for i, seed in enumerate(seeds, 1):
        print(f"  seed {seed} ({i}/{len(seeds)})")
        for cfg_path in configs:
            cfg = load_config(cfg_path)
            cfg.model.seed = seed
            cfg.model.mc_base_paths = 1000
            cfg.model.mc_boost_paths = 2000

            df, _ = load_data(cfg)
            results = run_walkforward(df, cfg)
            metrics = compute_metrics(results, cfg.model.horizons)

            for H in cfg.model.horizons:
                m = metrics["overlapping"].get(H, {})
                row: dict[str, float | int | str] = {
                    "seed": seed,
                    "config": Path(cfg_path).stem,
                    "ticker": cfg.data.ticker,
                    "threshold_mode": cfg.model.threshold_mode,
                    "horizon": H,
                    "n": int(m.get("n", 0)),
                    "brier_cal": float(m.get("brier_cal", np.nan)),
                    "bss_cal": float(m.get("bss_cal", np.nan)),
                    "auc_cal": float(m.get("auc_cal", np.nan)),
                    "separation_cal": float(m.get("separation_cal", np.nan)),
                    "event_rate": float(m.get("event_rate", np.nan)),
                }

                if "garch_projected" in results.columns:
                    gp = results["garch_projected"].astype(bool)
                    row["garch_projected_pct"] = float(gp.mean() * 100.0)

                if "jump_intensity_step" in results.columns:
                    ji = results["jump_intensity_step"].dropna()
                    if len(ji) > 0:
                        row["jump_intensity_mean"] = float(ji.mean())
                        row["jump_intensity_min"] = float(ji.min())
                        row["jump_intensity_max"] = float(ji.max())

                rows.append(row)

    out_path = OUT_DIR / "stage1_seed_stability.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    elapsed = time.perf_counter() - t0
    print(f"03_stage1_seed_stability completed in {elapsed / 60.0:.1f} min")
    print(f"saved: {out_path}")
    return out_path


def _stage_cv_and_gates() -> None:
    """CV + promotion-gate reports for cluster/jump/trend families."""
    families = {
        "cluster": [
            "configs/exp_suite/exp_cluster_legacy.yaml",
            "configs/exp_suite/exp_cluster_dyn_volscaled.yaml",
            "configs/exp_suite/exp_cluster_inst_fixed_multi.yaml",
            "configs/exp_suite/exp_cluster_regime_gated.yaml",
        ],
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

    print("\n=== 04_cv_and_promotion_gates ===")
    for family, paths in families.items():
        t0 = time.perf_counter()
        configs = [load_config(p) for p in paths]
        names = [Path(p).stem for p in paths]

        df, _ = load_data(configs[0])
        cv_results = expanding_window_cv(df, configs, names, n_folds=5)
        summary = compare_models(cv_results)
        gate_report = apply_promotion_gates(
            cv_results,
            gates={"bss_cal": 0.0, "auc_cal": 0.55, "ece_cal": 0.02},
        )

        cv_results.to_csv(OUT_DIR / f"cv_{family}_folds.csv", index=False)
        summary.to_csv(OUT_DIR / f"cv_{family}_summary.csv", index=False)
        gate_report.to_csv(OUT_DIR / f"cv_{family}_gates.csv", index=False)

        elapsed = time.perf_counter() - t0
        print(f"  {family}: {elapsed / 60.0:.1f} min")


def main() -> int:
    t_all = time.perf_counter()
    py = sys.executable

    try:
        _run_subprocess_stage("00_pytest", [py, "-m", "pytest", "tests", "-q"])
        _run_subprocess_stage("01_quick_validation", [py, "-u", "run_quick_validation.py"])
        _run_subprocess_stage("02_stress_suite", [py, "-u", "run_stress_suite.py"])
        _stage_seed_stability()
        _stage_cv_and_gates()
    except Exception as exc:
        print(f"\nFAILED: {exc}")
        return 1

    elapsed_all = time.perf_counter() - t_all
    print("\n=== COMPLETE ===")
    print(f"total elapsed: {elapsed_all / 3600.0:.2f} hours")
    print(f"outputs: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
