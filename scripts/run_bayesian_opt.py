"""
Bayesian optimization of hyperparameters using Optuna TPE sampler.

Searches over HMM, threshold, calibration, and GARCH parameters to
minimize mean pooled ECE across horizons via expanding-window CV.

Usage:
    python scripts/run_bayesian_opt.py cluster --n-trials 15
    python scripts/run_bayesian_opt.py jump --n-trials 15
    python scripts/run_bayesian_opt.py cluster --show-best
    python scripts/run_bayesian_opt.py cluster --apply
"""

from __future__ import annotations

import sys
import time
import warnings
import logging
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna
from optuna.samplers import TPESampler

from em_sde.config import load_config
from em_sde.data_layer import load_data
from em_sde.model_selection import expanding_window_cv, apply_promotion_gates_oof

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "cluster": "configs/exp_suite/exp_cluster_regime_gated.yaml",
    "jump": "configs/exp_suite/exp_jump_regime_gated.yaml",
}


def build_trial_config(trial: optuna.Trial, base_config_path: str):
    """Build a config with Optuna-suggested hyperparameters."""
    cfg = load_config(base_config_path)

    # --- HMM regime detection ---
    cfg.model.hmm_regime = trial.suggest_categorical("hmm_regime", [True, False])
    if cfg.model.hmm_regime:
        cfg.model.hmm_vol_blend = trial.suggest_float("hmm_vol_blend", 0.0, 0.6)
        cfg.model.hmm_refit_interval = trial.suggest_int("hmm_refit_interval", 21, 126)
    else:
        cfg.model.hmm_vol_blend = 0.0
        cfg.model.hmm_refit_interval = 63

    # --- Regime-conditional Student-t df ---
    cfg.model.mc_regime_t_df_low = trial.suggest_float("t_df_low", 5.0, 15.0)
    cfg.model.mc_regime_t_df_mid = trial.suggest_float("t_df_mid", 3.0, 8.0)
    cfg.model.mc_regime_t_df_high = trial.suggest_float("t_df_high", 2.5, 6.0)

    # --- Per-horizon threshold percentages ---
    cfg.model.regime_gated_fixed_pct_by_horizon = {
        5: trial.suggest_float("thr_5", 0.015, 0.04),
        10: trial.suggest_float("thr_10", 0.025, 0.055),
        20: trial.suggest_float("thr_20", 0.03, 0.06),
    }

    # --- Multi-feature calibration ---
    cfg.calibration.multi_feature_lr = trial.suggest_float("mf_lr", 0.002, 0.05, log=True)
    cfg.calibration.multi_feature_l2 = trial.suggest_float("mf_l2", 1e-5, 1e-2, log=True)
    cfg.calibration.multi_feature_min_updates = trial.suggest_int("mf_min_updates", 50, 200)

    # --- GARCH stationarity target ---
    cfg.model.garch_target_persistence = trial.suggest_float("garch_persistence", 0.95, 0.995)

    return cfg


def objective(trial: optuna.Trial, df, base_config_path: str) -> float:
    """Objective function: minimize mean pooled ECE across horizons."""
    t0 = time.perf_counter()

    cfg = build_trial_config(trial, base_config_path)
    cfg_name = f"trial_{trial.number}"

    try:
        cv_results, oof_df = expanding_window_cv(df, [cfg], [cfg_name], n_folds=5)
    except Exception as e:
        # If CV fails (e.g., degenerate config), return a bad value
        print(f"  Trial {trial.number} FAILED: {e}")
        return 1.0

    # Compute pooled ECE gates
    gates = apply_promotion_gates_oof(
        oof_df,
        gates={"bss_cal": 0.0, "auc_cal": 0.55, "ece_cal": 0.02},
        pooled_gate=True,
    )

    # Extract pooled ECE per horizon
    pooled = gates[gates["regime"] == "pooled"]
    ece_rows = pooled[pooled["metric"] == "ece_cal"]

    if len(ece_rows) == 0:
        return 1.0

    mean_ece = float(ece_rows["value"].mean())

    # Track secondary metrics
    n_pass = int(
        gates.groupby(["config_name", "horizon"])["all_gates_passed"].all().sum()
    )
    ece_dict = {
        int(row["horizon"]): float(row["value"])
        for _, row in ece_rows.iterrows()
    }

    trial.set_user_attr("n_horizons_pass", n_pass)
    trial.set_user_attr("ece_per_horizon", ece_dict)
    trial.set_user_attr("elapsed_min", round((time.perf_counter() - t0) / 60, 1))

    # Report per-horizon ECE
    ece_str = ", ".join(f"H={h}: {e:.4f}" for h, e in sorted(ece_dict.items()))
    print(
        f"  Trial {trial.number}: mean_ECE={mean_ece:.4f}, "
        f"pass={n_pass}/3, [{ece_str}] "
        f"({trial.user_attrs['elapsed_min']:.1f} min)"
    )

    return mean_ece


def run_optimization(config_name: str, n_trials: int) -> optuna.Study:
    """Run Bayesian optimization for a single config."""
    config_path = CONFIGS[config_name]
    db_path = OUT_DIR / f"optuna_{config_name}.db"

    print(f"\n{'='*60}")
    print(f"Bayesian Optimization: {config_name}")
    print(f"Config: {config_path}")
    print(f"Trials: {n_trials}")
    print(f"Storage: {db_path}")
    print(f"{'='*60}\n")

    # Load data once (shared across all trials)
    cfg = load_config(config_path)
    df, _ = load_data(cfg)
    print(f"Data: {len(df)} rows\n")

    study = optuna.create_study(
        study_name=f"ece_opt_{config_name}",
        direction="minimize",
        sampler=TPESampler(seed=42, n_startup_trials=min(5, n_trials)),
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    # Wrap objective with bound args
    def obj_fn(trial):
        return objective(trial, df, config_path)

    t0 = time.perf_counter()
    study.optimize(obj_fn, n_trials=n_trials)
    elapsed = (time.perf_counter() - t0) / 60

    print(f"\n{'='*60}")
    print(f"Optimization complete: {elapsed:.1f} min total")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best mean ECE: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in sorted(study.best_params.items()):
        print(f"  {k}: {v}")
    if "n_horizons_pass" in study.best_trial.user_attrs:
        print(f"Horizons passing: {study.best_trial.user_attrs['n_horizons_pass']}/3")
    if "ece_per_horizon" in study.best_trial.user_attrs:
        for h, e in sorted(study.best_trial.user_attrs["ece_per_horizon"].items()):
            status = "PASS" if e <= 0.02 else "FAIL"
            print(f"  H={h}: ECE={e:.4f} [{status}]")
    print(f"{'='*60}\n")

    # Save best params as YAML
    best_yaml = OUT_DIR / f"optuna_{config_name}_best.yaml"
    with open(best_yaml, "w") as f:
        yaml.dump(study.best_params, f, default_flow_style=False)
    print(f"Best params saved to: {best_yaml}")

    return study


def show_best(config_name: str) -> None:
    """Show best results from a previous optimization run."""
    db_path = OUT_DIR / f"optuna_{config_name}.db"
    if not db_path.exists():
        print(f"No study found at {db_path}")
        return

    study = optuna.load_study(
        study_name=f"ece_opt_{config_name}",
        storage=f"sqlite:///{db_path}",
    )

    print(f"\n{'='*60}")
    print(f"Study: {config_name} ({len(study.trials)} trials)")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best mean ECE: {study.best_value:.4f}")
    print(f"\nBest params:")
    for k, v in sorted(study.best_params.items()):
        print(f"  {k}: {v}")
    if "ece_per_horizon" in study.best_trial.user_attrs:
        print(f"\nPer-horizon ECE:")
        for h, e in sorted(study.best_trial.user_attrs["ece_per_horizon"].items()):
            status = "PASS" if e <= 0.02 else "FAIL"
            print(f"  H={h}: ECE={e:.4f} [{status}]")
    print(f"{'='*60}\n")

    # Show top 5 trials
    print("Top 5 trials:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else 999)
    for t in trials_sorted[:5]:
        n_pass = t.user_attrs.get("n_horizons_pass", "?")
        elapsed = t.user_attrs.get("elapsed_min", "?")
        print(f"  #{t.number}: ECE={t.value:.4f}, pass={n_pass}/3, {elapsed} min")


def apply_best(config_name: str) -> None:
    """Apply best params to the YAML config file."""
    db_path = OUT_DIR / f"optuna_{config_name}.db"
    if not db_path.exists():
        print(f"No study found at {db_path}")
        return

    study = optuna.load_study(
        study_name=f"ece_opt_{config_name}",
        storage=f"sqlite:///{db_path}",
    )

    config_path = CONFIGS[config_name]
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    best = study.best_params

    # Apply best params to config
    raw["model"]["hmm_regime"] = best["hmm_regime"]
    if best["hmm_regime"]:
        raw["model"]["hmm_vol_blend"] = round(best["hmm_vol_blend"], 4)
        raw["model"]["hmm_refit_interval"] = best["hmm_refit_interval"]
    else:
        raw["model"]["hmm_vol_blend"] = 0.0

    raw["model"]["mc_regime_t_df_low"] = round(best["t_df_low"], 1)
    raw["model"]["mc_regime_t_df_mid"] = round(best["t_df_mid"], 1)
    raw["model"]["mc_regime_t_df_high"] = round(best["t_df_high"], 1)
    raw["model"]["garch_target_persistence"] = round(best["garch_persistence"], 4)

    raw["model"]["regime_gated_fixed_pct_by_horizon"] = {
        5: round(best["thr_5"], 4),
        10: round(best["thr_10"], 4),
        20: round(best["thr_20"], 4),
    }

    raw["calibration"]["multi_feature_lr"] = round(best["mf_lr"], 6)
    raw["calibration"]["multi_feature_l2"] = round(best["mf_l2"], 6)
    raw["calibration"]["multi_feature_min_updates"] = best["mf_min_updates"]

    with open(config_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

    print(f"Applied best params to {config_path}")
    print(f"Best mean ECE: {study.best_value:.4f}")
    print("Run gate recheck to confirm:")
    print(f"  python scripts/run_gate_recheck.py {config_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bayesian optimization with Optuna")
    parser.add_argument("config", choices=["cluster", "jump"], help="Config to optimize")
    parser.add_argument("--n-trials", type=int, default=15, help="Number of trials")
    parser.add_argument("--show-best", action="store_true", help="Show best results")
    parser.add_argument("--apply", action="store_true", help="Apply best params to config")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.show_best:
        show_best(args.config)
        return 0

    if args.apply:
        apply_best(args.config)
        return 0

    run_optimization(args.config, args.n_trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
