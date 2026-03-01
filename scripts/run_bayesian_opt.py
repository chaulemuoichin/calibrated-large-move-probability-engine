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
import pandas as pd
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
    "spy": "configs/exp_suite/exp_spy_regime_gated.yaml",
    "aapl": "configs/exp_suite/exp_aapl_regime_gated.yaml",
    "googl": "configs/exp_suite/exp_googl_regime_gated.yaml",
}


def _compute_threshold_ranges(base_config_path: str):
    """Compute data-adaptive threshold search ranges based on realized event rates.

    Targets 5-20% event rate range for each horizon — the regime where
    calibration is most effective. Ranges scale automatically with dataset volatility.
    """
    cfg = load_config(base_config_path)
    df, _ = load_data(cfg)
    rets = np.log(df["price"] / df["price"].shift(1)).dropna().values

    ranges = {}
    for h in cfg.model.horizons:
        fwd = pd.Series(rets).rolling(h).sum().shift(-h).dropna().values
        abs_fwd = np.abs(fwd)
        # Target: 8-20% event rate range for calibration effectiveness.
        # The 8% floor ensures N_eff/N_params > 100x with 6 lean params.
        # Use P90 (10% rate) as upper bound — provides 2% margin above the
        # 8% guard to account for CV fold sampling variation.
        thr_low = float(np.percentile(abs_fwd, 80))   # 20% event rate
        thr_high = float(np.percentile(abs_fwd, 90))   # 10% event rate
        # Small margin on lower bound only; upper bound stays tight
        ranges[h] = (round(thr_low * 0.9, 4), round(thr_high, 4))
    return ranges


def build_trial_config(trial: optuna.Trial, base_config_path: str,
                       threshold_ranges: dict = None, lean: bool = True):
    """Build a config with Optuna-suggested hyperparameters.

    Args:
        lean: If True (default), only tune 6 high-impact parameters and fix
              the rest at sensible defaults.  This improves N_eff/N_params
              ratio by ~2.3x, reducing overfitting risk.
    """
    cfg = load_config(base_config_path)

    if lean:
        # === LEAN MODE: 6 tunable parameters ===
        # Fix well-understood params at defaults
        cfg.model.hmm_regime = False
        cfg.model.hmm_vol_blend = 0.0
        cfg.model.hmm_refit_interval = 63
        cfg.model.mc_regime_t_df_low = 10.0
        cfg.model.mc_regime_t_df_mid = 5.0
        cfg.model.mc_regime_t_df_high = 4.0
        cfg.model.har_rv = False
        cfg.model.har_rv_ridge_alpha = 0.01
        cfg.model.har_rv_refit_interval = 21
        cfg.calibration.multi_feature_min_updates = 63

        # Tune only 6 params: 3 thresholds + garch_persistence + mf_lr + mf_l2
        if threshold_ranges is None:
            threshold_ranges = {5: (0.015, 0.04), 10: (0.025, 0.055), 20: (0.03, 0.06)}
        cfg.model.regime_gated_fixed_pct_by_horizon = {
            h: trial.suggest_float(f"thr_{h}", lo, hi)
            for h, (lo, hi) in threshold_ranges.items()
        }
        cfg.model.garch_target_persistence = trial.suggest_float("garch_persistence", 0.95, 0.995)
        cfg.calibration.multi_feature_lr = trial.suggest_float("mf_lr", 0.002, 0.05, log=True)
        cfg.calibration.multi_feature_l2 = trial.suggest_float("mf_l2", 1e-5, 1e-2, log=True)
    else:
        # === FULL MODE: 14 tunable parameters (original) ===
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

        # --- Per-horizon threshold percentages (data-adaptive ranges) ---
        if threshold_ranges is None:
            threshold_ranges = {5: (0.015, 0.04), 10: (0.025, 0.055), 20: (0.03, 0.06)}
        cfg.model.regime_gated_fixed_pct_by_horizon = {
            h: trial.suggest_float(f"thr_{h}", lo, hi)
            for h, (lo, hi) in threshold_ranges.items()
        }

        # --- Multi-feature calibration ---
        cfg.calibration.multi_feature_lr = trial.suggest_float("mf_lr", 0.002, 0.05, log=True)
        cfg.calibration.multi_feature_l2 = trial.suggest_float("mf_l2", 1e-5, 1e-2, log=True)
        cfg.calibration.multi_feature_min_updates = trial.suggest_int("mf_min_updates", 50, 200)

        # --- GARCH stationarity target ---
        cfg.model.garch_target_persistence = trial.suggest_float("garch_persistence", 0.95, 0.995)

        # --- HAR-RV volatility model ---
        cfg.model.har_rv = trial.suggest_categorical("har_rv", [True, False])
        if cfg.model.har_rv:
            cfg.model.har_rv_ridge_alpha = trial.suggest_float("har_rv_ridge", 0.001, 0.1, log=True)
            cfg.model.har_rv_refit_interval = trial.suggest_int("har_rv_refit", 5, 63)
        else:
            cfg.model.har_rv_ridge_alpha = 0.01
            cfg.model.har_rv_refit_interval = 21

    return cfg


def objective(trial: optuna.Trial, df, base_config_path: str,
              threshold_ranges: dict = None, lean: bool = True) -> float:
    """Objective function: minimize mean pooled ECE across horizons."""
    t0 = time.perf_counter()

    cfg = build_trial_config(trial, base_config_path, threshold_ranges, lean=lean)
    cfg_name = f"trial_{trial.number}"

    try:
        cv_results, oof_df = expanding_window_cv(df, [cfg], [cfg_name], n_folds=5)
    except Exception as e:
        # If CV fails (e.g., degenerate config), return a bad value
        print(f"  Trial {trial.number} FAILED: {e}")
        return 1.0

    # --- Adaptive minimum event-rate guard ---
    # Compute minimum event rate needed for N_eff/N_params >= 100 (GREEN).
    # N_eff ≈ event_rate × n_samples × 2, so:
    #   min_rate = (target_ratio × n_params) / (2 × n_samples)
    # Floor at 3% absolute minimum (below this, calibration is meaningless).
    # Based on Vittinghoff & McCulloch (2007): 20 events per parameter minimum.
    n_params = 6 if lean else 14
    n_oof = len(oof_df)
    min_er_target = max((100 * n_params) / (2 * n_oof), 0.03)

    er_by_horizon = cv_results.groupby("horizon")["event_rate"].mean()
    min_er = float(er_by_horizon.min())
    if min_er < min_er_target:
        low = er_by_horizon[er_by_horizon < min_er_target]
        er_str = ", ".join(f"H={int(h)}: {r:.1%}" for h, r in low.items())
        print(f"  Trial {trial.number} REJECTED: event rate too low ({er_str}) [min={min_er_target:.1%} for N_eff/params>=100x]")
        trial.set_user_attr("rejected_reason", "low_event_rate")
        trial.set_user_attr("min_event_rate", round(min_er, 4))
        trial.set_user_attr("min_er_target", round(min_er_target, 4))
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


def holdout_evaluate(
    df_holdout,
    best_params: dict,
    base_config_path: str,
    config_name: str,
) -> dict:
    """
    Evaluate best BO params on a held-out test set that was never seen during
    optimization.  Returns dict with holdout ECE per horizon and overfit flag.
    """
    cfg = load_config(base_config_path)

    # Detect lean vs full mode based on which params are present
    is_lean = "hmm_regime" not in best_params

    if is_lean:
        # Lean mode: fix non-tuned params at defaults
        cfg.model.hmm_regime = False
        cfg.model.hmm_vol_blend = 0.0
        cfg.model.hmm_refit_interval = 63
        cfg.model.mc_regime_t_df_low = 10.0
        cfg.model.mc_regime_t_df_mid = 5.0
        cfg.model.mc_regime_t_df_high = 4.0
        cfg.model.har_rv = False
        cfg.model.har_rv_ridge_alpha = 0.01
        cfg.model.har_rv_refit_interval = 21
        cfg.calibration.multi_feature_min_updates = 63
    else:
        # Full mode: apply all params
        cfg.model.hmm_regime = best_params["hmm_regime"]
        if cfg.model.hmm_regime:
            cfg.model.hmm_vol_blend = best_params.get("hmm_vol_blend", 0.0)
            cfg.model.hmm_refit_interval = best_params.get("hmm_refit_interval", 63)
        else:
            cfg.model.hmm_vol_blend = 0.0
            cfg.model.hmm_refit_interval = 63

        cfg.model.mc_regime_t_df_low = best_params["t_df_low"]
        cfg.model.mc_regime_t_df_mid = best_params["t_df_mid"]
        cfg.model.mc_regime_t_df_high = best_params["t_df_high"]
        cfg.calibration.multi_feature_min_updates = best_params["mf_min_updates"]

        cfg.model.har_rv = best_params.get("har_rv", False)
        if cfg.model.har_rv:
            cfg.model.har_rv_ridge_alpha = best_params.get("har_rv_ridge", 0.01)
            cfg.model.har_rv_refit_interval = best_params.get("har_rv_refit", 21)
        else:
            cfg.model.har_rv_ridge_alpha = 0.01
            cfg.model.har_rv_refit_interval = 21

    # Common params (both lean and full)
    cfg.model.garch_target_persistence = best_params["garch_persistence"]
    cfg.model.regime_gated_fixed_pct_by_horizon = {
        5: best_params["thr_5"],
        10: best_params["thr_10"],
        20: best_params["thr_20"],
    }
    cfg.calibration.multi_feature_lr = best_params["mf_lr"]
    cfg.calibration.multi_feature_l2 = best_params["mf_l2"]

    name = f"holdout_{config_name}"

    # Run CV on holdout data (single fold = full holdout period)
    cv_results, oof_df = expanding_window_cv(
        df_holdout, [cfg], [name], n_folds=1, min_train_pct=0.5,
    )

    # Use relaxed min_events for holdout (fewer samples than full CV)
    gates = apply_promotion_gates_oof(
        oof_df,
        gates={"bss_cal": 0.0, "auc_cal": 0.55, "ece_cal": 0.02},
        pooled_gate=True,
        min_events=2,
        min_nonevents=2,
    )

    pooled = gates[gates["regime"] == "pooled"]
    ece_rows = pooled[pooled["metric"] == "ece_cal"]

    holdout_ece = {}
    for _, row in ece_rows.iterrows():
        if not np.isnan(row["value"]):
            holdout_ece[int(row["horizon"])] = float(row["value"])

    mean_ece = np.mean(list(holdout_ece.values())) if holdout_ece else float("nan")
    return {"holdout_ece": holdout_ece, "holdout_mean_ece": mean_ece}


def run_optimization(config_name: str, n_trials: int, holdout_pct: float = 0.2,
                     lean: bool = True) -> optuna.Study:
    """Run Bayesian optimization for a single config with holdout validation."""
    config_path = CONFIGS[config_name]
    db_path = OUT_DIR / f"optuna_{config_name}.db"
    mode = "lean (6 params)" if lean else "full (14 params)"

    print(f"\n{'='*60}")
    print(f"Bayesian Optimization: {config_name} [{mode}]")
    print(f"Config: {config_path}")
    print(f"Trials: {n_trials}")
    print(f"Holdout: last {holdout_pct:.0%} of data")
    print(f"Storage: {db_path}")
    print(f"{'='*60}\n")

    # Load data once (shared across all trials)
    cfg = load_config(config_path)
    df_full, _ = load_data(cfg)

    # --- Chronological train/holdout split ---
    n_total = len(df_full)
    n_train = int(n_total * (1 - holdout_pct))
    df_train = df_full.iloc[:n_train].copy()
    df_holdout = df_full.iloc[n_train:].copy()
    print(f"Data: {n_total} total rows")
    print(f"  Train (BO search): {len(df_train)} rows [{df_train.index[0].date()} to {df_train.index[-1].date()}]")
    print(f"  Holdout (overfit check): {len(df_holdout)} rows [{df_holdout.index[0].date()} to {df_holdout.index[-1].date()}]")
    print()

    study = optuna.create_study(
        study_name=f"ece_opt_{config_name}",
        direction="minimize",
        sampler=TPESampler(seed=42, n_startup_trials=min(5, n_trials)),
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    # Compute data-adaptive threshold ranges
    threshold_ranges = _compute_threshold_ranges(config_path)
    print(f"Threshold search ranges (data-adaptive):")
    for h, (lo, hi) in sorted(threshold_ranges.items()):
        print(f"  H={h}: [{lo:.4f}, {hi:.4f}]")
    print()

    # Wrap objective with train data only
    def obj_fn(trial):
        return objective(trial, df_train, config_path, threshold_ranges, lean=lean)

    t0 = time.perf_counter()
    study.optimize(obj_fn, n_trials=n_trials)
    elapsed = (time.perf_counter() - t0) / 60

    print(f"\n{'='*60}")
    print(f"Optimization complete: {elapsed:.1f} min total")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best mean ECE (train CV): {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in sorted(study.best_params.items()):
        print(f"  {k}: {v}")
    if "n_horizons_pass" in study.best_trial.user_attrs:
        print(f"Horizons passing (train): {study.best_trial.user_attrs['n_horizons_pass']}/3")
    if "ece_per_horizon" in study.best_trial.user_attrs:
        print(f"\n  Per-horizon ECE (train CV):")
        for h, e in sorted(study.best_trial.user_attrs["ece_per_horizon"].items()):
            status = "PASS" if e <= 0.02 else "FAIL"
            print(f"    H={h}: ECE={e:.4f} [{status}]")

    # --- Holdout validation ---
    print(f"\n{'='*60}")
    print(f"HOLDOUT VALIDATION (unseen {holdout_pct:.0%} of data)")
    print(f"{'='*60}")

    holdout_result = holdout_evaluate(
        df_holdout, study.best_params, config_path, config_name,
    )

    train_ece = study.best_value
    holdout_ece = holdout_result["holdout_mean_ece"]
    if np.isnan(holdout_ece):
        print("\n  Holdout ECE: N/A (too few events in holdout period)")
        print("  This means the holdout period had very few large moves.")
        print("  The overfit check is inconclusive — consider a longer holdout.\n")
        print(f"{'='*60}\n")
        best_yaml = OUT_DIR / f"optuna_{config_name}_best.yaml"
        with open(best_yaml, "w") as f:
            yaml.dump(study.best_params, f, default_flow_style=False)
        print(f"Best params saved to: {best_yaml}")
        return study

    gap = holdout_ece - train_ece
    gap_pct = (gap / train_ece * 100) if train_ece > 0 else 0

    print(f"\n  Train CV mean ECE:   {train_ece:.4f}")
    print(f"  Holdout mean ECE:    {holdout_ece:.4f}")
    print(f"  Gap:                 {gap:+.4f} ({gap_pct:+.1f}%)")
    print()

    for h, e in sorted(holdout_result["holdout_ece"].items()):
        train_h = study.best_trial.user_attrs.get("ece_per_horizon", {}).get(str(h), None)
        status = "PASS" if e <= 0.02 else "FAIL"
        train_str = f" (train: {train_h:.4f})" if train_h is not None else ""
        print(f"  H={h}: holdout ECE={e:.4f} [{status}]{train_str}")

    # Overfit assessment
    print()
    if gap_pct > 50:
        print("  WARNING: >50% gap — likely OVERFITTING. Consider fewer trials or wider regularization.")
    elif gap_pct > 25:
        print("  CAUTION: 25-50% gap — mild overfitting. Results may not fully generalize.")
    elif gap_pct < -10:
        print("  NOTE: Holdout better than train — possible regime shift or lucky holdout period.")
    else:
        print("  OK: Gap within 25% — params appear to generalize well.")

    print(f"{'='*60}\n")

    # Save results
    study.set_user_attr("holdout_mean_ece", holdout_ece)
    study.set_user_attr("holdout_ece_per_horizon", holdout_result["holdout_ece"])
    study.set_user_attr("holdout_gap_pct", round(gap_pct, 1))

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
        val = f"{t.value:.4f}" if t.value is not None else "NaN"
        print(f"  #{t.number}: ECE={val}, pass={n_pass}/3, {elapsed} min")


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

    # Detect lean vs full mode based on which params are present
    is_lean = "hmm_regime" not in best

    if is_lean:
        # Lean mode: only 6 params were tuned, fix the rest
        raw["model"]["hmm_regime"] = False
        raw["model"]["hmm_vol_blend"] = 0.0
        raw["model"]["hmm_refit_interval"] = 63
        raw["model"]["mc_regime_t_df_low"] = 10.0
        raw["model"]["mc_regime_t_df_mid"] = 5.0
        raw["model"]["mc_regime_t_df_high"] = 4.0
        raw["model"]["har_rv"] = False
        raw["model"]["har_rv_ridge_alpha"] = 0.01
        raw["model"]["har_rv_refit_interval"] = 21
        raw["calibration"]["multi_feature_min_updates"] = 63
    else:
        # Full mode: apply all params
        raw["model"]["hmm_regime"] = best["hmm_regime"]
        if best["hmm_regime"]:
            raw["model"]["hmm_vol_blend"] = round(best["hmm_vol_blend"], 4)
            raw["model"]["hmm_refit_interval"] = best["hmm_refit_interval"]
        else:
            raw["model"]["hmm_vol_blend"] = 0.0

        raw["model"]["mc_regime_t_df_low"] = round(best["t_df_low"], 1)
        raw["model"]["mc_regime_t_df_mid"] = round(best["t_df_mid"], 1)
        raw["model"]["mc_regime_t_df_high"] = round(best["t_df_high"], 1)
        raw["calibration"]["multi_feature_min_updates"] = best["mf_min_updates"]

        raw["model"]["har_rv"] = best.get("har_rv", False)
        if best.get("har_rv", False):
            raw["model"]["har_rv_ridge_alpha"] = round(best.get("har_rv_ridge", 0.01), 4)
            raw["model"]["har_rv_refit_interval"] = best.get("har_rv_refit", 21)
        else:
            raw["model"]["har_rv"] = False

    # Common params (both lean and full)
    raw["model"]["garch_target_persistence"] = round(best["garch_persistence"], 4)
    raw["model"]["regime_gated_fixed_pct_by_horizon"] = {
        5: round(best["thr_5"], 4),
        10: round(best["thr_10"], 4),
        20: round(best["thr_20"], 4),
    }
    raw["calibration"]["multi_feature_lr"] = round(best["mf_lr"], 6)
    raw["calibration"]["multi_feature_l2"] = round(best["mf_l2"], 6)

    with open(config_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

    mode = "lean (6 params)" if is_lean else "full (14 params)"
    print(f"Applied best params ({mode}) to {config_path}")
    print(f"Best mean ECE: {study.best_value:.4f}")
    print("Run gate recheck to confirm:")
    print(f"  python scripts/run_gate_recheck.py {config_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bayesian optimization with Optuna")
    parser.add_argument("config", choices=list(CONFIGS.keys()), help="Config to optimize")
    parser.add_argument("--n-trials", type=int, default=15, help="Number of trials")
    parser.add_argument("--show-best", action="store_true", help="Show best results")
    parser.add_argument("--apply", action="store_true", help="Apply best params to config")
    parser.add_argument("--full", action="store_true",
                        help="Use full 14-param search (default: lean 6-param)")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.show_best:
        show_best(args.config)
        return 0

    if args.apply:
        apply_best(args.config)
        return 0

    run_optimization(args.config, args.n_trials, lean=not args.full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
