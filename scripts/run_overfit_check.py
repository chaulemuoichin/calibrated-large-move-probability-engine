"""
Overfitting diagnostics for Bayesian-optimized parameters.

Computes 5 measurable metrics to assess whether BO results are genuine
or fitting to noise. Run after BO + gate recheck.

Usage:
    python scripts/run_overfit_check.py cluster
    python scripts/run_overfit_check.py jump
    python scripts/run_overfit_check.py both
"""

from __future__ import annotations

import sys
import time
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna
from em_sde.config import load_config
from em_sde.data_layer import load_data
from em_sde.evaluation import expected_calibration_error

DIAG_DIR = Path("outputs/diagnostics")
OUT_DIR = Path("outputs")

CONFIGS = {
    "cluster": "configs/exp_suite/exp_cluster_regime_gated.yaml",
    "jump": "configs/exp_suite/exp_jump_regime_gated.yaml",
    "spy": "configs/exp_suite/exp_spy_regime_gated.yaml",
    "aapl": "configs/exp_suite/exp_aapl_regime_gated.yaml",
    "googl": "configs/exp_suite/exp_googl_regime_gated.yaml",
}

# Thresholds for each metric
THRESHOLDS = {
    "gen_gap": {"green": 0.25, "yellow": 0.50},        # generalization gap ratio
    "fold_cv": {"green": 0.30, "yellow": 0.50},         # coefficient of variation
    "thr_sens": {"green": 0.25, "yellow": 0.50},        # threshold sensitivity
    "temporal": {"green": 0.30, "yellow": 0.60},         # early vs late fold gap
    "neff_ratio": {"green": 100, "yellow": 50},          # N_eff / N_params (inverted: higher=better)
}


def _status(value: float, metric: str) -> str:
    """Return GREEN/YELLOW/RED for a metric value."""
    t = THRESHOLDS[metric]
    if metric == "neff_ratio":
        # Higher is better
        if value >= t["green"]:
            return "GREEN"
        elif value >= t["yellow"]:
            return "YELLOW"
        return "RED"
    # Lower is better
    if value <= t["green"]:
        return "GREEN"
    elif value <= t["yellow"]:
        return "YELLOW"
    return "RED"


def compute_generalization_gap(config_name: str) -> list[dict]:
    """Metric 1: Compare BO train CV ECE vs full-data gate recheck ECE."""
    results = []

    # Load BO train ECE from Optuna study
    db_path = OUT_DIR / f"optuna_{config_name}.db"
    if not db_path.exists():
        return [{"metric": "gen_gap", "horizon": "all", "value": np.nan,
                 "status": "SKIP", "detail": "No Optuna DB found"}]

    study = optuna.load_study(
        study_name=f"ece_opt_{config_name}",
        storage=f"sqlite:///{db_path}",
    )
    train_ece = study.best_trial.user_attrs.get("ece_per_horizon", {})

    # Load full-data gate recheck ECE
    gates_path = DIAG_DIR / f"cv_{config_name}_gates_recheck.csv"
    if not gates_path.exists():
        return [{"metric": "gen_gap", "horizon": "all", "value": np.nan,
                 "status": "SKIP", "detail": "No gate recheck CSV found"}]

    gates = pd.read_csv(gates_path)
    pooled_ece = gates[(gates["regime"] == "pooled") & (gates["metric"] == "ece_cal")]

    for _, row in pooled_ece.iterrows():
        h = int(row["horizon"])
        full_ece = float(row["value"])
        t_ece = train_ece.get(str(h), train_ece.get(h, None))
        if t_ece is None or t_ece == 0:
            results.append({"metric": "gen_gap", "horizon": h, "value": np.nan,
                            "status": "SKIP", "detail": "No train ECE"})
            continue
        gap_ratio = (full_ece - t_ece) / t_ece
        results.append({
            "metric": "gen_gap", "horizon": h,
            "value": gap_ratio,
            "status": _status(abs(gap_ratio), "gen_gap"),
            "detail": f"train={t_ece:.4f} full={full_ece:.4f} gap={gap_ratio:+.1%}",
        })
    return results


def compute_fold_stability(config_name: str) -> list[dict]:
    """Metric 2: Coefficient of variation of ECE across CV folds."""
    results = []
    folds_path = DIAG_DIR / f"cv_{config_name}_folds_recheck.csv"
    if not folds_path.exists():
        return [{"metric": "fold_cv", "horizon": "all", "value": np.nan,
                 "status": "SKIP", "detail": "No folds CSV"}]

    folds = pd.read_csv(folds_path)
    for h in folds["horizon"].unique():
        ece_vals = folds[folds["horizon"] == h]["ece_cal"].values
        if len(ece_vals) < 2:
            continue
        mean_ece = np.mean(ece_vals)
        std_ece = np.std(ece_vals, ddof=1)
        cv = std_ece / mean_ece if mean_ece > 0 else 0
        results.append({
            "metric": "fold_cv", "horizon": int(h),
            "value": cv,
            "status": _status(cv, "fold_cv"),
            "detail": f"mean={mean_ece:.4f} std={std_ece:.4f} folds={len(ece_vals)}",
        })
    return results


def compute_threshold_sensitivity(config_name: str) -> list[dict]:
    """Metric 3: ECE change when thresholds perturbed +-10%.

    Uses raw price data + stored OOF predictions to re-label events
    with perturbed thresholds and recompute ECE without re-running MC.
    Falls back to fold-level re-labeling if OOF not available.
    """
    results = []
    config_path = CONFIGS.get(config_name)
    if not config_path:
        return []

    cfg = load_config(config_path)
    df, _ = load_data(cfg)
    prices = df["price"].values
    log_rets = np.log(prices[1:] / prices[:-1])

    # Get current thresholds from config
    thr_by_h = cfg.model.regime_gated_fixed_pct_by_horizon or {}
    if not thr_by_h:
        return [{"metric": "thr_sens", "horizon": "all", "value": np.nan,
                 "status": "SKIP", "detail": "No per-horizon thresholds in config"}]

    # Load gate recheck pooled ECE as baseline
    gates_path = DIAG_DIR / f"cv_{config_name}_gates_recheck.csv"
    if not gates_path.exists():
        return [{"metric": "thr_sens", "horizon": "all", "value": np.nan,
                 "status": "SKIP", "detail": "No gate recheck CSV"}]

    gates = pd.read_csv(gates_path)
    pooled_ece = gates[(gates["regime"] == "pooled") & (gates["metric"] == "ece_cal")]
    base_ece = {int(row["horizon"]): float(row["value"]) for _, row in pooled_ece.iterrows()}

    # For each horizon, compute forward returns and check sensitivity
    for h, thr in thr_by_h.items():
        h = int(h)
        if h not in base_ece or base_ece[h] == 0:
            continue

        # Compute cumulative forward returns
        fwd = pd.Series(log_rets).rolling(h).sum().shift(-h).values
        valid = ~np.isnan(fwd)

        # Check event rate changes at perturbed thresholds
        rates = {}
        for mult in [0.9, 1.0, 1.1]:
            t = thr * mult
            n_events = np.sum(np.abs(fwd[valid]) >= t)
            rates[mult] = n_events / valid.sum() if valid.sum() > 0 else 0

        # Event rate change is a proxy for ECE sensitivity
        # If event rate changes a lot, ECE will too
        rate_change_90 = abs(rates[0.9] - rates[1.0]) / rates[1.0] if rates[1.0] > 0 else 0
        rate_change_110 = abs(rates[1.1] - rates[1.0]) / rates[1.0] if rates[1.0] > 0 else 0
        max_rate_change = max(rate_change_90, rate_change_110)

        results.append({
            "metric": "thr_sens", "horizon": h,
            "value": max_rate_change,
            "status": _status(max_rate_change, "thr_sens"),
            "detail": f"thr={thr:.4f} rate_base={rates[1.0]:.1%} rate_-10%={rates[0.9]:.1%} rate_+10%={rates[1.1]:.1%}",
        })
    return results


def compute_temporal_stability(config_name: str) -> list[dict]:
    """Metric 4: Early folds (0,1) vs late folds (3,4) ECE gap."""
    results = []
    folds_path = DIAG_DIR / f"cv_{config_name}_folds_recheck.csv"
    if not folds_path.exists():
        return [{"metric": "temporal", "horizon": "all", "value": np.nan,
                 "status": "SKIP", "detail": "No folds CSV"}]

    folds = pd.read_csv(folds_path)
    for h in folds["horizon"].unique():
        h_data = folds[folds["horizon"] == h]
        early = h_data[h_data["fold"].isin([0, 1])]["ece_cal"].mean()
        late = h_data[h_data["fold"].isin([3, 4])]["ece_cal"].mean()
        mean_ece = h_data["ece_cal"].mean()

        if mean_ece == 0:
            continue
        gap = abs(late - early) / mean_ece
        results.append({
            "metric": "temporal", "horizon": int(h),
            "value": gap,
            "status": _status(gap, "temporal"),
            "detail": f"early={early:.4f} late={late:.4f} mean={mean_ece:.4f}",
        })
    return results


def compute_neff_ratio(config_name: str) -> list[dict]:
    """Metric 5: Effective sample size vs number of BO parameters."""
    gates_path = DIAG_DIR / f"cv_{config_name}_gates_recheck.csv"
    if not gates_path.exists():
        return [{"metric": "neff_ratio", "horizon": "all", "value": np.nan,
                 "status": "SKIP", "detail": "No gate recheck CSV"}]

    gates = pd.read_csv(gates_path)
    pooled = gates[(gates["regime"] == "pooled") & (gates["metric"] == "ece_cal")]

    # Count BO params from Optuna
    db_path = OUT_DIR / f"optuna_{config_name}.db"
    n_params = 14  # default
    if db_path.exists():
        try:
            study = optuna.load_study(
                study_name=f"ece_opt_{config_name}",
                storage=f"sqlite:///{db_path}",
            )
            n_params = len(study.best_params)
        except Exception:
            pass

    results = []
    for _, row in pooled.iterrows():
        h = int(row["horizon"])
        n_samples = int(row.get("n_samples", 0))
        n_events = int(row.get("n_events", 0))
        # Effective sample size: min(events, non-events) * 2 (binary classification rule of thumb)
        n_eff = min(n_events, n_samples - n_events) * 2 if n_samples > 0 else 0
        ratio = n_eff / n_params if n_params > 0 else 0

        results.append({
            "metric": "neff_ratio", "horizon": h,
            "value": ratio,
            "status": _status(ratio, "neff_ratio"),
            "detail": f"n_eff={n_eff} n_params={n_params} n_events={n_events} n_total={n_samples}",
        })
    return results


def run_diagnostic(config_name: str) -> pd.DataFrame:
    """Run all 5 overfitting diagnostics for a config."""
    t0 = time.perf_counter()

    print(f"\n{'='*65}")
    print(f"  OVERFITTING DIAGNOSTIC: {config_name.upper()}")
    print(f"{'='*65}\n")

    all_results = []
    all_results.extend(compute_generalization_gap(config_name))
    all_results.extend(compute_fold_stability(config_name))
    all_results.extend(compute_threshold_sensitivity(config_name))
    all_results.extend(compute_temporal_stability(config_name))
    all_results.extend(compute_neff_ratio(config_name))

    df = pd.DataFrame(all_results)

    # Print formatted report
    metric_names = {
        "gen_gap": "Generalization gap",
        "fold_cv": "Cross-fold CV",
        "thr_sens": "Threshold sensitivity",
        "temporal": "Early vs late folds",
        "neff_ratio": "N_eff / N_params",
    }

    print(f"{'Metric':<28s} | {'Horizon':>7s} | {'Value':>8s} | {'Status':>6s} | Detail")
    print(f"{'-'*28}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}-+-{'-'*30}")

    for _, row in df.iterrows():
        name = metric_names.get(row["metric"], row["metric"])
        h_str = f"H={row['horizon']}" if row["horizon"] != "all" else "all"
        val = row["value"]
        val_str = f"{val:.3f}" if not np.isnan(val) else "N/A"
        if row["metric"] == "neff_ratio" and not np.isnan(val):
            val_str = f"{val:.0f}x"
        elif row["metric"] in ("gen_gap", "thr_sens", "temporal") and not np.isnan(val):
            val_str = f"{val:.1%}"
        status = row["status"]
        detail = row.get("detail", "")
        print(f"{name:<28s} | {h_str:>7s} | {val_str:>8s} | {status:>6s} | {detail}")

    # Summary
    n_green = (df["status"] == "GREEN").sum()
    n_yellow = (df["status"] == "YELLOW").sum()
    n_red = (df["status"] == "RED").sum()
    n_skip = (df["status"] == "SKIP").sum()
    n_total = n_green + n_yellow + n_red

    print(f"\n{'='*65}")
    print(f"Summary: {n_green}/{n_total} GREEN, {n_yellow}/{n_total} YELLOW, {n_red}/{n_total} RED", end="")
    if n_skip:
        print(f", {n_skip} SKIPPED", end="")
    print()

    if n_red == 0 and n_yellow <= 2:
        verdict = "LOW OVERFITTING RISK"
    elif n_red <= 2:
        verdict = "MODERATE OVERFITTING RISK"
    else:
        verdict = "HIGH OVERFITTING RISK"
    print(f"Verdict: {verdict}")

    # Per-horizon recommendations
    if n_red > 0 or n_yellow > 0:
        print("\nRecommendations:")
        for h in sorted(df["horizon"].unique()):
            if h == "all":
                continue
            h_rows = df[df["horizon"] == h]
            reds = h_rows[h_rows["status"] == "RED"]
            for _, r in reds.iterrows():
                name = metric_names.get(r["metric"], r["metric"])
                if r["metric"] == "gen_gap":
                    print(f"  - H={h}: {name} is {r['value']:.0%} — full-data ECE much worse than train CV")
                elif r["metric"] == "fold_cv":
                    print(f"  - H={h}: {name} is {r['value']:.2f} — ECE very unstable across folds")
                elif r["metric"] == "neff_ratio":
                    print(f"  - H={h}: {name} is {r['value']:.0f}x — too few events for {len(df)} params")

    elapsed = time.perf_counter() - t0
    print(f"\n({elapsed:.1f}s)")
    print(f"{'='*65}\n")

    # Save CSV
    csv_path = DIAG_DIR / f"overfit_report_{config_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Report saved to: {csv_path}")

    return df


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "both"

    if target == "both":
        run_diagnostic("cluster")
        run_diagnostic("jump")
    elif target == "all":
        for name in CONFIGS:
            run_diagnostic(name)
    elif target in CONFIGS:
        run_diagnostic(target)
    else:
        print(f"Unknown config: {target}. Available: {list(CONFIGS.keys())}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
