"""
Stress Suite Runner — Runs all exp_suite configs and collects summary metrics.

Usage:
    python run_stress_suite.py
"""

import logging
import sys
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# Suppress noisy GARCH warnings that flood the output
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))

from em_sde.config import load_config
from em_sde.data_layer import load_data
from em_sde.backtest import run_walkforward, compute_backtest_analytics
from em_sde.evaluation import compute_metrics


def run_single(config_path: str) -> dict:
    """Run a single config and return summary metrics per horizon."""
    cfg = load_config(config_path)
    df, metadata = load_data(cfg)
    results = run_walkforward(df, cfg)
    metrics = compute_metrics(results, cfg.model.horizons)

    rows = []
    for H in cfg.model.horizons:
        m = metrics["overlapping"].get(H, {})
        row = {
            "config": Path(config_path).stem,
            "ticker": cfg.data.ticker,
            "threshold_mode": cfg.model.threshold_mode,
            "horizon": H,
            "n": m.get("n", 0),
            "brier_raw": m.get("brier_raw", np.nan),
            "brier_cal": m.get("brier_cal", np.nan),
            "bss_raw": m.get("bss_raw", np.nan),
            "bss_cal": m.get("bss_cal", np.nan),
            "auc_raw": m.get("auc_raw", np.nan),
            "auc_cal": m.get("auc_cal", np.nan),
            "event_rate": m.get("event_rate", np.nan),
            "separation_cal": m.get("separation_cal", np.nan),
            "logloss_cal": m.get("logloss_cal", np.nan),
            "n_eff": m.get("n_eff", np.nan),
        }

        # Check for regime-gated diagnostic columns
        if "threshold_regime" in results.columns:
            for mode in ["fixed_pct", "anchored_vol"]:
                count = (results["threshold_regime"] == mode).sum()
                row[f"regime_{mode}"] = count
        if "garch_projected" in results.columns:
            row["garch_projected_pct"] = results["garch_projected"].mean() * 100
        if "jump_intensity_step" in results.columns:
            row["jump_intensity_mean"] = results["jump_intensity_step"].mean()
            row["jump_intensity_std"] = results["jump_intensity_step"].std()

        rows.append(row)
    return rows


def main():
    config_dir = Path("configs/exp_suite")
    configs = sorted(config_dir.glob("*.yaml"))

    print("=" * 80)
    print("STRESS SUITE: 12-Config Validation Run")
    print("=" * 80)
    print(f"Found {len(configs)} configs")
    print()

    all_rows = []
    for i, cfg_path in enumerate(configs, 1):
        name = cfg_path.stem
        print(f"[{i:2d}/{len(configs)}] Running {name} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            rows = run_single(str(cfg_path))
            all_rows.extend(rows)
            elapsed = time.perf_counter() - t0
            print(f"OK ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"FAILED ({elapsed:.1f}s): {e}")
            traceback.print_exc()

    if not all_rows:
        print("No results collected!")
        return 1

    df = pd.DataFrame(all_rows)

    # Summary table
    print()
    print("=" * 80)
    print("SUMMARY: Key Metrics (H=5, 10, 20)")
    print("=" * 80)

    # Print header
    hdr = (
        f"{'Config':<40s} {'H':>3s} {'Brier':>8s} {'BSS':>8s} "
        f"{'AUC':>6s} {'Sep':>7s} {'EvtR':>6s} {'N':>5s}"
    )
    print(hdr)
    print("-" * len(hdr))

    for _, row in df.iterrows():
        line = (
            f"{row['config']:<40s} {int(row['horizon']):>3d} "
            f"{row['brier_cal']:>8.4f} {row['bss_cal']:>+8.4f} "
            f"{row['auc_cal']:>6.4f} {row['separation_cal']:>7.4f} "
            f"{row['event_rate']:>6.3f} {int(row['n']):>5d}"
        )
        print(line)

    # Regime-gated diagnostics
    regime_gated = df[df["config"].str.contains("regime_gated")]
    if not regime_gated.empty:
        print()
        print("=" * 80)
        print("REGIME-GATED DIAGNOSTICS")
        print("=" * 80)
        for _, row in regime_gated.iterrows():
            parts = [f"{row['config']:<40s} H={int(row['horizon'])}"]
            if "garch_projected_pct" in row and pd.notna(row.get("garch_projected_pct")):
                parts.append(f"  garch_projected={row['garch_projected_pct']:.1f}%")
            if "jump_intensity_mean" in row and pd.notna(row.get("jump_intensity_mean")):
                parts.append(
                    f"  jump_intensity={row['jump_intensity_mean']:.2f}±{row['jump_intensity_std']:.2f}"
                )
            if "regime_fixed_pct" in row and pd.notna(row.get("regime_fixed_pct")):
                parts.append(
                    f"  routing: fixed_pct={int(row['regime_fixed_pct'])}, "
                    f"anchored_vol={int(row.get('regime_anchored_vol', 0))}"
                )
            print("  ".join(parts))

    # Cross-pattern comparison table
    print()
    print("=" * 80)
    print("CROSS-PATTERN COMPARISON (BSS_cal at H=10)")
    print("=" * 80)

    h10 = df[df["horizon"] == 10].copy()
    if not h10.empty:
        pivot_bss = h10.pivot_table(
            index="config", columns="ticker", values="bss_cal", aggfunc="first"
        )
        # Reorder columns
        for col in ["CLUSTER", "JUMP", "TREND"]:
            if col not in pivot_bss.columns:
                pivot_bss[col] = np.nan
        pivot_bss = pivot_bss[["CLUSTER", "JUMP", "TREND"]]
        print(pivot_bss.to_string(float_format=lambda x: f"{x:+.4f}" if pd.notna(x) else "    -"))

        # AUC comparison
        print()
        print("CROSS-PATTERN COMPARISON (AUC_cal at H=10)")
        print("-" * 60)
        pivot_auc = h10.pivot_table(
            index="config", columns="ticker", values="auc_cal", aggfunc="first"
        )
        for col in ["CLUSTER", "JUMP", "TREND"]:
            if col not in pivot_auc.columns:
                pivot_auc[col] = np.nan
        pivot_auc = pivot_auc[["CLUSTER", "JUMP", "TREND"]]
        print(pivot_auc.to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "    -"))

    # Save CSV
    out_path = Path("outputs/stress_suite_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print()
    print(f"Results saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
