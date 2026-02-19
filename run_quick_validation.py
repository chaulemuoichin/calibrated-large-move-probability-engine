"""
Quick Validation Runner — Runs targeted configs with reduced MC paths for fast turnaround.

Compares the 3 new regime_gated configs against inst_fixed_multi baselines
on all 3 patterns (cluster, jump, trend) with 5000 MC paths instead of 30K.

Usage:
    python run_quick_validation.py
"""

import logging
import sys
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))

from em_sde.config import load_config
from em_sde.data_layer import load_data
from em_sde.backtest import run_walkforward, compute_backtest_analytics
from em_sde.evaluation import compute_metrics


# Focused config pairs: baseline vs regime_gated for each pattern
CONFIGS = [
    # Cluster pattern
    "configs/exp_suite/exp_cluster_inst_fixed_multi.yaml",
    "configs/exp_suite/exp_cluster_regime_gated.yaml",
    # Jump pattern
    "configs/exp_suite/exp_jump_inst_fixed_multi.yaml",
    "configs/exp_suite/exp_jump_regime_gated.yaml",
    # Trend pattern
    "configs/exp_suite/exp_trend_inst_fixed_multi.yaml",
    "configs/exp_suite/exp_trend_regime_gated.yaml",
]


def run_single(config_path: str, mc_override: int = 1000) -> list:
    """Run a single config with reduced MC paths and return summary metrics."""
    cfg = load_config(config_path)

    # Override MC paths for speed
    cfg.model.mc_base_paths = mc_override
    cfg.model.mc_boost_paths = mc_override * 2

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

        # Regime-gated diagnostic columns
        if "threshold_regime" in results.columns:
            for mode in ["fixed_pct", "anchored_vol"]:
                count = (results["threshold_regime"] == mode).sum()
                row[f"regime_{mode}"] = count
        if "garch_projected" in results.columns:
            proj = results["garch_projected"].astype(bool)
            row["garch_projected_pct"] = proj.mean() * 100
            row["garch_projected_count"] = proj.sum()
        if "jump_intensity_step" in results.columns:
            jis = results["jump_intensity_step"].dropna()
            if len(jis) > 0:
                row["jump_intensity_mean"] = jis.mean()
                row["jump_intensity_std"] = jis.std()
                row["jump_intensity_min"] = jis.min()
                row["jump_intensity_max"] = jis.max()

        rows.append(row)
    return rows


def main():
    print("=" * 80)
    print("QUICK VALIDATION: regime_gated vs inst_fixed_multi (1K MC paths)")
    print("=" * 80)
    print(f"Running {len(CONFIGS)} configs")
    print()

    all_rows = []
    for i, cfg_path in enumerate(CONFIGS, 1):
        name = Path(cfg_path).stem
        print(f"[{i}/{len(CONFIGS)}] {name} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            rows = run_single(cfg_path)
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

    # ── Main Summary Table ──────────────────────────────────────
    print()
    print("=" * 80)
    print("SUMMARY: All Metrics by Config and Horizon")
    print("=" * 80)
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

    # ── Head-to-Head Comparison ─────────────────────────────────
    print()
    print("=" * 80)
    print("HEAD-TO-HEAD: regime_gated vs inst_fixed_multi (H=10)")
    print("=" * 80)

    h10 = df[df["horizon"] == 10].copy()
    for ticker in ["CLUSTER", "JUMP", "TREND"]:
        ticker_df = h10[h10["ticker"] == ticker]
        if len(ticker_df) < 2:
            continue

        baseline = ticker_df[ticker_df["config"].str.contains("inst_fixed_multi")].iloc[0]
        new = ticker_df[ticker_df["config"].str.contains("regime_gated")].iloc[0]

        print(f"\n  {ticker} pattern:")
        for metric, higher_better in [
            ("brier_cal", False),
            ("bss_cal", True),
            ("auc_cal", True),
            ("separation_cal", True),
        ]:
            b_val = baseline[metric]
            n_val = new[metric]
            delta = n_val - b_val
            if higher_better:
                marker = "+" if delta > 0.001 else ("-" if delta < -0.001 else "=")
            else:
                marker = "+" if delta < -0.001 else ("-" if delta > 0.001 else "=")
            print(
                f"    {metric:<18s}  baseline={b_val:.4f}  regime_gated={n_val:.4f}  "
                f"delta={delta:+.4f}  [{marker}]"
            )

    # ── Regime-Gated Diagnostics ────────────────────────────────
    regime_gated = df[df["config"].str.contains("regime_gated")]
    if not regime_gated.empty:
        print()
        print("=" * 80)
        print("REGIME-GATED DIAGNOSTICS")
        print("=" * 80)
        for _, row in regime_gated.iterrows():
            parts = [f"  {row['config']:<40s} H={int(row['horizon'])}"]
            if pd.notna(row.get("garch_projected_pct")):
                parts.append(
                    f"garch_proj={row['garch_projected_pct']:.1f}% "
                    f"({int(row.get('garch_projected_count', 0))} steps)"
                )
            if pd.notna(row.get("jump_intensity_mean")):
                parts.append(
                    f"jump_int={row['jump_intensity_mean']:.2f}"
                    f"[{row['jump_intensity_min']:.2f}-{row['jump_intensity_max']:.2f}]"
                )
            if pd.notna(row.get("regime_fixed_pct")):
                total = int(row.get("regime_fixed_pct", 0)) + int(row.get("regime_anchored_vol", 0))
                if total > 0:
                    pct_fixed = int(row["regime_fixed_pct"]) / total * 100
                    pct_anch = int(row.get("regime_anchored_vol", 0)) / total * 100
                    parts.append(
                        f"routing: fixed={pct_fixed:.0f}% anchored={pct_anch:.0f}%"
                    )
            print("  ".join(parts))

    # ── Cross-Pattern BSS Pivot ─────────────────────────────────
    print()
    print("=" * 80)
    print("CROSS-PATTERN BSS_cal PIVOT (H=10)")
    print("=" * 80)
    if not h10.empty:
        pivot = h10.pivot_table(
            index="config", columns="ticker", values="bss_cal", aggfunc="first"
        )
        for col in ["CLUSTER", "JUMP", "TREND"]:
            if col not in pivot.columns:
                pivot[col] = np.nan
        pivot = pivot[["CLUSTER", "JUMP", "TREND"]]
        print(pivot.to_string(float_format=lambda x: f"{x:+.4f}" if pd.notna(x) else "    -"))

    # Save CSV
    out_path = Path("outputs/quick_validation_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print()
    print(f"\nResults saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
