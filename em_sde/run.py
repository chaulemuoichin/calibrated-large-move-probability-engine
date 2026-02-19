"""
CLI entry point for the Calibrated Large-Move Probability Engine.

Usage:
    python -m em_sde.run --config configs/large_move_3h.yaml
"""

import argparse
import logging
import math
import sys
import time
import uuid
from datetime import datetime

from .config import load_config
from .data_layer import load_data
from .backtest import run_walkforward, compute_backtest_analytics
from .evaluation import compute_metrics, compute_reliability, compute_risk_report
from .output import write_outputs
from .model_selection import expanding_window_cv, compare_models, apply_promotion_gates


def _date_str(value: object) -> str:
    """Format index values as YYYY-MM-DD strings for logs and console output."""
    text = str(value)
    return text[:10] if len(text) >= 10 else text


def _is_finite_number(value: object) -> bool:
    """Return True if value can be interpreted as a finite float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _horizon_phrase(H: int) -> str:
    """Convert trading-day horizon to a reader-friendly phrase."""
    if H == 1:
        return "next trading day"
    if H == 5:
        return "next ~1 week"
    if H == 10:
        return "next ~2 weeks"
    if H % 5 == 0 and H >= 5:
        weeks = H // 5
        return f"next ~{weeks} weeks"
    return f"next {H} trading days"


def _print_readable_risk_summary(results, horizons) -> None:
    """Print plain-English event risk summary for the latest prediction row."""
    if len(results) == 0:
        return

    latest = results.iloc[-1]
    as_of = _date_str(latest.get("date", "latest"))

    print()
    print("READABLE RISK SUMMARY (latest forecast)")
    print("-" * 60)

    for H in horizons:
        p_key = f"p_cal_{H}"
        thr_key = f"thr_{H}"
        p_val = latest.get(p_key)
        thr_val = latest.get(thr_key)

        if not _is_finite_number(p_val):
            continue

        p_pct = float(p_val) * 100.0
        horizon_text = _horizon_phrase(H)

        if _is_finite_number(thr_val):
            thr_pct = float(thr_val) * 100.0
            print(
                f"  As of {as_of}, estimated chance of a move >= {thr_pct:.2f}% "
                f"in the {horizon_text}: {p_pct:.1f}%"
            )
        else:
            print(
                f"  As of {as_of}, estimated event chance in the {horizon_text}: "
                f"{p_pct:.1f}%"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Calibrated Large-Move Probability Engine",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Custom run ID (default: auto-generated)",
    )
    parser.add_argument(
        "--compare", nargs="+", metavar="CONFIG",
        help="Compare multiple configs via expanding-window CV",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of CV folds for --compare mode",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("em_sde")

    # --compare mode: expanding-window CV across multiple configs
    if args.compare:
        return _run_compare(args)

    # Generate run ID
    run_id = args.run_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    print("=" * 60)
    print("Calibrated Large-Move Probability Engine")
    print("=" * 60)
    print(f"  run_id:  {run_id}")

    # Load config
    cfg = load_config(args.config)
    print(f"  source:  {cfg.data.source} ({cfg.data.ticker})")
    print(f"  horizons: {cfg.model.horizons}")
    print(f"  k:       {cfg.model.k}")
    print(f"  MC paths: {cfg.model.mc_base_paths}")
    print(f"  seed:    {cfg.model.seed}")
    print()

    # Load data
    logger.info("Loading data...")
    df, metadata = load_data(cfg)
    start_date = _date_str(df.index[0])
    end_date = _date_str(df.index[-1])
    logger.info("Loaded %d rows from %s to %s",
                len(df), start_date, end_date)
    if metadata.get("warnings"):
        for w in metadata["warnings"]:
            logger.warning("Data warning: %s", w)
    print(f"  data:    {len(df)} rows [{start_date} .. {end_date}]")
    print()

    # Run walk-forward backtest
    t_start = time.perf_counter()
    logger.info("Running walk-forward backtest...")
    results = run_walkforward(df, cfg)
    t_backtest = time.perf_counter() - t_start
    logger.info("Backtest produced %d prediction rows in %.1fs", len(results), t_backtest)

    # Evaluate
    logger.info("Computing evaluation metrics...")
    metrics = compute_metrics(results, cfg.model.horizons)
    reliability = compute_reliability(results, cfg.model.horizons)
    risk_report = compute_risk_report(results, cfg.model.horizons)
    bt_analytics = compute_backtest_analytics(results, cfg.model.horizons)

    # Attach extra analytics to metadata
    metadata["run_timing"] = {
        "backtest_seconds": round(t_backtest, 2),
        "timestamp": datetime.now().isoformat(),
    }
    metadata["risk_report"] = risk_report
    metadata["backtest_analytics"] = bt_analytics

    # Print results
    print("-" * 60)
    print("RESULTS: Brier Score (raw -> calibrated)")
    print("-" * 60)
    for H in cfg.model.horizons:
        overlap = metrics["overlapping"].get(H, {})
        br = overlap.get("brier_raw", float("nan"))
        bc = overlap.get("brier_cal", float("nan"))
        n = overlap.get("n", 0)
        improvement = ((br - bc) / br * 100) if br > 0 else 0
        print(f"  H={H:2d}:  {br:.6f} -> {bc:.6f}  ({improvement:+.1f}%)  [n={n}]")

    print()
    print("RESULTS: LogLoss (raw -> calibrated)")
    print("-" * 60)
    for H in cfg.model.horizons:
        overlap = metrics["overlapping"].get(H, {})
        lr = overlap.get("logloss_raw", float("nan"))
        lc = overlap.get("logloss_cal", float("nan"))
        improvement = ((lr - lc) / lr * 100) if lr > 0 else 0
        print(f"  H={H:2d}:  {lr:.6f} -> {lc:.6f}  ({improvement:+.1f}%)")

    print()
    print("RESULTS: Brier Skill Score (raw -> calibrated)")
    print("-" * 60)
    for H in cfg.model.horizons:
        overlap = metrics["overlapping"].get(H, {})
        bss_r = overlap.get("bss_raw", float("nan"))
        bss_c = overlap.get("bss_cal", float("nan"))
        n_eff = overlap.get("n_eff", float("nan"))
        n = overlap.get("n", 0)
        print(f"  H={H:2d}:  {bss_r:+.4f} -> {bss_c:+.4f}  [n={n}, n_eff={n_eff:.0f}]")

    print()
    print("RESULTS: AUC-ROC (raw -> calibrated)")
    print("-" * 60)
    for H in cfg.model.horizons:
        overlap = metrics["overlapping"].get(H, {})
        ar = overlap.get("auc_raw", float("nan"))
        ac = overlap.get("auc_cal", float("nan"))
        er = overlap.get("event_rate", float("nan"))
        sep = overlap.get("separation_cal", float("nan"))
        print(f"  H={H:2d}:  {ar:.4f} -> {ac:.4f}  [event_rate={er:.3f}, sep={sep:.4f}]")

    print()
    print("RESULTS: Risk Analytics")
    print("-" * 60)
    for H in cfg.model.horizons:
        rr = risk_report.get(H, {})
        if rr:
            print(
                f"  H={H:2d}:  VaR95={rr.get('var_95', 0):.4f}  "
                f"CVaR95={rr.get('cvar_95', 0):.4f}  "
                f"skew={rr.get('skewness', 0):+.3f}  "
                f"kurt={rr.get('kurtosis', 0):.2f}"
            )

    _print_readable_risk_summary(results, cfg.model.horizons)

    # Write outputs
    logger.info("Writing outputs...")
    out_dir = write_outputs(results, reliability, metrics, metadata, cfg, run_id, prices=df)

    print()
    print("-" * 60)
    print(f"  outputs: {out_dir}")
    print(f"  charts:  {out_dir / 'charts'}")
    print(f"  timing:  {t_backtest:.1f}s backtest")
    print("=" * 60)
    print("Done.")

    return 0


def _run_compare(args):
    """Run expanding-window CV comparison across multiple configs."""
    from pathlib import Path

    configs = []
    names = []
    for path in args.compare:
        cfg = load_config(path)
        configs.append(cfg)
        names.append(Path(path).stem)

    # Use the first config's data settings for loading
    logger = logging.getLogger("em_sde")
    logger.info("Compare mode: %d configs, %d folds", len(configs), args.cv_folds)

    print("=" * 60)
    print("Model Comparison (Expanding-Window CV)")
    print("=" * 60)
    print(f"  configs: {names}")
    print(f"  folds:   {args.cv_folds}")
    print()

    # Load data using first config
    df, metadata = load_data(configs[0])
    start_date = _date_str(df.index[0])
    end_date = _date_str(df.index[-1])
    print(f"  data:    {len(df)} rows [{start_date} .. {end_date}]")
    print()

    # Run CV
    cv_results = expanding_window_cv(df, configs, names, n_folds=args.cv_folds)

    # Summarize
    summary = compare_models(cv_results)

    print("-" * 60)
    print("CV RESULTS (mean +/- std across folds)")
    print("-" * 60)
    for row in summary.to_dict(orient="records"):
        print(
            f"  #{int(row['rank'])}  {row['config_name']:20s}  H={int(row['horizon']):2d}  "
            f"BSS={row['bss_cal_mean']:+.4f}±{row['bss_cal_std']:.4f}  "
            f"AUC={row['auc_cal_mean']:.4f}±{row['auc_cal_std']:.4f}  "
            f"Brier={row['brier_cal_mean']:.6f}  "
            f"[n={int(row['total_n'])}, folds={int(row['n_folds'])}]"
        )

    # Apply promotion gates if enabled
    if configs[0].calibration.promotion_gates_enabled:
        gates = {
            "bss_cal": configs[0].calibration.promotion_bss_min,
            "auc_cal": configs[0].calibration.promotion_auc_min,
            "ece_cal": configs[0].calibration.promotion_ece_max,
        }
        gate_report = apply_promotion_gates(cv_results, gates=gates)

        print()
        print("-" * 60)
        print("PROMOTION GATES (per regime bucket)")
        print("-" * 60)
        for row in gate_report.to_dict(orient="records"):
            status = "PASS" if row["passed"] else "FAIL"
            print(
                f"  [{status}] {row['config_name']:20s} H={int(row['horizon']):2d} "
                f"regime={row['regime']:10s} {row['metric']}={row['value']:.4f} "
                f"(threshold={row['threshold']:.4f}, margin={row['margin']:+.4f})"
            )

        # Summary
        if "all_gates_passed" in gate_report.columns:
            print()
            for (name, h), grp in gate_report.groupby(["config_name", "horizon"]):
                all_pass = grp["all_gates_passed"].iloc[0]
                verdict = "PROMOTED" if all_pass else "BLOCKED"
                print(f"  {name:20s} H={int(h):2d}: {verdict}")

    print()
    print("=" * 60)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
