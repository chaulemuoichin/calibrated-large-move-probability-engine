"""
Model selection framework: expanding-window CV plus governance gating.
"""

import logging
import re
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import PipelineConfig
from .backtest import run_walkforward
from .calibration import fit_offline_pooled_calibrator
from .evaluation import (
    brier_score, brier_skill_score, log_loss, auc_roc, separation,
    expected_calibration_error, effective_sample_size,
    crps_from_quantiles, pit_from_quantiles,
    pit_ks_statistic, central_interval_coverage_error,
    crps_per_sample_from_quantiles, paired_bootstrap_loss_diff_pvalue,
)
from .monte_carlo import QUANTILE_LEVELS

logger = logging.getLogger(__name__)


def _as_float_array(values: object) -> NDArray[np.float64]:
    """Return a 1D float64 numpy array from pandas/numpy array-like inputs."""
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _effective_sigma_column(results: pd.DataFrame, H: int) -> str | None:
    """Return the best available sigma column for horizon-aware OOF gating."""
    candidates = (
        f"sigma_forecast_{H}",
        "sigma_har_rv_1d",
        "sigma_garch_1d",
    )
    for col in candidates:
        if col in results.columns:
            return col
    return None


def _quantile_column_map(results: pd.DataFrame, H: int) -> dict[str, str]:
    """Map horizon-specific quantile columns to generic OOF names."""
    mapping: dict[str, str] = {}
    pattern = re.compile(rf"^q(\d{{2}})_{H}$")
    for col in results.columns:
        match = pattern.match(col)
        if match:
            mapping[col] = f"q{match.group(1)}"
    return mapping


def load_overfit_summary(overfit_report: str | pd.DataFrame | None) -> pd.DataFrame | None:
    """Load an overfit report or return a normalized DataFrame."""
    if overfit_report is None:
        return None
    if isinstance(overfit_report, pd.DataFrame):
        return overfit_report.copy()
    path = pd.io.common.stringify_path(overfit_report)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None
    return df


def _summarize_overfit_by_horizon(overfit_df: pd.DataFrame | None) -> dict[int, dict[str, object]]:
    """Summarize worst overfit status per horizon."""
    if overfit_df is None or len(overfit_df) == 0:
        return {}
    rank = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    summary: dict[int, dict[str, object]] = {}
    for h, grp in overfit_df.groupby("horizon"):
        try:
            h_int = int(h)
        except (TypeError, ValueError):
            continue
        statuses = [s for s in grp["status"].astype(str).tolist() if s in rank]
        if not statuses:
            continue
        worst = max(statuses, key=lambda s: rank[s])
        summary[h_int] = {"worst_status": worst, "worst_score": rank[worst]}
    return summary


def _era_label(value: object) -> str:
    """Bucket a date into a coarse research era."""
    dt = pd.Timestamp(value)
    if dt <= pd.Timestamp("2020-02-19"):
        return "pre_covid"
    if dt <= pd.Timestamp("2020-12-31"):
        return "covid_2020"
    if dt <= pd.Timestamp("2022-12-31"):
        return "tightening_2021_2022"
    return "post_2023"


def expanding_window_cv(
    df: pd.DataFrame,
    configs: List[PipelineConfig],
    config_names: List[str],
    n_folds: int = 5,
    min_train_pct: float = 0.4,
    fold_callback: Optional[Callable[[int, pd.DataFrame, pd.DataFrame], None]] = None,
) -> tuple:
    """
    Run expanding-window cross-validation over multiple configs.

    The data is split into n_folds test windows. For each fold,
    the model trains on all data up to the test start, then
    is evaluated on the test window only.

    Parameters
    ----------
    df : pd.DataFrame
        Full price data with DatetimeIndex and 'price' column.
    configs : list of PipelineConfig
        Model configurations to compare.
    config_names : list of str
        Human-readable names for each config.
    n_folds : int
        Number of expanding-window folds.
    min_train_pct : float
        Minimum fraction of data used for training in the first fold.
    fold_callback : callable, optional
        Invoked after each completed fold as
        ``fold_callback(fold_idx, cv_results_so_far, oof_so_far)``.
        The callback may raise an exception (e.g. ``optuna.TrialPruned``)
        to terminate CV early; the exception propagates to the caller.
        Callback return values are ignored.

    Returns
    -------
    cv_results : pd.DataFrame
        Fold-level summary metrics. Columns: config_name, fold, horizon,
        brier, bss, auc, logloss, separation, event_rate, n, sigma_mean.
    oof_df : pd.DataFrame
        Row-level out-of-fold predictions pooled across all folds.
        Always includes: config_name, fold, horizon, p_cal, y, sigma_1d.
        When available also includes realized_return and q01..q99 density columns.
        Here sigma_1d is the effective daily sigma proxy actually used for
        that horizon forecast, not necessarily the raw GARCH 1-day sigma.
    """
    n = len(df)
    min_train = int(n * min_train_pct)
    remaining = n - min_train
    fold_size = remaining // n_folds

    if fold_size < 50:
        logger.warning("Fold size %d is very small; results may be noisy", fold_size)

    rows = []
    oof_parts = []

    for fold_i in range(n_folds):
        train_end = min_train + fold_i * fold_size
        test_end = min(train_end + fold_size, n)

        logger.info(
            "CV fold %d/%d: train=[0:%d], test=[%d:%d]",
            fold_i + 1, n_folds, train_end, train_end, test_end,
        )

        for cfg, name in zip(configs, config_names):
            # Run walk-forward on data up to test_end
            df_fold = df.iloc[:test_end].copy()
            results = run_walkforward(df_fold, cfg)
            use_offline_calibration = bool(cfg.calibration.offline_pooled_calibration)

            # Extract test portion only (rows generated from train_end onward)
            test_dates = list(df.index[train_end:test_end])
            test_mask = results["date"].isin(test_dates)
            test_res = results[test_mask]

            if len(test_res) == 0:
                continue

            for H in cfg.model.horizons:
                y_col = f"y_{H}"
                if y_col not in test_res.columns:
                    continue

                sigma_col = _effective_sigma_column(test_res, H)
                sigma_1d_all = (
                    _as_float_array(test_res[sigma_col])
                    if sigma_col is not None
                    else np.full(len(test_res), np.nan)
                )

                y = _as_float_array(test_res[y_col])
                p_raw = _as_float_array(test_res[f"p_raw_{H}"])
                p_cal = _as_float_array(test_res[f"p_cal_{H}"])
                if use_offline_calibration:
                    train_cutoff_idx = train_end - H
                    train_res = results.iloc[0:0]
                    if train_cutoff_idx > 0:
                        train_cutoff_date = df.index[train_cutoff_idx - 1]
                        train_res = results[results["date"] <= train_cutoff_date]
                    effective_post_cal = (
                        cfg.calibration.post_cal_method
                        or ("histogram" if cfg.calibration.histogram_post_calibration else "none")
                    )
                    hist_bins = (cfg.calibration.histogram_n_bins_by_horizon or {}).get(H, cfg.calibration.histogram_n_bins)
                    hist_prior = (cfg.calibration.histogram_prior_strength_by_horizon or {}).get(
                        H, cfg.calibration.histogram_prior_strength,
                    )
                    offline_cal = fit_offline_pooled_calibrator(
                        train_res,
                        H,
                        multi_feature=cfg.calibration.multi_feature,
                        l2_reg=cfg.calibration.multi_feature_l2,
                        max_iter=cfg.calibration.offline_calibration_max_iter,
                        beta_calibration=cfg.calibration.beta_calibration,
                        earnings_aware=cfg.calibration.multi_feature and cfg.model.earnings_calendar and H <= 5,
                        implied_vol_aware=(
                            cfg.calibration.multi_feature
                            and cfg.model.implied_vol_enabled
                            and cfg.model.implied_vol_as_feature
                        ),
                        ohlc_aware=cfg.calibration.multi_feature and cfg.model.ohlc_features_enabled and all(
                            col in df_fold.columns for col in ("open", "high", "low")
                        ),
                        post_cal_method=effective_post_cal,
                        histogram_n_bins=hist_bins,
                        histogram_min_samples=cfg.calibration.histogram_min_samples,
                        histogram_prior_strength=hist_prior,
                        histogram_monotonic=cfg.calibration.histogram_monotonic,
                        histogram_interpolate=cfg.calibration.histogram_interpolate,
                    )
                    p_cal = offline_cal.calibrate_frame(test_res, H)

                mask = ~np.isnan(y)
                if mask.sum() < 10:
                    continue

                y_m = y[mask]
                p_raw_m = p_raw[mask]
                p_cal_m = p_cal[mask]
                sigma_1d_m = sigma_1d_all[mask]

                # Collect OOF row-level predictions
                oof_payload = {
                    "config_name": name,
                    "fold": fold_i,
                    "horizon": H,
                    "date": np.asarray(test_res["date"])[mask],
                    "p_raw": p_raw_m,
                    "p_cal": p_cal_m,
                    "y": y_m,
                    "sigma_1d": sigma_1d_m,
                }
                realized_col = f"realized_return_{H}"
                if realized_col in test_res.columns:
                    realized_all = _as_float_array(test_res[realized_col])[mask]
                    oof_payload["realized_return"] = realized_all
                if "earnings_proximity" in test_res.columns:
                    oof_payload["earnings_proximity"] = _as_float_array(test_res["earnings_proximity"])[mask]
                quantile_map = _quantile_column_map(test_res, H)
                for src_col, dst_col in quantile_map.items():
                    oof_payload[dst_col] = _as_float_array(test_res[src_col])[mask]
                oof_parts.append(pd.DataFrame(oof_payload))

                # Sigma mean for legacy regime bucketing
                sigma_mean = np.nan
                if sigma_col is not None:
                    sigma_mean = float(np.nanmean(_as_float_array(test_res[sigma_col])))

                rows.append({
                    "config_name": name,
                    "fold": fold_i,
                    "horizon": H,
                    "brier_raw": brier_score(p_raw_m, y_m),
                    "brier_cal": brier_score(p_cal_m, y_m),
                    "bss_raw": brier_skill_score(p_raw_m, y_m),
                    "bss_cal": brier_skill_score(p_cal_m, y_m),
                    "auc_raw": auc_roc(p_raw_m, y_m),
                    "auc_cal": auc_roc(p_cal_m, y_m),
                    "logloss_cal": log_loss(p_cal_m, y_m),
                    "separation_cal": separation(p_cal_m, y_m),
                    "ece_cal": expected_calibration_error(p_cal_m, y_m, adaptive=False),
                    "ece_cal_adaptive": expected_calibration_error(p_cal_m, y_m, adaptive=True),
                    "event_rate": float(np.nanmean(y_m)),
                    "n": int(mask.sum()),
                    "sigma_mean": sigma_mean,
                })

        if fold_callback is not None:
            cv_partial = pd.DataFrame(rows)
            oof_partial = (
                pd.concat(oof_parts, ignore_index=True)
                if oof_parts
                else pd.DataFrame(
                    columns=["config_name", "fold", "horizon", "p_raw", "p_cal", "y", "sigma_1d"]
                )
            )
            fold_callback(fold_i, cv_partial, oof_partial)

    cv_results = pd.DataFrame(rows)
    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame(
        columns=["config_name", "fold", "horizon", "p_raw", "p_cal", "y", "sigma_1d"]
    )
    return cv_results, oof_df


def compare_models(cv_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate CV results: mean +/- std across folds.
    Rank models by BSS (primary) and AUC (secondary).

    Parameters
    ----------
    cv_results : pd.DataFrame
        Output of expanding_window_cv.

    Returns
    -------
    summary : pd.DataFrame
        Aggregated comparison with ranking.
    """
    if len(cv_results) == 0:
        return pd.DataFrame()

    agg = cv_results.groupby(["config_name", "horizon"]).agg(
        bss_cal_mean=("bss_cal", "mean"),
        bss_cal_std=("bss_cal", "std"),
        auc_cal_mean=("auc_cal", "mean"),
        auc_cal_std=("auc_cal", "std"),
        brier_cal_mean=("brier_cal", "mean"),
        logloss_cal_mean=("logloss_cal", "mean"),
        n_folds=("fold", "count"),
        total_n=("n", "sum"),
    ).reset_index()

    # Rank by BSS descending (per horizon). Keep non-finite rows as unranked
    # instead of crashing comparison mode on sparse/failed variants.
    agg["rank"] = (
        agg.groupby("horizon")["bss_cal_mean"]
        .rank(ascending=False, method="min")
        .astype("Int64")
    )

    return agg.sort_values(["horizon", "rank"], na_position="last")


def compute_benchmark_report(
    oof_df: pd.DataFrame,
    baseline_config: str | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Benchmark OOF predictions against climatology or a named baseline config.
    """
    if len(oof_df) == 0:
        return pd.DataFrame()

    rows = []
    for horizon, h_group in oof_df.groupby("horizon"):
        baseline_group = None
        baseline_label = "climatology"
        if baseline_config is not None:
            bg = h_group[h_group["config_name"] == baseline_config].copy()
            if len(bg) > 0:
                baseline_group = bg.sort_values("date") if "date" in bg.columns else bg.reset_index(drop=True)
                baseline_label = baseline_config

        for config_name, grp in h_group.groupby("config_name"):
            grp_sorted = grp.sort_values("date") if "date" in grp.columns else grp.reset_index(drop=True)
            y = _as_float_array(grp_sorted["y"])
            p_model = _as_float_array(grp_sorted["p_cal"])
            model_brier_losses = (p_model - y) ** 2

            if baseline_group is None:
                p_base = np.full(len(grp_sorted), float(np.mean(y)))
                baseline_brier_losses = (p_base - y) ** 2
            else:
                if config_name == baseline_config:
                    continue
                if "date" in grp_sorted.columns and "date" in baseline_group.columns:
                    merged = grp_sorted.merge(
                        baseline_group[["date", "p_cal"]].rename(columns={"p_cal": "p_base"}),
                        on="date",
                        how="inner",
                    )
                else:
                    left = grp_sorted.reset_index(drop=True).copy()
                    right = baseline_group.reset_index(drop=True).copy()
                    n_common = min(len(left), len(right))
                    merged = left.iloc[:n_common].copy()
                    merged["p_base"] = _as_float_array(right.iloc[:n_common]["p_cal"])
                if len(merged) == 0:
                    continue
                y = _as_float_array(merged["y"])
                p_model = _as_float_array(merged["p_cal"])
                p_base = _as_float_array(merged["p_base"])
                model_brier_losses = (p_model - y) ** 2
                baseline_brier_losses = (p_base - y) ** 2

            model_brier = float(np.mean(model_brier_losses))
            baseline_brier = float(np.mean(baseline_brier_losses))
            brier_skill = np.nan if baseline_brier <= 1e-15 else float(1.0 - model_brier / baseline_brier)
            pvalue_brier = paired_bootstrap_loss_diff_pvalue(
                model_brier_losses, baseline_brier_losses, n_boot=n_boot, seed=seed,
            )

            row = {
                "config_name": config_name,
                "horizon": int(horizon),
                "baseline": baseline_label,
                "brier_model": model_brier,
                "brier_baseline": baseline_brier,
                "brier_skill": brier_skill,
                "brier_pvalue": pvalue_brier,
            }

            quantile_cols = [f"q{int(q * 100):02d}" for q in QUANTILE_LEVELS if f"q{int(q * 100):02d}" in grp_sorted.columns]
            if "realized_return" in grp_sorted.columns and len(quantile_cols) == len(QUANTILE_LEVELS):
                q_levels = np.array([int(col[1:]) / 100.0 for col in quantile_cols], dtype=np.float64)
                realized = _as_float_array(grp_sorted["realized_return"])
                quantiles = grp_sorted[quantile_cols].to_numpy(dtype=np.float64)
                model_crps_losses = crps_per_sample_from_quantiles(quantiles, q_levels, realized)

                if baseline_group is None:
                    base_q = np.quantile(realized[np.isfinite(realized)], q_levels).reshape(1, -1)
                    base_quantiles = np.repeat(base_q, len(realized), axis=0)
                else:
                    if "realized_return" not in baseline_group.columns or len(quantile_cols) != len(QUANTILE_LEVELS):
                        base_quantiles = None
                    else:
                        base_quantiles = baseline_group[quantile_cols].to_numpy(dtype=np.float64)
                        if len(base_quantiles) != len(model_crps_losses):
                            base_quantiles = None

                if base_quantiles is not None:
                    baseline_crps_losses = crps_per_sample_from_quantiles(base_quantiles, q_levels, realized)
                    model_crps = float(np.nanmean(model_crps_losses))
                    baseline_crps = float(np.nanmean(baseline_crps_losses))
                    row["crps_model"] = model_crps
                    row["crps_baseline"] = baseline_crps
                    row["crps_skill"] = np.nan if baseline_crps <= 1e-15 else float(1.0 - model_crps / baseline_crps)
                    row["crps_pvalue"] = paired_bootstrap_loss_diff_pvalue(
                        model_crps_losses, baseline_crps_losses, n_boot=n_boot, seed=seed,
                    )

            rows.append(row)

    return pd.DataFrame(rows)


def compute_pairwise_significance_report(
    oof_df: pd.DataFrame,
    metric: str = "brier",
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Pairwise significance report across configs using aligned OOF rows."""
    if len(oof_df) == 0 or oof_df["config_name"].nunique() < 2:
        return pd.DataFrame()

    rows = []
    names = sorted(oof_df["config_name"].unique())
    for horizon, h_group in oof_df.groupby("horizon"):
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                left_df = h_group[h_group["config_name"] == left]
                right_df = h_group[h_group["config_name"] == right]
                if "date" in left_df.columns and "date" in right_df.columns:
                    merged = left_df.merge(
                        right_df,
                        on=["date", "horizon", "fold"],
                        suffixes=("_left", "_right"),
                    )
                else:
                    left_tmp = left_df.reset_index(drop=True).copy()
                    right_tmp = right_df.reset_index(drop=True).copy()
                    n_common = min(len(left_tmp), len(right_tmp))
                    merged = left_tmp.iloc[:n_common].copy()
                    for col in right_tmp.columns:
                        merged[f"{col}_right"] = right_tmp.iloc[:n_common][col].values
                    for col in list(left_tmp.columns):
                        if col in merged.columns:
                            merged.rename(columns={col: f"{col}_left"}, inplace=True)
                if len(merged) == 0:
                    continue
                y = _as_float_array(merged["y_left"])
                if metric == "brier":
                    left_loss = (_as_float_array(merged["p_cal_left"]) - y) ** 2
                    right_loss = (_as_float_array(merged["p_cal_right"]) - y) ** 2
                else:
                    continue
                pvalue = paired_bootstrap_loss_diff_pvalue(left_loss, right_loss, n_boot=n_boot, seed=seed)
                rows.append({
                    "horizon": int(horizon),
                    "config_left": left,
                    "config_right": right,
                    "metric": metric,
                    "mean_loss_left": float(np.mean(left_loss)),
                    "mean_loss_right": float(np.mean(right_loss)),
                    "better_config": left if np.mean(left_loss) < np.mean(right_loss) else right,
                    "pvalue_left_beats_right": pvalue,
                })
    return pd.DataFrame(rows)


def compute_conditional_gate_report_oof(
    oof_df: pd.DataFrame,
    gates: dict | None = None,
    density_gates: dict | None = None,
    min_samples: int = 100,
    min_events: int = 30,
    min_nonevents: int = 30,
) -> pd.DataFrame:
    """Run pooled governance on conditional era/event slices."""
    if len(oof_df) == 0:
        return pd.DataFrame()

    frames = []
    if "date" in oof_df.columns:
        df_era = oof_df.copy()
        df_era["slice_type"] = "era"
        df_era["slice_value"] = pd.to_datetime(df_era["date"]).map(_era_label)
        frames.append(df_era)
    if "earnings_proximity" in oof_df.columns:
        df_event = oof_df.copy()
        df_event["slice_type"] = "event_state"
        prox = _as_float_array(df_event["earnings_proximity"])
        df_event["slice_value"] = np.where(prox >= 0.5, "near_earnings", "non_earnings")
        frames.append(df_event)
    if not frames:
        return pd.DataFrame()

    rows = []
    for sliced in frames:
        slice_type = sliced["slice_type"].iloc[0]
        for slice_value, subset in sliced.groupby("slice_value"):
            if len(subset) == 0:
                continue
            report = apply_promotion_gates_oof(
                subset.drop(columns=["slice_type", "slice_value"]),
                gates=gates,
                density_gates=density_gates,
                min_samples=min_samples,
                min_events=min_events,
                min_nonevents=min_nonevents,
                pooled_gate=True,
            )
            if len(report) == 0:
                continue
            pooled = report[report["regime"] == "pooled"].copy()
            pooled["slice_type"] = slice_type
            pooled["slice_value"] = slice_value
            rows.append(pooled)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def calibration_aic(p_cal: np.ndarray, y: np.ndarray, n_params: int) -> float:
    """AIC for calibration model: AIC = 2k - 2*LL."""
    mask = ~np.isnan(y) & ~np.isnan(p_cal)
    n = mask.sum()
    if n == 0:
        return np.nan
    ll = -log_loss(p_cal[mask], y[mask]) * n
    return float(2 * n_params - 2 * ll)


def calibration_bic(p_cal: np.ndarray, y: np.ndarray, n_params: int) -> float:
    """BIC for calibration model: BIC = k*ln(n) - 2*LL."""
    mask = ~np.isnan(y) & ~np.isnan(p_cal)
    n = mask.sum()
    if n == 0:
        return np.nan
    ll = -log_loss(p_cal[mask], y[mask]) * n
    return float(n_params * np.log(n) - 2 * ll)


def _bootstrap_ece_ci(
    p: np.ndarray, y: np.ndarray,
    n_bins: int = 10, n_boot: int = 1000,
    alpha: float = 0.05, seed: int = 42,
    block_size: int = 1,
) -> tuple:
    """Bootstrap confidence interval for equal-width ECE.

    When block_size > 1, uses circular block bootstrap to account for
    serial dependence in overlapping H-step predictions.
    """
    rng = np.random.default_rng(seed)
    n = len(p)
    ece_samples = np.empty(n_boot)

    if block_size <= 1:
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            ece_samples[b] = expected_calibration_error(
                p[idx], y[idx], n_bins=n_bins, adaptive=False,
            )
    else:
        n_blocks = max(1, (n + block_size - 1) // block_size)
        for b in range(n_boot):
            starts = rng.integers(0, n, size=n_blocks)
            idx = np.concatenate([
                np.arange(s, s + block_size) % n for s in starts
            ])[:n]
            ece_samples[b] = expected_calibration_error(
                p[idx], y[idx], n_bins=n_bins, adaptive=False,
            )

    lo = float(np.percentile(ece_samples, 100 * alpha / 2))
    hi = float(np.percentile(ece_samples, 100 * (1 - alpha / 2)))
    return lo, hi


def apply_promotion_gates_oof(
    oof_df: pd.DataFrame,
    gates: dict = None,
    density_gates: dict | None = None,
    min_samples: int = 100,
    min_events: int = 30,
    min_nonevents: int = 30,
    pooled_gate: bool = False,
    n_bo_params: int = 6,
    overfit_report: str | pd.DataFrame | None = None,
    require_overfit: bool = False,
) -> pd.DataFrame:
    """
    Apply promotion gates on pooled out-of-fold row-level predictions.

    Instead of bucketing folds by sigma_mean (noisy with 5 folds), this
    assigns each OOF row to a vol regime based on its own sigma_1d, then
    computes metrics on the pooled rows per regime. Provides much more
    statistical power and defensible pass/fail decisions.

    Parameters
    ----------
    oof_df : pd.DataFrame
        Row-level OOF predictions from expanding_window_cv.
        Required columns: config_name, horizon, p_cal, y, sigma_1d.
        Optional density columns: realized_return, q01..q99.
    gates : dict, optional
        Gate thresholds. Default: {"bss_cal": 0.0, "auc_cal": 0.55, "ece_cal": 0.02}
    density_gates : dict, optional
        Optional density-governance thresholds, e.g.
        {"crps_skill": 0.0, "pit_ks": 0.12, "tail_cov_error": 0.05}
    min_samples : int
        Minimum rows in a regime bucket for evaluation (else insufficient_data).
    min_events : int
        Minimum positive labels in a regime bucket for evaluation.
    min_nonevents : int
        Minimum negative labels in a regime bucket for evaluation.

    Returns
    -------
    gate_report : pd.DataFrame
        Columns: config_name, horizon, regime, metric, value, threshold,
                 passed, margin, n_samples, n_events, n_nonevents,
                 insufficient_reason, status,
                 ece_ci_low, ece_ci_high,
                 n_eff, neff_ratio, neff_warning, ece_gate_confidence,
                 promotion_status, all_gates_passed
    """
    if gates is None:
        gates = {"bss_cal": 0.0, "auc_cal": 0.55, "ece_cal": 0.02}
    if density_gates is None:
        density_gates = {}

    if len(oof_df) == 0:
        return pd.DataFrame()

    rows = []
    overfit_summary = _summarize_overfit_by_horizon(load_overfit_summary(overfit_report))

    for (config_name, horizon), group in oof_df.groupby(["config_name", "horizon"]):
        p_cal = _as_float_array(group["p_cal"])
        y = _as_float_array(group["y"])
        sigma = _as_float_array(group["sigma_1d"])
        realized_all = (
            _as_float_array(group["realized_return"])
            if "realized_return" in group.columns
            else np.full(len(group), np.nan)
        )
        quantile_cols = [
            f"q{int(q * 100):02d}"
            for q in QUANTILE_LEVELS
            if f"q{int(q * 100):02d}" in group.columns
        ]
        q_levels = (
            np.array([int(col[1:]) / 100.0 for col in quantile_cols], dtype=np.float64)
            if quantile_cols else np.array([], dtype=np.float64)
        )

        valid_sigma = sigma[~np.isnan(sigma)]
        if len(valid_sigma) < 3:
            regime_labels = np.full(len(sigma), "all")
        else:
            p33 = float(np.nanpercentile(valid_sigma, 33))
            p66 = float(np.nanpercentile(valid_sigma, 66))
            regime_labels = np.array([
                "all" if np.isnan(s)
                else "low_vol" if s < p33
                else "high_vol" if s > p66
                else "mid_vol"
                for s in sigma
            ])

        regime_list = sorted(set(regime_labels))
        if pooled_gate:
            regime_list = ["pooled"] + regime_list

        for regime in regime_list:
            if regime == "pooled":
                mask = np.ones(len(group), dtype=bool)
            else:
                mask = regime_labels == regime
            p_r = p_cal[mask]
            y_r = y[mask]
            realized_r = realized_all[mask]
            quantiles_r = (
                group.loc[mask, quantile_cols].to_numpy(dtype=np.float64)
                if quantile_cols else np.empty((mask.sum(), 0))
            )

            n_samples = len(y_r)
            n_events = int(np.sum(y_r))
            n_nonevents = n_samples - n_events
            # Use ACF-corrected N_eff on residuals (accounts for overlap)
            H_int = int(horizon) if isinstance(horizon, (int, float, np.integer, np.floating)) else 5
            n_eff_acf = effective_sample_size(y_r, H_int, p_cal=p_r)
            # Cap at min(events, nonevents) * 2 as upper bound
            n_eff = min(n_eff_acf, min(n_events, n_nonevents) * 2)
            neff_ratio = round(n_eff / n_bo_params, 1) if n_bo_params > 0 else 0.0
            neff_warning = "" if neff_ratio >= 100 else ("YELLOW" if neff_ratio >= 50 else "RED")

            if n_samples < min_samples:
                insufficient_reason = "too_few_samples"
            elif n_events < min_events:
                insufficient_reason = "too_few_events"
            elif n_nonevents < min_nonevents:
                insufficient_reason = "too_few_nonevents"
            else:
                insufficient_reason = "sufficient"
            insufficient = insufficient_reason != "sufficient"

            ece_ci_low, ece_ci_high = np.nan, np.nan
            if not insufficient:
                ece_ci_low, ece_ci_high = _bootstrap_ece_ci(p_r, y_r, block_size=H_int)

            base_status = "evaluated" if not (pooled_gate and regime != "pooled") else "diagnostic"

            for metric, threshold in gates.items():
                if insufficient:
                    rows.append({
                        "config_name": config_name,
                        "horizon": horizon,
                        "regime": regime,
                        "metric": metric,
                        "value": np.nan,
                        "threshold": threshold,
                        "passed": None,
                        "margin": np.nan,
                        "n_samples": n_samples,
                        "n_events": n_events,
                        "n_nonevents": n_nonevents,
                        "insufficient_reason": insufficient_reason,
                        "status": "insufficient_data",
                        "ece_ci_low": np.nan,
                        "ece_ci_high": np.nan,
                        "n_eff": n_eff,
                        "neff_ratio": neff_ratio,
                        "neff_warning": neff_warning,
                        "ece_gate_confidence": "",
                    })
                    continue

                if metric == "bss_cal":
                    value = float(brier_skill_score(p_r, y_r))
                    passed = value >= threshold
                    margin = round(value - threshold, 6)
                    ece_conf = ""
                elif metric == "auc_cal":
                    value = float(auc_roc(p_r, y_r))
                    passed = value >= threshold
                    margin = round(value - threshold, 6)
                    ece_conf = ""
                elif metric == "ece_cal":
                    value = float(expected_calibration_error(p_r, y_r, adaptive=False))
                    passed = value <= threshold
                    margin = round(threshold - value, 6)
                    if passed:
                        ece_conf = "fragile_pass" if ece_ci_high > threshold else "solid_pass"
                    else:
                        ece_conf = "fragile_fail" if ece_ci_low <= threshold else "solid_fail"
                else:
                    continue

                rows.append({
                    "config_name": config_name,
                    "horizon": horizon,
                    "regime": regime,
                    "metric": metric,
                    "value": round(value, 6),
                    "threshold": threshold,
                    "passed": bool(passed),
                    "margin": margin,
                    "n_samples": n_samples,
                    "n_events": n_events,
                    "n_nonevents": n_nonevents,
                    "insufficient_reason": "sufficient",
                    "status": base_status,
                    "ece_ci_low": round(ece_ci_low, 6) if np.isfinite(ece_ci_low) else np.nan,
                    "ece_ci_high": round(ece_ci_high, 6) if np.isfinite(ece_ci_high) else np.nan,
                    "n_eff": n_eff,
                    "neff_ratio": neff_ratio,
                    "neff_warning": neff_warning,
                    "ece_gate_confidence": ece_conf,
                })

            for metric, threshold in density_gates.items():
                density_available = (
                    not insufficient
                    and len(quantile_cols) >= 2
                    and np.isfinite(realized_r).sum() > 0
                    and quantiles_r.shape[0] == len(realized_r)
                )
                if not density_available:
                    rows.append({
                        "config_name": config_name,
                        "horizon": horizon,
                        "regime": regime,
                        "metric": metric,
                        "value": np.nan,
                        "threshold": threshold,
                        "passed": None,
                        "margin": np.nan,
                        "n_samples": n_samples,
                        "n_events": n_events,
                        "n_nonevents": n_nonevents,
                        "insufficient_reason": insufficient_reason if insufficient else "missing_density_inputs",
                        "status": ("insufficient_data" if insufficient else "unavailable_density")
                        if base_status == "evaluated" else "diagnostic",
                        "ece_ci_low": np.nan,
                        "ece_ci_high": np.nan,
                        "n_eff": n_eff,
                        "neff_ratio": neff_ratio,
                        "neff_warning": neff_warning,
                        "ece_gate_confidence": "",
                    })
                    continue

                if metric == "crps_skill":
                    model_crps = float(crps_from_quantiles(quantiles_r, q_levels, realized_r))
                    baseline_quantiles = np.quantile(realized_r[np.isfinite(realized_r)], q_levels).reshape(1, -1)
                    baseline_quantiles = np.repeat(baseline_quantiles, len(realized_r), axis=0)
                    baseline_crps = float(crps_from_quantiles(baseline_quantiles, q_levels, realized_r))
                    value = np.nan if not np.isfinite(baseline_crps) or abs(baseline_crps) < 1e-15 else float(1.0 - model_crps / baseline_crps)
                    passed = bool(np.isfinite(value) and value >= threshold)
                    margin = round(value - threshold, 6) if np.isfinite(value) else np.nan
                elif metric == "pit_ks":
                    pit = pit_from_quantiles(quantiles_r, q_levels, realized_r)
                    value = float(pit_ks_statistic(pit))
                    passed = bool(np.isfinite(value) and value <= threshold)
                    margin = round(threshold - value, 6) if np.isfinite(value) else np.nan
                elif metric == "tail_cov_error":
                    errs = []
                    for lower_q, upper_q in ((0.05, 0.95), (0.01, 0.99)):
                        err = central_interval_coverage_error(quantiles_r, q_levels, realized_r, lower_q, upper_q)
                        if np.isfinite(err):
                            errs.append(err)
                    value = float(max(errs)) if errs else np.nan
                    passed = bool(np.isfinite(value) and value <= threshold)
                    margin = round(threshold - value, 6) if np.isfinite(value) else np.nan
                else:
                    continue

                rows.append({
                    "config_name": config_name,
                    "horizon": horizon,
                    "regime": regime,
                    "metric": metric,
                    "value": round(value, 6) if np.isfinite(value) else np.nan,
                    "threshold": threshold,
                    "passed": passed if np.isfinite(value) else None,
                    "margin": margin,
                    "n_samples": n_samples,
                    "n_events": n_events,
                    "n_nonevents": n_nonevents,
                    "insufficient_reason": "sufficient",
                    "status": base_status,
                    "ece_ci_low": np.nan,
                    "ece_ci_high": np.nan,
                    "n_eff": n_eff,
                    "neff_ratio": neff_ratio,
                    "neff_warning": neff_warning,
                    "ece_gate_confidence": "",
                })

        if require_overfit or int(horizon) in overfit_summary:
            info = overfit_summary.get(int(horizon))
            if info is None:
                status = "missing_overfit"
                value = np.nan
                passed = None
                reason = "missing_overfit_report"
                margin = np.nan
            else:
                status = "evaluated"
                value = float(info["worst_score"])
                passed = bool(value <= 1.0)
                reason = "sufficient"
                margin = round(1.0 - value, 6)

            rows.append({
                "config_name": config_name,
                "horizon": horizon,
                "regime": "pooled" if pooled_gate else "all",
                "metric": "overfit_status",
                "value": value,
                "threshold": 1.0,
                "passed": passed,
                "margin": margin,
                "n_samples": len(group),
                "n_events": int(np.sum(y)),
                "n_nonevents": int(len(group) - np.sum(y)),
                "insufficient_reason": reason,
                "status": status,
                "ece_ci_low": np.nan,
                "ece_ci_high": np.nan,
                "n_eff": min(int(np.sum(y)), int(len(group) - np.sum(y))) * 2,
                "neff_ratio": np.nan,
                "neff_warning": "",
                "ece_gate_confidence": "",
            })

    report = pd.DataFrame(rows)

    if len(report) > 0:
        def _promotion_status(grp):
            primary = grp[grp["status"] != "diagnostic"]
            if len(primary) == 0:
                return "UNDECIDED"
            if primary["status"].isin(["insufficient_data", "unavailable_density", "missing_overfit"]).any():
                return "UNDECIDED"
            evaluated = primary[primary["status"] == "evaluated"]
            if len(evaluated) == 0:
                return "UNDECIDED"
            return "PASS" if bool(evaluated["passed"].fillna(False).all()) else "FAIL"

        summary = report.groupby(
            ["config_name", "horizon"]
        ).apply(_promotion_status, include_groups=False).reset_index(name="promotion_status")
        summary["all_gates_passed"] = summary["promotion_status"] == "PASS"
        report = report.merge(summary, on=["config_name", "horizon"])

    return report


def apply_promotion_gates(
    cv_results: pd.DataFrame,
    gates: dict = None,
) -> pd.DataFrame:
    """
    Apply hard promotion gates per regime bucket (legacy fold-level method).

    Splits CV results by vol regime (sigma_mean terciles across folds),
    then checks that every gate passes in every regime bucket.
    A model fails if ANY gate fails in ANY bucket.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Output of expanding_window_cv (must include ece_cal and sigma_mean).
    gates : dict, optional
        Gate thresholds. Default: {"bss_cal": 0.0, "auc_cal": 0.55, "ece_cal": 0.02}

    Returns
    -------
    gate_report : pd.DataFrame
        Columns: config_name, horizon, regime, metric, value, threshold,
                 passed, margin, all_gates_passed
    """
    if gates is None:
        gates = {"bss_cal": 0.0, "auc_cal": 0.55, "ece_cal": 0.02}

    if len(cv_results) == 0:
        return pd.DataFrame()

    rows = []

    for (config_name, horizon), group in cv_results.groupby(["config_name", "horizon"]):
        # Bucket folds by vol regime using sigma_mean terciles
        if "sigma_mean" in group.columns and group["sigma_mean"].notna().any():
            sigma_vals = group["sigma_mean"].values
            p33 = float(np.nanpercentile(sigma_vals, 33))
            p66 = float(np.nanpercentile(sigma_vals, 66))

            regimes = []
            for s in sigma_vals:
                if np.isnan(s):
                    regimes.append("all")
                elif s < p33:
                    regimes.append("low_vol")
                elif s > p66:
                    regimes.append("high_vol")
                else:
                    regimes.append("mid_vol")
            group = group.copy()
            group["regime"] = regimes
        else:
            group = group.copy()
            group["regime"] = "all"

        for regime, regime_group in group.groupby("regime"):
            for metric, threshold in gates.items():
                if metric not in regime_group.columns:
                    continue
                value = float(regime_group[metric].mean())

                if metric == "ece_cal":
                    passed = value <= threshold
                    margin = round(threshold - value, 6)
                else:
                    passed = value >= threshold
                    margin = round(value - threshold, 6)

                rows.append({
                    "config_name": config_name,
                    "horizon": horizon,
                    "regime": regime,
                    "metric": metric,
                    "value": round(value, 6),
                    "threshold": threshold,
                    "passed": bool(passed),
                    "margin": margin,
                })

    report = pd.DataFrame(rows)

    if len(report) > 0:
        summary = report.groupby(["config_name", "horizon"])["passed"].all().reset_index()
        summary.rename(columns={"passed": "all_gates_passed"}, inplace=True)
        report = report.merge(summary, on=["config_name", "horizon"])

    return report
