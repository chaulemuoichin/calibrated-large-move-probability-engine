"""
Validation tests for the Calibrated Large-Move Probability Engine.

Tests:
    1. Brownian increments mean/var
    2. GBM moment test
    3. Calibration improves Brier
    4. No-lookahead test
    5. Outputs schema test
"""

import sys
import json
import re
import tempfile
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional
from unittest.mock import patch

import numpy as np
import pandas as pd

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from em_sde.monte_carlo import simulate_gbm_terminal, simulate_garch_terminal, compute_move_probability, QUANTILE_LEVELS
from em_sde.calibration import OnlineCalibrator, RegimeCalibrator, MultiFeatureCalibrator, sigmoid, logit
from em_sde.evaluation import (
    brier_score, log_loss, auc_roc, separation, compute_metrics,
    brier_skill_score, effective_sample_size, crps_from_quantiles,
    value_at_risk, conditional_var, return_skewness, return_kurtosis,
    max_drawdown, compute_risk_report,
)
from em_sde.config import PipelineConfig, DataConfig, ModelConfig, CalibrationConfig, OutputConfig
from em_sde.garch import fit_garch, GarchResult, project_to_stationary
from em_sde.monte_carlo import compute_state_dependent_jumps
from em_sde.evaluation import expected_calibration_error
from em_sde.calibration import HistogramCalibrator
from em_sde.model_selection import apply_promotion_gates
from em_sde.backtest import run_walkforward
from em_sde.output import write_outputs
from em_sde.evaluation import compute_reliability


@contextmanager
def assert_raises(exc_type: type[BaseException], match: Optional[str] = None) -> Iterator[None]:
    """Minimal replacement for pytest.raises used in this test file."""
    try:
        yield
    except exc_type as exc:
        if match is not None and re.search(match, str(exc)) is None:
            raise AssertionError(
                f"Expected exception message to match '{match}', got '{exc}'"
            ) from exc
        return
    raise AssertionError(f"Expected {exc_type.__name__} to be raised")


def p_and_se(terminal_prices: np.ndarray, s0: float, threshold: float) -> tuple[float, float]:
    """Extract (p_raw, se) from a 2-or-3 tuple return."""
    result = compute_move_probability(terminal_prices, s0, threshold)
    return float(result[0]), float(result[1])


# ============================================================
# Test 1: Brownian Increments Mean/Var
# ============================================================

class TestBrownianIncrements:
    """Verify that the Euler-Maruyama increments have correct statistics."""

    def test_increments_mean(self):
        """Brownian increments with zero drift should have near-zero mean."""
        rng = np.random.default_rng(42)
        n_paths = 500_000
        sigma_1d = 0.01  # 1% daily vol
        sigma_year = sigma_1d * np.sqrt(252)
        dt = 1.0 / 252.0
        mu_year = 0.0

        # Single step increment in log space
        drift = (mu_year - 0.5 * sigma_year ** 2) * dt
        vol = sigma_year * np.sqrt(dt)
        Z = rng.standard_normal(n_paths)
        increments = drift + vol * Z

        # Expected mean of log-increment: (mu - 0.5*sigma^2)*dt
        expected_mean = drift
        observed_mean = np.mean(increments)

        # Should be within 3 SE
        se = vol / np.sqrt(n_paths)
        assert abs(observed_mean - expected_mean) < 4 * se, \
            f"Mean {observed_mean:.8f} too far from expected {expected_mean:.8f}"

    def test_increments_variance(self):
        """Brownian increments should have variance = sigma_year^2 * dt."""
        rng = np.random.default_rng(123)
        n_paths = 500_000
        sigma_1d = 0.015
        sigma_year = sigma_1d * np.sqrt(252)
        dt = 1.0 / 252.0

        vol = sigma_year * np.sqrt(dt)
        Z = rng.standard_normal(n_paths)
        increments = vol * Z

        expected_var = sigma_year ** 2 * dt
        observed_var = np.var(increments)

        # Chi-squared test: var * (n-1) / expected_var ~ chi2(n-1)
        # For large n, use normal approximation
        rel_error = abs(observed_var - expected_var) / expected_var
        assert rel_error < 0.01, \
            f"Variance {observed_var:.8f} deviates {rel_error*100:.2f}% from expected {expected_var:.8f}"


# ============================================================
# Test 2: GBM Moment Test
# ============================================================

class TestGBMMoments:
    """Verify GBM terminal distribution matches theoretical moments."""

    def test_gbm_mean(self):
        """E[S_T] = S_0 * exp(mu * T) for GBM."""
        S0 = 100.0
        sigma_1d = 0.01
        mu_year = 0.08
        H = 20  # 20 trading days
        T = H / 252.0
        n_paths = 500_000

        terminal = simulate_gbm_terminal(S0, sigma_1d, H, n_paths, mu_year, seed=42)

        expected_mean = S0 * np.exp(mu_year * T)
        observed_mean = np.mean(terminal)

        rel_error = abs(observed_mean - expected_mean) / expected_mean
        assert rel_error < 0.01, \
            f"GBM mean {observed_mean:.4f} vs expected {expected_mean:.4f} (err={rel_error*100:.2f}%)"

    def test_gbm_variance(self):
        """Var[S_T] = S_0^2 * exp(2*mu*T) * (exp(sigma^2*T) - 1)."""
        S0 = 100.0
        sigma_1d = 0.012
        mu_year = 0.05
        H = 10
        T = H / 252.0
        sigma_year = sigma_1d * np.sqrt(252)
        n_paths = 500_000

        terminal = simulate_gbm_terminal(S0, sigma_1d, H, n_paths, mu_year, seed=99)

        expected_var = S0 ** 2 * np.exp(2 * mu_year * T) * (np.exp(sigma_year ** 2 * T) - 1)
        observed_var = np.var(terminal)

        rel_error = abs(observed_var - expected_var) / expected_var
        assert rel_error < 0.05, \
            f"GBM variance {observed_var:.4f} vs expected {expected_var:.4f} (err={rel_error*100:.2f}%)"

    def test_gbm_zero_drift_symmetry(self):
        """With mu=0, log-returns should be symmetric around slightly negative mean."""
        S0 = 100.0
        sigma_1d = 0.01
        H = 5
        n_paths = 200_000

        terminal = simulate_gbm_terminal(S0, sigma_1d, H, n_paths, mu_year=0.0, seed=77)
        log_returns = np.log(terminal / S0)

        # Mean of log returns should be -0.5*sigma_year^2*T
        sigma_year = sigma_1d * np.sqrt(252)
        T = H / 252.0
        expected_log_mean = -0.5 * sigma_year ** 2 * T
        observed_log_mean = np.mean(log_returns)

        se = np.std(log_returns) / np.sqrt(n_paths)
        assert abs(observed_log_mean - expected_log_mean) < 4 * se, \
            f"Log-return mean {observed_log_mean:.8f} vs expected {expected_log_mean:.8f}"


# ============================================================
# Test 2b: Student-t Fat Tail Test
# ============================================================

class TestStudentTFatTails:
    """Verify that Student-t innovations produce fatter tails than Gaussian."""

    def test_t_distribution_more_extreme_moves(self):
        """Student-t simulation should produce more large moves than Gaussian."""
        S0 = 100.0
        sigma_1d = 0.01
        H = 20
        n_paths = 200_000
        threshold = 2.0 * sigma_1d * np.sqrt(H)  # k=2 threshold

        # Gaussian
        term_gauss = simulate_gbm_terminal(S0, sigma_1d, H, n_paths, seed=42, t_df=0)
        p_gauss, _ = p_and_se(term_gauss, S0, threshold)

        # Student-t with df=5
        term_t = simulate_gbm_terminal(S0, sigma_1d, H, n_paths, seed=42, t_df=5.0)
        p_t, _ = p_and_se(term_t, S0, threshold)

        assert p_t > p_gauss, \
            f"Student-t probability {p_t:.4f} should exceed Gaussian {p_gauss:.4f}"

    def test_t_distribution_unit_variance(self):
        """Scaled Student-t increments should have unit variance."""
        rng = np.random.default_rng(42)
        nu = 5.0
        n = 500_000
        # Same scaling used in monte_carlo.py
        Z = rng.standard_t(df=nu, size=n) * np.sqrt((nu - 2.0) / nu)
        observed_var = np.var(Z)
        assert abs(observed_var - 1.0) < 0.02, \
            f"Scaled t variance {observed_var:.4f} should be ~1.0"

    def test_gaussian_fallback_when_t_df_zero(self):
        """t_df=0 should produce identical results to default (Gaussian)."""
        S0 = 100.0
        sigma_1d = 0.01
        H = 10
        n_paths = 50_000

        term_default = simulate_gbm_terminal(S0, sigma_1d, H, n_paths, seed=42)
        term_zero = simulate_gbm_terminal(S0, sigma_1d, H, n_paths, seed=42, t_df=0)

        np.testing.assert_array_equal(term_default, term_zero)


# ============================================================
# Test 3: Calibration Improves Brier
# ============================================================

class TestCalibrationImprovesBrier:
    """Verify that online calibration reduces Brier score on biased predictions."""

    def test_calibration_reduces_brier_on_biased_data(self):
        """
        Generate systematically biased raw probabilities and verify
        that online calibration produces a lower Brier score.
        """
        rng = np.random.default_rng(42)
        n = 1000

        # True event rate ~15%
        true_prob = 0.15
        y = rng.binomial(1, true_prob, size=n).astype(float)

        # Raw predictions are systematically biased high (overconfident)
        p_raw = np.clip(rng.normal(0.30, 0.08, size=n), 0.01, 0.99)

        # Train calibrator on first half, evaluate on second half
        # min_updates=0 so calibration is active from the start
        cal = OnlineCalibrator(lr=0.05, adaptive_lr=False, min_updates=0)
        split = n // 2

        # Train phase
        for i in range(split):
            cal.update(p_raw[i], y[i])

        # Evaluate on held-out data
        p_cal_test = np.array([cal.calibrate(p) for p in p_raw[split:]])
        y_test = y[split:]
        p_raw_test = p_raw[split:]

        brier_raw = brier_score(p_raw_test, y_test)
        brier_cal = brier_score(p_cal_test, y_test)

        assert brier_cal < brier_raw, \
            f"Calibrated Brier {brier_cal:.6f} should be < raw Brier {brier_raw:.6f}"

    def test_calibration_identity_on_wellcalibrated(self):
        """If raw predictions are already calibrated, calibration should not degrade."""
        rng = np.random.default_rng(123)
        n = 500

        p_raw = rng.uniform(0.05, 0.50, size=n)
        y = (rng.uniform(size=n) < p_raw).astype(float)

        cal = OnlineCalibrator(lr=0.02, adaptive_lr=False, min_updates=0)
        for i in range(n):
            cal.update(p_raw[i], y[i])

        p_cal = np.array([cal.calibrate(p) for p in p_raw])
        brier_raw = brier_score(p_raw, y)
        brier_cal = brier_score(p_cal, y)

        # Should not be much worse (within 20%)
        assert brier_cal < brier_raw * 1.20, \
            f"Calibration degraded too much: {brier_cal:.6f} vs {brier_raw:.6f}"


# ============================================================
# Test 4: No-Lookahead Test
# ============================================================

class TestNoLookahead:
    """
    Verify strict no-lookahead: predictions at date t must be identical
    regardless of whether future data exists.
    """

    def _make_synthetic_df(self, n_days: int, seed: int = 42) -> pd.DataFrame:
        """Generate a synthetic price series."""
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0003, 0.012, size=n_days)
        log_prices = np.cumsum(returns)
        prices = 100.0 * np.exp(log_prices)
        prices = np.concatenate([[100.0], prices])
        dates = pd.bdate_range("2020-01-01", periods=n_days + 1)
        return pd.DataFrame({"price": prices}, index=dates)

    def _make_config(self, ensemble: bool = False) -> PipelineConfig:
        return PipelineConfig(
            data=DataConfig(source="synthetic", min_rows=252),
            model=ModelConfig(
                horizons=[5, 10],
                garch_window=300,
                garch_min_window=252,
                mc_base_paths=5000,  # small for speed
                mc_boost_paths=5000,
                seed=42,
            ),
            calibration=CalibrationConfig(
                lr=0.05,
                ensemble_enabled=ensemble,
                ensemble_weights=[0.6, 0.4] if not ensemble else [0.6, 0.4],
            ),
            output=OutputConfig(base_dir=tempfile.mkdtemp()),
        )

    def test_predictions_unchanged_with_extra_future_data(self):
        """
        Run backtest on data[0:N] and data[0:N+50].
        Predictions for the same dates must be identical.
        """
        df_long = self._make_synthetic_df(800, seed=42)
        cfg = self._make_config()

        # Run on shorter dataset
        df_short = df_long.iloc[:700].copy()
        results_short = run_walkforward(df_short, cfg)

        # Run on longer dataset
        results_long = run_walkforward(df_long, cfg)

        # Compare predictions for overlapping dates
        short_dates = list(results_short["date"].values)
        long_subset = results_long[results_long["date"].isin(short_dates)]

        # Merge on date
        merged = results_short.merge(
            long_subset, on="date", suffixes=("_short", "_long"),
        )

        assert len(merged) > 0, "No overlapping dates found"

        for H in cfg.model.horizons:
            raw_short = merged[f"p_raw_{H}_short"].to_numpy(dtype=float)
            raw_long = merged[f"p_raw_{H}_long"].to_numpy(dtype=float)

            # Raw MC probabilities should be identical (same seed, same input data)
            np.testing.assert_array_almost_equal(
                raw_short, raw_long, decimal=10,
                err_msg=f"H={H}: raw probabilities differ between short and long runs",
            )

            # Sigma should be identical
            sig_short = merged["sigma_garch_1d_short"].to_numpy(dtype=float)
            sig_long = merged["sigma_garch_1d_long"].to_numpy(dtype=float)
            np.testing.assert_array_almost_equal(
                sig_short, sig_long, decimal=10,
                err_msg=f"Sigma differs between short and long runs",
            )

    def test_labels_only_filled_for_resolvable(self):
        """Labels (y_H) should be NaN for predictions where t+H exceeds data."""
        df = self._make_synthetic_df(500, seed=42)
        cfg = self._make_config()
        results = run_walkforward(df, cfg)

        max_date = df.index[-1]
        for H in cfg.model.horizons:
            # Last H rows should have NaN labels
            last_rows = results.tail(H)
            is_nan = np.asarray(last_rows[f"y_{H}"].isna(), dtype=bool)
            assert bool(is_nan.all()), \
                f"H={H}: last {H} rows should have NaN labels"


# ============================================================
# Test 5: Outputs Schema Test
# ============================================================

class TestOutputsSchema:
    """Verify output files conform to the spec schema."""

    def _run_mini_pipeline(self) -> tuple:
        """Run a minimal pipeline and return (results, out_dir)."""
        rng = np.random.default_rng(42)
        n_days = 500
        returns = rng.normal(0.0003, 0.012, size=n_days)
        prices = 100.0 * np.exp(np.concatenate([[0], np.cumsum(returns)]))
        dates = pd.bdate_range("2020-01-01", periods=n_days + 1)
        df = pd.DataFrame({"price": prices}, index=dates)

        out_dir = tempfile.mkdtemp()
        cfg = PipelineConfig(
            data=DataConfig(source="synthetic", min_rows=252),
            model=ModelConfig(
                horizons=[5, 10, 20],
                garch_window=300,
                garch_min_window=252,
                mc_base_paths=2000,
                mc_boost_paths=2000,
                seed=42,
            ),
            calibration=CalibrationConfig(
                lr=0.05,
                ensemble_enabled=True,
                ensemble_weights=[0.5, 0.3, 0.2],
            ),
            output=OutputConfig(base_dir=out_dir, charts=True),
        )

        results = run_walkforward(df, cfg)
        metrics = compute_metrics(results, cfg.model.horizons)
        reliability = compute_reliability(results, cfg.model.horizons)
        run_id = "test_run"
        final_dir = write_outputs(results, reliability, metrics,
                                  {"ticker": "SYNTH"}, cfg, run_id)
        return results, final_dir, cfg

    def test_results_csv_columns(self):
        """results.csv must contain all required columns."""
        results, out_dir, cfg = self._run_mini_pipeline()

        csv_path = out_dir / "results.csv"
        assert csv_path.exists(), "results.csv not found"

        df = pd.read_csv(csv_path)
        required = ["date", "sigma_garch_1d", "sigma_source"]

        for H in cfg.model.horizons:
            required.extend([
                f"thr_{H}",
                f"p_raw_{H}", f"p_cal_{H}", f"mc_se_{H}",
                f"ci_low_{H}", f"ci_high_{H}", f"paths_used_{H}",
                f"y_{H}", f"realized_return_{H}",
                f"calib_a_{H}", f"calib_b_{H}",
            ])

        if cfg.calibration.ensemble_enabled:
            required.extend(["risk_combo", "p_meta20"])

        for col in required:
            assert col in df.columns, f"Missing required column: {col}"

    def test_summary_json_structure(self):
        """summary.json must contain required keys."""
        _, out_dir, _ = self._run_mini_pipeline()

        json_path = out_dir / "summary.json"
        assert json_path.exists(), "summary.json not found"

        with open(json_path) as f:
            summary = json.load(f)

        assert "run_id" in summary
        assert "config" in summary
        assert "unit_convention" in summary
        assert "metrics" in summary
        assert "overlapping" in summary["metrics"]
        assert "non_overlapping" in summary["metrics"]

        # Check unit convention documentation
        uc = summary["unit_convention"]
        assert "sigma_1d" in uc
        assert "sigma_year" in uc
        assert "simulation_dt" in uc

    def test_charts_generated(self):
        """All 4 required chart PNGs must be generated."""
        _, out_dir, _ = self._run_mini_pipeline()

        charts_dir = out_dir / "charts"
        required_charts = [
            "probability_timeseries.png",
            "reliability_diagram.png",
            "realized_return_hist.png",
            "rolling_brier.png",
        ]

        for chart in required_charts:
            assert (charts_dir / chart).exists(), f"Missing chart: {chart}"
            assert (charts_dir / chart).stat().st_size > 0, f"Empty chart: {chart}"

    def test_reliability_csv(self):
        """reliability.csv must exist and have correct structure."""
        _, out_dir, _ = self._run_mini_pipeline()

        rel_path = out_dir / "reliability.csv"
        assert rel_path.exists(), "reliability.csv not found"

        df = pd.read_csv(rel_path)
        if len(df) > 0:
            for col in ["horizon", "type", "bin_mid", "bin_count",
                         "mean_predicted", "mean_observed"]:
                assert col in df.columns, f"Missing column in reliability.csv: {col}"

    def test_probabilities_in_valid_range(self):
        """All probabilities must be in [0, 1]."""
        results, _, cfg = self._run_mini_pipeline()

        for H in cfg.model.horizons:
            for col in [f"p_raw_{H}", f"p_cal_{H}", f"ci_low_{H}", f"ci_high_{H}"]:
                vals = results[col].dropna().values
                assert np.all(vals >= 0), f"{col} has values < 0"
                assert np.all(vals <= 1), f"{col} has values > 1"

    def test_ci_contains_point_estimate(self):
        """Confidence intervals must contain the point estimate."""
        results, _, cfg = self._run_mini_pipeline()

        for H in cfg.model.horizons:
            p = results[f"p_raw_{H}"].values
            lo = results[f"ci_low_{H}"].values
            hi = results[f"ci_high_{H}"].values
            mask = ~np.isnan(p)

            assert np.all(lo[mask] <= p[mask] + 1e-10), f"ci_low > p_raw for H={H}"
            assert np.all(hi[mask] >= p[mask] - 1e-10), f"ci_high < p_raw for H={H}"


# ============================================================
# Additional unit tests for calibration math
# ============================================================

class TestCalibrationMath:

    def test_sigmoid_logit_inverse(self):
        """sigmoid and logit should be inverses."""
        for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            assert abs(sigmoid(logit(p)) - p) < 1e-10

    def test_calibrator_identity_init(self):
        """Fresh calibrator should return p_raw (warm-up or identity a=0,b=1)."""
        cal = OnlineCalibrator(min_updates=0)
        for p in [0.05, 0.2, 0.5, 0.8, 0.95]:
            assert abs(cal.calibrate(p) - p) < 1e-6, \
                f"Identity calibration failed for p={p}"

    def test_calibrator_warmup_returns_raw(self):
        """During warm-up period, calibrator should return p_raw unchanged."""
        cal = OnlineCalibrator(lr=0.05, min_updates=10)
        # Update a few times (less than min_updates)
        for _ in range(5):
            cal.update(0.1, 1.0)
        # Should still return p_raw since n_updates < min_updates
        assert abs(cal.calibrate(0.3) - 0.3) < 1e-10

    def test_adaptive_lr_decreases(self):
        """Adaptive lr should produce smaller updates over time."""
        cal = OnlineCalibrator(lr=0.1, adaptive_lr=True, min_updates=0)
        # First update
        cal.update(0.5, 1.0)
        a_after_1 = cal.a
        # Many more updates with same input to build n_updates
        for _ in range(99):
            cal.update(0.5, 0.0)
        a_before_last = cal.a
        cal.update(0.5, 1.0)
        a_after_last = cal.a
        # The change from the last update should be much smaller than the first
        last_change = abs(a_after_last - a_before_last)
        first_change = abs(a_after_1 - 0.0)
        assert last_change < first_change, \
            f"Adaptive lr not decreasing: first={first_change:.6f}, last={last_change:.6f}"


# ============================================================
# Test 7: GARCH Parameter Extraction
# ============================================================

class TestGarchParameterExtraction:
    """Verify GARCH parameter extraction from fit_garch()."""

    def test_garch_returns_result_object(self):
        """fit_garch() should return a GarchResult dataclass."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, size=500)
        result = fit_garch(returns, window=500, min_window=252)
        assert isinstance(result, GarchResult)
        assert isinstance(result.sigma_1d, float)
        assert isinstance(result.source, str)

    def test_garch_returns_parameters(self):
        """fit_garch() should return omega, alpha, beta when GARCH fits."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, size=500)
        result = fit_garch(returns, window=500, min_window=252, model_type="garch")
        assert result.omega is not None, "omega should not be None"
        assert result.alpha is not None, "alpha should not be None"
        assert result.beta is not None, "beta should not be None"
        assert result.omega > 0, f"omega should be positive, got {result.omega}"
        assert result.alpha >= 0, f"alpha should be non-negative, got {result.alpha}"
        assert result.beta >= 0, f"beta should be non-negative, got {result.beta}"

    def test_gjr_returns_gamma(self):
        """GJR-GARCH should return gamma parameter."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, size=500)
        result = fit_garch(returns, window=500, min_window=252, model_type="gjr")
        assert result.gamma is not None, "gamma should not be None for GJR"
        assert result.source == "gjr_garch"

    def test_ewma_fallback_null_parameters(self):
        """EWMA fallback should return None for GARCH parameters."""
        result = GarchResult(sigma_1d=0.01, source="ewma_fallback")
        assert result.source == "ewma_fallback"
        assert result.omega is None
        assert result.alpha is None
        assert result.beta is None
        assert result.gamma is None


# ============================================================
# Test 8: GARCH-in-Simulation
# ============================================================

class TestGarchInSimulation:
    """Verify GARCH-in-simulation produces variable p_raw."""

    def test_garch_sim_breaks_constant_praw(self):
        """
        GARCH-in-sim should produce different p_raw for different sigma_1d
        even with the same GARCH parameters, because starting from different
        initial variance states leads to different path dynamics.
        """
        H = 20
        n_paths = 100_000
        omega = 1e-6   # small intercept
        alpha = 0.10
        beta = 0.85

        # Low vol regime
        sigma_low = 0.008
        thr_low = 2.0 * sigma_low * np.sqrt(H)
        term_low = simulate_garch_terminal(
            100.0, sigma_low, H, n_paths, omega, alpha, beta, seed=42,
        )
        p_low, _ = p_and_se(term_low, 100.0, thr_low)

        # High vol regime
        sigma_high = 0.025
        thr_high = 2.0 * sigma_high * np.sqrt(H)
        term_high = simulate_garch_terminal(
            100.0, sigma_high, H, n_paths, omega, alpha, beta, seed=42,
        )
        p_high, _ = p_and_se(term_high, 100.0, thr_high)

        # p_raw should differ meaningfully (the whole point of the fix)
        assert abs(p_high - p_low) > 0.005, \
            f"p_raw still constant: p_low={p_low:.4f}, p_high={p_high:.4f}"

    def test_garch_sim_reduces_to_gbm_when_flat(self):
        """
        When alpha~0 and beta~0, GARCH-in-sim should approximate constant-vol GBM.
        """
        sigma_1d = 0.01
        H = 10
        n_paths = 200_000
        alpha = 0.001
        beta = 0.001
        omega = sigma_1d ** 2 * (1 - alpha - beta)

        term_garch = simulate_garch_terminal(
            100.0, sigma_1d, H, n_paths, omega, alpha, beta, seed=42,
        )
        term_gbm = simulate_gbm_terminal(
            100.0, sigma_1d, H, n_paths, seed=42,
        )

        # Means should be close (both E[S_T] ~ S0 with zero drift)
        rel_error = abs(np.mean(term_garch) - np.mean(term_gbm)) / np.mean(term_gbm)
        assert rel_error < 0.02, \
            f"GARCH-flat mean {np.mean(term_garch):.4f} vs GBM {np.mean(term_gbm):.4f}"

    def test_garch_sim_with_student_t(self):
        """GARCH-in-sim with Student-t should produce more extreme moves than Gaussian."""
        S0 = 100.0
        sigma_1d = 0.01
        H = 20
        n_paths = 100_000
        omega = 1e-6
        alpha = 0.10
        beta = 0.85
        threshold = 2.0 * sigma_1d * np.sqrt(H)

        term_gauss = simulate_garch_terminal(
            S0, sigma_1d, H, n_paths, omega, alpha, beta, seed=42, t_df=0,
        )
        p_gauss, _ = p_and_se(term_gauss, S0, threshold)

        term_t = simulate_garch_terminal(
            S0, sigma_1d, H, n_paths, omega, alpha, beta, seed=42, t_df=5.0,
        )
        p_t, _ = p_and_se(term_t, S0, threshold)

        assert p_t > p_gauss, \
            f"Student-t p_raw {p_t:.4f} should exceed Gaussian {p_gauss:.4f}"


# ============================================================
# Test 9: Jump-Diffusion
# ============================================================

class TestJumpDiffusion:
    """Verify jump-diffusion produces more extreme tails."""

    def test_jumps_increase_tail_probability(self):
        """Adding jumps should increase the probability of large moves."""
        S0, sigma_1d, H, n_paths = 100.0, 0.01, 20, 200_000
        threshold = 2.0 * sigma_1d * np.sqrt(H)
        omega = 1e-6
        alpha, beta = 0.10, 0.85

        # Without jumps
        term_no_jump = simulate_garch_terminal(
            S0, sigma_1d, H, n_paths, omega, alpha, beta, seed=42,
            jump_intensity=0.0,
        )
        p_no_jump, _ = p_and_se(term_no_jump, S0, threshold)

        # With jumps (high intensity for clear signal)
        term_jump = simulate_garch_terminal(
            S0, sigma_1d, H, n_paths, omega, alpha, beta, seed=42,
            jump_intensity=5.0, jump_mean=-0.03, jump_vol=0.05,
        )
        p_jump, _ = p_and_se(term_jump, S0, threshold)

        assert p_jump > p_no_jump, \
            f"Jumps should increase tail prob: {p_jump:.4f} vs {p_no_jump:.4f}"

    def test_no_jumps_identical_to_garch_only(self):
        """jump_intensity=0 should produce identical results to default (no jump params)."""
        S0, sigma_1d, H, n_paths = 100.0, 0.01, 10, 50_000
        omega, alpha, beta = 1e-6, 0.10, 0.85

        term1 = simulate_garch_terminal(
            S0, sigma_1d, H, n_paths, omega, alpha, beta, seed=42,
            jump_intensity=0.0,
        )
        term2 = simulate_garch_terminal(
            S0, sigma_1d, H, n_paths, omega, alpha, beta, seed=42,
        )
        np.testing.assert_array_equal(term1, term2)


# ============================================================
# Test 10: Backward Compatibility
# ============================================================

class TestBackwardCompatibility:
    """Verify all existing behavior is preserved when new features are disabled."""

    def test_default_config_has_features_disabled(self):
        """Default ModelConfig should have new features disabled."""
        cfg = ModelConfig()
        assert cfg.garch_in_sim is False
        assert cfg.jump_enabled is False
        assert cfg.garch_model_type == "garch"

    def test_config_validation_garch_model_type(self):
        """Invalid garch_model_type should raise."""
        cfg = PipelineConfig()
        cfg.model.garch_model_type = "invalid"
        with assert_raises(AssertionError):
            from em_sde.config import _validate
            _validate(cfg)

    def test_config_validation_jump_params(self):
        """Negative jump_vol should raise when jumps enabled."""
        cfg = PipelineConfig()
        cfg.model.jump_enabled = True
        cfg.model.jump_vol = -1.0
        with assert_raises(AssertionError):
            from em_sde.config import _validate
            _validate(cfg)


# ============================================================
# Test 11: Seed Independence Across Horizons
# ============================================================

class TestSeedIndependence:
    """Verify that different horizons get different RNG seeds."""

    def test_horizons_get_different_seeds(self):
        """The seed construction should produce distinct seeds per horizon."""
        from numpy.random import SeedSequence
        ss = SeedSequence(42)
        children = ss.spawn(3)
        seeds = [int(c.generate_state(1)[0]) + 100 for c in children]
        # All seeds must be distinct
        assert len(set(seeds)) == 3, \
            f"Horizons share seeds: {seeds}"

    def test_different_horizons_produce_different_paths(self):
        """MC simulation for H=5 and H=10 should use different RNG streams."""
        from numpy.random import SeedSequence, default_rng
        ss = SeedSequence(42)
        children = ss.spawn(3)
        idx = 300

        # Simulate what backtest does with the fixed code
        seeds = [int(c.generate_state(1)[0]) + idx for c in children]
        assert seeds[0] != seeds[1], \
            f"H=5 and H=10 got same seed: {seeds[0]}"
        assert seeds[1] != seeds[2], \
            f"H=10 and H=20 got same seed: {seeds[1]}"

        # Verify the actual RNG streams differ
        rng0 = default_rng(SeedSequence(seeds[0] % (2**31)))
        rng1 = default_rng(SeedSequence(seeds[1] % (2**31)))
        v0 = rng0.standard_normal(5)
        v1 = rng1.standard_normal(5)
        assert not np.allclose(v0, v1), \
            "Different horizons should produce different random draws"


# ============================================================
# Test 12: Calibration Safety Gate
# ============================================================

class TestCalibrationSafetyGate:
    """Verify the calibration safety gate mechanism."""

    def test_gate_disabled_by_default(self):
        """Safety gate should be off by default."""
        cal = OnlineCalibrator()
        assert cal.safety_gate is False
        assert cal.gated is False

    def test_gate_triggers_when_cal_hurts(self):
        """Gate should activate when calibration degrades Brier."""
        cal = OnlineCalibrator(
            lr=0.3, adaptive_lr=False, min_updates=0,
            safety_gate=True, gate_window=50,
        )
        # Deliberately train the calibrator to be badly miscalibrated:
        # feed it biased data that pushes a/b into bad territory
        for _ in range(30):
            cal.update(0.05, 1.0)  # push predictions up aggressively

        # Now feed correct data where p_raw=0.05 is about right
        # The calibrator has been warped, so p_cal will be wrong
        for _ in range(50):
            cal.update(0.05, 0.0)

        # After enough evidence, gate should be active
        # (cal Brier > raw Brier because calibrator is warped)
        assert cal.gated is True, \
            "Gate should activate when calibration hurts performance"

    def test_gate_returns_raw_when_active(self):
        """When gate is active, calibrate() should return p_raw."""
        cal = OnlineCalibrator(
            lr=0.1, adaptive_lr=False, min_updates=0,
            safety_gate=True, gate_window=20,
        )
        # Force gate active by manipulating internal state
        cal._gate_active = True
        p_raw = 0.07
        result = cal.calibrate(p_raw)
        assert result == p_raw, \
            f"Gated calibrator should return raw ({p_raw}), got {result}"

    def test_gate_passes_through_when_cal_helps(self):
        """Gate should remain off when calibration genuinely improves scores."""
        cal = OnlineCalibrator(
            lr=0.05, adaptive_lr=True, min_updates=0,
            safety_gate=True, gate_window=100,
        )
        # Feed BIASED data: raw predicts 0.05 but true rate is ~0.15
        # Calibration should learn to push predictions up, improving Brier
        rng = np.random.default_rng(42)
        for _ in range(300):
            p = 0.05
            y = 1.0 if rng.random() < 0.15 else 0.0
            cal.update(p, y)

        # Cal should be correcting the bias, so gate should NOT trigger
        assert cal.gated is False, \
            "Gate should not trigger when calibration fixes raw bias"

    def test_config_default_gate_disabled(self):
        """Default CalibrationConfig should have safety_gate disabled."""
        cfg = CalibrationConfig()
        assert cfg.safety_gate is False
        assert cfg.gate_window == 200


# ============================================================
# Test: AUC-ROC Metric
# ============================================================

class TestAUCROC:
    """Verify AUC-ROC computation."""

    def test_perfect_separation(self):
        """Perfect predictions should give AUC = 1.0."""
        y = np.array([1, 1, 1, 0, 0, 0], dtype=float)
        p = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert abs(auc_roc(p, y) - 1.0) < 1e-10

    def test_random_predictions(self):
        """Random predictions should give AUC ≈ 0.5."""
        rng = np.random.default_rng(42)
        n = 10000
        y = rng.integers(0, 2, n).astype(float)
        p = rng.random(n)
        auc = auc_roc(p, y)
        assert 0.45 < auc < 0.55, f"Random AUC should be ~0.5, got {auc}"

    def test_single_class_returns_nan(self):
        """If all labels are same class, AUC should be NaN."""
        y = np.array([1, 1, 1, 1], dtype=float)
        p = np.array([0.9, 0.8, 0.7, 0.6])
        assert np.isnan(auc_roc(p, y))

        y = np.array([0, 0, 0, 0], dtype=float)
        p = np.array([0.1, 0.2, 0.3, 0.4])
        assert np.isnan(auc_roc(p, y))


# ============================================================
# Test: Separation Metric
# ============================================================

class TestSeparationMetric:
    """Verify separation metric."""

    def test_positive_separation_with_biased_predictions(self):
        """Higher predictions for events should give positive separation."""
        y = np.array([1, 1, 0, 0], dtype=float)
        p = np.array([0.8, 0.7, 0.2, 0.1])
        sep = separation(p, y)
        assert sep > 0, f"Expected positive separation, got {sep}"
        assert abs(sep - 0.6) < 1e-10  # (0.75 - 0.15)

    def test_zero_separation_with_constant_predictions(self):
        """Constant predictions should give zero separation."""
        y = np.array([1, 1, 0, 0], dtype=float)
        p = np.array([0.5, 0.5, 0.5, 0.5])
        sep = separation(p, y)
        assert abs(sep) < 1e-10


# ============================================================
# Test: Regime Calibrator
# ============================================================

class TestRegimeCalibrator:
    """Verify regime-conditional calibration."""

    def test_assigns_different_regimes(self):
        """Different vol levels should map to different regime bins."""
        cal = RegimeCalibrator(n_bins=3, lr=0.05, min_updates=0)
        # Fill vol history with known distribution
        for v in np.linspace(0.005, 0.03, 100):
            cal.observe_vol(v)

        # Low vol should be regime 0
        assert cal._get_regime(0.006) == 0
        # High vol should be regime 2
        assert cal._get_regime(0.029) == 2

    def test_updates_correct_regime_calibrator(self):
        """Updates should go to the regime-specific calibrator."""
        cal = RegimeCalibrator(n_bins=3, lr=0.1, adaptive_lr=False, min_updates=0)
        # Fill vol history
        for v in np.linspace(0.005, 0.03, 100):
            cal.observe_vol(v)

        # Update in high-vol regime
        high_vol = 0.029
        cal.update(0.05, 1.0, high_vol)
        # Only regime 2 calibrator should have updates
        assert cal.calibrators[2].n_updates == 1
        assert cal.calibrators[0].n_updates == 0
        assert cal.calibrators[1].n_updates == 0

    def test_warmup_uses_default_bin(self):
        """During warmup, all assignments should go to bin 0."""
        cal = RegimeCalibrator(n_bins=3, lr=0.05, min_updates=0)
        # Only 10 observations (below warmup=50)
        for v in np.linspace(0.005, 0.03, 10):
            cal.observe_vol(v)

        # Even a high vol should map to bin 0 during warmup
        assert cal._get_regime(0.03) == 0

    def test_config_defaults(self):
        """Default CalibrationConfig should have regime_conditional disabled."""
        cfg = CalibrationConfig()
        assert cfg.regime_conditional is False
        assert cfg.regime_n_bins == 3


# ============================================================
# Test: Fixed Threshold Mode (Phase 1)
# ============================================================

class TestFixedThreshold:
    """Verify that fixed_pct threshold mode breaks self-referencing bias."""

    def test_fixed_pct_produces_varying_p_raw(self):
        """With fixed_pct threshold, p_raw should vary across vol regimes."""
        # Low vol regime
        terminal_low = simulate_gbm_terminal(100.0, 0.005, 10, 50000, 0.0, 42)
        p_low, _ = p_and_se(terminal_low, 100.0, 0.05)

        # High vol regime
        terminal_high = simulate_gbm_terminal(100.0, 0.025, 10, 50000, 0.0, 42)
        p_high, _ = p_and_se(terminal_high, 100.0, 0.05)

        # Fixed threshold: high vol should produce MUCH more events
        assert p_high > p_low + 0.05, \
            f"Expected p_high >> p_low with fixed threshold, got {p_high:.4f} vs {p_low:.4f}"

    def test_vol_scaled_produces_constant_p_raw(self):
        """With vol_scaled threshold, p_raw is nearly constant across regimes."""
        k = 2.0
        H = 10

        # Low vol
        sigma_low = 0.005
        thr_low = k * sigma_low * np.sqrt(H)
        terminal_low = simulate_gbm_terminal(100.0, sigma_low, H, 50000, 0.0, 42)
        p_low, _ = p_and_se(terminal_low, 100.0, thr_low)

        # High vol
        sigma_high = 0.025
        thr_high = k * sigma_high * np.sqrt(H)
        terminal_high = simulate_gbm_terminal(100.0, sigma_high, H, 50000, 0.0, 42)
        p_high, _ = p_and_se(terminal_high, 100.0, thr_high)

        # Vol-scaled: both should be ~4.55% (mathematical constant for k=2)
        assert abs(p_high - p_low) < 0.02, \
            f"Vol-scaled should produce similar p_raw, got {p_high:.4f} vs {p_low:.4f}"

    def test_config_defaults_backward_compatible(self):
        """Default config should use vol_scaled (legacy behavior)."""
        cfg = ModelConfig()
        assert cfg.threshold_mode == "vol_scaled"
        assert cfg.fixed_threshold_pct == 0.05

    def test_config_validation_threshold_mode(self):
        """Invalid threshold_mode should raise."""
        cfg = PipelineConfig()
        cfg.model.threshold_mode = "invalid"
        with assert_raises(AssertionError, match="threshold_mode"):
            from em_sde.config import _validate
            _validate(cfg)


# ============================================================
# Test: Anchored Vol Threshold (Phase 1B)
# ============================================================

class TestAnchoredVolThreshold:
    """Verify anchored_vol threshold behavior."""

    def test_anchored_vol_is_walk_forward_safe(self):
        """Unconditional vol computed from expanding window uses only past data."""
        rng = np.random.default_rng(42)
        all_returns = rng.normal(0, 0.015, 1000)

        # Simulate expanding-window unconditional vol at two points
        sigma_at_500 = float(np.std(all_returns[:500]))
        sigma_at_600 = float(np.std(all_returns[:600]))

        # The 100 extra data points should barely change the expanding-window estimate
        relative_change = abs(sigma_at_600 - sigma_at_500) / sigma_at_500
        assert relative_change < 0.05, \
            f"Expanding-window vol should be stable, changed {relative_change:.4f}"

    def test_anchored_creates_discrimination(self):
        """
        With anchored_vol threshold, high-vol periods should produce
        higher MC exceedance probability than low-vol periods.
        """
        k = 2.0
        H = 10

        # Simulate long-run vol
        sigma_uncond = 0.012  # long-run average

        # Threshold is anchored to unconditional vol
        thr_anchored = k * sigma_uncond * np.sqrt(H)

        # Simulate with low conditional vol
        terminal_low = simulate_gbm_terminal(100.0, 0.008, H, 50000, 0.0, 42)
        p_low, _ = p_and_se(terminal_low, 100.0, thr_anchored)

        # Simulate with high conditional vol
        terminal_high = simulate_gbm_terminal(100.0, 0.020, H, 50000, 0.0, 42)
        p_high, _ = p_and_se(terminal_high, 100.0, thr_anchored)

        # High vol should produce more exceedances vs fixed threshold
        assert p_high > p_low + 0.03, \
            f"Anchored should discriminate: p_high={p_high:.4f} vs p_low={p_low:.4f}"


# ============================================================
# Test: Multi-Feature Calibrator (Phase 2)
# ============================================================

class TestMultiFeatureCalibrator:
    """Verify multi-feature online logistic calibration."""

    def test_identity_init(self):
        """Initial calibration should approximate identity mapping."""
        cal = MultiFeatureCalibrator(min_updates=0)
        # With w=[0,1,0,0,0,0], calibrate should return ~p_raw
        p_raw = 0.06
        p_cal = cal.calibrate(p_raw, 0.01, 0.0, 1.0, 0.0)
        assert abs(p_cal - p_raw) < 0.001, f"Expected ~{p_raw}, got {p_cal}"

    def test_warmup_returns_raw(self):
        """During warmup, should return p_raw unchanged."""
        cal = MultiFeatureCalibrator(min_updates=100)
        p_raw = 0.08
        # Only 10 updates
        for _ in range(10):
            cal.update(0.05, 0.0, 0.01, 0.0, 1.0, 0.0)
        p_cal = cal.calibrate(p_raw, 0.01, 0.0, 1.0, 0.0)
        assert p_cal == p_raw

    def test_l2_prevents_weight_explosion(self):
        """L2 regularization should keep weights bounded."""
        cal = MultiFeatureCalibrator(lr=0.1, l2_reg=0.01, min_updates=0)
        # Feed extreme inputs repeatedly
        for _ in range(500):
            cal.update(0.99, 1.0, 0.05, 0.02, 3.0, 0.01)
        # Weights should not explode
        max_w = float(np.max(np.abs(cal.w)))
        assert max_w < 100.0, f"Weights exploded: max |w| = {max_w}"

    def test_gradient_clipping(self):
        """Gradient clipping should prevent single-step jumps."""
        cal = MultiFeatureCalibrator(lr=1.0, l2_reg=0.0, min_updates=0)
        w_before = cal.w.copy()
        # Extreme update
        cal.update(0.001, 1.0, 0.1, 0.05, 5.0, 0.03)
        w_after = cal.w.copy()
        # Change should be bounded by clipping
        w_change = float(np.linalg.norm(w_after - w_before))
        assert w_change < 20.0, f"Update too large without clipping: {w_change}"

    def test_state_returns_weights(self):
        """State should include full weight vector."""
        cal = MultiFeatureCalibrator()
        state = cal.state()
        assert "weights" in state
        assert len(state["weights"]) == 6
        assert state["b"] == 1.0  # initial logit weight


# ============================================================
# Test: Brier Skill Score (Phase 3A)
# ============================================================

class TestBrierSkillScore:
    """Verify Brier Skill Score computation."""

    def test_climatology_gives_zero_bss(self):
        """Predicting the event rate for every sample should give BSS = 0."""
        rng = np.random.default_rng(42)
        y = (rng.random(1000) < 0.05).astype(float)
        p_bar = float(y.mean())
        p = np.full_like(y, p_bar)
        bss = brier_skill_score(p, y)
        assert abs(bss) < 1e-10, f"Expected BSS=0 for climatology, got {bss}"

    def test_perfect_predictions_give_positive_bss(self):
        """Perfect predictions should give BSS = 1."""
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        p = y.copy()  # perfect
        bss = brier_skill_score(p, y)
        assert abs(bss - 1.0) < 1e-10, f"Expected BSS=1, got {bss}"

    def test_worse_than_climatology_gives_negative_bss(self):
        """Anti-correlated predictions should give BSS < 0."""
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        p = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        bss = brier_skill_score(p, y)
        assert bss < 0, f"Expected BSS<0, got {bss}"


# ============================================================
# Test: Effective Sample Size (Phase 3B)
# ============================================================

class TestEffectiveSampleSize:
    """Verify effective sample size for overlapping windows."""

    def test_iid_neff_equals_n(self):
        """For iid data (H=1), N_eff should equal N."""
        rng = np.random.default_rng(42)
        y = (rng.random(500) < 0.05).astype(float)
        n_eff = effective_sample_size(y, H=1)
        assert abs(n_eff - 500.0) < 1.0

    def test_overlapping_neff_less_than_n(self):
        """For overlapping H>1 labels, N_eff should be < N."""
        # Create autocorrelated binary labels (simulating overlapping windows)
        rng = np.random.default_rng(42)
        base = rng.random(500)
        # Create autocorrelation by smoothing
        smoothed = np.convolve(base, np.ones(10) / 10, mode="same")
        y = (smoothed > 0.5).astype(float)
        n_eff = effective_sample_size(y, H=10)
        assert n_eff < len(y), f"Expected N_eff < N, got {n_eff} >= {len(y)}"


# ============================================================
# Test: Wilson Reliability CI (Phase 3C)
# ============================================================

class TestReliabilityCI:
    """Verify Wilson confidence intervals in reliability diagram."""

    def _make_synthetic_df(self, n_days=500, seed=42):
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0003, 0.012, n_days)
        prices = 100.0 * np.exp(np.cumsum(returns))
        dates = pd.bdate_range("2020-01-01", periods=n_days)
        return pd.DataFrame({"price": prices}, index=dates)

    def _make_config(self):
        cfg = PipelineConfig()
        cfg.data.source = "synthetic"
        cfg.model.mc_base_paths = 5000
        cfg.model.mc_boost_paths = 5000
        cfg.model.horizons = [5, 10, 20]
        cfg.model.seed = 42
        cfg.output.charts = False
        return cfg

    def test_ci_columns_present(self):
        """compute_reliability should include ci_low and ci_high columns."""
        df = self._make_synthetic_df(500, seed=42)
        cfg = self._make_config()
        results = run_walkforward(df, cfg)
        reliability = compute_reliability(results, cfg.model.horizons)
        if len(reliability) > 0:
            assert "ci_low" in reliability.columns
            assert "ci_high" in reliability.columns

    def test_ci_bounds_valid(self):
        """CI bounds should be in [0, 1] and ci_low <= ci_high."""
        df = self._make_synthetic_df(500, seed=42)
        cfg = self._make_config()
        results = run_walkforward(df, cfg)
        reliability = compute_reliability(results, cfg.model.horizons)
        if len(reliability) > 0:
            assert (reliability["ci_low"] >= 0).all()
            assert (reliability["ci_high"] <= 1).all()
            assert (reliability["ci_low"] <= reliability["ci_high"]).all()


# ============================================================
# Test: CRPS from Quantiles (Phase 3D)
# ============================================================

class TestCRPS:
    """Verify CRPS computation from quantile summaries."""

    def test_crps_narrow_beats_wide(self):
        """Narrow distribution centered on observation should have lower CRPS."""
        q_levels = QUANTILE_LEVELS

        # Narrow distribution centered at 0
        narrow_q = np.quantile(np.random.default_rng(42).normal(0, 0.01, 10000), q_levels)
        # Wide distribution centered at 0
        wide_q = np.quantile(np.random.default_rng(42).normal(0, 0.10, 10000), q_levels)

        realized = np.array([0.0])
        crps_narrow = crps_from_quantiles(narrow_q.reshape(1, -1), q_levels, realized)
        crps_wide = crps_from_quantiles(wide_q.reshape(1, -1), q_levels, realized)

        assert crps_narrow < crps_wide, \
            f"Narrow CRPS ({crps_narrow:.6f}) should beat wide ({crps_wide:.6f})"


# ============================================================
# Test: Quantile Storage (Phase 3D)
# ============================================================

class TestQuantileStorage:
    """Verify MC quantile return from compute_move_probability."""

    def test_returns_quantiles_when_requested(self):
        """compute_move_probability with return_quantiles=True should return 3 values."""
        terminal = simulate_gbm_terminal(100.0, 0.01, 10, 10000, 0.0, 42)
        result = compute_move_probability(terminal, 100.0, 0.05, return_quantiles=True)
        assert len(result) == 3
        p_raw, se, quantiles = result
        assert len(quantiles) == len(QUANTILE_LEVELS)
        # Quantiles should be sorted
        assert all(quantiles[i] <= quantiles[i + 1] for i in range(len(quantiles) - 1))

    def test_no_quantiles_by_default(self):
        """Default behavior should return 2 values (p_raw, se)."""
        terminal = simulate_gbm_terminal(100.0, 0.01, 10, 10000, 0.0, 42)
        result = compute_move_probability(terminal, 100.0, 0.05)
        assert len(result) == 2


# ============================================================
# Test: Model Selection Framework (Phase 4)
# ============================================================

class TestModelSelection:
    """Verify expanding-window CV and model comparison."""

    def test_calibration_aic_bic(self):
        """More complex model should have higher BIC penalty."""
        from em_sde.model_selection import calibration_aic, calibration_bic

        rng = np.random.default_rng(42)
        y = (rng.random(500) < 0.05).astype(float)
        p = np.full(500, 0.05)  # constant prediction

        aic_2 = calibration_aic(p, y, n_params=2)
        aic_6 = calibration_aic(p, y, n_params=6)
        bic_2 = calibration_bic(p, y, n_params=2)
        bic_6 = calibration_bic(p, y, n_params=6)

        # More params = higher AIC/BIC (worse) for same fit
        assert aic_6 > aic_2, f"6-param AIC ({aic_6}) should > 2-param ({aic_2})"
        assert bic_6 > bic_2, f"6-param BIC ({bic_6}) should > 2-param ({bic_2})"

    def test_compare_models_output_shape(self):
        """compare_models should produce valid output from CV results."""
        from em_sde.model_selection import compare_models

        cv_data = pd.DataFrame([
            {"config_name": "A", "fold": 0, "horizon": 5, "bss_cal": 0.01, "auc_cal": 0.55,
             "brier_cal": 0.04, "logloss_cal": 0.18, "separation_cal": 0.01, "brier_raw": 0.04,
             "bss_raw": 0.0, "auc_raw": 0.50, "event_rate": 0.05, "n": 100},
            {"config_name": "B", "fold": 0, "horizon": 5, "bss_cal": 0.05, "auc_cal": 0.65,
             "brier_cal": 0.035, "logloss_cal": 0.16, "separation_cal": 0.02, "brier_raw": 0.04,
             "bss_raw": 0.0, "auc_raw": 0.50, "event_rate": 0.05, "n": 100},
        ])
        summary = compare_models(cv_data)
        assert len(summary) == 2
        assert "rank" in summary.columns
        # B should rank higher (better BSS)
        rank_values = np.asarray(summary.loc[summary["config_name"] == "B", "rank"], dtype=int)
        assert rank_values.size == 1
        assert int(rank_values[0]) == 1


# ============================================================
# Test: Config Defaults for New Features
# ============================================================

class TestNewConfigDefaults:
    """Verify backward compatibility of new config fields."""

    def test_threshold_mode_default(self):
        cfg = ModelConfig()
        assert cfg.threshold_mode == "vol_scaled"
        assert cfg.fixed_threshold_pct == 0.05
        assert cfg.store_quantiles is False

    def test_multi_feature_default(self):
        cfg = CalibrationConfig()
        assert cfg.multi_feature is False
        assert cfg.multi_feature_lr == 0.01
        assert cfg.multi_feature_l2 == 1e-4
        assert cfg.multi_feature_min_updates == 100


# ============================================================
# Test: Data Quality Pipeline
# ============================================================

class TestDataQuality:
    """Verify data quality checks in data_layer."""

    def test_outlier_detection_finds_extremes(self):
        """IQR outlier detection should flag extreme returns."""
        from em_sde.data_layer import detect_outliers
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 1000))
        # Inject clear outliers
        returns.iloc[100] = 0.50   # +50% return
        returns.iloc[200] = -0.40  # -40% return
        result = detect_outliers(returns, iqr_multiplier=5.0)
        assert result["n_outliers"] >= 2, \
            f"Should detect injected outliers, got {result['n_outliers']}"

    def test_outlier_detection_clean_data(self):
        """Clean normal data should have few or no outliers."""
        from em_sde.data_layer import detect_outliers
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 1000))
        result = detect_outliers(returns, iqr_multiplier=5.0)
        # With 5x IQR, very few outliers expected in normal data
        assert result["n_outliers"] < 5

    def test_stale_price_detection(self):
        """Stale price detection should find consecutive unchanged prices."""
        from em_sde.data_layer import detect_stale_prices
        prices = pd.Series(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 101.0, 102.0],
            index=pd.bdate_range("2020-01-01", periods=8),
        )
        result = detect_stale_prices(prices, max_consecutive=5)
        assert result["n_stale_periods"] >= 1

    def test_stale_price_no_false_positive(self):
        """Varying prices should not trigger stale detection."""
        from em_sde.data_layer import detect_stale_prices
        rng = np.random.default_rng(42)
        prices = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))),
            index=pd.bdate_range("2020-01-01", periods=200),
        )
        result = detect_stale_prices(prices, max_consecutive=5)
        assert result["n_stale_periods"] == 0

    def test_data_gap_detection(self):
        """Should detect gaps exceeding max_gap_days business days."""
        from em_sde.data_layer import detect_data_gaps
        # Create dates with a 2-week gap
        dates1 = pd.bdate_range("2020-01-01", periods=50)
        dates2 = pd.bdate_range("2020-04-01", periods=50)  # big gap
        dates = dates1.append(dates2)
        df = pd.DataFrame({"price": range(len(dates))}, index=dates)
        result = detect_data_gaps(df, max_gap_days=5)
        assert result["n_gaps"] >= 1

    def test_full_quality_report_structure(self):
        """run_data_quality_checks should return all expected keys."""
        from em_sde.data_layer import run_data_quality_checks
        rng = np.random.default_rng(42)
        n = 500
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        dates = pd.bdate_range("2020-01-01", periods=n)
        df = pd.DataFrame({"price": prices}, index=dates)
        returns = pd.Series(df["price"].to_numpy(dtype=float), index=df.index).pct_change().dropna()
        report = run_data_quality_checks(df, returns)
        assert "outliers" in report
        assert "stale_prices" in report
        assert "data_gaps" in report
        assert "return_statistics" in report
        stats = report["return_statistics"]
        assert "skewness" in stats
        assert "kurtosis" in stats
        assert "n_trading_days" in stats


# ============================================================
# Test: Risk Analytics
# ============================================================

class TestRiskAnalytics:
    """Verify VaR, CVaR, skewness, kurtosis, max drawdown."""

    def test_var_95_positive(self):
        """VaR at 95% should be positive for normal returns."""
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.001, 0.02, 1000)
        var = value_at_risk(returns, 0.95)
        assert var > 0, f"VaR should be positive, got {var}"

    def test_cvar_exceeds_var(self):
        """CVaR (expected shortfall) should be >= VaR."""
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.001, 0.02, 1000)
        var = value_at_risk(returns, 0.95)
        cvar = conditional_var(returns, 0.95)
        assert cvar >= var - 1e-10, \
            f"CVaR ({cvar}) should be >= VaR ({var})"

    def test_var_99_exceeds_var_95(self):
        """99% VaR should be >= 95% VaR."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 5000)
        var95 = value_at_risk(returns, 0.95)
        var99 = value_at_risk(returns, 0.99)
        assert var99 >= var95 - 1e-10, \
            f"VaR99 ({var99}) should be >= VaR95 ({var95})"

    def test_skewness_negative_for_left_skew(self):
        """Left-skewed distribution should have negative skewness."""
        rng = np.random.default_rng(42)
        # Create left-skewed data (log-normal flipped)
        x = -np.abs(rng.lognormal(0, 1, 5000))
        skew = return_skewness(x)
        assert skew < 0, f"Expected negative skewness for left-skewed data, got {skew}"

    def test_kurtosis_positive_for_heavy_tails(self):
        """Student-t(3) should have positive excess kurtosis."""
        rng = np.random.default_rng(42)
        x = rng.standard_t(df=3, size=10000)
        kurt = return_kurtosis(x)
        assert kurt > 0, f"Expected positive excess kurtosis for t(3), got {kurt}"

    def test_kurtosis_near_zero_for_normal(self):
        """Normal distribution should have near-zero excess kurtosis."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50000)
        kurt = return_kurtosis(x)
        assert abs(kurt) < 0.3, f"Expected near-zero kurtosis for normal, got {kurt}"

    def test_max_drawdown_known_case(self):
        """Max drawdown should match a known scenario."""
        prices = np.array([100.0, 110.0, 90.0, 95.0, 80.0, 105.0])
        dd = max_drawdown(prices)
        # Peak at 110, trough at 80 -> drawdown = (110-80)/110 = 0.2727
        expected = (110.0 - 80.0) / 110.0
        assert abs(dd["max_drawdown"] - expected) < 0.01, \
            f"Expected drawdown ~{expected:.4f}, got {dd['max_drawdown']:.4f}"

    def test_compute_risk_report_structure(self):
        """compute_risk_report should return expected keys per horizon."""
        rng = np.random.default_rng(42)
        n = 300
        results = pd.DataFrame({
            "realized_return_5": rng.normal(0, 0.03, n),
            "realized_return_10": rng.normal(0, 0.05, n),
        })
        report = compute_risk_report(results, [5, 10])
        assert 5 in report
        assert 10 in report
        assert "var_95" in report[5]
        assert "cvar_95" in report[5]
        assert "skewness" in report[5]
        assert "kurtosis" in report[5]


# ============================================================
# Test: Backtest Analytics
# ============================================================

class TestBacktestAnalytics:
    """Verify backtest analytics (hit rate, turnover, precision/recall)."""

    def test_hit_rate_perfect_signal(self):
        """Perfect signal should have 100% hit rate."""
        from em_sde.backtest import compute_backtest_analytics
        results = pd.DataFrame({
            "p_cal_5": [0.9, 0.9, 0.1, 0.1],
            "y_5": [1.0, 1.0, 0.0, 0.0],
        })
        analytics = compute_backtest_analytics(results, [5], signal_threshold=0.5)
        assert analytics[5]["hit_rate"] == 1.0
        assert analytics[5]["precision"] == 1.0
        assert analytics[5]["recall"] == 1.0

    def test_hit_rate_no_signal(self):
        """No predictions above threshold should give NaN hit rate."""
        from em_sde.backtest import compute_backtest_analytics
        results = pd.DataFrame({
            "p_cal_5": [0.1, 0.1, 0.1, 0.1],
            "y_5": [1.0, 0.0, 0.0, 0.0],
        })
        analytics = compute_backtest_analytics(results, [5], signal_threshold=0.5)
        assert np.isnan(analytics[5]["hit_rate"])

    def test_signal_turnover_constant_signal(self):
        """Constant signal should have zero turnover."""
        from em_sde.backtest import compute_backtest_analytics
        results = pd.DataFrame({
            "p_cal_5": [0.05, 0.05, 0.05, 0.05],
            "y_5": [0.0, 0.0, 0.0, 0.0],
        })
        analytics = compute_backtest_analytics(results, [5])
        assert analytics[5]["signal_turnover"] == 0.0

    def test_signal_turnover_varying_signal(self):
        """Varying signal should have positive turnover."""
        from em_sde.backtest import compute_backtest_analytics
        results = pd.DataFrame({
            "p_cal_5": [0.05, 0.15, 0.05, 0.15],
            "y_5": [0.0, 0.0, 1.0, 0.0],
        })
        analytics = compute_backtest_analytics(results, [5])
        assert analytics[5]["signal_turnover"] > 0


# ============================================================
# Test: GARCH Diagnostics
# ============================================================

class TestGarchDiagnostics:
    """Verify GARCH stationarity and diagnostics."""

    def test_stationary_garch(self):
        """GARCH with alpha+beta < 1 should be stationary."""
        from em_sde.garch import garch_diagnostics
        diag = garch_diagnostics(omega=1e-6, alpha=0.10, beta=0.85)
        assert diag["is_stationary"] is True
        assert diag["persistence"] == 0.95
        assert diag["half_life_days"] is not None
        assert diag["half_life_days"] > 0

    def test_nonstationary_garch(self):
        """GARCH with alpha+beta >= 1 should be non-stationary."""
        from em_sde.garch import garch_diagnostics
        diag = garch_diagnostics(omega=1e-6, alpha=0.15, beta=0.90)
        assert diag["is_stationary"] is False
        assert diag["persistence"] >= 1.0

    def test_gjr_persistence_includes_gamma(self):
        """GJR-GARCH persistence should include gamma/2."""
        from em_sde.garch import garch_diagnostics
        diag = garch_diagnostics(
            omega=1e-6, alpha=0.05, beta=0.85, gamma=0.10, model_type="gjr",
        )
        # persistence = 0.05 + 0.85 + 0.10/2 = 0.95
        assert abs(diag["persistence"] - 0.95) < 1e-10

    def test_unconditional_vol_positive(self):
        """Stationary GARCH should have positive unconditional vol."""
        from em_sde.garch import garch_diagnostics
        diag = garch_diagnostics(omega=1e-6, alpha=0.10, beta=0.85)
        assert diag["unconditional_vol"] is not None
        assert diag["unconditional_vol"] > 0

    def test_half_life_ordering(self):
        """Higher persistence should have longer half-life."""
        from em_sde.garch import garch_diagnostics
        diag_low = garch_diagnostics(omega=1e-6, alpha=0.10, beta=0.80)
        diag_high = garch_diagnostics(omega=1e-6, alpha=0.10, beta=0.88)
        assert diag_high["half_life_days"] > diag_low["half_life_days"]

    def test_fit_garch_includes_diagnostics(self):
        """fit_garch result should include diagnostics when GARCH fits."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, size=500)
        result = fit_garch(returns, window=500, min_window=252)
        if result.source != "ewma_fallback":
            assert result.diagnostics is not None
            assert "persistence" in result.diagnostics
            assert "is_stationary" in result.diagnostics


# ============================================================
# Test: Calibration Convergence Diagnostics
# ============================================================

class TestCalibrationConvergence:
    """Verify calibration convergence diagnostics."""

    def test_online_convergence_diagnostics(self):
        """OnlineCalibrator should provide convergence diagnostics."""
        cal = OnlineCalibrator(lr=0.05, adaptive_lr=True, min_updates=0)
        rng = np.random.default_rng(42)
        for _ in range(200):
            cal.update(0.05, 1.0 if rng.random() < 0.05 else 0.0)
        diag = cal.convergence_diagnostics()
        assert "velocity_a" in diag
        assert "velocity_b" in diag
        assert "is_converged" in diag
        assert diag["n_updates"] == 200

    def test_convergence_velocity_decreases(self):
        """Calibrator velocity should decrease as it converges."""
        cal = OnlineCalibrator(lr=0.05, adaptive_lr=True, min_updates=0)
        rng = np.random.default_rng(42)
        # First batch
        for _ in range(100):
            cal.update(0.05, 1.0 if rng.random() < 0.05 else 0.0)
        diag_early = cal.convergence_diagnostics()
        # Second batch (more updates, should be more stable)
        for _ in range(400):
            cal.update(0.05, 1.0 if rng.random() < 0.05 else 0.0)
        diag_late = cal.convergence_diagnostics()
        # Velocity should decrease with adaptive lr
        assert diag_late["velocity_a"] < diag_early["velocity_a"], \
            "Velocity should decrease over time with adaptive lr"

    def test_mf_convergence_diagnostics(self):
        """MultiFeatureCalibrator should provide convergence diagnostics."""
        cal = MultiFeatureCalibrator(lr=0.01, min_updates=0)
        rng = np.random.default_rng(42)
        for _ in range(150):
            cal.update(0.05, 1.0 if rng.random() < 0.05 else 0.0,
                       0.01, 0.0, 1.0, 0.0)
        diag = cal.convergence_diagnostics()
        assert "weight_velocity" in diag
        assert "max_weight" in diag
        assert "is_converged" in diag
        assert diag["n_updates"] == 150


# ============================================================
# Test: Public API (__init__.py)
# ============================================================

class TestPublicAPI:
    """Verify __init__.py exports."""

    def test_version(self):
        import em_sde
        assert hasattr(em_sde, "__version__")
        assert em_sde.__version__ == "2.0.0"

    def test_all_exports_exist(self):
        import em_sde
        for name in em_sde.__all__:
            assert hasattr(em_sde, name), f"Missing export: {name}"

    def test_key_classes_importable(self):
        from em_sde import (
            PipelineConfig, OnlineCalibrator, MultiFeatureCalibrator,
            GarchResult, run_walkforward, brier_score, value_at_risk,
            ewma_volatility,
        )
        # Verify they're the real classes/functions, not None
        assert PipelineConfig is not None
        assert OnlineCalibrator is not None
        assert callable(ewma_volatility)


# ============================================================
# Test: U4 — Stationarity-Constrained GARCH Projection
# ============================================================

class TestStationarityProjection:
    """Tests for project_to_stationary()."""

    def test_projection_reduces_persistence(self):
        omega, alpha, beta = 1e-6, 0.15, 0.90
        o, a, b, g = project_to_stationary(omega, alpha, beta, target_persistence=0.98)
        new_p = a + b
        assert new_p < 1.0
        assert abs(new_p - 0.98) < 0.001

    def test_projection_preserves_alpha_beta_ratio(self):
        omega, alpha, beta = 1e-6, 0.15, 0.90
        ratio_before = alpha / beta
        o, a, b, g = project_to_stationary(omega, alpha, beta, target_persistence=0.98)
        ratio_after = a / b
        assert abs(ratio_before - ratio_after) < 1e-10

    def test_gjr_projection_preserves_gamma_alpha_ratio(self):
        omega, alpha, beta, gamma = 1e-6, 0.10, 0.85, 0.12
        ratio_before = gamma / alpha
        o, a, b, gm = project_to_stationary(
            omega, alpha, beta, gamma, model_type="gjr", target_persistence=0.98,
        )
        ratio_after = gm / a
        assert abs(ratio_before - ratio_after) < 1e-10
        # Verify GJR persistence
        new_p = a + b + gm / 2.0
        assert abs(new_p - 0.98) < 0.001

    def test_no_projection_when_stationary(self):
        omega, alpha, beta = 1e-6, 0.10, 0.85
        o, a, b, g = project_to_stationary(omega, alpha, beta, target_persistence=0.98)
        assert a == alpha
        assert b == beta

    def test_projection_preserves_omega_when_no_anchor(self):
        """Backward compat: omega unchanged when variance_anchor is not provided."""
        omega = 1e-6
        o, a, b, g = project_to_stationary(omega, 0.15, 0.90, target_persistence=0.98)
        assert o == omega

    def test_projection_recomputes_omega_with_variance_anchor(self):
        """When variance_anchor is provided, omega is recomputed for variance targeting."""
        omega = 1e-6
        alpha, beta = 0.15, 0.90
        sigma_1d = 0.02  # 2% daily vol
        variance_anchor = sigma_1d ** 2
        target_p = 0.98

        o, a, b, g = project_to_stationary(
            omega, alpha, beta,
            target_persistence=target_p,
            variance_anchor=variance_anchor,
        )

        # omega_new = variance_anchor * (1 - target_persistence)
        expected_omega = variance_anchor * (1.0 - target_p)
        assert abs(o - expected_omega) < 1e-12, f"omega {o} != expected {expected_omega}"

        # V_inf = omega_new / (1 - new_persistence) should equal variance_anchor
        new_persistence = a + b
        v_inf = o / (1.0 - new_persistence)
        assert abs(v_inf - variance_anchor) < 1e-10, \
            f"V_inf {v_inf} != variance_anchor {variance_anchor}"

    def test_gjr_projection_recomputes_omega_with_variance_anchor(self):
        """GJR projection with variance_anchor also recomputes omega correctly."""
        omega, alpha, beta, gamma = 1e-6, 0.10, 0.85, 0.12
        sigma_1d = 0.025
        variance_anchor = sigma_1d ** 2
        target_p = 0.98

        o, a, b, gm = project_to_stationary(
            omega, alpha, beta, gamma,
            model_type="gjr",
            target_persistence=target_p,
            variance_anchor=variance_anchor,
        )

        expected_omega = variance_anchor * (1.0 - target_p)
        assert abs(o - expected_omega) < 1e-12

        # GJR persistence: a + b + gm/2
        new_persistence = a + b + gm / 2.0
        assert abs(new_persistence - target_p) < 0.001
        v_inf = o / (1.0 - new_persistence)
        assert abs(v_inf - variance_anchor) < 1e-10

    def test_ewma_fallback_computes_actual_ewma(self):
        """ewma_volatility is public and returns a reasonable value."""
        from em_sde.garch import ewma_volatility

        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, size=500)

        sigma_ewma = ewma_volatility(returns, span=252)

        assert sigma_ewma > 0
        assert sigma_ewma < 1.0
        # Should be near true vol of 0.01
        assert abs(sigma_ewma - 0.01) < 0.003

    def test_projected_garch_simulation_mean_reverts_to_anchor(self):
        """After projection with variance_anchor, simulation mean-reverts to anchor."""
        from em_sde.monte_carlo import simulate_garch_terminal

        sigma_1d = 0.02
        variance_anchor = sigma_1d ** 2
        # Non-stationary params
        omega_orig = 1e-6
        alpha_orig = 0.15
        beta_orig = 0.90

        # Project with variance anchor
        omega_p, alpha_p, beta_p, _ = project_to_stationary(
            omega_orig, alpha_orig, beta_orig,
            target_persistence=0.98,
            variance_anchor=variance_anchor,
        )

        # Simulate long GARCH paths
        terminal = simulate_garch_terminal(
            S0=100.0,
            sigma_1d=sigma_1d,
            H=100,
            n_paths=50_000,
            omega=omega_p,
            alpha=alpha_p,
            beta=beta_p,
            gamma=0.0,
            mu_year=0.0,
            seed=42,
        )

        # Realized vol over many paths should be in neighborhood of sigma_1d
        sim_returns = terminal / 100.0 - 1.0
        realized_vol_daily = np.std(sim_returns) / np.sqrt(100)
        ratio = realized_vol_daily / sigma_1d
        assert 0.5 < ratio < 2.0, \
            f"Realized daily vol {realized_vol_daily:.6f} too far from anchor {sigma_1d:.6f} (ratio={ratio:.2f})"

    def test_default_config_enables_constraint(self):
        cfg = ModelConfig()
        assert cfg.garch_stationarity_constraint is True


# ============================================================
# Test: U2 — State-Dependent Jump Model
# ============================================================

class TestStateDependentJumps:
    """Tests for compute_state_dependent_jumps()."""

    def test_interpolation_at_low_vol(self):
        vol_hist = np.linspace(0.005, 0.030, 300)
        sigma_low = float(np.percentile(vol_hist, 25))
        result = compute_state_dependent_jumps(
            sigma_low, vol_hist, (1.0, -0.01, 0.03), (4.0, -0.04, 0.06),
        )
        assert abs(result[0] - 1.0) < 0.05

    def test_interpolation_at_high_vol(self):
        vol_hist = np.linspace(0.005, 0.030, 300)
        sigma_high = float(np.percentile(vol_hist, 75))
        result = compute_state_dependent_jumps(
            sigma_high, vol_hist, (1.0, -0.01, 0.03), (4.0, -0.04, 0.06),
        )
        assert abs(result[0] - 4.0) < 0.05

    def test_interpolation_midpoint(self):
        vol_hist = np.linspace(0.005, 0.030, 300)
        sigma_mid = float(np.percentile(vol_hist, 50))
        result = compute_state_dependent_jumps(
            sigma_mid, vol_hist, (1.0, -0.01, 0.03), (4.0, -0.04, 0.06),
        )
        assert abs(result[0] - 2.5) < 0.3

    def test_warmup_returns_midpoint(self):
        vol_hist = np.array([0.01, 0.02])
        result = compute_state_dependent_jumps(
            0.015, vol_hist, (1.0, -0.01, 0.03), (4.0, -0.04, 0.06),
        )
        assert abs(result[0] - 2.5) < 0.01
        assert abs(result[1] - (-0.025)) < 0.01

    def test_disabled_by_default(self):
        cfg = ModelConfig()
        assert cfg.jump_state_dependent is False

    def test_validation_requires_jump_enabled(self):
        from em_sde.config import _validate
        cfg = PipelineConfig()
        cfg.model.jump_state_dependent = True
        cfg.model.jump_enabled = False
        with assert_raises(AssertionError):
            _validate(cfg)


# ============================================================
# Test: U1 — Regime-Gated Threshold Routing
# ============================================================

class TestRegimeGatedThreshold:
    """Tests for RegimeRouter."""

    def test_router_warmup_uses_default(self):
        from em_sde.backtest import RegimeRouter
        router = RegimeRouter(
            warmup=100, low_mode="fixed_pct",
            mid_mode="fixed_pct", high_mode="anchored_vol",
        )
        for v in np.linspace(0.005, 0.03, 50):
            router.observe(v)
        assert router.get_threshold_mode(0.03) == "fixed_pct"  # still in warmup

    def test_router_routes_by_vol_regime(self):
        from em_sde.backtest import RegimeRouter
        router = RegimeRouter(
            warmup=100, low_mode="fixed_pct",
            mid_mode="vol_scaled", high_mode="anchored_vol",
        )
        for v in np.linspace(0.005, 0.030, 300):
            router.observe(v)
        assert router.get_threshold_mode(0.006) == "fixed_pct"
        assert router.get_threshold_mode(0.015) == "vol_scaled"
        assert router.get_threshold_mode(0.029) == "anchored_vol"

    def test_regime_gated_config_validation(self):
        from em_sde.config import _validate
        cfg = PipelineConfig()
        cfg.model.threshold_mode = "regime_gated"
        cfg.model.regime_gated_low_mode = "invalid"
        with assert_raises(AssertionError):
            _validate(cfg)

    def test_regime_gated_backward_compatible(self):
        cfg = ModelConfig()
        assert cfg.threshold_mode == "vol_scaled"

    def test_regime_gated_is_warmed_up_property(self):
        from em_sde.backtest import RegimeRouter
        router = RegimeRouter(warmup=50)
        assert not router.is_warmed_up
        for v in np.linspace(0.01, 0.02, 50):
            router.observe(v)
        assert router.is_warmed_up


# ============================================================
# Test: U3 — AUC/Separation Calibration Guardrail
# ============================================================

class TestDiscriminationGuardrail:
    """Tests for discrimination gate on calibrators."""

    def test_discrimination_gate_default_config(self):
        cfg = CalibrationConfig()
        assert cfg.gate_on_discrimination is True

    def test_gate_triggers_on_inverted_auc(self):
        # lr=0 so calibrator never adjusts — p_cal stays as inverted p_raw
        cal = OnlineCalibrator(
            lr=0.0, adaptive_lr=False, min_updates=0,
            safety_gate=False,
            gate_on_discrimination=True,
            gate_auc_threshold=0.50,
            gate_discrimination_window=50,
        )
        rng = np.random.default_rng(42)
        for _ in range(60):
            y = 1.0 if rng.random() < 0.3 else 0.0
            p_raw = 0.9 if y == 0.0 else 0.1  # inverted signal
            cal.update(p_raw, y)
        assert cal._discrimination_gate_active, "Gate should trigger on inverted AUC"

    def test_gate_stays_off_on_good_discrimination(self):
        cal = OnlineCalibrator(
            lr=0.01, adaptive_lr=True, min_updates=0,
            gate_on_discrimination=True,
            gate_auc_threshold=0.50,
            gate_discrimination_window=50,
        )
        rng = np.random.default_rng(42)
        for _ in range(100):
            y = 1.0 if rng.random() < 0.1 else 0.0
            p_raw = 0.15 if y == 1.0 else 0.05  # correct direction
            cal.update(p_raw, y)
        assert not cal._discrimination_gate_active

    def test_gate_returns_raw_when_active(self):
        cal = OnlineCalibrator(
            lr=0.1, min_updates=0,
            gate_on_discrimination=True,
            gate_discrimination_window=20,
        )
        cal._discrimination_gate_active = True
        result = cal.calibrate(0.07)
        assert result == 0.07

    def test_multifeature_has_discrimination_gate(self):
        cal = MultiFeatureCalibrator(
            min_updates=0,
            gate_on_discrimination=True,
            gate_discrimination_window=50,
        )
        assert hasattr(cal, '_discrimination_gate_active')
        assert not cal._discrimination_gate_active


# ============================================================
# Test: U5 — Promotion Gates and ECE
# ============================================================

class TestExpectedCalibrationError:
    """Tests for expected_calibration_error()."""

    def test_perfect_calibration_low_ece(self):
        rng = np.random.default_rng(42)
        p = rng.uniform(0.0, 1.0, 10000)
        y = (rng.random(10000) < p).astype(float)
        ece = expected_calibration_error(p, y)
        assert ece < 0.02, f"ECE should be near zero for perfect cal, got {ece}"

    def test_constant_bias_positive_ece(self):
        y = np.zeros(1000)
        y[:50] = 1.0  # 5% event rate
        p = np.full(1000, 0.20)  # predict 20% but rate is 5%
        ece = expected_calibration_error(p, y)
        assert ece > 0.10, f"Biased predictions should have high ECE, got {ece}"

    def test_adaptive_is_default(self):
        """Adaptive binning is the default (adaptive=True)."""
        rng = np.random.default_rng(99)
        p = rng.uniform(0.0, 1.0, 500)
        y = (rng.random(500) < p).astype(float)
        ece_default = expected_calibration_error(p, y)
        ece_adaptive = expected_calibration_error(p, y, adaptive=True)
        assert ece_default == ece_adaptive

    def test_equal_width_backward_compat(self):
        """adaptive=False reproduces original equal-width behavior."""
        rng = np.random.default_rng(42)
        p = rng.uniform(0.0, 1.0, 10000)
        y = (rng.random(10000) < p).astype(float)
        ece_ew = expected_calibration_error(p, y, adaptive=False)
        assert ece_ew < 0.02, f"Equal-width ECE should be low for perfect cal, got {ece_ew}"

    def test_adaptive_detects_bias_in_skewed_predictions(self):
        """Adaptive ECE correctly detects calibration errors in skewed distributions.

        When predictions cluster in a narrow range, equal-width ECE may mask
        genuine calibration errors by putting everything in one bin.  Adaptive
        ECE spreads weight across bins and can expose finer-grained bias.
        """
        rng = np.random.default_rng(77)
        n = 5000
        # All predictions near 0.15, but true event rate is 0.03
        p = np.clip(rng.normal(0.15, 0.02, n), 0.01, 0.30)
        y = (rng.random(n) < 0.03).astype(float)

        ece_adaptive = expected_calibration_error(p, y, adaptive=True)
        # Should detect the ~12pp bias
        assert ece_adaptive > 0.05, (
            f"Adaptive ECE should detect systematic bias, got {ece_adaptive:.4f}"
        )

    def test_adaptive_identical_predictions(self):
        """Adaptive ECE handles all-identical predictions gracefully."""
        p = np.full(100, 0.10)
        y = np.zeros(100)
        y[:10] = 1.0  # exactly 10% event rate
        ece = expected_calibration_error(p, y, adaptive=True)
        # |0.10 - 0.10| = 0.0
        assert ece < 0.01, f"Identical predictions matching event rate: ECE={ece}"


class TestPromotionGates:
    """Tests for apply_promotion_gates()."""

    def test_passing_model(self):
        cv_data = pd.DataFrame([
            {"config_name": "A", "fold": 0, "horizon": 5,
             "bss_cal": 0.05, "auc_cal": 0.60, "ece_cal": 0.01, "sigma_mean": 0.012},
            {"config_name": "A", "fold": 1, "horizon": 5,
             "bss_cal": 0.03, "auc_cal": 0.58, "ece_cal": 0.015, "sigma_mean": 0.018},
        ])
        report = apply_promotion_gates(cv_data)
        assert report["all_gates_passed"].all()

    def test_failing_model_negative_bss(self):
        cv_data = pd.DataFrame([
            {"config_name": "B", "fold": 0, "horizon": 5,
             "bss_cal": -0.02, "auc_cal": 0.60, "ece_cal": 0.01, "sigma_mean": 0.012},
            {"config_name": "B", "fold": 1, "horizon": 5,
             "bss_cal": -0.03, "auc_cal": 0.58, "ece_cal": 0.015, "sigma_mean": 0.025},
        ])
        report = apply_promotion_gates(cv_data)
        assert not report["all_gates_passed"].all()

    def test_custom_gates(self):
        cv_data = pd.DataFrame([
            {"config_name": "C", "fold": 0, "horizon": 5,
             "bss_cal": 0.10, "auc_cal": 0.70, "ece_cal": 0.005, "sigma_mean": 0.015},
        ])
        report = apply_promotion_gates(
            cv_data, gates={"bss_cal": 0.05, "auc_cal": 0.65, "ece_cal": 0.01},
        )
        assert report["all_gates_passed"].all()

    def test_report_shows_margin(self):
        cv_data = pd.DataFrame([
            {"config_name": "D", "fold": 0, "horizon": 5,
             "bss_cal": 0.03, "auc_cal": 0.52, "ece_cal": 0.01, "sigma_mean": 0.015},
        ])
        report = apply_promotion_gates(cv_data)
        assert "margin" in report.columns


class TestHistogramCalibrator:
    """Tests for HistogramCalibrator (bin-level bias correction)."""

    def test_no_correction_before_min_samples(self):
        hc = HistogramCalibrator(n_bins=20, min_samples_per_bin=30)
        # Feed 10 samples (below threshold)
        for _ in range(10):
            hc.update(0.10, 0.0)
        # Should return input unchanged
        assert hc.calibrate(0.10) == 0.10

    def test_correction_reduces_bias(self):
        hc = HistogramCalibrator(n_bins=20, min_samples_per_bin=10)
        # Systematically predict 0.20 but true rate is 0.05
        for _ in range(50):
            hc.update(0.20, 0.0)
        for _ in range(3):
            hc.update(0.20, 1.0)  # ~5.7% event rate
        corrected = hc.calibrate(0.20)
        # Correction should pull prediction down toward observed rate
        assert corrected < 0.20, f"Expected correction below 0.20, got {corrected}"
        assert corrected > 0.0, f"Corrected probability should be positive, got {corrected}"

    def test_well_calibrated_no_correction(self):
        hc = HistogramCalibrator(n_bins=20, min_samples_per_bin=30)
        rng = np.random.default_rng(42)
        # Feed well-calibrated data: predict 0.50, event rate 50%
        for _ in range(100):
            y = float(rng.random() < 0.50)
            hc.update(0.50, y)
        corrected = hc.calibrate(0.50)
        # Should be close to 0.50 (minimal correction)
        assert abs(corrected - 0.50) < 0.05, (
            f"Well-calibrated input should have minimal correction, got {corrected}"
        )

    def test_output_clipped_to_01(self):
        hc = HistogramCalibrator(n_bins=20, min_samples_per_bin=5)
        # Extreme case: predict 0.02 but all events
        for _ in range(10):
            hc.update(0.02, 1.0)
        corrected = hc.calibrate(0.02)
        assert 0.0 <= corrected <= 1.0, f"Output must be in [0, 1], got {corrected}"

    def test_online_calibrator_with_histogram_post_cal(self):
        """OnlineCalibrator with histogram_post_cal=True produces valid output."""
        from em_sde.calibration import OnlineCalibrator
        cal = OnlineCalibrator(lr=0.05, min_updates=5, histogram_post_cal=True)
        rng = np.random.default_rng(42)
        for _ in range(100):
            p_raw = float(rng.uniform(0.01, 0.50))
            y = float(rng.random() < 0.10)
            p_cal = cal.calibrate(p_raw)
            assert 0.0 <= p_cal <= 1.0, f"p_cal out of range: {p_cal}"
            cal.update(p_raw, y)

    def test_histogram_calibrator_import(self):
        """HistogramCalibrator is importable from public API."""
        from em_sde import HistogramCalibrator as HC
        hc = HC()
        assert hc.n_bins == 20


if __name__ == "__main__":
    raise SystemExit(
        subprocess.call([sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"])
    )
