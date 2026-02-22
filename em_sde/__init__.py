"""
em_sde: Calibrated Large-Move Probability Engine

Production-quality quantitative research framework for estimating
the probability of large two-sided price moves using GARCH(1,1)
volatility forecasting, Euler-Maruyama GBM simulation, and
online probability calibration with walk-forward backtesting.
"""

__version__ = "2.0.0"

__all__ = [
    # Config
    "PipelineConfig",
    "DataConfig",
    "ModelConfig",
    "CalibrationConfig",
    "OutputConfig",
    "load_config",
    # Data
    "load_data",
    "run_data_quality_checks",
    # GARCH
    "fit_garch",
    "GarchResult",
    "garch_diagnostics",
    "project_to_stationary",
    "ewma_volatility",
    "garch_term_structure_vol",
    # Monte Carlo
    "simulate_gbm_terminal",
    "simulate_garch_terminal",
    "compute_move_probability",
    "compute_state_dependent_jumps",
    # Calibration
    "HistogramCalibrator",
    "IsotonicCalibrator",
    "OnlineCalibrator",
    "RegimeCalibrator",
    "MultiFeatureCalibrator",
    "RegimeMultiFeatureCalibrator",
    # Backtest
    "run_walkforward",
    "compute_backtest_analytics",
    # Evaluation
    "brier_score",
    "brier_skill_score",
    "expected_calibration_error",
    "log_loss",
    "auc_roc",
    "separation",
    "effective_sample_size",
    "crps_from_quantiles",
    "compute_metrics",
    "compute_reliability",
    "compute_risk_report",
    "value_at_risk",
    "conditional_var",
    "max_drawdown",
    # Model Selection
    "expanding_window_cv",
    "compare_models",
    "apply_promotion_gates",
    "apply_promotion_gates_oof",
    "calibration_aic",
    "calibration_bic",
    # Output
    "write_outputs",
]

from .config import (
    PipelineConfig,
    DataConfig,
    ModelConfig,
    CalibrationConfig,
    OutputConfig,
    load_config,
)
from .data_layer import load_data, run_data_quality_checks
from .garch import fit_garch, GarchResult, garch_diagnostics, project_to_stationary, ewma_volatility, garch_term_structure_vol
from .monte_carlo import simulate_gbm_terminal, simulate_garch_terminal, compute_move_probability, compute_state_dependent_jumps
from .calibration import HistogramCalibrator, IsotonicCalibrator, OnlineCalibrator, RegimeCalibrator, MultiFeatureCalibrator, RegimeMultiFeatureCalibrator
from .backtest import run_walkforward, compute_backtest_analytics
from .evaluation import (
    brier_score,
    brier_skill_score,
    expected_calibration_error,
    log_loss,
    auc_roc,
    separation,
    effective_sample_size,
    crps_from_quantiles,
    compute_metrics,
    compute_reliability,
    compute_risk_report,
    value_at_risk,
    conditional_var,
    max_drawdown,
)
from .model_selection import expanding_window_cv, compare_models, apply_promotion_gates, apply_promotion_gates_oof, calibration_aic, calibration_bic
from .output import write_outputs
