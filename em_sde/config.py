"""Configuration loading and validation."""

import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    source: str = "yfinance"
    ticker: str = "SPY"
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    fallback_to_synthetic: bool = True
    min_rows: int = 756
    max_retry: int = 5
    cache_max_age_days: int = 7
    csv_path: Optional[str] = None
    synthetic_days: int = 2520
    synthetic_seed: int = 12345


@dataclass
class ModelConfig:
    k: float = 2.0
    mu_year: float = 0.0
    horizons: List[int] = field(default_factory=lambda: [5, 10, 20])
    garch_window: int = 756
    garch_min_window: int = 252
    mc_base_paths: int = 100_000
    mc_boost_paths: int = 200_000
    mc_boost_threshold: float = 0.02
    seed: int = 42
    t_df: float = 5.0  # Student-t degrees of freedom for fat tails (0 or inf = Gaussian)
    # GARCH-in-simulation: evolve vol dynamics within MC paths
    garch_in_sim: bool = False         # Enable GARCH vol dynamics within MC paths
    garch_model_type: str = "garch"    # "garch" or "gjr" (GJR-GARCH for leverage effect)
    # Stationarity constraint: project non-stationary GARCH params before simulation
    garch_stationarity_constraint: bool = True
    garch_target_persistence: float = 0.98
    garch_fallback_to_ewma: bool = False  # if True, use EWMA instead of projection
    # Jump-diffusion (Merton model)
    jump_enabled: bool = False         # Enable Merton jump-diffusion
    jump_intensity: float = 2.0        # lambda: expected jumps per year
    jump_mean: float = -0.02           # mu_J: mean jump size (log-space, negative = crash bias)
    jump_vol: float = 0.04             # sigma_J: jump size volatility
    # State-dependent jump parameters (conditional on realized-vol regime)
    jump_state_dependent: bool = False
    jump_low_intensity: float = 1.0
    jump_low_mean: float = -0.01
    jump_low_vol: float = 0.03
    jump_high_intensity: float = 4.0
    jump_high_mean: float = -0.04
    jump_high_vol: float = 0.06
    # Threshold mode: controls how the large-move threshold is defined
    threshold_mode: str = "vol_scaled"  # "vol_scaled" | "fixed_pct" | "anchored_vol" | "regime_gated"
    fixed_threshold_pct: float = 0.05  # absolute return threshold (used when threshold_mode="fixed_pct")
    # Regime-gated threshold routing (routes between modes based on vol regime)
    regime_gated_low_mode: str = "fixed_pct"
    regime_gated_mid_mode: str = "fixed_pct"
    regime_gated_high_mode: str = "anchored_vol"
    regime_gated_warmup: int = 252
    regime_gated_vol_window: int = 252
    regime_gated_fixed_pct_by_horizon: Optional[dict] = None  # per-horizon override: {5: 0.03, 10: 0.04}
    # MC vol term structure: use GARCH h-step average vol instead of 1-step forecast
    mc_vol_term_structure: bool = False
    # Regime-conditional Student-t degrees of freedom for MC innovations
    mc_regime_t_df: bool = False
    mc_regime_t_df_low: float = 8.0    # thinner tails in low-vol
    mc_regime_t_df_mid: float = 5.0    # default
    mc_regime_t_df_high: float = 4.0   # heavier tails in high-vol
    # HMM regime detection
    hmm_regime: bool = False              # Enable HMM-based regime detection
    hmm_n_regimes: int = 2               # Number of HMM states (2 = low/high vol)
    hmm_vol_blend: float = 0.5           # Blend weight: 0=pure GARCH, 1=pure HMM sigma
    hmm_refit_interval: int = 20         # Refit HMM every N days (not every day — too slow)
    store_quantiles: bool = False      # store MC return quantiles for CRPS evaluation


@dataclass
class CalibrationConfig:
    lr: float = 0.05
    adaptive_lr: bool = True   # scale lr by 1/sqrt(1 + n_updates)
    min_updates: int = 50      # warm-up: don't calibrate until this many labels resolve
    safety_gate: bool = False  # auto-fallback to raw when cal Brier > raw Brier
    gate_window: int = 200     # rolling window for safety gate comparison
    # Discrimination guardrail: gate on rolling AUC and separation
    gate_on_discrimination: bool = True
    gate_auc_threshold: float = 0.50
    gate_separation_threshold: float = 0.0
    gate_discrimination_window: int = 200
    regime_conditional: bool = False  # enable per-vol-regime calibration
    regime_n_bins: int = 3            # number of vol-regime bins
    multi_feature: bool = False       # enable multi-feature online logistic calibration
    multi_feature_lr: float = 0.01   # learning rate for multi-feature calibrator
    multi_feature_l2: float = 1e-4   # L2 regularization strength
    multi_feature_min_updates: int = 100  # warmup for multi-feature calibrator
    multi_feature_regime_conditional: bool = False  # per-regime MF calibrators (requires multi_feature=true)
    histogram_post_calibration: bool = True  # legacy flag; use post_cal_method instead
    post_cal_method: str = ""               # "histogram" | "isotonic" | "none"; overrides histogram_post_calibration when set
    histogram_n_bins: int = 10               # number of equal-width bins for histogram/isotonic post-cal
    histogram_min_samples: int = 15          # min effective samples per bin before correction activates
    histogram_prior_strength: float = 15.0   # Bayesian shrinkage (histogram only): correction *= count/(count + prior_strength)
    histogram_monotonic: bool = True         # monotonic PAV enforcement on bin corrections (histogram only)
    ensemble_enabled: bool = True
    ensemble_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    # Promotion gates for model selection
    promotion_gates_enabled: bool = False
    promotion_bss_min: float = 0.0
    promotion_auc_min: float = 0.55
    promotion_ece_max: float = 0.02
    promotion_pooled_gate: bool = False  # use pooled ECE gate (all regimes combined) as primary
    # Neural calibrator (MLP): replaces multi-feature + histogram stack
    calibration_method: str = ""         # "online" | "multi_feature" | "neural" | "" (legacy: uses multi_feature flag)
    neural_hidden_size: int = 8          # MLP hidden layer width
    neural_lr: float = 0.005             # learning rate (lower than MF due to more params)
    neural_l2: float = 1e-4              # L2 regularization
    neural_min_updates: int = 100        # warmup before calibration active
    neural_regime_conditional: bool = False  # separate MLP per vol regime


@dataclass
class OutputConfig:
    base_dir: str = "outputs"
    charts: bool = True


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(path: str) -> PipelineConfig:
    """Load and validate configuration from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = PipelineConfig()

    if "data" in raw:
        for k, v in raw["data"].items():
            if v is not None and hasattr(cfg.data, k):
                setattr(cfg.data, k, v)

    if "model" in raw:
        for k, v in raw["model"].items():
            if v is not None and hasattr(cfg.model, k):
                setattr(cfg.model, k, v)

    if "calibration" in raw:
        for k, v in raw["calibration"].items():
            if v is not None and hasattr(cfg.calibration, k):
                setattr(cfg.calibration, k, v)

    if "output" in raw:
        for k, v in raw["output"].items():
            if v is not None and hasattr(cfg.output, k):
                setattr(cfg.output, k, v)

    _validate(cfg)
    return cfg


def _validate(cfg: PipelineConfig):
    """Validate configuration constraints."""
    assert cfg.data.source in ("yfinance", "csv", "synthetic"), \
        f"Invalid source: {cfg.data.source}"
    assert cfg.data.min_rows >= 252, "min_rows must be >= 252"
    assert cfg.model.k > 0, "k must be positive"
    assert cfg.model.garch_min_window >= 252, "garch_min_window must be >= 252"
    assert cfg.model.mc_base_paths >= 1000, "mc_base_paths must be >= 1000"
    assert all(h > 0 for h in cfg.model.horizons), "horizons must be positive"
    assert 0 < cfg.calibration.lr < 1, "lr must be in (0, 1)"

    if cfg.calibration.ensemble_enabled:
        assert len(cfg.calibration.ensemble_weights) == len(cfg.model.horizons), \
            "ensemble_weights length must match horizons"
        assert abs(sum(cfg.calibration.ensemble_weights) - 1.0) < 1e-6, \
            "ensemble_weights must sum to 1.0"

    if cfg.calibration.regime_conditional:
        assert cfg.calibration.regime_n_bins >= 2, "regime_n_bins must be >= 2"

    if cfg.calibration.multi_feature_regime_conditional:
        assert cfg.calibration.multi_feature, \
            "multi_feature_regime_conditional requires multi_feature=true"

    if cfg.calibration.post_cal_method:
        assert cfg.calibration.post_cal_method in ("histogram", "isotonic", "none"), \
            f"post_cal_method must be 'histogram', 'isotonic', or 'none', got {cfg.calibration.post_cal_method!r}"

    if cfg.calibration.calibration_method:
        assert cfg.calibration.calibration_method in ("online", "multi_feature", "neural"), \
            f"calibration_method must be 'online', 'multi_feature', or 'neural', got {cfg.calibration.calibration_method!r}"
    if cfg.calibration.neural_regime_conditional:
        assert cfg.calibration.calibration_method == "neural", \
            "neural_regime_conditional requires calibration_method='neural'"

    assert cfg.model.threshold_mode in ("vol_scaled", "fixed_pct", "anchored_vol", "regime_gated"), \
        f"threshold_mode must be 'vol_scaled', 'fixed_pct', 'anchored_vol', or 'regime_gated', got {cfg.model.threshold_mode}"
    if cfg.model.threshold_mode == "fixed_pct":
        assert cfg.model.fixed_threshold_pct > 0, "fixed_threshold_pct must be positive"
    if cfg.model.threshold_mode == "regime_gated":
        _valid_sub = ("vol_scaled", "fixed_pct", "anchored_vol")
        assert cfg.model.regime_gated_low_mode in _valid_sub, \
            f"regime_gated_low_mode must be one of {_valid_sub}"
        assert cfg.model.regime_gated_mid_mode in _valid_sub, \
            f"regime_gated_mid_mode must be one of {_valid_sub}"
        assert cfg.model.regime_gated_high_mode in _valid_sub, \
            f"regime_gated_high_mode must be one of {_valid_sub}"
        assert cfg.model.regime_gated_warmup >= 50, "regime_gated_warmup must be >= 50"

    if cfg.model.hmm_regime:
        assert cfg.model.hmm_n_regimes >= 2, "hmm_n_regimes must be >= 2"
        assert 0.0 <= cfg.model.hmm_vol_blend <= 1.0, "hmm_vol_blend must be in [0, 1]"
        assert cfg.model.hmm_refit_interval >= 1, "hmm_refit_interval must be >= 1"

    assert cfg.model.garch_model_type in ("garch", "gjr"), \
        f"garch_model_type must be 'garch' or 'gjr', got {cfg.model.garch_model_type}"
    if cfg.model.jump_enabled:
        assert cfg.model.jump_intensity >= 0, "jump_intensity must be non-negative"
        assert cfg.model.jump_vol >= 0, "jump_vol must be non-negative"
    if cfg.model.jump_state_dependent:
        assert cfg.model.jump_enabled, "jump_state_dependent requires jump_enabled=True"
        assert cfg.model.jump_low_intensity >= 0, "jump_low_intensity must be non-negative"
        assert cfg.model.jump_high_intensity >= 0, "jump_high_intensity must be non-negative"
        assert cfg.model.jump_low_vol >= 0, "jump_low_vol must be non-negative"
        assert cfg.model.jump_high_vol >= 0, "jump_high_vol must be non-negative"

    if cfg.data.source == "csv":
        assert cfg.data.csv_path is not None, "csv_path required when source=csv"
