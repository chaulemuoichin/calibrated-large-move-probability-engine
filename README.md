# Calibrated Large-Move Probability Engine

A production-ready quantitative research framework for forecasting **two-sided large price move probabilities** using:

- rolling **GARCH(1,1)** or **GJR-GARCH(1,1,1)** volatility forecasts,
- **Euler-Maruyama Monte Carlo** with GARCH-in-simulation volatility dynamics,
- optional **Student-t fat tails** and **Merton jump-diffusion**,
- **multi-feature online probability calibration** in strict walk-forward mode,
- **fixed-threshold mode** that eliminates self-referencing threshold bias.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-105%20passing-brightgreen)
![Model](https://img.shields.io/badge/model-GARCH%20%2B%20GJR%20%2B%20Jumps%20%2B%20MC%20%2B%20Calibration-informational)

## Overview

This engine estimates:

*P*(|*R*<sub>H</sub>| &ge; threshold)

for multiple horizons `H` (default `[5, 10, 20]` trading days).

**Three threshold modes** control what "large move" means:

- `vol_scaled` (legacy): threshold = `k * sigma_1d * sqrt(H)` -- scales with current vol
- `fixed_pct` (recommended): threshold = fixed absolute return (e.g., 5%) -- genuine discrimination
- `anchored_vol`: threshold uses long-run unconditional vol -- quasi-fixed goalpost

The system is designed for reproducible research:

- no-lookahead walk-forward backtesting,
- deterministic seeds,
- cached market data,
- expanding-window cross-validation for model comparison,
- complete run artifacts (`results.csv`, `summary.json`, reliability and charts).

## Core Features

### Modeling

- **GARCH(1,1) / GJR-GARCH(1,1,1) volatility forecasting** with automatic EWMA fallback.
- **GARCH-in-simulation** for per-step volatility dynamics inside Monte Carlo paths.
- **GJR leverage effect** where negative shocks can amplify future volatility more than positive shocks.
- **Merton jump-diffusion** for Poisson-arrival jumps independent from diffusion noise.
- **Euler-Maruyama simulation** in log-space with constant-vol or GARCH-evolving volatility.
- **Fat-tail mode** with variance-normalized Student-t innovations (`t_df > 2`).
- **Adaptive path boosting** when recent event frequency is very low.
- **Three threshold modes**: `vol_scaled`, `fixed_pct`, and `anchored_vol`.

### Calibration Features

- **Online logistic calibration**: `p_cal = sigmoid(a + b * logit(p_raw))`.
- **Multi-feature calibration**: 6-feature online logistic with vol level, vol trend, vol ratio, and vol-of-vol.
- **Warm-up control** (`min_updates`) before calibration activates.
- **Adaptive learning rate** (`lr / sqrt(1 + n_updates)`) for stability.
- **L2 regularization** and **gradient clipping** for multi-feature stability.
- **Safety gate** with auto-fallback to raw probabilities when calibration worsens Brier score.
- **Regime-conditional calibration** with separate calibrators for low/mid/high volatility regimes.
- Optional **ensemble/meta calibration** across horizons.

### Evaluation and Validation

- **Brier Score** and **Brier Skill Score** (BSS vs. climatology baseline).
- **LogLoss** (binary cross-entropy).
- **AUC-ROC** (discrimination/ranking power).
- **Separation**: `mean(p|event) - mean(p|non-event)`.
- **Effective sample size** correcting for overlapping-window autocorrelation.
- **CRPS** (Continuous Ranked Probability Score) from stored MC quantiles.
- **Wilson confidence intervals** on reliability diagram bins.
- **AIC/BIC** information criteria for calibration model complexity.
- Overlapping and non-overlapping evaluation windows.

### Risk Analytics

- **Value-at-Risk (VaR)** at 95% and 99% confidence from realized returns.
- **Conditional VaR / Expected Shortfall (CVaR)** -- average tail loss beyond VaR.
- **Return skewness and excess kurtosis** -- distribution shape diagnostics.
- **Maximum drawdown** -- peak-to-trough loss measurement.
- Per-horizon risk report in structured output.

### Data Quality

- **Outlier detection** via IQR method (5x multiplier, flags truly extreme moves).
- **Stale price detection** (consecutive unchanged prices indicating data feed issues).
- **Data gap detection** (missing business days beyond threshold).
- **Return statistics** (mean, std, skewness, kurtosis, min/max) in structured report.

### Model Diagnostics

- **GARCH stationarity check**: persistence = alpha + beta (+ gamma/2 for GJR) < 1.
- **Volatility half-life**: days for shock to decay 50%.
- **Unconditional variance/volatility** from GARCH parameters.
- **Calibration convergence diagnostics**: parameter velocity, range, convergence flag.

### Model Selection

- **Expanding-window cross-validation** across multiple configs.
- **Automatic ranking** by BSS and AUC across CV folds.
- CLI `--compare` mode for head-to-head evaluation.

## Pipeline

```text
Price Data -> Daily Returns -> GARCH/GJR-GARCH -> sigma_1d, omega, alpha, beta, gamma
                                                -> threshold (fixed_pct | anchored_vol | vol_scaled)
                                                -> Monte Carlo (GARCH-in-sim + jumps)
                                                -> p_raw(H) + optional quantiles
                                                -> Multi-feature or online calibration -> p_cal(H)
                                                -> Queue-based label resolution (no lookahead)
```

## Mathematical Specification

### 1) Threshold Modes

**Vol-scaled (legacy)**:
*thr*<sub>H</sub> = *k* &middot; &sigma;<sub>1d</sub> &middot; &radic;*H*

**Fixed percentage (recommended)**:
*thr*<sub>H</sub> = *fixed_threshold_pct* (e.g., 0.05 for 5%)

**Anchored vol**:
*thr*<sub>H</sub> = *k* &middot; &sigma;<sub>unconditional</sub> &middot; &radic;*H*

where `sigma_unconditional` is the expanding-window standard deviation of all past returns.

### 2) Event Definition

For prediction date `t` and horizon `H`:

*y*<sub>H</sub>(t) = I(|*S*<sub>t+H</sub> / *S*<sub>t</sub> - 1| &ge; *thr*<sub>H</sub>)

### 3) Multi-Feature Calibration

*p*<sub>cal</sub> = &sigma;(**w**<sup>T</sup> **x**)

where:
**x** = [1, logit(*p*<sub>raw</sub>), &sigma;<sub>1d</sub>, &Delta;&sigma;<sub>20d</sub>, *vol_ratio*, *vol_of_vol*]

Online SGD with L2 regularization:
**w** &larr; **w** + &eta; [(*y* - *p*<sub>cal</sub>)**x** - &lambda;**w**]

### 4) Brier Skill Score

*BSS* = 1 - (*Brier*<sub>model</sub> / *Brier*<sub>climatology</sub>)

where `Brier_climatology = p_bar * (1 - p_bar)`. BSS > 0 means the model beats predicting the historical event rate.

## Installation

### Prerequisites

- Python 3.10+

### Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Quick Start

```bash
# Run with fixed-threshold config (recommended)
python -m em_sde.run --config configs/spy_fixed.yaml --run-id run_spy_fixed

# Run with legacy vol-scaled config
python -m em_sde.run --config configs/spy.yaml --run-id run_spy

# Compare multiple configs via expanding-window CV
python -m em_sde.run --compare configs/spy_fixed.yaml configs/spy.yaml --cv-folds 5
```

### Preset Configs

**Fixed-threshold (recommended for production):**

```bash
python -m em_sde.run --config configs/spy_fixed.yaml --run-id run_spy_fixed
python -m em_sde.run --config configs/goog_fixed.yaml --run-id run_goog_fixed
python -m em_sde.run --config configs/tsla_fixed.yaml --run-id run_tsla_fixed
```

**Legacy vol-scaled (for comparison):**

```bash
python -m em_sde.run --config configs/spy.yaml --run-id run_spy
python -m em_sde.run --config configs/goog_garchsim.yaml --run-id run_goog
python -m em_sde.run --config configs/tsla_garchsim.yaml --run-id run_tsla
```

## Configuration

All settings are YAML-based in `configs/`.

### Model Settings

- `k`: volatility-scaled threshold multiplier
- `horizons`: forecast windows in trading days
- `threshold_mode`: `"vol_scaled"` | `"fixed_pct"` | `"anchored_vol"`
- `fixed_threshold_pct`: absolute return threshold (default `0.05` = 5%)
- `garch_in_sim`: enable GARCH volatility dynamics within MC paths
- `garch_model_type`: `"garch"` or `"gjr"`
- `jump_enabled`: enable Merton jump-diffusion
- `jump_intensity`, `jump_mean`, `jump_vol`: jump parameters
- `t_df`: Student-t degrees of freedom (`0` for Gaussian)
- `store_quantiles`: store MC return quantiles for CRPS evaluation
- `seed`: reproducibility seed

### Calibration Settings

- `lr`, `adaptive_lr`, `min_updates`
- `safety_gate`: fallback to raw when calibrated Brier degrades
- `multi_feature`: enable 6-feature online logistic calibration
- `multi_feature_lr`, `multi_feature_l2`, `multi_feature_min_updates`
- `regime_conditional`: enable per-vol-regime calibration
- `regime_n_bins`: number of volatility regime bins
- `ensemble_enabled`, `ensemble_weights`

## Output Artifacts

Each run writes to `outputs/<run_id>/`:

```text
outputs/<run_id>/
  results.csv
  reliability.csv
  summary.json
  charts/
    probability_timeseries.png
    reliability_diagram.png       (with Wilson CI bands)
    realized_return_hist.png
    rolling_brier.png
    probability_vs_price.png
    volatility_regime.png
    signal_heatmap.png
```

### `results.csv` Key Columns

| Column | Meaning |
| --- | --- |
| `date` | Prediction timestamp |
| `sigma_garch_1d` | Daily volatility forecast |
| `thr_{H}` | Event threshold for horizon `H` |
| `p_raw_{H}` | Raw Monte Carlo probability |
| `p_cal_{H}` | Calibrated probability |
| `mc_se_{H}` | Monte Carlo standard error |
| `y_{H}` | Resolved label (`1`, `0`, or unresolved `NaN`) |
| `delta_sigma` | 20-day vol change |
| `vol_ratio` | Realized vol / forecast vol |
| `vol_of_vol` | Rolling std of sigma |
| `q{NN}_{H}` | Return quantiles (if `store_quantiles: true`) |

## Evaluation Metrics

| Metric | What it measures | Good value |
| --- | --- | --- |
| **Brier Score** | Calibration accuracy | Lower is better |
| **BSS** | Skill vs. climatology | > 0 (positive = beats base rate) |
| **AUC-ROC** | Discrimination/ranking | > 0.5 (1.0 perfect) |
| **Separation** | Event vs non-event prob gap | > 0 (positive = events rank higher) |
| **N_eff** | Effective sample size | Corrects for overlap autocorrelation |
| **CRPS** | Full distribution accuracy | Lower is better |
| **LogLoss** | Cross-entropy | Lower is better |
| **VaR (95/99)** | Tail loss at confidence level | Context-dependent |
| **CVaR / ES** | Expected loss beyond VaR | Context-dependent |
| **Skewness** | Return distribution asymmetry | Negative = left tail risk |
| **Kurtosis** | Tail heaviness (excess) | > 0 = heavier than normal |

## Testing

This project includes **105 unit tests** in `tests/test_framework.py`, covering:

- Brownian increment statistics and GBM moment consistency,
- Student-t fat-tail behavior,
- calibration math and adaptive learning,
- strict no-lookahead invariants,
- output schema and probability bounds,
- GARCH parameter extraction (standard + GJR + EWMA fallback),
- GARCH-in-simulation dynamics and Merton jump-diffusion,
- seed independence across horizons,
- calibration safety gate behavior,
- **fixed-threshold discrimination** (Phase 1 core validation),
- **anchored-vol walk-forward safety** and discrimination,
- **multi-feature calibrator**: identity init, warmup, L2, gradient clipping,
- **Brier Skill Score**: climatology=0, perfect=1, anti-correlated<0,
- **effective sample size**: iid vs autocorrelated,
- **Wilson CI** coverage and bounds,
- **CRPS** from quantiles,
- **MC quantile storage** return/shape,
- **model selection**: AIC/BIC, compare_models ranking,
- **data quality**: outlier detection, stale prices, data gaps, return statistics,
- **risk analytics**: VaR, CVaR, skewness, kurtosis, max drawdown,
- **backtest analytics**: hit rate, signal turnover, precision/recall,
- **GARCH diagnostics**: stationarity, persistence, half-life, unconditional vol,
- **calibration convergence**: parameter velocity, convergence detection,
- **public API**: `__init__.py` exports, version, importability,
- backward compatibility and config validation.

Run tests:

```bash
python -m pytest tests/ -v
```

## Project Layout

```text
em_sde/
  __init__.py            # Public API with __all__ exports (v2.0.0)
  backtest.py            # Walk-forward engine + backtest analytics (hit rate, turnover)
  calibration.py         # OnlineCalibrator + RegimeCalibrator + MultiFeatureCalibrator + convergence diagnostics
  config.py              # YAML config loading and validation
  data_layer.py          # yfinance / CSV / synthetic data loading + data quality pipeline
  evaluation.py          # Brier, BSS, LogLoss, AUC-ROC, separation, CRPS, N_eff, VaR, CVaR, risk report
  garch.py               # GARCH / GJR-GARCH fitting with EWMA fallback + stationarity diagnostics
  model_selection.py     # Expanding-window CV and model comparison
  monte_carlo.py         # Euler-Maruyama MC with GARCH-in-sim + jumps + quantiles
  output.py              # CSV/JSON output + chart generation (Wilson CI bands)
  run.py                 # CLI entry point (--config and --compare modes) + timing
configs/
  spy_fixed.yaml         # SPY: fixed 5% threshold + GJR + multi-feature (recommended)
  goog_fixed.yaml        # GOOG: fixed 5% threshold + GJR + multi-feature
  tsla_fixed.yaml        # TSLA: fixed 5% + GJR + jumps + multi-feature
  spy.yaml               # SPY: legacy vol-scaled baseline
  goog_garchsim.yaml     # GOOG: legacy GJR GARCH-in-sim
  tsla_garchsim.yaml     # TSLA: legacy GJR GARCH-in-sim + jumps
  csv_example.yaml       # Template for custom CSV data
  synthetic_example.yaml # Pure synthetic GBM
outputs/
  <run_id>/...
tests/
  test_framework.py      # 75 unit tests
calibrated_large_move_probability_engine.py
requirements.txt
README.md
```

## Known Limitations

1. Event labels use close-to-close forward returns (no intraday path labeling).
2. Transaction costs and slippage are not modeled (this is a probability engine, not an execution simulator).
3. Multi-feature calibration requires sufficient label resolution before activation (100+ updates by default).
4. Overlapping windows inflate apparent sample size; use N_eff or non-overlapping metrics for proper inference.

## Practical Notes

- **Start with `fixed_pct` threshold mode** -- it eliminates the self-referencing bias that makes vol-scaled AUC ~0.50.
- Enable `garch_in_sim: true` and `garch_model_type: "gjr"` for all assets.
- Use `multi_feature: true` to leverage vol trend and forecast error signals.
- Use jump-diffusion mainly for crash-prone names (`jump_enabled: true`).
- If calibration degrades metrics, `safety_gate: true` auto-reverts to raw probabilities.
- Use `--compare` mode to systematically evaluate model variants via expanding-window CV.
- When EWMA fallback triggers, simulation falls back to constant-vol behavior.

## Disclaimer

Research and educational use only. Not investment advice.
