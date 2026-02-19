# Calibrated Large-Move Probability Engine

A production-ready quantitative research framework for forecasting **two-sided large price move probabilities** using:

- rolling **GARCH(1,1)** or **GJR-GARCH(1,1,1)** volatility forecasts,
- **Euler-Maruyama Monte Carlo** with GARCH-in-simulation volatility dynamics,
- optional **Student-t fat tails** and **state-dependent Merton jump-diffusion**,
- **multi-feature online probability calibration** in strict walk-forward mode,
- **regime-gated threshold routing** to switch threshold policy by vol percentile.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-133%20passing-brightgreen)
![Model](https://img.shields.io/badge/model-GARCH%20%2B%20GJR%20%2B%20Jumps%20%2B%20MC%20%2B%20Calibration-informational)

## Why This Exists ?!!

You are an investor. You think a stock is undervalued. You want to buy.
The painful question is simple: **is today already a good entry?**

```text
if price goes up after you buy
-> good for you
  -> you will feel like a genius

if price goes down hard right after you buy
-> you may doubt yourself
  -> bad exit & realized loss
    -> loss of confidence                                                          
      -> rushed next decision
        -> another loss   
          -> either posting "market is rigged" on Reddit & quit or repeating the loop 
```

This project is built to break that loop by turning entry fear into a measurable probability forecast.

## Practical Use Case

Use this as an entry-risk decision support tool.

- For each date and horizon, the engine estimates the probability of a defined event.
- In this project, the default event is a large two-sided price move.
- The event definition is configurable, so you can adapt thresholds and horizons to your own process.

If out-of-sample calibration remains strong, the output can support decisions such as scaling in, waiting for better conditions, or adding protection, instead of reacting emotionally after the fact.

**Final goal:** estimate the likelihood of an event using a strict walk-forward calibration forecasting loop.

## Overview

This engine estimates:

*P*(|*R*<sub>H</sub>| &ge; threshold)

for multiple horizons `H` (default `[5, 10, 20]` trading days), under strict no-lookahead walk-forward constraints.

**Four threshold modes** control what "large move" means:

- `vol_scaled` (legacy): threshold = `k * sigma_1d * sqrt(H)` -- scales with current vol
- `fixed_pct` (recommended): threshold = fixed absolute return (e.g., 5%) -- genuine discrimination
- `anchored_vol`: threshold uses long-run unconditional vol -- quasi-fixed goalpost
- `regime_gated`: routes between low/mid/high threshold policies by rolling vol percentile

The system is designed for reproducible research:

- no-lookahead walk-forward backtesting,
- deterministic seeds,
- cached market data,
- expanding-window cross-validation for model comparison,
- complete run artifacts (`results.csv`, `summary.json`, reliability and charts).

## Recent Updates (U1-U5)

- **U1: Regime-Gated Threshold Routing** (`threshold_mode: "regime_gated"`, default OFF).
- **U2: State-Dependent Jump Model** (`jump_state_dependent: true`, default OFF).
- **U3: AUC/Separation Calibration Guardrail** (`gate_on_discrimination: true`, default ON).
- **U4: Stationarity-Constrained GARCH Projection** (`garch_stationarity_constraint: true`, default ON).
- **U5: Promotion Gates per Regime Bucket** (`promotion_gates_enabled: true`, default OFF).

Backward compatibility is preserved: U3/U4 are safety no-ops unless triggered, while U1/U2/U5 are opt-in behavior changes.

## How It Works (Plain English)

The system reads recent price behavior to understand whether the market has been calm or jumpy.  
It then creates many possible future price stories instead of relying on one guess.  
From those stories, it measures how often a move large enough to matter to you actually happens.  
As real outcomes arrive, it checks whether earlier percentages were too optimistic or too conservative.  
It keeps correcting itself over time so the next set of percentages is more trustworthy.  
The result is a clear probability output you can use to decide buy now, wait, or scale in gradually.

## Core Features

### Modeling

- **GARCH(1,1) / GJR-GARCH(1,1,1) volatility forecasting** with automatic EWMA fallback.
- **GARCH-in-simulation** for per-step volatility dynamics inside Monte Carlo paths.
- **Stationarity-constrained GARCH projection** to target persistence when fitted params are non-stationary.
- **GJR leverage effect** where negative shocks can amplify future volatility more than positive shocks.
- **Merton jump-diffusion** for Poisson-arrival jumps independent from diffusion noise.
- **State-dependent jump routing** that interpolates jump parameters between low-vol and high-vol regimes.
- **Euler-Maruyama simulation** in log-space with constant-vol or GARCH-evolving volatility.
- **Fat-tail mode** with variance-normalized Student-t innovations (`t_df > 2`).
- **Adaptive path boosting** when recent event frequency is very low.
- **Four threshold modes**: `vol_scaled`, `fixed_pct`, `anchored_vol`, and `regime_gated`.

### Calibration Features

- **Online logistic calibration**: `p_cal = sigmoid(a + b * logit(p_raw))`.
- **Multi-feature calibration**: 6-feature online logistic with vol level, vol trend, vol ratio, and vol-of-vol.
- **Warm-up control** (`min_updates`) before calibration activates.
- **Adaptive learning rate** (`lr / sqrt(1 + n_updates)`) for stability.
- **L2 regularization** and **gradient clipping** for multi-feature stability.
- **Safety gate** with auto-fallback to raw probabilities when calibration worsens Brier score.
- **Discrimination guardrail** that disables calibration when rolling AUC or separation degrades below thresholds.
- **Regime-conditional calibration** with separate calibrators for low/mid/high volatility regimes.
- Optional **ensemble/meta calibration** across horizons.

### Evaluation and Validation

- **Brier Score** and **Brier Skill Score** (BSS vs. climatology baseline).
- **LogLoss** (binary cross-entropy).
- **AUC-ROC** (discrimination/ranking power).
- **Separation**: `mean(p|event) - mean(p|non-event)`.
- **ECE** (Expected Calibration Error) for bin-level calibration quality.
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
- **Promotion gates per vol-regime bucket** with hard thresholds on BSS, AUC, and ECE.
- CLI `--compare` mode for head-to-head evaluation.

## Pipeline

```text
Price Data -> Daily Returns -> GARCH/GJR-GARCH -> sigma_1d, omega, alpha, beta, gamma
                                                -> stationarity projection (optional safety)
                                                -> threshold (fixed_pct | anchored_vol | vol_scaled | regime_gated)
                                                -> Monte Carlo (GARCH-in-sim + static/state-dependent jumps)
                                                -> p_raw(H) + optional quantiles
                                                -> Multi-feature or online calibration + discrimination gate -> p_cal(H)
                                                -> Queue-based label resolution (no lookahead)
```

## Mathematical Specification

### 1) Threshold Modes

**Vol-scaled (legacy)**:
*thr*<sub>H</sub> = *k* &middot; &sigma;<sub>1d</sub> &middot; &radic;*H*

**Fixed percentage (recommended default policy)**:
*thr*<sub>H</sub> = *fixed_threshold_pct* (e.g., 0.05 for 5%)

**Anchored vol**:
*thr*<sub>H</sub> = *k* &middot; &sigma;<sub>unconditional</sub> &middot; &radic;*H*

where `sigma_unconditional` is the expanding-window standard deviation of past returns.

**Regime-gated**:
mode(*t*) &isin; {low, mid, high} based on rolling percentile of &sigma;<sub>1d</sub>(t), then
*thr*<sub>H</sub>(t) = `thr_mode(mode(t), H)` using configured low/mid/high sub-modes.

### 2) Event Definition

For prediction date `t` and horizon `H`:

*y*<sub>H</sub>(t) = I(|*S*<sub>t+H</sub> / *S*<sub>t</sub> - 1| &ge; *thr*<sub>H</sub>(t))

### 3) Stationarity-Constrained GARCH Projection (U4)

Persistence:

- GARCH(1,1): `p = alpha + beta`
- GJR-GARCH(1,1,1): `p = alpha + beta + gamma/2`

If `p >= 1`, project coefficients with scale `s = target_persistence / p`:

- `alpha' = s * alpha`
- `beta' = s * beta`
- `gamma' = s * gamma` (GJR only)
- `omega` unchanged

### 4) State-Dependent Jumps (U2)

Jump parameters `(lambda, mu_J, sigma_J)` are linearly interpolated between low-vol and high-vol parameter sets using current vol percentile in the rolling vol history.

### 5) Multi-Feature Calibration

*p*<sub>cal</sub> = &sigma;(**w**<sup>T</sup> **x**)

where:
**x** = [1, logit(*p*<sub>raw</sub>), &sigma;<sub>1d</sub>, &Delta;&sigma;<sub>20d</sub>, *vol_ratio*, *vol_of_vol*]

Online SGD with L2 regularization:
**w** &larr; **w** + &eta; [(*y* - *p*<sub>cal</sub>)**x** - &lambda;**w**]

### 6) Discrimination Guardrail (U3)

On a rolling window, if calibrated discrimination degrades:

- `AUC_cal < gate_auc_threshold` or
- `Separation_cal < gate_separation_threshold`

then calibration is gated and raw probability is passed through.

### 7) Brier Skill Score and Promotion Gates (U5)

*BSS* = 1 - (*Brier*<sub>model</sub> / *Brier*<sub>climatology</sub>)

where `Brier_climatology = p_bar * (1 - p_bar)`.

In `--compare` mode, optional promotion gates are enforced per vol-regime bucket:

- `BSS_cal >= promotion_bss_min`
- `AUC_cal >= promotion_auc_min`
- `ECE_cal <= promotion_ece_max`

A config is blocked if any gate fails in any bucket.

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

# Run regime-gated experiment (opt-in U1 behavior)
python -m em_sde.run --config configs/exp_suite/exp_cluster_regime_gated.yaml --run-id exp_cluster_regime_gated

# Compare multiple configs via expanding-window CV
python -m em_sde.run --compare configs/spy_fixed.yaml configs/spy.yaml configs/exp_suite/exp_cluster_regime_gated.yaml --cv-folds 5
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

**Regime-gated experimental suite:**

```bash
python -m em_sde.run --config configs/exp_suite/exp_cluster_regime_gated.yaml --run-id exp_cluster_regime_gated
python -m em_sde.run --config configs/exp_suite/exp_jump_regime_gated.yaml --run-id exp_jump_regime_gated
python -m em_sde.run --config configs/exp_suite/exp_trend_regime_gated.yaml --run-id exp_trend_regime_gated
```

## Configuration

All settings are YAML-based in `configs/`.

### Model Settings

- `k`: volatility-scaled threshold multiplier
- `horizons`: forecast windows in trading days
- `threshold_mode`: `"vol_scaled"` | `"fixed_pct"` | `"anchored_vol"` | `"regime_gated"`
- `fixed_threshold_pct`: absolute return threshold (default `0.05` = 5%)
- `regime_gated_low_mode`, `regime_gated_mid_mode`, `regime_gated_high_mode`: sub-mode routing for U1
- `regime_gated_warmup`, `regime_gated_vol_window`: warmup/history for regime classification
- `garch_in_sim`: enable GARCH volatility dynamics within MC paths
- `garch_model_type`: `"garch"` or `"gjr"`
- `garch_stationarity_constraint` (default `true`): enable U4 stationarity projection
- `garch_target_persistence` (default `0.98`): target persistence for projected params
- `garch_fallback_to_ewma` (default `false`): fallback policy when non-stationary
- `jump_enabled`: enable Merton jump-diffusion
- `jump_intensity`, `jump_mean`, `jump_vol`: jump parameters
- `jump_state_dependent` (default `false`): enable U2 interpolation by vol regime
- `jump_low_*`, `jump_high_*`: low/high regime jump parameter sets for U2
- `t_df`: Student-t degrees of freedom (`0` for Gaussian)
- `store_quantiles`: store MC return quantiles for CRPS evaluation
- `seed`: reproducibility seed

### Calibration Settings

- `lr`, `adaptive_lr`, `min_updates`
- `safety_gate`: fallback to raw when calibrated Brier degrades
- `gate_on_discrimination` (default `true`): enable U3 AUC/separation guardrail
- `gate_auc_threshold` (default `0.50`), `gate_separation_threshold` (default `0.0`)
- `gate_discrimination_window` (default `200`)
- `multi_feature`: enable 6-feature online logistic calibration
- `multi_feature_lr`, `multi_feature_l2`, `multi_feature_min_updates`
- `regime_conditional`: enable per-vol-regime calibration
- `regime_n_bins`: number of volatility regime bins
- `ensemble_enabled`, `ensemble_weights`
- `promotion_gates_enabled` (default `false`): enable U5 hard gates in `--compare`
- `promotion_bss_min` (default `0.0`)
- `promotion_auc_min` (default `0.55`)
- `promotion_ece_max` (default `0.02`)

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
| `sigma_source` | Volatility source (`garch`, `gjr_garch`, `ewma_fallback`, projected variants) |
| `garch_projected` | Whether U4 projection was applied on that step |
| `thr_{H}` | Event threshold for horizon `H` |
| `threshold_regime` | Active threshold sub-mode when `threshold_mode: regime_gated` |
| `p_raw_{H}` | Raw Monte Carlo probability |
| `p_cal_{H}` | Calibrated probability |
| `mc_se_{H}` | Monte Carlo standard error |
| `y_{H}` | Resolved label (`1`, `0`, or unresolved `NaN`) |
| `delta_sigma` | 20-day vol change |
| `vol_ratio` | Realized vol / forecast vol |
| `vol_of_vol` | Rolling std of sigma |
| `jump_intensity_step` | Per-step jump intensity when state-dependent jumps are enabled |
| `jump_mean_step` | Per-step jump mean when state-dependent jumps are enabled |
| `jump_vol_step` | Per-step jump vol when state-dependent jumps are enabled |
| `q{NN}_{H}` | Return quantiles (if `store_quantiles: true`) |

## Evaluation Metrics

| Metric | What it measures | Good value |
| --- | --- | --- |
| **Brier Score** | Calibration accuracy | Lower is better |
| **BSS** | Skill vs. climatology | > 0 (positive = beats base rate) |
| **AUC-ROC** | Discrimination/ranking | > 0.5 (1.0 perfect) |
| **Separation** | Event vs non-event prob gap | > 0 (positive = events rank higher) |
| **ECE** | Calibration gap across bins | < 0.02 (institutional gate default) |
| **N_eff** | Effective sample size | Corrects for overlap autocorrelation |
| **CRPS** | Full distribution accuracy | Lower is better |
| **LogLoss** | Cross-entropy | Lower is better |
| **VaR (95/99)** | Tail loss at confidence level | Context-dependent |
| **CVaR / ES** | Expected loss beyond VaR | Context-dependent |
| **Skewness** | Return distribution asymmetry | Negative = left tail risk |
| **Kurtosis** | Tail heaviness (excess) | > 0 = heavier than normal |

## Testing

This project includes **133 unit tests** in `tests/test_framework.py`, covering:

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
- **stationarity projection (U4)**: non-stationary parameter projection and target persistence behavior,
- **state-dependent jumps (U2)**: interpolation behavior across vol regimes,
- **regime-gated threshold routing (U1)**: low/mid/high mode assignment and warmup handling,
- **discrimination guardrails (U3)**: calibration gating on rolling AUC/separation,
- **promotion gates (U5)**: per-regime hard gate enforcement in compare mode,
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
  backtest.py            # Walk-forward engine + RegimeRouter + backtest analytics
  calibration.py         # Online/Regime/MultiFeature calibrators + discrimination gate
  config.py              # YAML config loading and validation
  data_layer.py          # yfinance / CSV / synthetic data loading + data quality pipeline
  evaluation.py          # Brier, BSS, LogLoss, AUC-ROC, separation, ECE, CRPS, N_eff, risk metrics
  garch.py               # GARCH/GJR fitting + stationarity diagnostics + parameter projection
  model_selection.py     # Expanding-window CV + promotion gates
  monte_carlo.py         # Euler-Maruyama MC with GARCH-in-sim + state-dependent jumps + quantiles
  output.py              # CSV/JSON output + chart generation (Wilson CI bands)
  run.py                 # CLI entry point (--config/--compare) + optional promotion-gate report
configs/
  spy_fixed.yaml         # SPY: fixed 5% threshold + GJR + multi-feature (recommended)
  goog_fixed.yaml        # GOOG: fixed 5% threshold + GJR + multi-feature
  tsla_fixed.yaml        # TSLA: fixed 5% + GJR + jumps + multi-feature
  spy.yaml               # SPY: legacy vol-scaled baseline
  goog_garchsim.yaml     # GOOG: legacy GJR GARCH-in-sim
  tsla_garchsim.yaml     # TSLA: legacy GJR GARCH-in-sim + jumps
  exp_suite/exp_cluster_regime_gated.yaml # Regime-gated clustered-vol experiment
  exp_suite/exp_jump_regime_gated.yaml    # Regime-gated jump-regime experiment
  exp_suite/exp_trend_regime_gated.yaml   # Regime-gated trending-regime experiment
  csv_example.yaml       # Template for custom CSV data
  synthetic_example.yaml # Pure synthetic GBM
outputs/
  <run_id>/...
tests/
  test_framework.py      # 133 unit tests
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
- For adaptive threshold control, use `threshold_mode: "regime_gated"` with explicit low/mid/high sub-modes.
- Enable `garch_in_sim: true` and `garch_model_type: "gjr"` for volatile assets.
- Keep `garch_stationarity_constraint: true` (default) to avoid unstable forward variance dynamics.
- Use `multi_feature: true` to leverage vol trend and forecast error signals.
- Use jump-diffusion mainly for crash-prone names (`jump_enabled: true`), and `jump_state_dependent: true` only when you have enough regime variation.
- Keep `gate_on_discrimination: true` (default) to prevent calibration from degrading ranking quality.
- If calibration degrades metrics, `safety_gate: true` auto-reverts to raw probabilities.
- Use `--compare` mode to systematically evaluate model variants via expanding-window CV.
- For institutional promotion decisions, enable `promotion_gates_enabled: true` and enforce BSS/AUC/ECE by regime bucket.
- When EWMA fallback triggers, simulation falls back to constant-vol behavior.

## Disclaimer

Research and educational use only. Not investment advice.
