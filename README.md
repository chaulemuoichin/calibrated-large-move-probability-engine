# Calibrated Large-Move Probability Engine

Estimates the probability of large stock price moves over 1-2 week horizons, with online self-correction as outcomes arrive.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-312%20passing-brightgreen)

## Why This Exists

You want to buy a stock. The hard question: **is today a good entry?**

```text
price drops hard right after you buy
  -> you doubt yourself -> bad exit -> rushed next trade -> another loss
```

This system replaces guesswork with a number:

> "There is a 12% chance the price moves more than 5% in the next two weeks."

If that number is trustworthy, you can size positions, hedge, or wait with confidence.

## How It Works

Full methodology: [METHODOLOGY.md](METHODOLOGY.md)

1. **Estimate current volatility.** Fit a GJR-GARCH model to recent returns, with optional implied-vol blending.
2. **Simulate many futures.** Generate 30,000+ Monte Carlo price paths with GARCH-in-sim dynamics, Student-t fat tails, and Merton jump-diffusion.
3. **Count large moves.** What fraction of simulated paths exceed the threshold? That's the raw probability.
4. **Calibrate against history.** Multi-feature online logistic regression + histogram post-calibration with Bayesian shrinkage.
5. **Output a calibrated probability.** The final number reflects both current conditions and the model's track record.

## Quick Start

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# Run a backtest
python -m em_sde.run --config configs/spy_fixed.yaml --run-id my_first_run

# Compare configs with cross-validation
python -m em_sde.run --compare configs/spy_fixed.yaml configs/goog_fixed.yaml --cv-folds 5

# Generate a live prediction (after backtest builds state)
python -m em_sde.run --predict-now --config configs/exp_suite/exp_spy_regime_gated.yaml --save-state
```

Output goes to `outputs/<run_id>/`: results CSV, summary JSON, and charts.

## Live Prediction Mode

The system can generate forward-looking predictions from saved state, without rerunning the full backtest:

```bash
# Build checkpoint from backtest, then predict
python -m em_sde.run --predict-now --config configs/exp_suite/exp_spy_regime_gated.yaml --save-state

# Predict from existing checkpoint (~seconds)
python -m em_sde.run --predict-now --config configs/exp_suite/exp_spy_regime_gated.yaml --state-dir outputs/state/spy

# Daily scheduled predictions for all tickers
python scripts/daily_predict.py
```

State checkpoints contain serialized calibrators, GARCH parameters, and metadata in JSON format. The prediction engine loads a checkpoint, fits GARCH on the latest price history, runs MC simulation, and applies the saved calibrators.

## Key Metrics

| Metric | Meaning | Target |
| ------ | ------- | ------ |
| **Brier Skill Score** | Improvement over naive base-rate predictor | > 0 |
| **AUC** | Can the model rank event days above non-event days? | >= 0.55 |
| **ECE** | Are predicted probabilities accurate across all levels? | < 0.02 |

All p-values are Benjamini-Hochberg FDR-corrected. 95% BCa bootstrap CIs on all metrics. N_eff computed on prediction residuals.

## Current Results

| Ticker | H=5 | H=10 | H=20 |
|--------|-----|------|------|
| SPY    | PASS | PASS | FAIL (ECE=0.024) |
| GOOGL  | PASS | PASS | PASS |
| AMZN   | PASS | PASS | PASS |
| NVDA   | PASS | FAIL | FAIL |

**7/8 primary-horizon tests pass.** Full results: [RESULTS.md](RESULTS.md)

## Configuration

Settings live in YAML files under `configs/`.

**Threshold mode** (what counts as a "large move"):
- `fixed_pct` -- fixed return threshold (e.g., 5%). Recommended.
- `regime_gated` -- switches between fixed_pct and anchored_vol by vol regime.

**Volatility model:**
- `gjr` (GJR-GARCH captures leverage effect: drops amplify future vol more than rallies)
- GARCH-in-simulation for path-level vol dynamics

**Optional features:**
- Merton jump-diffusion for crash-prone assets
- Student-t innovations with regime-conditional degrees of freedom
- Multi-feature calibration (6-8 features + L2 regularization)
- Histogram post-calibration with Bayesian shrinkage + PAV monotonicity
- Options-implied volatility blending (VIX data)
- Earnings calendar proximity (H<=5, single stocks)

### Preset Configs

| Config | Description |
| ------ | ----------- |
| `spy_fixed.yaml` | SPY, fixed 5% threshold (quick start) |
| `goog_fixed.yaml` | GOOGL, fixed 5% threshold |
| `tsla_fixed.yaml` | TSLA, fixed 5% threshold + jump-diffusion |
| `exp_suite/exp_*` | BO-tuned production configs per ticker |

## Project Structure

```text
em_sde/                Core library
  config.py              YAML config system
  data_layer.py          Data loading, caching, quality checks
  garch.py               Volatility estimation (GJR-GARCH), state export/import
  monte_carlo.py         MC simulation (GARCH-in-sim, jumps, fat tails)
  calibration.py         Multi-feature calibration + histogram, state serialization
  backtest.py            Walk-forward engine, resolution queues, state checkpointing
  evaluation.py          BSS, AUC, ECE, N_eff, bootstrap CIs, FDR correction
  model_selection.py     Cross-validation, promotion gates
  predict.py             Live prediction engine (checkpoint load/save)
  resolve.py             Async label resolution for pending predictions
  run.py                 CLI entry point (backtest, --predict-now)
  output.py              Results and charts

configs/               Configuration
  exp_suite/             BO-tuned production configs (SPY, GOOGL, AMZN, NVDA)
  examples/              Example configs (CSV, synthetic)
  archive/               Inactive configs (AAPL, synthetic patterns)
  *.yaml                 Preset quick-start configs

scripts/               Runners
  run_bayesian_opt.py    Optuna hyperparameter optimization
  run_gate_recheck.py    5-fold CV gate validation
  run_overfit_check.py   Overfitting diagnostics
  baselines.py           Five baseline models (hist freq, GARCH-CDF, IV-BS, logistic, gradient boosting)
  run_ablation_study.py  7-variant component ablation
  run_temporal_holdout.py  Temporal hold-out (train pre-2020, test 2020-2025)
  run_paper_results.py   LaTeX tables with FDR-corrected p-values and CIs
  run_economic_significance.py  Portfolio analysis + transaction cost sensitivity
  generate_paper_figures.py  Publication-quality figures
  daily_predict.py       Daily prediction runner with scheduling support

paper/                 Academic paper
  main.tex               LaTeX source
  reproduce.py           Full reproduction script
  README.md              Reproduction instructions

data/                  Price data
  *_daily.csv            Real ticker data (SPY, GOOGL, AMZN, NVDA, AAPL)
  vix_history.csv        VIX/VIX9D/VIX3M implied vol data
  synthetic/             Synthetic pattern datasets

tests/                 312 unit tests
```

## Validation & Optimization

```bash
# Bayesian optimization (lean mode, thresholds locked)
python scripts/run_bayesian_opt.py spy --n-trials 12
python scripts/run_bayesian_opt.py spy --apply

# 5-fold CV gate validation
python -u scripts/run_gate_recheck.py spy

# Overfitting diagnostics
python scripts/run_overfit_check.py spy
```

## Academic Paper

A full paper is included in `paper/` with LaTeX source and a reproduction script.

```bash
# Reproduce all tables and figures (~2-4 hours)
python paper/reproduce.py --all
```

Seven baselines, 7-variant ablation, temporal hold-out, economic significance with transaction cost sensitivity, and publication-quality figures.

## Known Limitations

1. Primary claims scoped to H=5 and H=10 (1-2 weeks). H=20 calibration degrades for most tickers.
2. Calibration requires ~100 resolved outcomes to activate. Early predictions are uncalibrated.
3. Overlapping prediction windows create correlated samples. Always check N_eff, not raw count.
4. Transaction cost model is approximate (fixed proportional costs only).
5. Direction agnostic -- predicts magnitude, not direction.

## Disclaimer

Research and educational use only. Not investment advice.
