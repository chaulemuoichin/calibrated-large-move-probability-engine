# Calibrated Large-Move Probability Engine

Estimates the probability of large stock price moves over 1-4 week horizons, with online self-correction as outcomes arrive.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-296%20passing-brightgreen)

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

1. **Estimate current volatility.** Fit a GARCH/GJR or HAR-family model to recent returns, with optional OHLC range anchoring and implied-vol blending.
2. **Simulate many futures.** Generate thousands of Monte Carlo price paths using current vol conditions, optional jumps, and scheduled-event variance.
3. **Count large moves.** What fraction of simulated paths exceed the threshold? That's the raw probability.
4. **Calibrate against history.** Compare past predictions to actual outcomes and adjust systematically, either online or with train-fold-only offline pooled calibration.
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

# Compare two configs with cross-validation
python -m em_sde.run --compare configs/spy_fixed.yaml configs/goog_fixed.yaml --cv-folds 5
```

Output goes to `outputs/<run_id>/`: results CSV, summary JSON, and charts.

## Output

| File | Contents |
| ---- | -------- |
| `results.csv` | Daily predictions: raw/calibrated probabilities, realized labels, volatility |
| `summary.json` | Evaluation metrics (Brier, AUC, ECE) |
| `reliability.csv` | Calibration curve data |
| `artifact_manifest.json` | Dataset hash, config fingerprint, and artifact lineage |
| `data_snapshot.csv` | Persisted research snapshot when enabled |
| `charts/` | Probability time series, reliability diagram, vol regime, rolling accuracy |

### Key Metrics

| Metric | Meaning | Target |
| ------ | ------- | ------ |
| **Brier Skill Score** | Improvement over naive base-rate predictor | > 0 |
| **AUC** | Can the model rank event days above non-event days? | >= 0.55 |
| **ECE** | Are predicted probabilities accurate across all levels? | < 0.02 |

## Configuration

Settings live in YAML files under `configs/`.

**Threshold mode** (what counts as a "large move"):

- `fixed_pct` — fixed return threshold (e.g., 5%). Recommended.
- `anchored_vol` — slowly-moving threshold tied to historical average vol.
- `regime_gated` — switches between modes based on current vol regime.

**Volatility model:**

- `garch` or `gjr` (GJR captures leverage effect: drops amplify future vol more than rallies)
- GARCH-in-simulation for path-level vol dynamics

**Optional features:**

- Jump-diffusion (Merton) for crash-prone stocks
- Multi-feature calibration (6 features + L2 regularization)
- Offline pooled calibration on train folds (`offline_pooled_calibration`)
- Student-t fat-tailed innovations with regime-conditional degrees of freedom
- Earnings calendar proximity for single-stock short-horizon calibration
- Scheduled event jump variance for earnings-driven horizons
- OHLC-derived realized-state features and optional hybrid variance blending
- Regime-gated threshold routing

### Preset Configs

| Config | Description |
| ------ | ----------- |
| `spy_fixed.yaml` | SPY, fixed 5% threshold (recommended starting point) |
| `goog_fixed.yaml` | GOOGL, fixed 5% threshold |
| `tsla_fixed.yaml` | TSLA, fixed 5% threshold + jump-diffusion |

## Project Structure

```text
README.md              This file
METHODOLOGY.md         Full technical methodology with math
RESULTS.md             Validation results with pass/fail status
requirements.txt
data/                  Price data (CSV)
em_sde/                Core library
  data_layer.py          Data loading, caching, quality checks
  garch.py               Volatility estimation (GARCH/GJR)
  monte_carlo.py         Monte Carlo simulation (paths, jumps, fat tails)
  calibration.py         Probability calibration (multi-feature + histogram)
  backtest.py            Walk-forward engine (no lookahead)
  evaluation.py          Scoring (Brier, AUC, ECE, VaR, CRPS)
  model_selection.py     Cross-validation, promotion gates
  config.py              YAML config system
  output.py              Results and charts
  run.py                 CLI entry point
configs/               YAML configuration presets
scripts/               Operational runners
  run_bayesian_opt.py    Optuna hyperparameter optimization
  run_gate_recheck.py    5-fold CV gate validation
  run_overfit_check.py   Overfitting diagnostics
  run_full_institutional.py  Full validation battery
  run_stress_suite.py    Stress testing
  baselines.py           Formal baseline models for comparison
  run_ablation_study.py  Component ablation study
  run_temporal_holdout.py  Temporal hold-out evaluation
  run_paper_results.py   Paper table generation
  run_economic_significance.py  Economic impact analysis
  generate_paper_figures.py  Publication figures
paper/                 Academic paper
  main.tex               LaTeX source
  references.bib         BibTeX references
  reproduce.py           Full reproduction script
tests/                 296 unit tests
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Validation & Optimization

```bash
# Bayesian optimization (lean mode, thresholds locked by default)
python scripts/run_bayesian_opt.py spy --n-trials 12
python scripts/run_bayesian_opt.py spy --apply

# Optional: also tune the threshold panel
python scripts/run_bayesian_opt.py spy --n-trials 12 --tune-thresholds

# 5-fold CV gate validation
python -u scripts/run_gate_recheck.py spy

# Overfitting diagnostics
python scripts/run_overfit_check.py spy

# Full institutional battery
python -u scripts/run_full_institutional.py
```

## Academic Paper

A full academic paper is included in the `paper/` directory with LaTeX source, BibTeX references, and a reproduction script.

```bash
# Reproduce all paper tables and figures
python paper/reproduce.py --all

# Compile the paper
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Paper infrastructure in `scripts/`:
- `baselines.py` -- four formal baseline models
- `run_ablation_study.py` -- systematic component ablation
- `run_temporal_holdout.py` -- temporal hold-out evaluation
- `run_paper_results.py` -- LaTeX-ready tables with significance tests
- `run_economic_significance.py` -- economic impact analysis
- `generate_paper_figures.py` -- publication-quality figures

## Known Limitations

1. Many bundled CSVs are still close-only. OHLC-derived features and hybrid variance only activate when `open/high/low` are present in the dataset.
2. No transaction costs or market impact - this is a probability tool, not a trading system.
3. Calibration requires ~100 resolved outcomes to activate. Early predictions are uncalibrated.
4. Single-name implied vol is still only as good as the supplied free-data proxy. If no ticker-specific IV is available, the system falls back to historical-only behavior.
5. Overlapping prediction windows create correlated samples. Always check effective sample size (N_eff), not raw count.

## Disclaimer

Research and educational use only. Not investment advice.
