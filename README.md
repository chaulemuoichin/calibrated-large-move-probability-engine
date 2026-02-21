# Calibrated Large-Move Probability Engine

A system that estimates the probability of large price moves over the next 1-4 weeks, and keeps correcting itself as real outcomes arrive.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-147%20passing-brightgreen)

## Why This Exists

You are an investor. You think a stock is undervalued. You want to buy.
The painful question is simple: **is today already a good entry?**

```text
if price goes up after you buy
-> good for you

if price goes down hard right after you buy
-> you doubt yourself
  -> bad exit & realized loss
    -> rushed next decision
      -> another loss
        -> either posting "market is rigged" on Reddit or repeating the loop
```

This project breaks that loop. Instead of guessing, it gives you a number:

> "There is a 12% chance the price moves more than 5% in the next two weeks."

If that number is trustworthy (and the system works hard to make it trustworthy), you can make a calmer decision — buy now, wait, or scale in gradually.

## How It Works (Plain English)

1. **Read the recent past.** The system looks at recent price behavior to understand how calm or jumpy the market has been.

2. **Imagine many possible futures.** Instead of making one guess about where the price will go, it generates thousands of realistic future price paths based on current conditions.

3. **Count the bad outcomes.** Out of all those imagined futures, how many had a price move large enough to matter? That fraction is the raw probability.

4. **Learn from past mistakes.** As real outcomes arrive (did the big move actually happen?), the system checks whether its earlier estimates were too high or too low, and adjusts.

5. **Output a calibrated probability.** The final number accounts for both current market conditions and the system's track record of accuracy.

The result is a clear, actionable probability you can use to inform your timing decisions.

## Quick Start

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# Run a backtest
python -m em_sde.run --config configs/spy_fixed.yaml --run-id my_first_run

# Compare two model configs
python -m em_sde.run --compare configs/spy_fixed.yaml configs/spy.yaml --cv-folds 5
```

Output goes to `outputs/<run_id>/` and includes a results CSV, summary JSON, and charts.

## What You Get

Each run produces:

| Output | What it is |
| ------ | ---------- |
| `results.csv` | Daily predictions with raw and calibrated probabilities, realized labels, volatility |
| `summary.json` | All evaluation metrics (accuracy, discrimination, calibration quality) |
| `reliability.csv` | Calibration curve data for plotting |
| `charts/` | Probability time series, reliability diagram, volatility regime, rolling accuracy |

### Key Metrics

| Metric | What it tells you | Good value |
| ------ | ----------------- | ---------- |
| **Brier Score** | Overall accuracy of probabilities | Lower is better |
| **Brier Skill Score (BSS)** | How much better than always guessing the average | > 0 means useful |
| **AUC** | Can the model tell events from non-events? | > 0.5 (1.0 = perfect) |
| **ECE** | Are the probabilities well-calibrated across the board? | < 0.02 |

## Configuration

All settings live in YAML files under `configs/`. Key choices:

**What counts as a "large move"?**

- `fixed_pct`: a fixed return threshold (e.g., 5%) — recommended
- `regime_gated`: automatically switches strategy based on current volatility

**How to model volatility?**

- `garch` or `gjr` (GJR captures the fact that drops amplify future volatility more than rallies)
- GARCH-in-simulation for path-level volatility dynamics

**Optional features:**

- Jump-diffusion for crash-prone stocks
- Multi-feature calibration for richer probability correction
- Fat-tailed innovations (Student-t)

### Preset Configs

| Config | Description |
| ------ | ----------- |
| `spy_fixed.yaml` | SPY with fixed 5% threshold (recommended starting point) |
| `goog_fixed.yaml` | GOOG with fixed 5% threshold |
| `tsla_fixed.yaml` | TSLA with fixed 5% threshold + jumps (volatile stock) |
| `spy.yaml` | SPY legacy vol-scaled baseline (for comparison) |

## For Collaborators

**Want to understand the full methodology?** Read [METHODOLOGY.md](METHODOLOGY.md). It explains every step from raw data to final probability, with the math, the assumptions, and the places where things can go wrong. It is specifically written so you can spot issues without reading the source code.

**Want to run tests?**

```bash
python -m pytest tests/ -v
```

147 tests covering the full pipeline: simulation math, calibration logic, no-lookahead guarantees, evaluation metrics, and more.

**Project layout:**

```text
em_sde/              Core library
  data_layer.py        Data loading, caching, quality checks
  garch.py             Volatility estimation (GARCH/GJR)
  monte_carlo.py       Price simulation (MC paths)
  calibration.py       Probability calibration (online learning)
  backtest.py          Walk-forward engine (no lookahead)
  evaluation.py        Scoring metrics (Brier, AUC, ECE, ...)
  model_selection.py   Cross-validation, model comparison
  config.py            YAML config system
  output.py            Results output and charts
  run.py               CLI entry point
configs/             YAML configuration presets
tests/               147 unit tests
METHODOLOGY.md       Full technical methodology (start here for review)
```

## Known Limitations

1. Uses close-to-close returns only (no intraday data).
2. No transaction costs or market impact modeling — this is a probability tool, not a trading simulator.
3. Calibration needs ~100 resolved outcomes to activate. Early predictions are uncalibrated.
4. Overlapping prediction windows create correlated samples. Always check effective sample size (`N_eff`), not raw count.

## Disclaimer

Research and educational use only. Not investment advice.
