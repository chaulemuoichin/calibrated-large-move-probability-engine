# Quickstart

Get a calibrated probability for SPY in 5 minutes.

## 1. Install

```bash
git clone <repo-url> && cd calibrated-large-move-probability-engine
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Python 3.10+ required. All paths in this doc assume the project root as working directory.

## 2. Run a backtest

```bash
python -m em_sde.run --config configs/spy_fixed.yaml --run-id hello_spy
```

Output lands in [outputs/hello_spy/](outputs/hello_spy/): a results CSV, summary JSON, reliability chart, and a state checkpoint.

Expected runtime: ~3-5 minutes on a modern laptop.

## 3. Generate a live prediction

Once you have a checkpoint from step 2:

```bash
python -m em_sde.run --predict-now --config configs/spy_fixed.yaml --state-dir outputs/hello_spy/state
```

You will see something like:

```
SPY  H=5   p_cal = 0.127   (base rate 0.089)
SPY  H=10  p_cal = 0.164   (base rate 0.118)
```

Interpretation: *"The model estimates a 12.7% probability that SPY moves more than the configured threshold over the next 5 trading days."*

## 4. Validate the numbers

```bash
python scripts/run_gate_recheck.py spy        # 5-fold CV gate check (~20-40 min)
python scripts/run_overfit_check.py spy       # overfitting diagnostics (~1 min)
```

A config is considered **trustworthy** only when it passes all three gates:

| Gate | Meaning                                   | Target        |
|------|-------------------------------------------|---------------|
| BSS  | Skill vs. climatological base rate        | > 0           |
| AUC  | Ranks event days above non-event days     | >= 0.55       |
| ECE  | Predicted probability matches reality     | <= 0.02       |

## 5. Tune (optional)

```bash
python scripts/run_bayesian_opt.py spy --n-trials 15       # find better hyperparams
python scripts/run_bayesian_opt.py spy --apply             # write into the YAML
python scripts/run_gate_recheck.py spy                     # confirm gates still pass
```

BO uses **constrained Bayesian optimization**: infeasible trials (ECE > 0.02, BSS < 0, AUC < 0.55) are pruned early, cutting typical BO runtime by 2-4x. See [METHODOLOGY.md](METHODOLOGY.md#bayesian-optimization) for details.

## Where things live

| Path                        | What                                         |
|-----------------------------|----------------------------------------------|
| [em_sde/](em_sde/)          | Core library (GARCH, MC, calibration, CV)    |
| [scripts/](scripts/)        | CLI runners (BO, gate checks, paper, live)   |
| [configs/](configs/)        | YAML configs per ticker                      |
| [data/](data/)              | Price CSVs and VIX data                      |
| [tests/](tests/)            | 345+ unit tests (`pytest tests/`)            |
| [paper/](paper/)            | LaTeX source + reproduction script           |
| [outputs/](outputs/)        | Run artifacts (gitignored)                   |

## Next steps

- New to the project? Read [README.md](README.md) and skim [METHODOLOGY.md](METHODOLOGY.md).
- Want to contribute? See [CONTRIBUTING.md](CONTRIBUTING.md).
- Want to verify the claims independently? See the live verification dashboard setup in README.md.
- Curious about a design decision? Check [CLAUDE.md](CLAUDE.md) for the current state and rationale.

## Troubleshooting

**`ImportError: No module named 'arch'`** - run `pip install -r requirements.txt` inside the activated venv.

**`FileNotFoundError: data/spy_daily.csv`** - run from project root, not from `scripts/` or `em_sde/`.

**Gates fail after BO** - expected for volatile tickers (NVDA). BO maximizes train-CV ECE; holdout may degrade. Check `scripts/run_overfit_check.py <ticker>` for diagnostics.

**Windows: `python` not found** - use `.venv\Scripts\python.exe` explicitly.
