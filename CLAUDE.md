# CLAUDE.md

## Project Goal

Build an **institutional-grade calibrated large-move probability engine** that can be used in real-life investment decisions. The system estimates the probability of large two-sided price moves over 1-4 week horizons, self-corrects as outcomes arrive, and passes rigorous statistical validation gates.

**The output must be trustworthy enough to inform real capital allocation decisions.**

## Non-Negotiable Rules

1. **No lookahead bias.** Walk-forward only. Predictions are queued and resolved only after H days pass. Never use future data for any computation.
2. **ECE gate stays at 0.02.** This is the hard promotion threshold. Do not relax it.
3. **Promotion gates are mandatory.** Every config must pass: BSS >= 0.0, AUC >= 0.55, ECE <= 0.02 (pooled OOF row-level).
4. **No leakage.** Train/test splits are strictly chronological. Expanding-window CV only.
5. **Always check N_eff, not raw N.** Overlapping predictions create correlated samples. Effective sample size governs all statistical conclusions.
6. **Backward compatibility behind flags.** New features use flags (e.g., `lean` mode in BO). Don't break existing configs or tests.
7. **Run tests before declaring anything done.** `python -m pytest tests/ -v` must pass (191+ tests).
8. **Adaptive event-rate guard in BO.** `min_rate = max((100 × n_params) / (2 × n_oof), 3%)`. Adapts to dataset size — larger datasets get a looser guard, smaller ones stricter. Ensures N_eff/N_params >= 100x (GREEN). Floor at 3% absolute minimum. Per Vittinghoff & McCulloch (2007): 20 events per parameter minimum.
9. **Use all available data from IPO/inception.** All configs must use the earliest available data for the ticker. Never truncate history arbitrarily.
10. **Always update docs after any change.** When making code changes, results, or architectural decisions, update ALL relevant docs: `CLAUDE.md` (current state, rules, context for next session), `METHODOLOGY.md` (technical details), `RESULTS.md` (honest numbers), `README.md` (if user-facing). Never leave docs stale — the next Claude session depends on accurate CLAUDE.md.

## Architecture & Pipeline

```
Raw Prices -> GARCH/GJR Vol -> Monte Carlo Sim -> p_raw -> Calibration -> p_cal
```

### Core Modules (`em_sde/`)

| Module | Purpose |
|--------|---------|
| `config.py` | YAML config loading, dataclass validation |
| `data_layer.py` | Price loading (yfinance/CSV/synthetic), quality checks, caching |
| `garch.py` | GARCH(1,1)/GJR, EWMA fallback, stationarity projection, HAR-RV (inactive) |
| `monte_carlo.py` | GBM/GARCH-in-sim, Merton jump-diffusion, Student-t fat tails, state-dependent jumps |
| `calibration.py` | Online Platt, multi-feature (6 features + L2), regime-conditional multi-feature, histogram post-cal |
| `backtest.py` | Walk-forward engine, resolution queue, regime-gated thresholds |
| `evaluation.py` | Brier, BSS, AUC, ECE (adaptive bins), VaR, CVaR, CRPS, N_eff |
| `model_selection.py` | Expanding-window CV (5-fold), OOF row-level promotion gates (pooled + per-regime) |
| `output.py` | CSV/JSON results, reliability curves, charts |
| `run.py` | CLI entry point, single-run and --compare modes |

### Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `run_bayesian_opt.py` | Optuna TPE hyperparameter optimization. Lean mode (6 params) or full (14 params). `--apply` writes best to YAML. |
| `run_gate_recheck.py` | Re-run promotion gates with OOF row-level evaluation |
| `run_overfit_check.py` | 5-metric overfitting diagnostics (gen gap, CV stability, threshold sensitivity, temporal stability, N_eff/N_params) |
| `run_full_institutional.py` | Full validation battery |
| `run_stress_suite.py` | Stress testing |

### Config System (`configs/`)

- Preset configs: `spy_fixed.yaml`, `goog_fixed.yaml`, `tsla_fixed.yaml`
- Experimental/BO-tuned: `configs/exp_suite/exp_{ticker}_regime_gated.yaml`
- Real ticker data: SPY (from 1993), AAPL (from 2000), GOOGL (from 2004-08-19)
- Threshold modes: `fixed_pct` (recommended), `regime_gated`, `anchored_vol`

## Key Technical Concepts

**Promotion Gates (the quality bar):**
- Row-level OOF: pool all out-of-fold predictions across 5 CV folds
- Pooled ECE gate is primary; per-regime is diagnostic
- Must pass ALL: BSS >= 0.0, AUC >= 0.55, ECE <= 0.02
- Minimum guards: n >= 100, events >= 30, non-events >= 30

**N_eff / N_params Ratio (overfitting constraint):**
- N_eff = min(events, non-events) * 2 (corrected for overlap)
- N_params = number of BO-tuned hyperparameters
- GREEN: ratio > 100x. YELLOW: 50-100x. RED: < 50x.
- This is the binding constraint. More data or fewer params to improve it.

**Lean BO Mode (anti-overfitting):**
- Only tunes 6 params: thr_5, thr_10, thr_20, garch_persistence, mf_lr, mf_l2
- Fixes at defaults: t_df_low=10.0, t_df_mid=5.0, t_df_high=4.0, mf_min_updates=63, har_rv=false
- Use `--full` flag to access the original 14-param search

**Data-Adaptive Thresholds:**
- `_compute_threshold_ranges()` derives BO search bounds from realized return percentiles
- Lower bound: P80 * 0.9 (targets ~20% event rate)
- Upper bound: P90 (targets ~10% event rate, no margin)
- Ensures all thresholds in search range produce 10-24% event rates

## Removed Features (do NOT re-implement)

These were tried, tested, and proven to not work for our use case. Do not add them back.

| Feature | Why Removed |
|---------|-------------|
| **HMM regime detection** (`hmm_regime`) | Added complexity without measurable gain. HAR-RV (if enabled) handles vol regimes better. HMM fitting was unstable with limited data. |
| **NeuralCalibrator / RegimeNeuralCalibrator** | MLP calibrator (8 hidden units) was never used in production. Multi-feature linear + L2 is simpler, more stable, and passes gates. Neural adds parameters without proven benefit. |
| **RegimeCalibrator** (`regime_conditional`) | Vol-percentile-binned calibration was superseded by `RegimeMultiFeatureCalibrator` which integrates regime-gating into multi-feature directly. |
| **IsotonicCalibrator** | Isotonic regression post-cal was superseded by histogram binning with Bayesian shrinkage + PAV monotonicity, which is more stable for small-N bins. |
| **`vol_scaled` threshold mode** | Self-referential: threshold = `k × σ × √H` moves with current vol, creating circular predictions. Produces AUC ≈ 0.50 (no discrimination). Use `fixed_pct` instead. |

**What works (the proven stack):**
- **Volatility**: GARCH(1,1)/GJR with EWMA fallback + stationarity projection
- **Simulation**: GARCH-in-sim Monte Carlo with Student-t fat tails
- **Calibration**: Multi-feature linear (6 features + L2) with histogram post-calibration
- **Thresholds**: `fixed_pct` or `regime_gated` (routes to fixed_pct/anchored_vol by vol regime)

## Common Commands

```bash
# Run tests
python -m pytest tests/ -v

# Single backtest
python -m em_sde.run --config configs/spy_fixed.yaml --run-id my_run

# Compare configs with CV
python -m em_sde.run --compare configs/spy_fixed.yaml configs/spy.yaml --cv-folds 5

# Bayesian optimization (lean mode, default)
python scripts/run_bayesian_opt.py spy --n-trials 12
python scripts/run_bayesian_opt.py spy --apply

# Gate recheck after BO
python -u scripts/run_gate_recheck.py spy

# Overfitting diagnostics
python scripts/run_overfit_check.py spy
python scripts/run_overfit_check.py all

# Full institutional validation
python -u scripts/run_full_institutional.py
```

## Current State (2026-03-01)

### Ticker Status
- **SPY**: 6538 rows (2000-2025), **3/3 gates PASS**. Working. Thresholds: H=5 5.3%, H=10 3.3%, H=20 5.0%.
- **AAPL**: 6538 rows (2000-2025), **1/3 gates PASS** (H=5 only). BO with adaptive guard found thr_5=6.2% (PASS, BSS=+0.004, AUC=0.637, ECE=0.010), thr_10=11.2% (FAIL, BSS=-0.043), thr_20=14.7% (FAIL, BSS=-0.010). H=10/H=20 event rates too low (2.4%, 4.0%). Overfitting: 4 RED, 6 YELLOW, 5 GREEN.
- **GOOGL**: 5376 rows (2004-2025), **2/3 gates PASS** (H=10, H=20). BO with adaptive guard found thr_5=5.2% (FAIL, ECE=0.027), thr_10=9.2% (PASS, BSS=+0.017, AUC=0.661, ECE=0.020), thr_20=12.7% (PASS, BSS=+0.014, AUC=0.642, ECE=0.012). BSS positive everywhere — model has real skill. H=5 ECE borderline (0.027 vs 0.02 gate). Overfitting: 3 GREEN, 10 YELLOW, 2 RED.

### Recent Changes (2026-03-01)
1. **Adaptive event-rate guard**: `min_rate = max((100 × n_params) / (2 × n_oof), 3%)`. Adapts to dataset size.
2. **Threshold range tightened**: Upper bound = P90 (no margin), lower bound = P80 * 0.9.
3. **Per-regime gate minimums raised**: min_samples=100, min_events=30, min_nonevents=30.
4. **All yfinance configs**: Now use data from IPO/inception.
5. **Dead code cleanup**: Removed HMM, NeuralCalibrator, RegimeCalibrator, IsotonicCalibrator, vol_scaled. ~820 LOC removed. Tests: 227 → 191 (36 tests for removed code).

### Next Steps
1. AAPL: Run more BO trials (4-8) for H=10/H=20, or investigate model improvements
2. GOOGL: Gate recheck in progress; likely needs more BO trials
3. Consider: Whether single-stock BSS failure at longer horizons is fundamental

### Standard Workflow
```bash
# 1. Bayesian optimization
python scripts/run_bayesian_opt.py <ticker> --n-trials N
# 2. Apply best params to config
python scripts/run_bayesian_opt.py <ticker> --apply
# 3. Gate recheck (5-fold CV, ~40 min)
python -u scripts/run_gate_recheck.py <ticker>
# 4. Overfitting diagnostics (~1 min)
python scripts/run_overfit_check.py <ticker>
# 5. Update RESULTS.md and CLAUDE.md with results
```

### Python Environment
- Python is at `.venv/Scripts/python.exe` (NOT `python` or `python3` on this Windows machine)
- All commands: `.venv/Scripts/python.exe -m pytest tests/ -v`, etc.

## Quality Standards

- Every probability output must be validated against promotion gates before any claim of "working"
- Overfitting diagnostics must show GREEN on N_eff/N_params before trusting BO results
- Generalization gap < 25% between BO train ECE and full-data ECE
- Cross-fold CV stability (coefficient of variation) < 0.50
- Always report results honestly, including FAILs. Never hide bad numbers.

## What "Institutional Grade" Means Here

1. **Calibration**: When the model says 15%, it happens ~15% of the time (ECE < 0.02)
2. **Discrimination**: The model separates events from non-events better than chance (AUC > 0.55)
3. **Skill**: Better than naive climatological base rate (BSS > 0)
4. **Robustness**: Results hold across time periods (expanding-window CV, not just in-sample)
5. **Honest uncertainty**: N_eff and overfitting diagnostics are tracked and reported
6. **Reproducibility**: Seed-controlled MC, deterministic CV splits, versioned configs
