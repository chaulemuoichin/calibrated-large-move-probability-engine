# CLAUDE.md

## Project Goal

Institutional-grade calibrated large-move probability engine: estimate the probability of two-sided price moves over 1–4 week horizons, self-correct as outcomes arrive, pass rigorous validation gates. **Output must be trustworthy enough to inform real capital allocation.**

## Non-Negotiable Rules

1. **No lookahead.** Walk-forward only. Predictions queue at `t`, resolve at `t+H`. Never read future data.
2. **Promotion gates are mandatory.** Every config must pass pooled OOF row-level: `BSS ≥ 0`, `AUC ≥ 0.55`, `ECE ≤ 0.02`. Do not relax.
3. **No leakage.** Train/test splits strictly chronological. Expanding-window CV only.
4. **Check N_eff, not N.** Overlapping predictions are correlated. Residual-based `effective_sample_size(y, H, p_cal)` is the statistical-power anchor.
5. **Backward compatibility behind flags.** New behavior ships disabled by default behind a YAML flag.
6. **Tests must pass before "done".** `python -m pytest tests/` (356+ tests).
7. **Adaptive event-rate guard in BO.** `min_rate = max((100 × n_params) / (2 × n_oof), 3%)`. Enforces N_eff/N_params ≥ 100× (GREEN). Per Vittinghoff & McCulloch (2007): ≥20 events per parameter.
8. **Use all available data.** Configs use the earliest available data for the ticker. No arbitrary truncation.
9. **Update docs in the same PR.** This file (current state + rules), METHODOLOGY.md (technical), RESULTS.md (numbers), README.md (user-facing).

## Architecture

```
Prices -> GARCH/GJR Vol -> Monte Carlo -> p_raw -> 3-layer Calibration -> p_cal
                                                          ^
                                        realized outcomes-+   (walk-forward only)
```

### Core modules (`em_sde/`)

| Module | Purpose |
|--------|---------|
| `config.py` | YAML loading + dataclass validation |
| `data_layer.py` | Prices (yfinance/CSV/synthetic), implied vol, earnings, quality checks, cache |
| `garch.py` | GARCH(1,1)/GJR, EWMA fallback, stationarity projection, state export |
| `monte_carlo.py` | GBM + GARCH-in-sim, Merton jumps, Student-t tails, regime-conditional df |
| `calibration.py` | Online Platt → multi-feature logistic (6–8 features, L2) → histogram/PAV post-cal |
| `backtest.py` | Walk-forward loop, resolution queue, threshold routing, checkpoint save |
| `evaluation.py` | Brier, BSS, AUC, ECE (adaptive), CRPS, PIT, N_eff (residual ACF), BCa/block bootstrap, FDR |
| `model_selection.py` | Expanding-window CV, OOF row-level promotion gates, pooled + per-regime, fold callback |
| `predict.py` | Live prediction engine (loads checkpoint) |
| `resolve.py` | Async label resolution, ticker-filtered |
| `ledger.py` | Append-only JSONL forecast/resolution ledger |
| `live_metrics.py` | Cumulative OOS metrics from resolved forecasts |
| `run.py` | CLI: single-run, `--compare`, `--predict-now`, `--save-state` |

### Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `run_bayesian_opt.py` | **Constrained** Optuna TPE (constraints on ECE/AUC/BSS) + MedianPruner + fold-0 fast-fail. Lean mode (3–6 params) or `--full` (7+). Studies versioned by flags. |
| `run_gate_recheck.py` | Re-run pooled OOF promotion gates |
| `run_overfit_check.py` | 5-metric diagnostics (gen gap, CV stability, threshold sensitivity, temporal stability, N_eff/N_params) |
| `run_full_institutional.py` | Full validation battery |
| `baselines.py` | 7 baselines: hist freq, GARCH-CDF, IV Black-Scholes, feature logistic, gradient boosting, VIX rule, straddle |
| `run_ablation_study.py` | 7-variant ablation with FDR-adjusted paired block bootstrap |
| `run_temporal_holdout.py` | Online-adaptive hold-out (pre/post 2020) |
| `run_paper_results.py` | LaTeX tables with CIs + FDR-adjusted p-values |
| `run_economic_significance.py` | Risk-managed portfolio + selective hedging + cost sensitivity |
| `generate_paper_figures.py` | Reliability diagrams, ablation heatmaps |
| `daily_predict.py` | Daily scheduled predictions (cron/Task Scheduler) |
| `publish_live_forecasts.py` / `resolve_live_forecasts.py` / `update_live_verification.py` | Live verification ledger |
| `build_live_verification_site.py` | Static HTML dashboard |
| `anchor_ledger.py` | Tamper-evident SHA-256 anchoring via git tags |

### Configs (`configs/`)

- Presets: `spy_fixed.yaml`, `goog_fixed.yaml`, `tsla_fixed.yaml`; BO-tuned under `configs/exp_suite/`.
- Tickers: SPY (1993+), AAPL (2000+), GOOGL (2004-08-19+), AMZN (1997+), NVDA (1999+).
- Threshold modes: `fixed_pct` (default), `regime_gated`, `anchored_vol`.
- Threshold locking: `lock_threshold_panel: true` freezes thresholds in BO; `--tune-thresholds` opts in.

## Key concepts

**Promotion gates.** Pool OOF predictions row-level across 5 CV folds. Pooled ECE is the primary gate; per-regime rows are diagnostic. Minimum guards per regime: n ≥ 100, events ≥ 30, non-events ≥ 30. Each row reports `n_eff`, `neff_ratio`, `neff_warning` (GREEN >100×, YELLOW 50–100×, RED <50×). ECE rows also carry `solid_pass` / `fragile_pass` / `solid_fail` / `fragile_fail` based on bootstrap CI.

**Constrained BO.** Objective = mean OOF ECE. Constraints `[ECE-0.02, 0.55-AUC, -BSS]` fed to Optuna's TPE `constraints_func`. Feasibility dominates objective. `MedianPruner(n_startup_trials=5, n_warmup_steps=2)` + per-fold callback + fold-0 fast-fail (ECE > 0.10) typically prune 30–50% of trials; 2–4× wall-clock speedup. Study key hashes flags including `bo_formulation=constrained_v1`.

**Lean BO mode (default, anti-overfitting).** Tunes 3 params (`multi_feature_lr`, `multi_feature_l2`, `garch_target_persistence`) with thresholds frozen. `--tune-thresholds` adds `thr_{5,10,20}`. `--full` adds HAR-RV + regime t_df.

**Data-adaptive thresholds.** BO bounds derived from realized return percentiles: `[P80 × 0.9, P90]`, targeting 10–24% event rate.

## Removed features (do NOT re-implement)

| Feature | Why removed |
|---------|-------------|
| **HMM regime detection** | Unstable fit on <2k obs; no measurable gain over HAR-RV / rolling vol percentile |
| **Neural / MLP calibrator** | Multi-feature linear + L2 is simpler, more stable, passes gates. No proven benefit. |
| **RegimeCalibrator** (vol-bin post-cal) | Superseded by `RegimeMultiFeatureCalibrator` (regime-gating integrated) |
| **IsotonicCalibrator** | Superseded by histogram binning + Bayesian shrinkage + PAV monotonicity |
| **`vol_scaled` threshold mode** | Self-referential; AUC ≈ 0.50 (no discrimination) |
| **FHS** (`fhs_enabled`) | Resampled residuals have fewer tails than Student-t(5); stationarity projection drops residuals; bypasses regime t_df. Regressed SPY 3/3→2/3, GOOGL 2/3→1/3. |
| **GARCH ensemble** | Averaged sigma ≠ simulation dynamics (GJR mean-reverts to GJR's unconditional). Horizon-growing mismatch. Pooled EGARCH residuals contaminate FHS. |

**What works (the proven stack):**

- **Vol**: GARCH(1,1)/GJR + EWMA fallback + stationarity projection + horizon-average term structure.
- **Sim**: GARCH-in-sim MC with regime-conditional Student-t + Merton jumps + optional IV blend.
- **Calibration**: Online Platt → multi-feature logistic (logit(p_raw), σ_d, Δσ_20, σ_r/σ_d, vol-of-vol, earnings?, IV ratio?) → histogram/PAV post-cal. Regime-conditional variant available.
- **Thresholds**: `fixed_pct` (recommended) or `regime_gated`.

**Earnings calendar** (`earnings_calendar: true`): proximity feature (0–1) in multi-feature calibrator. **Active only for H ≤ 5** — at longer H the signal becomes noise. Horizon gating automatic in `backtest.py`. (Dubinsky et al., 2019; Savor & Wilson, 2016)

**Options-implied vol** (`implied_vol_enabled: true`): blends into MC sigma; adds `implied_vol_ratio` as a calibration feature. Horizon-matched (VIX9D → interpolated → VIX). Staleness guard: >5 business days old → skip. Default blend weight 0.3. Requires `implied_vol_csv_path`.

## Common commands

```bash
# Tests
python -m pytest tests/ -v

# Single backtest / comparison
python -m em_sde.run --config configs/spy_fixed.yaml --run-id my_run
python -m em_sde.run --compare configs/spy_fixed.yaml configs/goog_fixed.yaml --cv-folds 5

# BO → apply → gate recheck → overfit check (the standard workflow)
python scripts/run_bayesian_opt.py <ticker> --n-trials 12
python scripts/run_bayesian_opt.py <ticker> --apply
python -u scripts/run_gate_recheck.py <ticker>
python scripts/run_overfit_check.py <ticker>

# Full paper reproduction (~2–4 h)
python paper/reproduce.py --all

# Live prediction
python -m em_sde.run --predict-now --config configs/spy_fixed.yaml --state-dir outputs/state/spy
python scripts/daily_predict.py        # scheduled daily

# Live verification ledger
python scripts/update_live_verification.py --publish   # publish + resolve + rebuild
python scripts/anchor_ledger.py                        # tamper-evident anchor
```

## Current State

**Primary scope: H=5 and H=10 (1–2 weeks). H=20 exploratory.**

| Ticker | Rows (range) | Primary (H=5, 10) | Exploratory (H=20) | Notes |
|--------|--------------|-------------------|---------------------|-------|
| SPY    | 6,538 (2000–2025) | **2/2 PASS** | ECE 0.024 barely fails | Strong BSS/AUC all H |
| GOOGL  | 5,376 (2004–2025) | **2/2 PASS** | PASS | IV blend was the H=5 unlock |
| AMZN   | 7,202 (1997–2025) | **2/2 PASS** | PASS | Best calibration (H=10 ECE=0.0029) |
| NVDA   | 6,777 (1999–2025) | 1/2 (H=5 only)   | FAIL | Needs more BO trials |

**Totals: 7/8 primary-horizon tests PASS.** Full numbers in [RESULTS.md](RESULTS.md).

**Next steps:**

1. SPY + implied vol at H=20 (VIX data exists; ECE=0.024 may drop under 0.02).
2. NVDA: run 7+ more BO trials to find lower thresholds → positive BSS.
3. Submit to IJF or Quantitative Finance.

## Recent changes

Most recent only. Full history via `git log`.

**2026-04-19 — Workspace + constrained BO + paper polish**
- Purged `test_runtime_artifacts/` (~200 MB).
- New onboarding: `QUICKSTART.md` + trimmed `CONTRIBUTING.md`.
- **Constrained BO** with Optuna `constraints_func` + `MedianPruner` + fold-0 fast-fail (2–4× speedup). `expanding_window_cv` gained `fold_callback`. Study key: `bo_formulation=constrained_v1`.
- Paper: 819 → 626 lines. New §3.6 constrained BO. Ancillary diagnostics moved to Appendix A. GitHub URL hyperlinked.
- Paper §2 + METHODOLOGY.md §13: new **comparison table** vs Black-Scholes / GARCH-CDF / HMM / feature-ML / VaR-CVaR.
- METHODOLOGY.md: 1084 → 287 lines. Disabled-feature detail moved here (this file).
- Tests: **356 passed** (was 345; 11 new for fold-callback + constrained BO).

## Python environment

- Interpreter: `.venv/Scripts/python.exe` (Windows — **not** `python`/`python3`).
- All commands: `.venv/Scripts/python.exe -m pytest tests/ -v`, etc.

## Quality standards

- Probability output validated against promotion gates before any "working" claim.
- GREEN on N_eff/N_params before trusting BO results.
- Generalization gap < 25% between BO-train ECE and full-data ECE.
- Cross-fold CV CoV < 0.50.
- Report honestly, including FAILs.

## What "institutional grade" means

1. **Calibration.** When the model says 15%, it happens ~15% of the time (ECE < 0.02).
2. **Discrimination.** Events rank above non-events (AUC > 0.55).
3. **Skill.** Beats climatology (BSS > 0).
4. **Robustness.** Holds across time periods under expanding-window CV.
5. **Honest uncertainty.** N_eff + overfitting diagnostics reported, not hidden.
6. **Reproducibility.** Seed-controlled MC, deterministic CV, versioned configs.
