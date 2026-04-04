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
7. **Run tests before declaring anything done.** `python -m pytest tests/ -v` must pass (296+ tests).
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
| `data_layer.py` | Price loading (yfinance/CSV/synthetic), implied vol loading, earnings dates, quality checks, caching |
| `garch.py` | GARCH(1,1)/GJR, EWMA fallback, stationarity projection, HAR-RV (inactive) |
| `monte_carlo.py` | GBM/GARCH-in-sim, Merton jump-diffusion, Student-t fat tails, state-dependent jumps |
| `calibration.py` | Online Platt, multi-feature (6-8 features + L2), regime-conditional multi-feature, histogram post-cal |
| `backtest.py` | Walk-forward engine, resolution queue, regime-gated thresholds, implied vol blending |
| `evaluation.py` | Brier, BSS, AUC, ECE (adaptive bins), VaR, CVaR, CRPS, PIT, tail coverage, N_eff |
| `model_selection.py` | Expanding-window CV (5-fold), OOF row-level promotion gates (pooled + per-regime), density gates, benchmark reports, conditional gate reports |
| `output.py` | CSV/JSON results, reliability curves, charts |
| `run.py` | CLI entry point, single-run and --compare modes |

### Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `run_bayesian_opt.py` | Optuna TPE hyperparameter optimization. Lean mode (6 params) or full (14 params). `--apply` writes best to YAML. Studies are versioned by feature flags — changing config flags auto-starts a fresh study. |
| `run_gate_recheck.py` | Re-run promotion gates with OOF row-level evaluation |
| `run_overfit_check.py` | 5-metric overfitting diagnostics (gen gap, CV stability, threshold sensitivity, temporal stability, N_eff/N_params) |
| `run_full_institutional.py` | Full validation battery |
| `run_stress_suite.py` | Stress testing |
| `baselines.py` | Four formal baseline models (hist freq, GARCH-CDF, implied-vol BS, feature logistic) |
| `run_ablation_study.py` | Systematic 7-variant ablation study with significance testing |
| `run_temporal_holdout.py` | Temporal hold-out evaluation (train pre-2020, test post-2020) |
| `run_paper_results.py` | Generate LaTeX-ready tables with p-values and CIs |
| `run_economic_significance.py` | Risk-managed portfolio and selective hedging analysis |
| `generate_paper_figures.py` | Publication-quality reliability diagrams, heatmaps, charts |

### Config System (`configs/`)

- Preset configs: `spy_fixed.yaml`, `goog_fixed.yaml`, `tsla_fixed.yaml`
- Experimental/BO-tuned: `configs/exp_suite/exp_{ticker}_regime_gated.yaml`
- Real ticker data: SPY (from 1993), AAPL (from 2000), GOOGL (from 2004-08-19), AMZN (from 1997), NVDA (from 1999)
- Threshold modes: `fixed_pct` (recommended), `regime_gated`, `anchored_vol`
- Threshold locking: `lock_threshold_panel: true` (default) freezes thresholds in BO; `--tune-thresholds` opts back in

## Key Technical Concepts

**Promotion Gates (the quality bar):**
- Row-level OOF: pool all out-of-fold predictions across 5 CV folds
- Pooled ECE gate is primary; per-regime is diagnostic
- Must pass ALL: BSS >= 0.0, AUC >= 0.55, ECE <= 0.02
- Optional density gates: CRPS skill >= 0, PIT KS <= 0.12, tail coverage error <= 0.05
- Optional robustness gate: worst overfit status <= YELLOW
- Minimum guards: n >= 100, events >= 30, non-events >= 30
- Each gate row reports: N_eff, neff_ratio, neff_warning (GREEN/YELLOW/RED)
- ECE gates report confidence annotations: solid_pass, fragile_pass, solid_fail, fragile_fail (based on bootstrap CI)

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
| **FHS** (`fhs_enabled`) | Resampled GARCH standardized residuals have fewer tail events than parametric Student-t(5). Stationarity projection silently drops residuals, causing inconsistent innovation distribution across time. Regime-conditional t-df (which was BO-tuned) gets bypassed. Combined stack regressed SPY 3/3→2/3, GOOGL 2/3→1/3. |
| **GARCH Ensemble** (`garch_ensemble`) | Averaging sigma from GARCH(1,1)+GJR+EGARCH creates sigma/dynamics mismatch: averaged sigma initializes sim, but GJR dynamics mean-revert to GJR's unconditional level. Mismatch grows with horizon. Pooled EGARCH residuals contaminate FHS distribution. |

**What works (the proven stack):**
- **Volatility**: GARCH(1,1)/GJR with EWMA fallback + stationarity projection
- **Simulation**: GARCH-in-sim Monte Carlo with Student-t fat tails (regime-conditional df), optional implied vol blending
- **Calibration**: Multi-feature linear (6-8 features + L2) with per-horizon histogram post-calibration + interpolation. Earnings calendar (7th feature) for H≤5 on single stocks. Implied vol ratio (8th feature) when options data available.
- **Thresholds**: `fixed_pct` or `regime_gated` (routes to fixed_pct/anchored_vol by vol regime)

**Earnings Calendar** (`earnings_calendar: true`): Adds earnings proximity feature (0-1 scale) to multi-feature calibrator. **Only active for H≤5** — at longer horizons the signal becomes noise and adds a parameter without improving discrimination. Horizon-conditional gating is automatic in `backtest.py`. (Dubinsky et al., 2019; Savor & Wilson, 2016)

**Options-Implied Volatility** (`implied_vol_enabled: true`): Blends options-implied vol into MC sigma and adds implied_vol_ratio as a calibration feature. Accepts VIX data (SPY) or generic IV CSVs (single stocks). Horizon-matched: H=5→VIX9D, H=10→interpolated, H=20→VIX. Staleness guard skips data >5 business days old. Default blend weight 30%. Disabled by default — requires `implied_vol_csv_path`.

## Common Commands

```bash
# Run tests
python -m pytest tests/ -v

# Single backtest
python -m em_sde.run --config configs/spy_fixed.yaml --run-id my_run

# Compare configs with CV
python -m em_sde.run --compare configs/spy_fixed.yaml configs/goog_fixed.yaml --cv-folds 5

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

# --- Paper Infrastructure ---
# Ablation study
python scripts/run_ablation_study.py spy    # single ticker
python scripts/run_ablation_study.py all    # all tickers

# Temporal hold-out evaluation
python scripts/run_temporal_holdout.py spy --cutoff 2019-12-31

# Paper results tables (main + baselines with significance)
python scripts/run_paper_results.py all

# Economic significance analysis
python scripts/run_economic_significance.py all

# Publication figures
python scripts/generate_paper_figures.py

# Full paper reproduction (all of the above)
python paper/reproduce.py --all
```

## Current State (2026-04-03)

### Ticker Status (Legacy Point-Metric Gates)

- **SPY**: 6538 rows (2000-2025), **2/3 gates PASS** (H=5, H=10). Thresholds: H=5 3.72%, H=10 5.54%, H=20 7.05%. H=20 ECE=0.024 barely fails. BSS positive across all horizons.
- **GOOGL**: 5376 rows (2004-2025), **3/3 gates PASS**. Earnings + implied vol (VIX proxy) enabled. Thresholds: H=5 5.22%, H=10 9.24%, H=20 12.65%. Implied vol was the key unlock — H=5 ECE dropped 0.022→0.0076.
- **AMZN**: 7202 rows (1997-2025), **3/3 gates PASS**. Earnings + implied vol enabled. Thresholds: H=5 8.52%, H=10 15.99%, H=20 20.50%. Best calibration: H=10 ECE=0.0029. AUC >0.70 all horizons.
- **NVDA**: 6777 rows (1999-2025), **1/3 gates PASS** (H=5 only). Earnings + implied vol enabled. Thresholds: H=5 9.84%, H=10 18.04%, H=20 23.68%. H=10/H=20 BSS negative. Only 5 BO trials — needs more.

**Density governance (CRPS/PIT/tail) mostly failing.** Even best tickers fail CRPS skill at longer horizons. The MC engine produces calibrated binary probabilities but the full return distribution needs work.

### Recent Changes (2026-04-03)

**Academic paper infrastructure (Phase 1-3):**

1. **Baselines module** (`scripts/baselines.py`): Four formal baseline models — Historical Frequency (rolling event rate), GARCH-CDF (parametric, no MC), Implied-Vol Black-Scholes, Feature Logistic Regression. All run through the same evaluation pipeline.
2. **Ablation study** (`scripts/run_ablation_study.py`): Systematic 7-variant ablation (Base GBM → +GARCH-in-Sim → +Student-t → +MF Cal → +Histogram → +Implied Vol → Full). Paired bootstrap significance vs base variant.
3. **Temporal hold-out** (`scripts/run_temporal_holdout.py`): Train through 2019-12-31, test 2020-2025. Per-era breakdown (COVID, tightening, post-2023). Runs baselines on same split.
4. **Significance testing** (`scripts/run_paper_results.py`): Generates LaTeX-ready tables with paired bootstrap p-values, confidence intervals, gate pass/fail. Tables output to `outputs/paper/tables/`.
5. **Economic significance** (`scripts/run_economic_significance.py`): Risk-managed portfolio (reduce weight when p > threshold) and selective hedging analysis. Sharpe, max drawdown, CVaR comparisons.
6. **Publication figures** (`scripts/generate_paper_figures.py`): Multi-panel reliability diagrams, ablation heatmaps, baseline comparison charts, rolling ECE plots. Publication-quality PDF/PNG.
7. **Full LaTeX paper** (`paper/main.tex`): ~16-page paper with abstract, intro, related work (20 citations), methodology, experimental setup, results, discussion, conclusion. BibTeX references in `paper/references.bib`.
8. **Reproducibility package** (`paper/reproduce.py`): Single script to regenerate all paper tables and figures. `--all` flag or individual `--main-results`, `--ablation`, `--holdout`, `--economic`, `--figures`.
9. **296 tests still passing.**

### Previous Changes (2026-03-10)

**Anti-overfitting methodology improvements:**

1. **Gen gap sign fix** (`run_overfit_check.py`): `abs(gap_ratio)` → `max(gap_ratio, 0.0)`. Negative gaps (model improving on full data) no longer falsely flagged RED.
2. **Temporal stability direction fix** (`run_overfit_check.py`): Only flags when late folds are worse; improvement (late < early) maps to GREEN.
3. **N_eff columns in gate reports** (`model_selection.py`): Every gate row now reports `n_eff`, `neff_ratio`, `neff_warning` (GREEN/YELLOW/RED).
4. **ECE confidence annotations** (`model_selection.py`): Bootstrap CI determines `solid_pass`, `fragile_pass`, `solid_fail`, `fragile_fail`.
5. **N_eff soft penalty in BO** (`run_bayesian_opt.py`): When min N_eff ratio < 100x, adds `0.01 * shortfall²` to objective — steers BO toward statistically robust regions.
6. **Holdout min_events raised** (`run_bayesian_opt.py`): `min_events=2` → `min_events=10`, `min_nonevents=2` → `min_nonevents=10`.

**Density governance stack (via Codex):**

1. **Density gates**: CRPS skill, PIT KS statistic, tail coverage error — optional stricter gates beyond point metrics.
2. **Benchmark reports**: Bootstrap p-value testing model vs climatology.
3. **Conditional gate reports**: Per-era (pre-covid, covid-2020, tightening, post-2023) and per-event-state breakdowns.
4. **Threshold locking**: `lock_threshold_panel: true` by default; `--tune-thresholds` to opt in.
5. **Offline pooled calibration**: Research path — batch calibrator on train rows, apply to held-out fold.
6. **Overfit report integration**: Overfit diagnostic status can block promotion via `require_overfit` flag.
7. **Scheduled jump variance**: Earnings-driven jump variance injection (research, not yet proven).
8. **296 tests passing** (was 241).

### Previous Changes (2026-03-09)

1. AMZN 3/3 gates PASS. NVDA 1/3 gates PASS (H=5).
2. New tickers: AMZN (7202 rows, 1997-2025) and NVDA (6777 rows, 1999-2025).

### Previous Changes (2026-03-08)

1. GOOGL 3/3 gates PASS with implied vol enabled.
2. VIX data: `data/vix_history.csv` (5421 rows, 2004-2025).

### Next Steps

1. **Run paper reproduction** — `python paper/reproduce.py --all` to generate all tables/figures with real numbers
2. **SPY + implied vol** — VIX data exists, H=20 ECE=0.024 may drop under 0.02
3. **NVDA: Run more BO trials** (7+ more) to find lower thresholds → positive BSS
4. **Update paper tables** with actual reproduction numbers (current tables have representative values from RESULTS.md)
5. **Submit to IJF or Quantitative Finance** — best venue fit for calibration + walk-forward methodology

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
