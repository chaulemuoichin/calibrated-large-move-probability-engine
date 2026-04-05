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
7. **Run tests before declaring anything done.** `python -m pytest tests/ -v` must pass (312+ tests).
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
| `predict.py` | Live prediction engine with state checkpoint loading |
| `resolve.py` | Async label resolution for live predictions |
| `run.py` | CLI entry point, single-run, --compare, and --predict-now modes |

### Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `run_bayesian_opt.py` | Optuna TPE hyperparameter optimization. Lean mode (6 params) or full (14 params). `--apply` writes best to YAML. Studies are versioned by feature flags — changing config flags auto-starts a fresh study. |
| `run_gate_recheck.py` | Re-run promotion gates with OOF row-level evaluation |
| `run_overfit_check.py` | 5-metric overfitting diagnostics (gen gap, CV stability, threshold sensitivity, temporal stability, N_eff/N_params) |
| `run_full_institutional.py` | Full validation battery |
| `run_stress_suite.py` | Stress testing |
| `baselines.py` | Seven baseline models (hist freq, GARCH-CDF, implied-vol BS, feature logistic, gradient boosting, VIX threshold rule, market-implied straddle) |
| `run_robustness_analysis.py` | Blind spot analysis: sharpness, conditional ECE, universal thresholds, cross-asset correlation, AAPL failure |
| `run_ablation_study.py` | Systematic 7-variant ablation study with significance testing |
| `run_temporal_holdout.py` | Temporal hold-out evaluation (train pre-2020, test post-2020) |
| `run_paper_results.py` | Generate LaTeX-ready tables with p-values and CIs |
| `run_economic_significance.py` | Risk-managed portfolio and selective hedging analysis |
| `generate_paper_figures.py` | Publication-quality reliability diagrams, heatmaps, charts |
| `daily_predict.py` | Daily prediction runner with scheduling support |

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

# --- Live Prediction ---
# Generate prediction for today (uses checkpoint if available)
python -m em_sde.run --predict-now --config configs/spy_fixed.yaml --state-dir outputs/state/spy

# Build checkpoint from backtest, then predict
python -m em_sde.run --predict-now --config configs/spy_fixed.yaml --save-state

# Daily scheduled predictions for all tickers
python scripts/daily_predict.py
```

## Current State (2026-04-04)

### Ticker Status

**Primary scope: H=5 and H=10. H=20 is exploratory.**

- **SPY**: 6538 rows (2000-2025), **Primary 2/2 PASS** (H=5, H=10). H=20 ECE=0.024 barely fails. BSS positive across all horizons.
- **GOOGL**: 5376 rows (2004-2025), **Primary 2/2 PASS**, **Exploratory 1/1 PASS**. Implied vol was the key unlock for H=5.
- **AMZN**: 7202 rows (1997-2025), **Primary 2/2 PASS**, **Exploratory 1/1 PASS**. Best calibration: H=10 ECE=0.0029.
- **NVDA**: 6777 rows (1999-2025), **Primary 1/2** (H=5 only). H=10/H=20 BSS negative. Only 5 BO trials — needs more.

**Summary: 7/8 primary-horizon tests pass. Paper scope correctly scoped to H=5/H=10.**

### Recent Changes (2026-04-04)

**Critical bug fixes (from dual-persona Codex review):**

1. **Economic significance lookahead fix** (`run_economic_significance.py`): `dates[i+1]` → `dates[i]` in both risk-managed and selective hedging strategies. Previous code used tomorrow's prediction for today's trade decision. Also fixed drawdown reduction sign.
2. **resolve.py ticker filtering** (`resolve.py`, `daily_predict.py`): Added `ticker` parameter to prevent cross-ticker contamination in shared prediction log. Added `mark_resolved` tracking to prevent re-resolution.
3. **Baseline date alignment** (`run_paper_results.py`): Replaced blind `[:n_common]` truncation with proper date-based merge for paired bootstrap comparisons. Full model and baselines now aligned on matching dates.
4. **Baseline label leakage fix** (`baselines.py`): Feature logistic and gradient boosting baselines now only train on labels where `tt + H <= t` (outcome observable at prediction time). Previously used `tt + H < n` which leaked future labels. Added CalibratedClassifierCV (Platt scaling) to gradient boosting.
5. **Block bootstrap** (`evaluation.py`): Added `block_size` parameter to `paired_bootstrap_loss_diff_pvalue` and `bootstrap_metric_ci`. Circular block bootstrap preserves temporal structure for overlapping predictions.
6. **ACF-corrected N_eff in gates** (`model_selection.py`): Promotion gates now use `effective_sample_size(y, H, p_cal)` (residual-based ACF correction) instead of naive `min(events, nonevents) * 2`.
7. **RegimeMultiFeatureCalibrator serialization** (`calibration.py`): Added `export_state()`/`from_state()` — was missing, causing all BO trials and gate rechecks to crash.
8. **SPY + implied vol tested**: H=5 ECE=0.0088, H=10 ECE=0.0092 (PASS). H=20 ECE=0.0265 (still fails). Implied vol doesn't help SPY H=20.
9. **Robustness analysis** (`scripts/run_robustness_analysis.py`): New script covering 8 blind spots — sharpness, conditional ECE, universal thresholds, cross-asset correlation, AAPL failure, VIX/straddle baselines.
10. **Two new baselines** (`baselines.py`): VIX threshold rule and market-implied straddle probability.
11. **312 tests passing**.

**Second-pass fixes (from Codex re-review):**

12. **Block bootstrap actually enabled** (`run_paper_results.py`, `run_temporal_holdout.py`): All paper scripts now pass `block_size=H` to bootstrap functions. Previously used default `block_size=1` (i.i.d.) despite having block bootstrap support.
13. **FDR on baseline table** (`run_paper_results.py`): Baseline comparison table now gets Benjamini-Hochberg FDR correction on all p-values, matching the main results table.
14. **Baseline target fairness** (`run_paper_results.py`): Baselines now evaluated against the full model's y labels (same target definition). Previously baselines used fixed-threshold labels while the full model used regime-gated thresholds.
15. **Temporal holdout honesty** (`run_temporal_holdout.py`, `paper/main.tex`): Docstring and paper now explicitly describe this as an online-adaptive holdout, not a frozen-model test. Paper Section 4.4 updated.
16. **reproduce.py FDR** (`paper/reproduce.py`): Reproduction script now applies FDR correction to main results table.
17. **Calibrator online update** (`predict.py`, `daily_predict.py`): `PredictionEngine.update_calibrators()` feeds resolved outcomes back to live calibrators. `daily_predict.py` calls this after resolution.
18. **Paper wording tightened** (`paper/main.tex`): Removed "zero lookahead bias" and "exact same code" overclaims. Added limitations for target-design dependence, online-adaptive holdout, IV proxies, and live engine parity gap. Placeholder repo URL replaced. Bootstrap described as "block bootstrap" throughout.

**Third-pass fixes (from Codex recheck):**

19. **Live update semantics** (`predict.py`): `update_calibrators()` now uses `p_raw` (not `p_cal`) and full feature vector (sigma, delta_sigma, vol_ratio, vol_of_vol, earnings_proximity, implied_vol_ratio), matching backtest `_resolve_predictions_mf`.
20. **Resolve-before-predict** (`daily_predict.py`): Causal ordering fixed — resolve past predictions and update calibrators BEFORE generating new predictions. Previously predicted first, resolved after.
21. **Robustness date alignment** (`run_robustness_analysis.py`): Conditional ECE and cross-asset correlation now use actual prediction dates from `results["date"]` instead of `df.index[:n_results]` (which included warmup dates).
22. **Ablation block bootstrap** (`run_ablation_study.py`): Ablation p-values now use `block_size=H`.
23. **Governance ECE CI block bootstrap** (`model_selection.py`): `_bootstrap_ece_ci` now accepts `block_size` parameter; promotion gates pass `H` as block size.
24. **reproduce.py baseline FDR** (`paper/reproduce.py`): Standalone baseline reproduction now applies FDR correction, matching the paper claim.
25. **Holdout description** (`paper/main.tex`): Removed misleading "only pre-cutoff resolved labels" claim. Now explicitly states online-adaptive holdout with post-cutoff labels resolving and feeding back.
26. **Threshold disclosure** (`paper/main.tex`): Methodology now describes both fixed and regime-gated thresholds. Notes baselines evaluated on full model's target labels.
27. **Conclusion fixed** (`paper/main.tex`): "five baselines" → "seven baselines". Config hash claim removed. Novelty reframed as systems engineering.
28. **312 tests passing**.

**Statistical rigor + live prediction mode + honest scoping:**

1. **N_eff fix** (`evaluation.py`): ACF now computed on prediction residuals `(p_cal - y)` instead of binary labels. Removes upward bias in effective sample size estimates.
2. **Bootstrap CIs** (`evaluation.py`): BCa bootstrap CIs for BSS, AUC, and ECE. All paper tables now report `metric [95% CI]`.
3. **FDR correction** (`evaluation.py`, `run_paper_results.py`, `run_ablation_study.py`): Benjamini-Hochberg FDR correction across all hypothesis tests. Tables show both raw and adjusted p-values.
4. **ECE detailed** (`evaluation.py`): Per-bin sample counts reported alongside ECE for reviewer transparency.
5. **Gradient Boosting baseline** (`baselines.py`): HistGradientBoostingClassifier on vol features with Platt scaling — strongest fair baseline. Walk-forward expanding-window, same splits.
6. **Calibrator serialization** (`calibration.py`): `export_state()`/`from_state()` on OnlineCalibrator, MultiFeatureCalibrator, HistogramCalibrator, RegimeMultiFeatureCalibrator. JSON format.
7. **GARCH state persistence** (`garch.py`): `export_state()`/`from_state()` on GarchResult.
8. **Backtest state checkpoint** (`backtest.py`): Final calibrator and GARCH states saved in `result_df.attrs`.
9. **Live prediction engine** (`predict.py`): Loads checkpoint, fetches latest prices, runs MC, applies calibrators, returns `PredictionResult` per horizon.
10. **Async label resolution** (`resolve.py`): Resolves past predictions as outcomes become available. Ticker-filtered, with resolved-row tracking.
11. **`--predict-now` CLI** (`run.py`): Generate live predictions from CLI. Loads checkpoint or runs backtest first.
12. **Transaction cost model** (`run_economic_significance.py`): Sensitivity sweep at 0/5/10/20 bps per trade. No lookahead bias.
13. **Daily scheduling** (`scripts/daily_predict.py`): Cron-ready script for daily predictions with logging.
14. **312 tests passing** (was 296). 16 new tests for all Phase 1-2 features.

### Previous Changes (2026-04-03)

**Academic paper infrastructure (Phase 1-3):**

1. **Baselines module** (`scripts/baselines.py`): Five formal baseline models — Historical Frequency, GARCH-CDF, Implied-Vol Black-Scholes, Feature Logistic Regression, Gradient Boosting. All walk-forward.
2. **Ablation study** (`scripts/run_ablation_study.py`): Systematic 7-variant ablation with FDR-corrected paired bootstrap significance.
3. **Temporal hold-out** (`scripts/run_temporal_holdout.py`): Train through 2019-12-31, test 2020-2025. Per-era breakdown.
4. **Significance testing** (`scripts/run_paper_results.py`): LaTeX tables with bootstrap CIs, FDR-adjusted p-values.
5. **Economic significance** (`scripts/run_economic_significance.py`): Risk-managed portfolio + selective hedging + transaction cost sensitivity.
6. **Publication figures** (`scripts/generate_paper_figures.py`): Multi-panel reliability diagrams, ablation heatmaps, charts.
7. **Full LaTeX paper** (`paper/main.tex`): ~16-page paper with references.
8. **Reproducibility package** (`paper/reproduce.py`): Single script to regenerate all paper artifacts.

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
4. **Submit to IJF or Quantitative Finance** — best venue fit for calibration + walk-forward methodology

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
