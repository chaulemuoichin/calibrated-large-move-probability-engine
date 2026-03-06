# Validation Results

## Current Status

| Ticker | Data | Rows | Gates Passed | Status |
|--------|------|------|-------------|--------|
| **SPY** | 2000-2025 | 6,538 | **3/3** | H=5, H=10, H=20 all PASS |
| **AAPL** | 2000-2025 | 6,538 | **1/3** | H=5 PASS. H=10, H=20 FAIL (BSS < 0) |
| **GOOGL** | 2004-2025 | 5,376 | **2/3** | H=10, H=20 PASS. H=5 FAIL (ECE = 0.027) |

**Promotion gates** (all must pass per horizon): ECE <= 0.02, BSS > 0, AUC > 0.55

All results are from 5-fold expanding-window cross-validation with pooled out-of-fold evaluation. Lean BO mode (6 tuned parameters).

---

## SPY (S&P 500 ETF)

6,538 daily observations, annualized vol ~16%. 8 BO trials (Lean, 6 params).

| Horizon | Threshold | ECE | BSS | AUC | Status |
|---------|-----------|-----|-----|-----|--------|
| H=5 | 5.30% | 0.0062 | +0.012 | 0.62 | **PASS** |
| H=10 | 3.25% | 0.0167 | +0.008 | 0.58 | **PASS** |
| H=20 | 4.99% | 0.0121 | +0.005 | 0.57 | **PASS** |

SPY is the strongest performer. Deep liquidity and strong vol-clustering make it ideal for GARCH-based forecasting.

## AAPL (Apple Inc.)

6,538 daily observations, annualized vol ~30%. 2 BO trials with adaptive event-rate guard.

| Horizon | Threshold | ECE | BSS | AUC | Event Rate | Status |
|---------|-----------|-----|-----|-----|------------|--------|
| H=5 | 6.24% | 0.0098 | +0.004 | 0.637 | 6.6% | **PASS** |
| H=10 | 11.19% | 0.0158 | -0.043 | 0.598 | 2.4% | FAIL (BSS) |
| H=20 | 14.73% | 0.0111 | -0.010 | 0.650 | 4.0% | FAIL (BSS) |

H=5 passes all gates. H=10 and H=20 have good ECE and AUC but negative BSS: the model underperforms the naive base-rate at longer horizons. Root cause: BO-selected thresholds produce event rates of only 2.4-4.0%, leaving too few positive samples for the calibrator to learn a reliable mapping.

Overfitting risk: HIGH. N_eff/N_params: H=5 86x (YELLOW), H=10 31x (RED), H=20 51x (YELLOW).

## GOOGL (Alphabet Inc.)

5,376 daily observations (from 2004 IPO), annualized vol ~28%. 2 BO trials with adaptive event-rate guard.

| Horizon | Threshold | ECE | BSS | AUC | Event Rate | Status |
|---------|-----------|-----|-----|-----|------------|--------|
| H=5 | 5.22% | 0.0273 | +0.010 | 0.626 | 9.4% | FAIL (ECE) |
| H=10 | 9.24% | 0.0197 | +0.017 | 0.661 | 5.9% | **PASS** |
| H=20 | 12.65% | 0.0122 | +0.014 | 0.642 | 6.9% | **PASS** |

H=10 and H=20 pass all gates with positive BSS (real forecasting skill). H=5 has good discrimination and skill but ECE = 0.027 barely exceeds the 0.02 gate.

Overfitting risk: MODERATE. N_eff/N_params: H=5 101x (GREEN), H=10 62x (YELLOW), H=20 72x (YELLOW).

---

## Synthetic Baselines

### Cluster Dataset (leverage-clustering pattern)

3,200 observations, annualized vol ~11%.

| Horizon | ECE | BSS | AUC | Status |
|---------|-----|-----|-----|--------|
| H=5 | 0.0112 | +0.205 | 0.874 | PASS |
| H=10 | 0.0170 | +0.180 | 0.874 | PASS |
| H=20 | 0.0243 | +0.095 | 0.701 | FAIL (ECE) |

### Jump-Crash Dataset (synthetic crash pattern)

3,200 observations, annualized vol ~28%, skew -2.7, kurtosis 16.3.

| Horizon | ECE | BSS | AUC | Status |
|---------|-----|-----|-----|--------|
| H=5 | 0.0123 | -0.013 | 0.510 | FAIL |
| H=10 | 0.0155 | -0.012 | 0.549 | FAIL |
| H=20 | 0.0473 | -0.010 | 0.521 | FAIL |

Crashes are inherently unpredictable with vol-only models. Score: 0/3.

---

## Anti-Overfitting Strategy

### Problem

14 BO parameters on ~3,500 rows produced:
- N_eff/N_params ratios: 12-31x (need 100x+)
- Generalization gaps up to +75%
- CV instability: coefficient of variation 0.43-0.83

### Solution

**1. Extend data to full available history.**
SPY/AAPL: 2012 -> 2000 (+3,000 rows). GOOGL: 2012 -> 2004 IPO (+1,800 rows). Roughly 2x more events.

**2. Reduce BO parameters from 14 to 6 (Lean Mode).**

| Tuned (6) | Fixed at Defaults |
|-----------|-------------------|
| thr_5, thr_10, thr_20 | hmm_regime = false |
| garch_persistence | t_df_low=10, t_df_mid=5, t_df_high=4 |
| mf_lr, mf_l2 | mf_min_updates = 63, har_rv = false |

Combined effect: N_eff/N_params from 20-30x -> 55-110x.

---

## Interpreting Forecasts

### Output format

For each trading day:

```
p_cal(H, threshold) = calibrated probability that |return over H days| >= threshold
```

Example with SPY config:
- `p_cal(H=5, 5.30%)` = probability of a >5.30% move in either direction over 5 trading days
- `p_cal(H=20, 4.99%)` = probability of a >4.99% move over 20 trading days

### What "calibrated" means

ECE of 0.006 (SPY H=5) means: when the model says 15%, the actual frequency is 14.4-15.6%.

| Model output | Actual frequency (ECE ~0.01) | Implication |
|-------------|------------------------------|-------------|
| 5% | 4-6% | Normal conditions |
| 15% | 14-16% | Slightly elevated risk |
| 35% | 34-36% | High risk, consider hedging |
| 60% | 59-61% | Extreme, significant de-risking warranted |

### Practical examples

**SPY p_5d = 8%**: 8% chance of a >5.3% weekly move. Calm market, standard positioning.

**AAPL p_5d = 25%**: 25% chance of a >6.2% weekly move. Elevated risk near earnings. Consider hedging.

**SPY p_20d = 55%**: More likely than not that SPY moves >5% this month. Vol clustering in effect. De-risk or hedge.

### Caveats

1. **Two-sided.** Predicts |move| >= threshold, not direction.
2. **Regime lag.** GARCH responds to vol changes with 1-2 day delay.
3. **Ticker-specific thresholds.** Always re-run BO for new tickers.
4. **Risk model, not trading signal.** Designed for position sizing and hedging, not directional bets.

---

## Adding a New Ticker

```bash
# 1. Prepare CSV with Date index + Close column, or use yfinance
# 2. Create config YAML (copy exp_spy_regime_gated.yaml, adjust ticker/csv_path/start)
# 3. Run lean BO
python scripts/run_bayesian_opt.py <ticker> --n-trials 12
# 4. Apply best params
python scripts/run_bayesian_opt.py <ticker> --apply
# 5. Validate
python -u scripts/run_gate_recheck.py <ticker>
python scripts/run_overfit_check.py <ticker>
```

---

## Gate Progression

| Session | Score | Key Change |
|---------|-------|------------|
| Baseline | 0/6 | Vol-scaled thresholds, basic online calibrator |
| +Histogram post-cal | 0/6 | Bayesian shrinkage + PAV monotonicity |
| +Multi-feature | 1/6 | Cluster H=5 PASS |
| +BO threshold tuning | 2/6 | Cluster H=10 PASS |
| +Real tickers (14p BO) | 5/9 | SPY 3/3, AAPL 1/3, GOOGL 1/3 |
| +Lean BO + extended data | 4/9 | Event rates too low on some horizons |
| +Adaptive event-rate guard | 6/9 | SPY 3/3, AAPL 1/3, GOOGL 2/3 |

---

## Validation Standards

1. **Calibration.** When the model says 15%, it happens ~15% of the time (ECE < 0.02).
2. **Discrimination.** The model separates events from non-events better than chance (AUC > 0.55).
3. **Skill.** Better than naive base-rate prediction (BSS > 0).
4. **Robustness.** Results hold across 20+ years including dot-com crash, 2008 GFC, COVID, and 2022.
5. **Honest uncertainty.** N_eff and overfitting diagnostics are tracked and reported.
6. **Reproducibility.** Seed-controlled MC, deterministic CV splits, versioned configs.

---

## Disclaimer

Research and educational use only. Not investment advice.
