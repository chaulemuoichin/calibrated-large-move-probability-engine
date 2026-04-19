# Validation Results

## Current Status (2026-04-04)

**Scope:** Primary claims are H=5 and H=10 (1-2 week horizons). H=20 is exploratory.

| Ticker | Data | Rows | Primary (H=5,10) | Exploratory (H=20) | Notes |
|--------|------|------|-------------------|---------------------|-------|
| **SPY** | 2000-2025 | 6,538 | **2/2 PASS** | FAIL (ECE=0.024) | Strong BSS/AUC at all horizons |
| **GOOGL** | 2004-2025 | 5,376 | **2/2 PASS** | PASS (ECE=0.018) | Implied vol key unlock for H=5 |
| **AMZN** | 1997-2025 | 7,202 | **2/2 PASS** | PASS (ECE=0.012) | Best calibration overall |
| **NVDA** | 1999-2025 | 6,777 | **1/2** (H=5 only) | FAIL (ECE=0.040) | Needs more BO trials |

**Summary:** 7/8 primary-horizon tests pass. 2/4 exploratory H=20 tests pass.

**Promotion gates:** ECE <= 0.02, BSS > 0, AUC >= 0.55. All three must pass simultaneously.

**Statistical rigor:** All p-values are Benjamini-Hochberg FDR-corrected. N_eff computed on prediction residuals (not binary labels). 95% BCa bootstrap CIs on BSS, AUC, ECE. Seven baselines (hist freq, GARCH-CDF, IV Black-Scholes, feature logistic, gradient boosting, VIX threshold rule, market-implied straddle) with paired block-bootstrap tests (block size = H).

**Method positioning:** A qualitative comparison against Black-Scholes/IV, GARCH-CDF, HMM, feature ML, and VaR/CVaR approaches is in [paper/main.tex](paper/main.tex) Section 2 (Table 1) and [METHODOLOGY.md §13](METHODOLOGY.md#13-how-this-differs-from-alternative-approaches).

All results from 5-fold expanding-window cross-validation with pooled out-of-fold evaluation.

---

## SPY (S&P 500 ETF)

6,538 daily observations, annualized vol ~16%.

| Horizon | Threshold | ECE | BSS | AUC | Status |
|---------|-----------|-----|-----|-----|--------|
| H=5 | 3.72% | 0.0103 | +0.074 | 0.758 | **PASS** |
| H=10 | 5.54% | 0.0147 | +0.046 | 0.715 | **PASS** |
| H=20 | 7.05% | 0.0240 | +0.058 | 0.738 | FAIL (ECE) |

Strong BSS and AUC across all horizons. H=20 ECE exceeds the 0.02 gate by 0.004. Implied vol (VIX) being tested -- may unlock H=20.

## GOOGL (Alphabet Inc.)

5,376 daily observations (from 2004 IPO), annualized vol ~28%. Implied vol (VIX proxy, 30% blend) enabled.

| Horizon | Threshold | ECE | BSS | AUC | Event Rate | Status |
|---------|-----------|-----|-----|-----|------------|--------|
| H=5 | 5.22% | 0.0076 | +0.027 | 0.659 | 9.4% | **PASS** |
| H=10 | 9.24% | 0.0114 | +0.018 | 0.650 | 5.9% | **PASS** |
| H=20 | 12.65% | 0.0181 | +0.022 | 0.671 | 6.9% | **PASS** |

All gates pass. Implied vol was the key unlock for H=5 -- ECE dropped from 0.022 to 0.0076.

## AMZN (Amazon.com Inc.)

7,202 daily observations (from 1997 IPO), annualized vol ~55%. Implied vol (VIX proxy) enabled.

| Horizon | Threshold | ECE | BSS | AUC | Event Rate | Status |
|---------|-----------|-----|-----|-----|------------|--------|
| H=5 | 8.52% | 0.0070 | +0.033 | 0.707 | 5.1% | **PASS** |
| H=10 | 15.99% | 0.0029 | +0.022 | 0.718 | 2.1% | **PASS** |
| H=20 | 20.50% | 0.0118 | +0.012 | 0.715 | 2.3% | **PASS** |

All gates pass. Best calibration: H=10 ECE of 0.0029. AUC >0.70 across all horizons despite high volatility. 28 years of history spanning dot-com, GFC, and COVID enables effective calibration.

## NVDA (NVIDIA Corp.)

6,777 daily observations (from 1999), annualized vol ~60%. Only 5 BO trials completed (needs 12+).

| Horizon | Threshold | ECE | BSS | AUC | Event Rate | Status |
|---------|-----------|-----|-----|-----|------------|--------|
| H=5 | 9.84% | 0.0165 | +0.017 | 0.665 | 7.2% | **PASS** |
| H=10 | 18.04% | 0.0127 | -0.010 | 0.588 | 4.4% | FAIL (BSS) |
| H=20 | 23.68% | 0.0395 | -0.010 | 0.595 | 6.9% | FAIL (BSS+ECE) |

H=5 passes. H=10 and H=20 have negative BSS -- thresholds too high (18-24%) from insufficient BO exploration. More trials in progress.

---

## Anti-Overfitting Strategy

**Problem:** 14 BO parameters on ~3,500 rows produced N_eff/N_params ratios of 12-31x (need 100x+).

**Solution:**
1. **Extended data** to full available history (2x more events).
2. **Lean BO mode** -- 6 tunable params (from 14). N_eff/N_params from 20-30x to 55-110x.
3. **N_eff soft penalty** in BO objective steers toward statistically robust regions.
4. **Adaptive event-rate guard** rejects trials with insufficient positive samples.

---

## Interpreting Forecasts

For each trading day: `p_cal(H, threshold)` = calibrated probability that |return over H days| >= threshold.

Example with SPY:
- `p_cal(H=5, 3.72%) = 0.08` -- 8% chance of a >3.7% weekly move. Normal conditions.
- `p_cal(H=5, 3.72%) = 0.25` -- 25% chance. Elevated risk, consider hedging.
- `p_cal(H=5, 3.72%) = 0.60` -- Extreme. Significant de-risking warranted.

ECE of 0.010 means: when the model says 15%, the actual frequency is 14-16%.
