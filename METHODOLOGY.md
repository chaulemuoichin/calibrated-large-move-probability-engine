# Methodology: How This System Produces a Probability

This document explains every step from raw price data to final calibrated probability output. It is written so that a collaborator can spot errors, question assumptions, or propose improvements without reading the full source code.

If something here does not match the code, the code wins and this document needs updating.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Step 1: Data Ingestion and Quality Checks](#2-step-1-data-ingestion-and-quality-checks)
3. [Step 2: Volatility Estimation (GARCH)](#3-step-2-volatility-estimation-garch)
4. [Step 3: Defining "Large Move" (Threshold)](#4-step-3-defining-large-move-threshold)
5. [Step 4: Simulating Future Prices (Monte Carlo)](#5-step-4-simulating-future-prices-monte-carlo)
6. [Step 5: Computing Raw Probability](#6-step-5-computing-raw-probability)
7. [Step 6: Calibrating the Probability](#7-step-6-calibrating-the-probability)
8. [Step 7: Walk-Forward Backtest (No Lookahead)](#8-step-7-walk-forward-backtest-no-lookahead)
9. [Step 8: Evaluation Metrics](#9-step-8-evaluation-metrics)
10. [Step 9: Model Selection and Promotion](#10-step-9-model-selection-and-promotion)
11. [Mathematical Reference](#11-mathematical-reference)
12. [Known Limitations and Assumptions](#12-known-limitations-and-assumptions)
13. [Bayesian Hyperparameter Optimization](#13-bayesian-hyperparameter-optimization)
14. [Statistical Rigor](#14-statistical-rigor-2026-04-04)
15. [Live Prediction System](#15-live-prediction-system-2026-04-04)
16. [File Map](#16-file-map)
17. [References](#17-references)

---

## 1. Pipeline Overview

```
Raw Prices
  |
  v
Daily Returns  ──>  GARCH/GJR-GARCH fit  ──>  sigma_1d (today's vol forecast)
               ──>  HAR-RV fit (optional) ──>  sigma_1d/5d/22d (horizon-specific)
                                                   |
                                                   v
                                            Threshold ("what counts as a large move?")
                                                   |
                                                   v
                                            Monte Carlo Simulation (thousands of possible futures)
                                                   |
                                                   v
                                            p_raw = fraction of futures where the move is large enough
                                                   |
                                                   v
                                            Calibration (adjust p_raw using past accuracy)
                                                   |
                                                   v
                                            p_cal = final calibrated probability
```

At every step, the system only uses data available up to today. It never peeks at future prices. This is enforced by a resolution queue that waits for outcomes to arrive before updating the calibrator.

---

## 2. Step 1: Data Ingestion and Quality Checks

**File:** `em_sde/data_layer.py`

### What goes in

Daily price data for a single ticker. Three sources are supported:

| Source | Description |
|--------|-------------|
| `yfinance` | Downloads from Yahoo Finance with retry logic and local Parquet cache |
| `csv` | Reads a local CSV file (auto-detects `Close` / `Adj Close` / `Price`, and preserves OHLCV when present) |
| `synthetic` | Generates fake GBM prices for testing |

### What comes out

A DataFrame with a canonical `price` column indexed by date. When available, `open`, `high`, `low`, and `volume` are preserved for OHLC-aware features. Minimum 756 rows (~3 years of trading days).

### Quality checks (warning-only by default, blocking when `strict_validation: true`)

| Check | What it catches |
|-------|----------------|
| Outlier detection | Returns beyond 5x the interquartile range |
| Stale prices | 5+ consecutive identical closes (data feed issue) |
| Data gaps | More than 5 missing business days in a row |
| Split-like moves | Unadjusted corporate actions / bad vendor adjustments |
| OHLC consistency | Impossible bars (e.g. `high < low`, close outside range) |
| Return statistics | Reports skewness, kurtosis, min/max for manual review |

The loader also records a dataset hash and can persist a research snapshot / artifact manifest for run lineage.

### Where to look for problems

- If the ticker has stock splits that Yahoo didn't adjust, volatility estimates will be wrong.
- If a CSV only contains `price`, OHLC-derived features and hybrid variance are automatically inactive.
- If `min_rows` is set too low, GARCH fitting may be unstable.
- The cache has a 7-day expiration. Stale cache during fast-moving markets could matter.

---

## 3. Step 2: Volatility Estimation (GARCH)

**File:** `em_sde/garch.py`

The goal is to answer: **how volatile is this stock right now?** The answer is a single number, `sigma_1d`, the estimated daily volatility in decimal (e.g., 0.015 = 1.5% per day).

### GARCH(1,1)

The standard model ([Bollerslev, 1986](https://doi.org/10.1016/0304-4076(86)90063-1) | [Wikipedia](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity#GARCH)). Volatility tomorrow depends on today's volatility and today's surprise:

$$\sigma_{t+1}^{2} = \omega + \alpha \epsilon_t^{2} + \beta \sigma_t^{2}$$

- **omega**: a small constant floor (prevents vol from collapsing to zero)
- **alpha**: how much today's surprise matters (higher = more reactive)
- **beta**: how much yesterday's vol persists (higher = slower to change)
- **shock**: the unexpected part of today's return

The model is **stationary** (well-behaved long-term) when $\alpha + \beta < 1$. In that case, volatility eventually settles to a long-run level:

$$\sigma_{\mathrm{long\_run}} = \sqrt{\frac{\omega}{1-\alpha-\beta}}$$

### GJR-GARCH (Leverage Effect)

An extension ([Glosten, Jagannathan & Runkle, 1993](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x) | [Wikipedia](https://en.wikipedia.org/wiki/GJR-GARCH_model)) that lets negative surprises (drops) have more impact than positive ones:

$$\sigma_{t+1}^{2} = \omega + \left(\alpha + \gamma I_{\epsilon_t<0}\right)\epsilon_t^2 + \beta \sigma_t^2$$

where `I[drop]` is 1 when the shock was negative. This captures the empirical fact that markets react more to bad news than good news.

Stationarity condition: $\alpha + \beta + \gamma/2 < 1$.

### EWMA Fallback

If GARCH fitting fails (optimizer doesn't converge), the system falls back to [Exponentially Weighted Moving Average](https://en.wikipedia.org/wiki/Moving_average#Exponentially_weighted_moving_variance_and_standard_deviation) volatility ([RiskMetrics, J.P. Morgan 1996](https://en.wikipedia.org/wiki/RiskMetrics)) with a 252-day span:

$$\sigma_t^2 = \sum_{j=1}^{m} w_j r_{t-j}^2,\quad \sum_{j=1}^{m} w_j = 1$$

This always produces a number but loses the shock-response dynamics of GARCH.

### Fitting Procedure

1. Take the most recent 756 daily returns (configurable)
2. Fit GARCH or GJR-GARCH using the `arch` library
3. Extract the one-step-ahead volatility forecast (`sigma_1d`)
4. Extract the fitted parameters (`omega`, `alpha`, `beta`, `gamma`)
5. Compute diagnostics (persistence, stationarity, half-life)

### Stationarity Projection (U4)

**Problem:** Sometimes the fitted parameters have $\alpha + \beta \ge 1$, meaning volatility would explode over time in simulation.

**Fix:** Scale all the shock-response parameters down proportionally to hit a target persistence (default 0.98), and recompute `omega` so the long-run variance equals the current forecast:

$$s = \frac{0.98}{\phi_{\mathrm{current}}},\quad \alpha_{\mathrm{new}} = s\alpha,\quad \beta_{\mathrm{new}} = s\beta,\quad \omega_{\mathrm{new}} = \sigma_{1d}^2(1-0.98)$$

This preserves the relative importance of each parameter while anchoring the simulation to current volatility.

### Where to look for problems

- If $\alpha + \beta$ is frequently >= 1, the underlying data may have structural breaks.
- If `omega_new` becomes very small, the GARCH simulation becomes essentially a martingale — volatility doesn't mean-revert.
- The EWMA fallback loses all shock-asymmetry information. When it triggers often, GJR features are wasted.

### GARCH Vol Term Structure (mc_vol_term_structure)

When `mc_vol_term_structure: true`, the MC simulation uses a **horizon-adjusted initial volatility** instead of the raw 1-step-ahead GARCH forecast. This addresses the vol mean-reversion bias: when current volatility is elevated, a 1-step forecast overstates the average volatility over longer horizons (H=10, H=20) because GARCH conditional variance mean-reverts toward the unconditional level.

The term-structure average is computed analytically:

$$\mathbb{E}\!\left[\sigma_{t+h}^{2}\right] = \sigma_{\mathrm{unc}}^{2} + \phi^{h}\left(\sigma_t^{2}-\sigma_{\mathrm{unc}}^{2}\right)$$

$$\sigma_{\mathrm{avg}}(H) = \sqrt{\frac{1}{H}\sum_{h=1}^{H}\mathbb{E}\!\left[\sigma_{t+h}^{2}\right]}$$

For GARCH(1,1), $\phi=\alpha+\beta$. For GJR-GARCH(1,1), $\phi=\alpha+\beta+\frac{\gamma}{2}$. Walk-forward safe (uses only fitted params).

**File:** `em_sde/garch.py` (`garch_term_structure_vol`)

### HAR-RV Volatility Model (`har_rv`)

When `har_rv: true`, a Heterogeneous AutoRegressive Realized Variance model ([Corsi, 2009](https://doi.org/10.1093/jjfinec/nbp001) | [Wikipedia](https://en.wikipedia.org/wiki/Heterogeneous_autoregressive_model)) replaces GARCH as the primary sigma engine. This directly addresses the root cause of `p_raw` bias: GARCH persistence (~0.97) keeps sigma elevated after vol spikes, but HAR-RV uses actual realized variance which mean-reverts at the correct speed.

**Model**: Three separate ridge regressions, one per forecast horizon:

$$RV_{t+1}^{(h)} = \beta_0 + \beta_1 \cdot RV_{1d,t} + \beta_2 \cdot RV_{5d,t} + \beta_3 \cdot RV_{22d,t}$$

Where:
- $RV_{1d} = r_t^2$ (daily realized variance)
- $RV_{5d} = \frac{1}{5}\sum_{i=0}^{4} r_{t-i}^2$ (weekly average)
- $RV_{22d} = \frac{1}{22}\sum_{i=0}^{21} r_{t-i}^2$ (monthly average)

**Three regressions produce horizon-specific forecasts**:
- **1-day target**: $RV_{t+1}$ → `sigma_1d`
- **5-day target**: $\text{mean}(RV_{t+1} \ldots RV_{t+5})$ → `sigma_5d`
- **22-day target**: $\text{mean}(RV_{t+1} \ldots RV_{t+22})$ → `sigma_22d`

This is the key advantage over GARCH: each forecast horizon gets its own optimized sigma, eliminating the horizon-conflicted blend problem that limited HMM.

**Horizon-specific sigma mapping**:
- H=5: uses `sigma_5d`
- H=10: interpolates between `sigma_5d` and `sigma_22d`
- H=20: uses `sigma_22d`

**Integration with existing pipeline**:
- GARCH is still fitted (provides omega/alpha/beta for GARCH-in-sim MC dynamics)
- HAR-RV overrides `sigma_1d` and `sigma_per_h` for MC initial conditions
- HMM can still blend on top if enabled
- Multi-feature calibration features (vol_ratio, delta_sigma) use HAR-RV sigma

**Config fields** (in `model` section):

- `har_rv: true` — enable HAR-RV sigma engine
- `har_rv_min_window: 252` — minimum returns for fitting
- `har_rv_refit_interval: 21` — refit every N days
- `har_rv_ridge_alpha: 0.01` — ridge regularization strength

**File:** `em_sde/garch.py` (`fit_har_rv`, `HarRvResult`, `compute_realized_variance`)

---

## 4. Step 3: Defining "Large Move" (Threshold)

**File:** `em_sde/backtest.py`

Before we can estimate the probability of a "large move," we need to define what that means. The threshold is the minimum absolute return that counts as a large move.

### Three Threshold Modes

| Mode | Formula | When to use |
|------|---------|-------------|
| `fixed_pct` | $\tau = c$ (e.g., $c=5\%$) | Best for genuine discrimination — the goalpost doesn't move |
| `anchored_vol` | $\tau = k\,\sigma_{\mathrm{unconditional}}\sqrt{H}$ | Slowly-moving goalpost using historical average vol |
| `regime_gated` | Routes between above modes based on current vol percentile | Adaptive: uses different strategy in calm vs volatile markets |

**Regime-gated routing logic:**

```
vol_percentile = rank of today's sigma in the last 252 sigma values

if vol_percentile < 25th percentile:    use low-vol mode  (default: fixed_pct)
if vol_percentile > 75th percentile:    use high-vol mode (default: anchored_vol)
otherwise:                               use mid-vol mode  (default: fixed_pct)
```

### The Event Definition

For a prediction made on date `t` with horizon `H` trading days:

$$\mathrm{event}_t^{(H)}=\begin{cases}1,& \left|\frac{P_{t+H}}{P_t}-1\right|\ge \tau \\ 0,& \text{otherwise}\end{cases}$$

This is **two-sided**: both large up-moves and large down-moves count.

### Where to look for problems

- `fixed_pct` can produce very different event rates across regimes. A 5% threshold might fire 30% of the time during a crash but 1% during calm markets. The calibrator has to handle this imbalance.
- `regime_gated` has a 252-day warmup. During warmup it uses the mid-vol mode, which might not be optimal.
- `regime_gated_fixed_pct_by_horizon` allows per-horizon threshold overrides (e.g., lower threshold at H=5 to ensure evaluable event counts in low-vol regimes). Default: uses the global `fixed_threshold_pct` for all horizons.

---

## 5. Step 4: Simulating Future Prices (Monte Carlo)

**File:** `em_sde/monte_carlo.py`

We generate thousands of possible future price paths (default: 100,000) and look at where they end up. Each path is one possible future.

### Constant-Volatility GBM

The simplest model ([Geometric Brownian Motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion) | [Euler–Maruyama method](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)). Each daily step:

$$\log P_{t+1} = \log P_t + \left(\mu-\frac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\,Z_t$$

where:
- `drift` = annualized expected return (default 0)
- `vol` = annualized volatility = sigma_1d * sqrt(252)
- `dt` = 1/252 (one trading day)
- `Z` = random standard normal draw

After $H$ steps, convert back: $P_{\mathrm{terminal}} = \exp(X_{\mathrm{final}})$.

### GARCH-in-Simulation

Instead of constant volatility, each path evolves its own volatility using the GARCH equation:

```
For each day in the path:
  1. Use current path-specific sigma for today's step
  2. Simulate a price move using that sigma
  3. Update sigma for tomorrow using the GARCH equation
```

This means each simulated future has its own volatility story — some paths see vol spike, others see it calm down. This is more realistic than assuming vol stays constant.

### Jump-Diffusion (Merton Model)

Adds rare, sudden jumps on top of the diffusion ([Merton, 1976](https://doi.org/10.1016/0304-405X(76)90022-2) | [Wikipedia](https://en.wikipedia.org/wiki/Jump_diffusion#In_economics_and_finance)):

On any given day, there is a small probability of a jump:

$$N_{\mathrm{jump}} \sim \mathrm{Poisson}(\lambda/252)$$

If a jump happens:

$$J_t \sim \mathcal{N}(\mu_J,\sigma_J),\quad \log P_t \leftarrow \log P_t + J_t$$

The drift is compensated so that adding jumps doesn't change the expected return:

$$d_{\mathrm{adj}} = \lambda\left(e^{\mu_J + \frac{1}{2}\sigma_J^2} - 1\right)$$

### State-Dependent Jumps (U2)

Jump parameters vary with the current volatility regime:

$$t = \mathrm{clip}\!\left(\frac{\sigma_{1d}-\sigma_{p25}}{\sigma_{p75}-\sigma_{p25}},\,0,\,1\right)$$

$$\lambda_t = \lambda_{\mathrm{low}} + t\left(\lambda_{\mathrm{high}}-\lambda_{\mathrm{low}}\right)$$
$$\mu_{J,t} = \mu_{J,\mathrm{low}} + t\left(\mu_{J,\mathrm{high}}-\mu_{J,\mathrm{low}}\right)$$
$$\sigma_{J,t} = \sigma_{J,\mathrm{low}} + t\left(\sigma_{J,\mathrm{high}}-\sigma_{J,\mathrm{low}}\right)$$

In calm markets (low `t`): fewer, smaller jumps. In stressed markets (high `t`): more frequent, larger jumps.

### Fat Tails ([Student-t](https://en.wikipedia.org/wiki/Student%27s_t-distribution))

Instead of drawing `Z` from a standard normal, draw from a Student-t distribution and rescale to unit variance:

$$Z_{\mathrm{fat}} = Z_t\sqrt{\frac{\nu-2}{\nu}},\quad Z_t \sim t_{\nu}$$

This produces more extreme daily moves than a normal distribution, matching empirical return distributions better.

### Regime-Conditional Student-t Degrees of Freedom (mc_regime_t_df)

When `mc_regime_t_df: true`, the Student-t degrees of freedom for MC innovations adapt to the current vol regime. High-vol periods use heavier tails (lower t_df) to better capture crash dynamics; low-vol periods use thinner tails (higher t_df) closer to Gaussian. Regime assignment uses the same rolling vol percentile logic as RegimeRouter (walk-forward safe).

Defaults: low_vol t_df=8.0, mid_vol t_df=5.0, high_vol t_df=4.0. Configured via `mc_regime_t_df_low`, `mc_regime_t_df_mid`, `mc_regime_t_df_high`.

### Filtered Historical Simulation (FHS) — disabled

> **Status: disabled by default.** Tested and found to degrade results at longer horizons. Kept in codebase behind `fhs_enabled` flag for reference.

FHS resamples standardized residuals $z_t = \epsilon_t / \hat{\sigma}_t$ from the GARCH fit instead of drawing from a parametric distribution ([Barone-Adesi et al., 1999](https://doi.org/10.1016/S0378-4266(98)00091-4)). In theory this captures the true innovation distribution without distributional assumptions.

In practice, ~750 residuals from a rolling GARCH window contain fewer tail events than a Student-t(5) distribution. This truncates the simulated tail mass, reducing discrimination for large moves. The effect compounds over multi-step horizons (H=10, H=20). Additionally, stationarity projection drops the residuals, causing inconsistent fallback between FHS and parametric across time — the calibrator cannot learn a stable mapping when the input distribution switches unpredictably.

### GARCH Ensemble — disabled

> **Status: disabled by default.** Tested and found to degrade results due to sigma/dynamics mismatch. Kept behind `garch_ensemble` flag for reference.

The ensemble averages sigma_1d from GARCH(1,1), GJR-GARCH, and EGARCH ([Ranjan & Gneiting, 2010](https://doi.org/10.1111/j.1467-9868.2009.00726.x)). However, simulation dynamics still use GJR parameters (omega, alpha, beta, gamma), which mean-revert to GJR's unconditional variance — not the ensemble average. This mismatch grows with horizon length, systematically biasing multi-step simulations.

### Where to look for problems

- **Path count**: 100K paths give Monte Carlo standard error ~0.001 for p=0.05. Reducing to 1K (for speed) gives SE ~0.007 — still usable for ranking but noisy for absolute levels.
- **GARCH-in-sim with non-stationary params**: If stationarity projection fails or is disabled, volatility can explode inside paths. The variance floor (`1e-12`) prevents NaN but produces unrealistic paths.
- **Jump compensator**: If `mu_J` is very negative and `lambda` is high, the compensator is large and the diffusion drift becomes strongly positive, which may look odd.
- **dt = 1/252 assumption**: Ignores weekends, holidays, overnight gaps. These matter more for short horizons (H=5).

---

## 6. Step 5: Computing Raw Probability

**File:** `em_sde/monte_carlo.py` (`compute_move_probability`)

After simulation, we have terminal prices for each path. The raw probability is simply:

$$r_i = \frac{P^{(i)}_{\mathrm{terminal}}}{P_0}-1,\quad p_{\mathrm{raw}}=\frac{1}{N_{\mathrm{paths}}}\sum_{i=1}^{N_{\mathrm{paths}}}\mathbf{1}\{|r_i|\ge \tau\}$$

With Monte Carlo standard error:

$$\mathrm{SE}=\sqrt{\frac{p_{\mathrm{raw}}(1-p_{\mathrm{raw}})}{N_{\mathrm{paths}}}}$$

### Adaptive Path Boosting

When events are rare (event rate < 2% over the last year), the system doubles the path count to reduce noise:

```
if recent_event_rate < 0.02:
    n_paths = mc_boost_paths (default: 200,000)
else:
    n_paths = mc_base_paths (default: 100,000)
```

### Where to look for problems

- `p_raw` near 0 or 1 can cause numerical issues downstream in calibration (logit of 0 or 1 is undefined). The code clamps probabilities to `[1e-7, 1 - 1e-7]`.
- If the threshold is too low relative to vol, `p_raw` approaches 1.0 and the forecast becomes uninformative.
- If the threshold is too high, `p_raw` approaches 0.0 and the calibrator has almost no events to learn from.

---

## 7. Step 6: Calibrating the Probability

**File:** `em_sde/calibration.py`

Raw Monte Carlo probabilities are systematically biased. Calibration learns a correction mapping from realized outcomes.

### Online Platt Scaling (Basic Calibrator)

The simplest calibrator ([Platt, 1999](https://www.researchgate.net/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines) | [Wikipedia](https://en.wikipedia.org/wiki/Platt_scaling)) maps raw to calibrated via a logistic function:

$$p_{\mathrm{cal}}=\sigma\!\left(a+b\,\mathrm{logit}(p_{\mathrm{raw}})\right)$$

where $\sigma(x)=\frac{1}{1+e^{-x}}$ and $\mathrm{logit}(p)=\log\!\left(\frac{p}{1-p}\right)$.

**Initialization:** $a=0,\ b=1$ (identity — calibrated = raw).

**Online update** (when an outcome `y` arrives):

$$e_t = y_t - p_{\mathrm{cal},t},\quad a \leftarrow a + \eta e_t,\quad b \leftarrow b + \eta e_t\,\mathrm{logit}(p_{\mathrm{raw},t})$$

Learning rate decays over time: $\eta_t = \eta_0/\sqrt{1+n_{\mathrm{updates}}}$.

The calibrator only starts adjusting after `min_updates` (default 50) outcomes have arrived.

### Offline Pooled Calibration (`offline_pooled_calibration`)

For research mode, the system can fit a batch calibrator on each **train fold only** and apply it to the held-out fold. This keeps the walk-forward split honest while reducing the noise of purely online updates on sparse event horizons.

The offline calibrator uses the same feature family as the online multi-feature calibrator, but solves a penalized logistic problem with IRLS on resolved train rows only. Optional features that are missing on a given ticker (earnings, implied-vol ratio, OHLC realized-state features) fall back to neutral defaults instead of propagating NaNs into the calibrated probability.

### Multi-Feature Calibrator

Uses 6 features instead of just `logit(p_raw)` (7 features when earnings calendar is active for the horizon):

$$x_t=\left[1,\ \mathrm{logit}\!\left(p_t^{\mathrm{raw}}\right),\ 100\,\sigma_{d,t},\ 100\,\Delta\sigma_{20,t},\ \frac{\sigma_{r,t}}{\sigma_{d,t}},\ 100\,v_{ov,t},\ (\mathrm{earn}_{t})\right]$$

Here, $p_t^{\mathrm{raw}}\equiv p_{\mathrm{raw},t}$, $\sigma_{d,t}\equiv\sigma_{1d,t}$, $\sigma_{r,t}\equiv\sigma_{\mathrm{realized},t}$, $v_{ov,t}$ corresponds to `vol_of_vol_t`, and $\mathrm{earn}_{t}$ is the earnings proximity feature (optional, 0-1 scale).

$$p_{\mathrm{cal},t}=\sigma\!\left(w^\top x_t\right)$$

Updated via SGD with L2 regularization (prevents overfitting) and gradient clipping (prevents explosions). Requires 100 outcomes before activating.

### Earnings Calendar Feature (earnings_calendar)

When `earnings_calendar: true`, an earnings proximity feature is added to the multi-feature calibrator **for short horizons only (H ≤ 5)**. Earnings announcements produce significantly larger moves than non-earnings days ([Dubinsky et al., 2019](https://doi.org/10.1093/rfs/hhy018) | [Savor & Wilson, 2016](https://doi.org/10.1111/jofi.12351)), making this a strong calibration signal at weekly timescales.

$$\mathrm{earn}_{t} = \max\!\left(0,\, 1 - \frac{\left|d^{\mathrm{earn}}_{t}\right|}{20}\right)$$

Here, $d^{\mathrm{earn}}_{t}$ is the number of calendar days to the nearest earnings date. The feature equals 1.0 on an earnings day and decays linearly to 0.0 at 20 calendar days away. At longer horizons (H=10, H=20), prediction windows almost always overlap with an earnings date, so the feature becomes uninformative noise. The horizon restriction is enforced automatically in the backtest engine.

Walk-forward safe: earnings dates are publicly announced weeks in advance. Data sourced from yfinance with local CSV caching (30-day expiry).

### Regime-Conditional Multi-Feature Calibration

When `multi_feature_regime_conditional=true`, the system maintains one `MultiFeatureCalibrator` per volatility regime bin (low/mid/high vol). Regime assignment uses the same rolling sigma_1d percentile approach as `RegimeCalibrator` (walk-forward safe). Each regime-specific calibrator learns its own weight vector independently, so calibration adapts to the distinct probability-outcome relationship within each vol environment. Enabled via config flag; default off for backward compatibility.

### Safety Gate (Brier-based)

If calibration makes things worse, stop doing it:

```
if rolling_brier(p_cal) > rolling_brier(p_raw):
    return p_raw instead of p_cal
```

### Discrimination Guardrail (U3)

If the model loses the ability to rank events correctly, bypass calibration:

```
if rolling_AUC < 0.50 or rolling_separation < 0.0:
    return p_raw instead of p_cal
```

This prevents the calibrator from actively inverting the signal during regime changes.

### Histogram Post-Calibration (P0-2)

After the Platt/logistic mapping, an optional second pass corrects residual bin-level bias using [histogram binning calibration](https://en.wikipedia.org/wiki/Calibration_(statistics)#Calibration_in_classification) with [Bayesian shrinkage](https://en.wikipedia.org/wiki/Shrinkage_(statistics)) and [Pool Adjacent Violators (PAV)](https://en.wikipedia.org/wiki/Isotonic_regression) for monotonicity:

```
For each of 10 equal-width bins over [0, 1]:
    Track running mean(p_cal) and mean(y) for samples in that bin.

At prediction time:
    raw_correction = mean_pred_in_bin - mean_obs_in_bin
    shrinkage = count / (count + prior_strength)    (default prior_strength = 15)
    correction = raw_correction * shrinkage
    p_final = p_cal - correction    (clipped to [0, 1])

Monotonic enforcement (PAV):
    After computing all bin corrections, apply Pool Adjacent Violators
    to ensure corrected bin-center values are non-decreasing.
    This guarantees the mapping preserves probability rankings (AUC invariant).
```

This directly targets calibration error (ECE) rather than squared error (Brier). The 10-bin grid is aligned with the ECE evaluation bins. A minimum of 15 samples per bin is required before correction activates.

Bayesian shrinkage dampens the correction when bin sample counts are low. At 15 samples (activation threshold), shrinkage = 0.50. At 63 samples (typical per-regime bin), shrinkage ~ 0.81. At 190 samples, shrinkage ~ 0.93. This prevents noisy overcorrection in sparse bins without requiring decay (which reduces effective sample size and amplifies noise).

Monotonic enforcement via the Pool Adjacent Violators (PAV) algorithm ensures that higher raw predictions always map to higher corrected predictions. When adjacent bin corrections would violate this ordering, PAV pools them to their average. This provides two benefits: it guarantees AUC cannot be damaged by histogram corrections, and it reduces variance by pooling noisy adjacent bins. Enabled by default via `histogram_post_calibration: true` and `histogram_monotonic: true` in the calibration config.

**File:** `em_sde/calibration.py` (`HistogramCalibrator`)

### Per-Horizon Histogram Settings

Histogram post-calibration supports per-horizon bin counts and prior strengths via config overrides (`histogram_n_bins_by_horizon`, `histogram_prior_strength_by_horizon`). At long horizons (H=20), events are sparser, so fewer bins and stronger priors reduce noise:

```yaml
histogram_n_bins_by_horizon:
  5: 15    # more bins for abundant H=5 data
  10: 10   # moderate
  20: 7    # fewer bins for sparse H=20
histogram_prior_strength_by_horizon:
  5: 5.0   # light shrinkage (many samples)
  10: 10.0
  20: 25.0 # strong shrinkage (few samples)
```

When `histogram_interpolate: true`, the correction between adjacent bins is linearly interpolated instead of piecewise-constant, reducing staircase artifacts at bin boundaries that inflate ECE.

### Options-Implied Volatility (`implied_vol_enabled`)

When enabled, the system blends options-implied volatility into the Monte Carlo simulation and optionally adds an implied vol ratio as a calibration feature. This captures forward-looking market expectations that historical GARCH cannot.

**Data input:** CSV with date index and implied vol columns. Two formats supported:
- **VIX format** (percentage points): columns `VIX`, `VIX9D`, `VIX3M` — auto-scaled to decimal
- **Decimal format**: columns `iv_9d`, `iv_30d`, `iv_3m` — used directly

**Horizon matching:**

| Horizon | Implied Vol Source | Fallback |
|---------|-------------------|----------|
| H <= 5  | `iv_9d` (9-day)    | `iv_30d` |
| H <= 10 | Interpolate `iv_9d` to `iv_30d` | `iv_30d` alone |
| H <= 22 | `iv_30d` (30-day)  | `iv_3m`  |
| H > 22  | `iv_3m` (3-month)  | `iv_30d` |

**Sigma blending:** The per-horizon sigma used for MC simulation is a weighted average:

```
sigma_blend = (1 - w) * sigma_hist + w * sigma_implied_daily
```

where `sigma_implied_daily = IV_annualized / sqrt(252)` and `w` = `implied_vol_blend` (default 0.3).

**Calibration feature:** When `implied_vol_as_feature: true`, the ratio `sigma_implied / sigma_hist` is added as an additional feature in the multi-feature calibrator. Values > 1 indicate the market expects more volatility than GARCH forecasts; values < 1 indicate less.

**Walk-forward safety:**
- Uses most recent available IV data on or before the current date
- Staleness guard: skips blending if IV data is more than 5 business days old
- Feature defaults to 1.0 (neutral) when no data available

**Config:**
```yaml
model:
  implied_vol_enabled: true
  implied_vol_csv_path: data/vix_daily.csv
  implied_vol_blend: 0.3        # 30% implied, 70% historical
  implied_vol_as_feature: true  # add ratio to MF calibrator
```

**File:** `em_sde/data_layer.py` (`load_implied_vol`, `get_implied_vol_for_horizon`), `em_sde/backtest.py` (blending + feature wiring)

### Scheduled Jump Variance (`scheduled_jump_variance`)

When enabled, the forecast distribution adds a residual scheduled-event variance term when an earnings date falls inside the prediction horizon:

$$\mathrm{Var}_{\mathrm{total}}(H) = \mathrm{Var}_{\mathrm{diffusive}}(H) + \mathrm{Var}_{\mathrm{scheduled}}(H)$$

`Var_scheduled` is estimated from trailing historical earnings events for that ticker. If there are too few past events, the term falls back to zero. This lets the generator, not just the calibrator, react to known scheduled jump risk.

### Hybrid Variance Blend (`hybrid_variance_enabled`)

When OHLC data is available, the engine can blend the physical sigma forecast with a 20-day Parkinson-style range anchor in variance space:

$$\sigma_{\mathrm{hybrid}}^{2} = (1-w)\sigma_{\mathrm{physical}}^{2} + w\sigma_{\mathrm{range}}^{2}$$

This is only active when the dataset actually contains `open/high/low`. On close-only CSVs, the hybrid path is a no-op by design.

### Where to look for problems

- **Cold start**: The first 50-100 predictions are uncalibrated. If the evaluation window is short, this dominates the metrics.
- **Learning rate**: Too high causes oscillation. Too low causes the calibrator to never catch up with regime changes. The adaptive decay helps but doesn't solve structural shifts.
- **Multi-feature overfitting**: With 6-8 features and online learning, the calibrator can chase noise. L2 regularization mitigates this, but the default `1e-4` may be too weak for very noisy data.
- **Gate hysteresis**: Once the safety gate or discrimination gate activates, there's no explicit re-entry logic — it re-evaluates every step. This can cause rapid on/off switching.
- **Histogram calibrator + Platt interaction**: The histogram post-pass learns from the Platt-scaled output at update time (not prediction time). If Platt parameters shift significantly between prediction and resolution, the histogram correction may lag. This is mitigated by adaptive lr decay on the Platt stage.
- **Implied vol data gaps**: If the IV CSV has gaps (holidays, missing data), the staleness guard skips blending for those days. Verify IV coverage matches the price data date range.

---

## 8. Step 7: Walk-Forward Backtest (No Lookahead)

**File:** `em_sde/backtest.py`

The backtest simulates what would happen if you ran this system in real-time, one day at a time, without ever looking at future data.

### The Loop

For each trading day `t` from warmup to the end of data:

```
1. RESOLVE: Check if any past predictions have matured.
   If prediction from day (t - H) is pending and day t has arrived:
     - Compute actual return over those H days
     - Label as event (1) or non-event (0)
     - Feed the label to the calibrator to learn from

2. FIT: Estimate today's volatility using only data up to day t.

3. PROJECT: If GARCH params are non-stationary, apply stationarity projection.

4. THRESHOLD: Compute what "large move" means for each horizon.

5. SIMULATE: Run Monte Carlo to get p_raw for each horizon.

6. CALIBRATE: Apply the calibrator to get p_cal for each horizon.

7. QUEUE: Store the prediction for future resolution (when the outcome is known).
```

### Resolution Queue

The key mechanism preventing lookahead. Each prediction goes into a queue:

```
PendingPrediction = {
    row_index,        (which output row)
    price_index,      (which date's price is the starting point)
    p_raw,            (the raw probability)
    threshold,        (what threshold was used)
}
```

The prediction stays in the queue until `H` trading days have passed. Only then is the actual return computed and the calibrator updated.

### Final Resolution Pass

After the main loop, a final pass resolves any remaining predictions whose horizons extend beyond the last date in the data.

### Where to look for problems

- **Overlapping predictions**: H=20 predictions overlap by 19 days. Outcomes are correlated. This inflates apparent sample size. The system computes `N_eff` to correct for this.
- **GARCH refitting cost**: GARCH is refit every single day on the full window. This is the main computational bottleneck.
- **Thread parallelism**: MC simulations for different horizons run in parallel threads. This is safe because each uses independent seeds.

---

## 9. Step 8: Evaluation Metrics

**File:** `em_sde/evaluation.py`

After the backtest, we measure how good the predictions were.

### Primary: Calibration Accuracy

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| **[Brier Score](https://en.wikipedia.org/wiki/Brier_score)** ([Brier, 1950](https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2)) | $\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2$ | Average squared error. Lower = better. Range [0, 1]. |
| **Brier Skill Score** | $\mathrm{BSS}=1-\frac{\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2}{\bar{y}(1-\bar{y})}$ | How much better than always predicting the event rate. BSS > 0 = useful. |
| **[Log Loss](https://en.wikipedia.org/wiki/Cross-entropy#Cross-entropy_loss_function_and_logistic_regression)** | $-\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]$ | Penalizes confident wrong predictions heavily. Lower = better. |
| **[ECE](https://en.wikipedia.org/wiki/Calibration_(statistics))** ([Naeini et al., 2015](https://people.cs.pitt.edu/~milos/research/2015/AAAI_Calibration.pdf)) | $\sum_{k=1}^{K}\frac{n_k}{N}\lvert\hat{p}_k-\hat{y}_k\rvert$ | Measures systematic over/under-confidence. Lower = better. |

### Secondary: Discrimination Quality

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| **[AUC-ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)** | $\int \mathrm{TPR}\,d(\mathrm{FPR})$ | Can the model rank events above non-events? 0.5 = random. 1.0 = perfect. |
| **Separation** | $\mathbb{E}[p\mid y=1]-\mathbb{E}[p\mid y=0]$ | Average probability gap between events and non-events. |

### Sample Size Correction

| Metric | Purpose |
|--------|---------|
| **N_eff** | Effective sample size correcting for autocorrelation in overlapping windows |
| **Non-overlapping evaluation** | Every H-th row only — independent samples, smaller but cleaner |

### Risk Analytics

| Metric | What it measures |
|--------|-----------------|
| VaR (95/99) | How bad is a typical bad day? |
| CVaR / Expected Shortfall | How bad is the average really-bad day? |
| Skewness | Is the return distribution tilted? (Negative = left-tail risk) |
| Excess Kurtosis | Are tails fatter than normal? (> 0 = fat tails) |
| Max Drawdown | Largest peak-to-trough decline |

### Where to look for problems

- **BSS near zero**: The model isn't beating the base rate. Check if the threshold mode is creating near-constant event rates (vol_scaled does this).
- **AUC near 0.5**: No discrimination. The model assigns similar probabilities to events and non-events. Check threshold mode and calibration warmup.
- **ECE > 0.02**: Systematic miscalibration. ECE uses adaptive (quantile-based) bins by default, which avoids pathological inflation when predictions cluster in a narrow range. If ECE is still high with adaptive bins, the calibrator genuinely needs more data or the signal is too noisy. Equal-width bins are available via `adaptive=False` for backward compatibility.
- **N_eff much smaller than N**: Heavy overlap autocorrelation. Non-overlapping metrics may tell a different story.

---

## 10. Step 9: Model Selection and Promotion

**File:** `em_sde/model_selection.py`

### Expanding-Window Cross-Validation

Uses [walk-forward validation](https://en.wikipedia.org/wiki/Walk_forward_optimization) (expanding window variant of [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))) to respect temporal ordering. Splits the data into folds:

```
Fold 1:  Train on [0% - 40%],  Test on [40% - 52%]
Fold 2:  Train on [0% - 52%],  Test on [52% - 64%]
Fold 3:  Train on [0% - 64%],  Test on [64% - 76%]
Fold 4:  Train on [0% - 76%],  Test on [76% - 88%]
Fold 5:  Train on [0% - 88%],  Test on [88% - 100%]
```

Each fold runs a full walk-forward backtest. Metrics are computed on the test portion only.

### Promotion Gates (U5)

The active governance path is **row-level pooled OOF gating**. CV folds are pooled at the row level, and when `promotion_pooled_gate: true` the pooled row becomes the primary pass/fail gate while per-regime slices remain diagnostic:

```
Pool all out-of-fold (OOF) row-level predictions across CV folds.
Assign each row to low / mid / high vol regime using its own sigma_1d tercile.

Primary point-metric gates:
  BSS  >= 0.00    (must beat climatology)
  AUC  >= 0.55    (must have some discrimination)
  ECE  <= 0.02    (must be reasonably calibrated)

Optional density gates:
  CRPS skill       >= 0.00
  PIT KS statistic <= 0.12
  Tail cov error   <= 0.05

Optional robustness gate:
  worst overfit status <= YELLOW

Minimum sample guards per regime:
  n_samples >= 100
  n_events  >= 30   (positive labels)
  n_nonevents >= 30 (negative labels)
Insufficiency reason is tracked: too_few_samples, too_few_events, too_few_nonevents.

With pooled mode enabled:
  PASS       — all regimes evaluated, all gates pass
  FAIL       — any evaluated gate fails
  UNDECIDED  — any regime has insufficient data for evaluation

Diagnostic regime rows do not block promotion.
```

Row-level regime assignment gives ~500-700 OOF rows per regime (vs 1-2 fold-level means in the legacy approach). Bootstrap confidence intervals for ECE are reported for defensibility.

**N_eff tracking (2026-03-10):** Every gate report row now includes `n_eff` (= min(events, nonevents) × 2), `neff_ratio` (= n_eff / n_bo_params), and `neff_warning` (GREEN >100x, YELLOW 50-100x, RED <50x). This makes statistical power visible per-horizon, per-regime.

**ECE confidence annotations:** When ECE is evaluated, bootstrap CI determines a confidence tag:

- `solid_pass`: ECE passes and upper CI bound is also below the gate threshold
- `fragile_pass`: ECE passes but upper CI bound exceeds the gate threshold (could flip with more data)
- `solid_fail`: ECE fails and lower CI bound is also above the gate threshold
- `fragile_fail`: ECE fails but lower CI bound is below the gate threshold (close to passing)

Implementation note:
- `python -m em_sde.run --compare ...` uses **row-level OOF** gates (`apply_promotion_gates_oof`) and respects `promotion_pooled_gate`.
- `scripts/run_gate_recheck.py` and `scripts/run_full_institutional.py` use the same pooled OOF governance path and can also enforce density / overfit gates.

### Pooled ECE Gate (`promotion_pooled_gate`)

Per-regime ECE evaluation with ~200 samples per bucket has estimation noise of ~0.02–0.03, which is the same magnitude as the 0.02 gate threshold. This makes per-regime ECE gates statistically underpowered.

When `promotion_pooled_gate: true`, the gate evaluator adds a **pooled** regime that includes ALL OOF samples for a given (config, horizon) pair (~600+ samples). The pooled evaluation becomes the **primary gate** for promotion decisions:

```text
With pooled_gate=True:
  1. Compute BSS, AUC, ECE, and any enabled density gates on ALL OOF samples (regime="pooled")
  2. Pooled rows have status="evaluated" → determine pass/fail
  3. Per-regime rows have status="diagnostic" → reported but not blocking
  4. Insufficient per-regime data does NOT trigger UNDECIDED

Tri-state decision (pooled mode):
  PASS  — all pooled gates pass
  FAIL  — any pooled gate fails
  (UNDECIDED only if pooled itself has insufficient data, which is rare)
```

This increases statistical power substantially: ECE estimation noise drops from ~0.02–0.03 (per-regime) to ~0.01 (pooled), making the 0.02 threshold meaningful. Per-regime breakdowns remain available as diagnostics for identifying regime-specific calibration issues.

### Where to look for problems

- **5 folds on 3200 rows**: Each test fold has ~384 rows. At H=20, only ~19 non-overlapping observations per fold. OOF pooling across folds gives ~1900 total rows, improving statistical power.
- **Expanding window**: Later folds have more training data and may perform better simply due to data quantity, not model quality.
- **Overlapping predictions**: OOF rows from adjacent folds may share overlapping prediction horizons. This inflates effective sample size. Always check bootstrap CIs, not just point estimates.

---

## 11. Mathematical Reference

### GARCH(1,1) Variance Equation

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

### GJR-GARCH(1,1) Variance Equation

$$\sigma_t^2 = \omega + \left(\alpha + \gamma I_{\epsilon_{t-1}<0}\right)\epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

### GBM Log-Price Discretization (Euler-Maruyama)

$$X_{t+1} = X_t + \left(\mu - \frac{1}{2}\sigma_{\mathrm{ann}}^2\right)\Delta t + \sigma_{\mathrm{ann}}\sqrt{\Delta t}\,Z_t,\quad Z_t \sim \mathcal{N}(0,1)$$

$$X_t = \log S_t,\quad \Delta t = \frac{1}{252}$$

### Merton Jump-Diffusion Drift Compensation

$$\mu_{\mathrm{drift}} = \mu_{\mathrm{ann}} - \lambda\left(e^{\mu_J + \frac{1}{2}\sigma_J^2} - 1\right) - \frac{1}{2}\sigma_{\mathrm{ann}}^2$$

### Stationarity Projection with Variance Targeting

Define persistence as:

$$\phi = \alpha + \beta \quad \text{(GARCH(1,1))}$$

$$\phi = \alpha + \beta + \frac{1}{2}\gamma \quad \text{(GJR-GARCH(1,1))}$$

If $\phi \ge 1$, project to target persistence $\phi_{\mathrm{target}}$ (default $0.98$):

$$s = \frac{\phi_{\mathrm{target}}}{\phi},\quad \alpha^{\prime} = s\alpha,\quad \beta^{\prime} = s\beta,\quad \gamma^{\prime} = s\gamma,\quad \omega^{\prime} = \sigma_{1d}^{2}\left(1-\phi_{\mathrm{target}}\right)$$

### Logistic Calibration (Platt Scaling)

$$p_{\mathrm{cal}} = \sigma\left(a + b\,\mathrm{logit}(p_{\mathrm{raw}})\right)$$

$$\sigma(x) = \frac{1}{1+e^{-x}},\quad \mathrm{logit}(p) = \log\left(\frac{p}{1-p}\right)$$

### Brier Skill Score

$$\mathrm{BSS} = 1 - \frac{\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2}{\bar{y}(1-\bar{y})}$$

### Expected Calibration Error (Adaptive Binning)

Let $B_1,\dots,B_K$ be adaptive (quantile) bins of predictions.

$$\mathrm{ECE} = \sum_{k=1}^{K}\frac{|B_k|}{N}\left|\frac{1}{|B_k|}\sum_{i\in B_k}p_i - \frac{1}{|B_k|}\sum_{i\in B_k}y_i\right|$$

Adaptive (quantile-based) bins are the default. Equal-width bins over [0, 1] are available via `adaptive=False`.

---

## 12. Known Limitations and Assumptions

1. **Many bundled datasets are still close-only.** OHLC-aware features and hybrid variance only activate when `open/high/low` are actually present. With close-only data, the engine falls back to close-based behavior.

2. **No transaction costs or market impact.** This is a probability engine, not a trading simulator.

3. **GARCH persistence bias.** GARCH persistence (~0.97) keeps sigma elevated after vol spikes, causing p_raw to be systematically biased upward. HAR-RV (when enabled) directly addresses this by using realized variance which mean-reverts at the correct speed and provides horizon-specific sigma forecasts. HMM regime detection can complement HAR-RV but its sigma blending is horizon-conflicted when used alone.

4. **Overlapping predictions are correlated.** A bad week shows up in H=5, H=10, and H=20 predictions simultaneously. Always check N_eff, not raw N.

5. **Calibration cold start.** The first ~100 predictions are essentially uncalibrated. Early backtest performance is unreliable.

6. **Jump parameters are fixed or linearly interpolated.** Real jump behavior is more complex (clustering, contagion). The Merton model is a first-order approximation.

7. **Monte Carlo variance.** At 100K paths, the standard error for p=0.05 is about 0.0007. At 1K paths (fast mode), it's 0.007. For close comparisons between models, use high path counts.

8. **Student-t tail calibration.** The degrees of freedom `t_df` is a config input, not estimated from data. Misspecification can bias tail probabilities.

---

## 13. Bayesian Hyperparameter Optimization

**File:** `scripts/run_bayesian_opt.py`

The system includes an [Optuna](https://optuna.org/)-based [Bayesian optimization](https://en.wikipedia.org/wiki/Bayesian_optimization) script using the [TPE (Tree-structured Parzen Estimator)](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html) sampler ([Bergstra et al., 2011](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html)) for jointly tuning hyperparameters.

### Search Space

**Lean mode (default, 3 params with frozen threshold panel)** — recommended for anti-overfitting:
- **Calibration**: `multi_feature_lr` [0.002, 0.05], `multi_feature_l2` [1e-5, 1e-2]
- **GARCH**: `garch_target_persistence` [0.95, 0.995]

If `lock_threshold_panel: false` or `--tune-thresholds` is explicitly used, the optimizer also tunes per-horizon thresholds `thr_5`, `thr_10`, `thr_20`.

**Full mode (`--full`, 7 params with frozen thresholds)** — adds:
- **HAR-RV**: `har_rv` (on/off), `har_rv_ridge` [0.001, 0.1], `har_rv_refit` [5, 63]
- **Student-t df**: `t_df_low` [5, 15], `t_df_mid` [3, 8], `t_df_high` [2.5, 6]

### Objective

Minimize **mean pooled ECE across all horizons** via 5-fold expanding-window CV. The optimizer uses Optuna's TPE (Tree-structured Parzen Estimator) sampler with SQLite-backed persistence for resume capability.

**N_eff soft penalty (2026-03-10):** After computing mean ECE, the objective checks the minimum N_eff/N_params ratio across all horizons. If below 100x (GREEN threshold), a quadratic penalty is added: `mean_ece += 0.01 × ((100 - ratio) / 100)²`. This steers BO toward threshold/parameter combinations with adequate statistical power, without hard-rejecting borderline trials.

### Usage

```bash
python scripts/run_bayesian_opt.py cluster --n-trials 15   # ~5 hours
python scripts/run_bayesian_opt.py cluster --show-best      # view results
python scripts/run_bayesian_opt.py cluster --apply           # write best to YAML
```

Each config (cluster, jump) is optimized independently since they have different data characteristics.

### 13.1 Overfitting Diagnostics

After BO optimization, run the overfitting diagnostic to assess whether results generalize:

```bash
python scripts/run_overfit_check.py cluster   # or jump, or both
```

This computes 5 metrics with GREEN/YELLOW/RED thresholds:

| Metric | What it measures | GREEN | YELLOW | RED |
|--------|-----------------|-------|--------|-----|
| Generalization gap | Train CV ECE vs full-data ECE | <25% | 25-50% | >50% |
| Cross-fold CV | ECE stability across 5 CV folds | <0.30 | 0.30-0.50 | >0.50 |
| Threshold sensitivity | Event rate change at threshold +/-10% | <25% | 25-50% | >50% |
| Early vs late folds | Temporal stability (expanding window) | <30% | 30-60% | >60% |
| N_eff / N_params | Sample size adequacy for tuned params | >100x | 50-100x | <50x |

Key insight: the **N_eff / N_params ratio** is the most fundamental constraint. With ~70-120 large-move events per horizon and 12-14 BO parameters, the ratio is only 10-20x, well below the 100x rule of thumb. This means BO has limited statistical budget and must be used conservatively.

**Direction-aware scoring (2026-03-10 fixes):**

- **Generalization gap**: Uses `max(gap_ratio, 0.0)` instead of `abs(gap_ratio)`. A negative gap means the model *improves* on full data (more training data helps) — this is expected and healthy, so it maps to GREEN rather than being falsely flagged.
- **Temporal stability**: Uses `max(raw_gap, 0.0)` so that late folds being *better* than early folds (model improves with more data) maps to GREEN. Only flags when late folds are worse, which indicates genuine degradation.

### 13.2 Adaptive Event-Rate Guard

**File:** `scripts/run_bayesian_opt.py`, `objective()` function

**Problem:** The BO optimizer can select thresholds that produce event rates far below the 5-20% target range. When this happens, calibration has too few positive samples to learn from, and BSS degrades because the base-rate estimator becomes competitive. For example, AAPL BO found thresholds producing 0.6% event rate at H=10 — only 24 events in 3,870 samples. A hardcoded 8% floor rejected all AAPL trials even with tight threshold ranges, because different tickers and dataset sizes need different minimums.

**Solution:** After CV completes in each trial, the objective function computes the minimum event rate needed for N_eff/N_params >= 100x (GREEN zone), adapting to the actual dataset size and number of BO parameters:

```python
n_params = max(len(trial.params), 1)
n_oof = len(oof_df)
min_er_target = max((100 * n_params) / (2 * n_oof), 0.03)

er_by_horizon = cv_results.groupby("horizon")["event_rate"].mean()
if float(er_by_horizon.min()) < min_er_target:
    return 1.0  # reject trial
```

**The formula:** `min_rate = max((100 × n_params) / (2 × n_oof_samples), 3%)`. This derives from:
- N_eff ≈ event_rate × n_samples × 2 (when events are the minority class)
- Require N_eff / n_params >= 100 (GREEN zone)
- Solving: event_rate >= (100 × n_params) / (2 × n_samples)
- Floor at 3% absolute minimum (below this, calibration is meaningless regardless of dataset size)

**Examples by dataset size (legacy 6-parameter threshold-tuning case):**

| Dataset | OOF samples | Adaptive min ER | Events at min | N_eff/N_params |
|---------|-------------|-----------------|---------------|----------------|
| SPY     | ~6,538      | 4.6%            | ~301          | 100x           |
| AAPL    | ~5,230      | 5.7%            | ~298          | 100x           |
| GOOGL   | ~4,300      | 7.0%            | ~301          | 100x           |
| Small   | ~3,000      | 10.0%           | ~300          | 100x           |

**Why adaptive, not fixed:** A hardcoded threshold (e.g., 8%) is too strict for large datasets and too lenient for small ones. The adaptive guard ensures every ticker gets exactly the statistical power its data can support — no more, no less. This is not overfitting: the guard depends only on dataset size (a fixed property), not on model performance. Based on [Vittinghoff & McCulloch (2007)](https://doi.org/10.1002/sim.2691): minimum 20 events per tuned parameter.

**Per-regime gate minimums** (in `apply_promotion_gates_oof`): min_samples=100, min_events=30, min_nonevents=30. These ensure per-regime diagnostic metrics are meaningful (previous defaults of 30/5/5 were too lenient for institutional-grade conclusions).

**Diagnostic output:** Rejected trials log which horizons failed and their event rates, and store `rejected_reason` and `min_event_rate` as Optuna user attributes for post-hoc analysis.

---

## 14. Statistical Rigor (2026-04-04)

### 14.1 Residual-Based N_eff

The effective sample size calculation now computes autocorrelation on prediction residuals `r_t = p_cal(t) - y_t` instead of binary labels. Binary labels (0/1) have artificially low autocorrelation because the discrete distribution masks temporal dependence. Residuals are continuous and capture the actual overlap-induced correlation structure. This reduces N_eff estimates by ~30-40% compared to the old label-based method.

**File:** `em_sde/evaluation.py`, `effective_sample_size()` — accepts optional `p_cal` parameter.

### 14.2 Bootstrap Confidence Intervals

BCa (bias-corrected accelerated) bootstrap 95% CIs are computed for BSS, AUC, and ECE using `bootstrap_metric_ci()`. The BCa method uses jackknife acceleration to correct for skewness and bias in the bootstrap distribution, providing better coverage than percentile-based CIs especially with small samples.

**File:** `em_sde/evaluation.py`, `bootstrap_metric_ci()`.

### 14.3 Benjamini-Hochberg FDR Correction

With 12+ ticker-horizon tests (or 21 ablation tests), raw p-values are inflated by multiplicity. The Benjamini-Hochberg procedure controls the false discovery rate at alpha=0.05. All paper tables report both raw and FDR-adjusted p-values.

**File:** `em_sde/evaluation.py`, `apply_fdr_correction()`.

### 14.4 Per-Bin ECE Counts

`expected_calibration_error_detailed()` returns bin-level sample counts alongside the ECE value. Paper tables report `min_bin_n` so reviewers can assess whether ECE is driven by well-populated or sparse bins.

### 14.5 Gradient Boosting Baseline

A 5th baseline using `HistGradientBoostingClassifier` (sklearn) with walk-forward expanding-window training. Features: vol_20d, delta_sigma, vol_ratio, vol_of_vol, ret_5d. Refit every 63 days. Tests whether flexible ML on raw features can match the MC+calibration stack.

**File:** `scripts/baselines.py`, `gradient_boosting_baseline()`.

---

## 15. Live Prediction System (2026-04-04)

### 15.1 State Checkpointing

At the end of a walk-forward run, the system serializes:
- **Calibrator states** (weight vectors, histogram corrections, learning rates) → `calibrators.json`
- **GARCH parameters** (omega, alpha, beta, gamma, last sigma) → `garch_state.json`
- **Metadata** (thresholds, config, ticker) → `metadata.json`

Stored in `outputs/state/{ticker}/`. Files: `em_sde/calibration.py` (export_state/from_state), `em_sde/garch.py` (export_state/from_state), `em_sde/backtest.py` (saves to result_df.attrs).

### 15.2 Prediction Engine

`em_sde/predict.py` provides `PredictionEngine`:
- `from_checkpoint(state_dir)` loads saved state
- `predict(prices, horizons, n_paths, seed)` generates calibrated probabilities in seconds
- Fits GARCH on full history (warm-started), runs MC simulation, applies saved calibrators
- Returns `Dict[int, PredictionResult]` with p_cal, p_raw, sigma_1d, threshold, metadata

### 15.3 Asynchronous Label Resolution

`em_sde/resolve.py` tracks pending predictions and resolves outcomes:
- `append_prediction()` logs a prediction to CSV
- `resolve_predictions()` checks which past predictions can be resolved (t + H has passed)

### 15.4 CLI --predict-now Mode

`em_sde/run.py` supports `--predict-now --config <path> [--state-dir <path>]`:
- With state-dir: loads checkpoint, predicts immediately (~seconds)
- Without state-dir: runs full backtest first, saves state, then predicts

### 15.5 Daily Scheduling

`scripts/daily_predict.py` runs predictions for all configured tickers:
1. Loads/builds PredictionEngine per ticker
2. Logs predictions to CSV audit trail
3. Resolves past predictions with outcomes
4. Outputs JSON summary

Schedule via Windows Task Scheduler (4:30 PM ET daily) or Unix cron.

---

## 16. File Map

| File | What it does |
|------|-------------|
| `em_sde/data_layer.py` | Loads prices, implied vol data, earnings dates, caches, validates, runs quality checks |
| `em_sde/garch.py` | Fits GARCH/GJR, EWMA fallback, stationarity projection, state export/import |
| `em_sde/monte_carlo.py` | Simulates price paths (GBM, GARCH-in-sim, jumps), computes p_raw |
| `em_sde/calibration.py` | Online/multi-feature/regime-MF calibrators, histogram post-calibration, state serialization |
| `em_sde/backtest.py` | Walk-forward loop, resolution queues, threshold routing, implied vol blending, state checkpointing |
| `em_sde/evaluation.py` | Brier, BSS, AUC, ECE, N_eff (residual-based), bootstrap CIs, FDR correction, paired bootstrap |
| `em_sde/model_selection.py` | Cross-validation, model comparison, promotion gates, density gates, benchmark/conditional reports |
| `em_sde/predict.py` | Live prediction engine with checkpoint load/save |
| `em_sde/resolve.py` | Asynchronous label resolution for pending predictions |
| `em_sde/config.py` | YAML config loading and validation |
| `em_sde/output.py` | CSV/JSON output, chart generation |
| `em_sde/run.py` | CLI entry point (--predict-now, --save-state modes) |
| `scripts/run_bayesian_opt.py` | Optuna Bayesian optimization for hyperparameter search |
| `scripts/run_gate_recheck.py` | Re-run CV gates for diagnostic evaluation |
| `scripts/run_overfit_check.py` | Overfitting diagnostics (5 metrics, GREEN/YELLOW/RED) |
| `scripts/baselines.py` | Five baseline models (hist freq, GARCH-CDF, IV-BS, feature logistic, gradient boosting) |
| `scripts/daily_predict.py` | Daily prediction runner with scheduling support |
| `tests/test_framework.py` | 312 unit tests |

---

## 17. References

| Method | Paper | Link |
|--------|-------|------|
| GARCH(1,1) | Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity." *J. Econometrics* 31(3), 307-327. | [DOI](https://doi.org/10.1016/0304-4076(86)90063-1) |
| GJR-GARCH | Glosten, L., Jagannathan, R. & Runkle, D. (1993). "On the relation between the expected value and the volatility of the nominal excess return on stocks." *J. Finance* 48(5), 1779-1801. | [DOI](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x) |
| HAR-RV | Corsi, F. (2009). "A simple approximate long-memory model of realized volatility." *J. Financial Econometrics* 7(2), 174-196. | [DOI](https://doi.org/10.1093/jjfinec/nbp001) |
| Jump-diffusion | Merton, R. (1976). "Option pricing when underlying stock returns are discontinuous." *J. Financial Economics* 3(1-2), 125-144. | [DOI](https://doi.org/10.1016/0304-405X(76)90022-2) |
| Platt scaling | Platt, J. (1999). "Probabilistic outputs for support vector machines." *Advances in Large Margin Classifiers*, 61-74. | [Paper](https://www.researchgate.net/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines) |
| ECE | Naeini, M.P., Cooper, G.F. & Hauskrecht, M. (2015). "Obtaining well calibrated probabilities using Bayesian binning." *AAAI 2015*. | [Paper](https://people.cs.pitt.edu/~milos/research/2015/AAAI_Calibration.pdf) |
| Brier score | Brier, G.W. (1950). "Verification of forecasts expressed in terms of probability." *Monthly Weather Review* 78(1), 1-3. | [DOI](https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2) |
| TPE sampler | Bergstra, J., Bardenet, R., Bengio, Y. & Kégl, B. (2011). "Algorithms for hyper-parameter optimization." *NeurIPS 2011*. | [Paper](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html) |
| Events per parameter | Vittinghoff, E. & McCulloch, C.E. (2007). "Relaxing the rule of ten events per variable in logistic and Cox regression." *Am J Epidemiology* 165(6), 710-718. | [DOI](https://doi.org/10.1002/sim.2691) |
| Isotonic regression (PAV) | Barlow, R.E. et al. (1972). *Statistical Inference Under Order Restrictions*. Wiley. | [Wikipedia](https://en.wikipedia.org/wiki/Isotonic_regression) |
| Filtered Historical Sim | Barone-Adesi, G., Giannopoulos, K. & Vosper, L. (1999). "VaR without correlations for portfolios of derivative securities." *J. Futures Markets* 19(5), 583-602. | [DOI](https://doi.org/10.1016/S0378-4266(98)00091-4) |
| Model averaging | Ranjan, R. & Gneiting, T. (2010). "Combining probability forecasts." *JRSS-B* 72(1), 71-91. | [DOI](https://doi.org/10.1111/j.1467-9868.2009.00726.x) |
| EGARCH | Nelson, D.B. (1991). "Conditional heteroskedasticity in asset returns." *Econometrica* 59(2), 347-370. | [DOI](https://doi.org/10.2307/2938260) |
| Earnings and tails | Dubinsky, A., Johannes, M., Kaeck, A. & Seeger, N. (2019). "Option pricing of earnings announcement risks." *RFS* 32(2), 646-687. | [DOI](https://doi.org/10.1093/rfs/hhy018) |
| Earnings and risk | Savor, P. & Wilson, M. (2016). "Earnings announcements and systematic risk." *J. Finance* 71(1), 83-138. | [DOI](https://doi.org/10.1111/jofi.12351) |
