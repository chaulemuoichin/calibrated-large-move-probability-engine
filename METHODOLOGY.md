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
13. [File Map](#13-file-map)

---

## 1. Pipeline Overview

```
Raw Prices
  |
  v
Daily Returns  ──>  GARCH/GJR-GARCH fit  ──>  sigma_1d (today's vol forecast)
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

Daily closing prices for a single ticker. Three sources are supported:

| Source | Description |
|--------|-------------|
| `yfinance` | Downloads from Yahoo Finance with retry logic and local Parquet cache |
| `csv` | Reads a local CSV file (auto-detects column names like `Close`, `Price`, `Adj Close`) |
| `synthetic` | Generates fake GBM prices for testing |

### What comes out

A DataFrame with a `price` column (positive floats) indexed by date. Minimum 756 rows (~3 years of trading days).

### Quality checks (warnings, not blocking)

| Check | What it catches |
|-------|----------------|
| Outlier detection | Returns beyond 5x the interquartile range |
| Stale prices | 5+ consecutive identical closes (data feed issue) |
| Data gaps | More than 5 missing business days in a row |
| Return statistics | Reports skewness, kurtosis, min/max for manual review |

### Where to look for problems

- If the ticker has stock splits that Yahoo didn't adjust, volatility estimates will be wrong.
- If `min_rows` is set too low, GARCH fitting may be unstable.
- The cache has a 7-day expiration. Stale cache during fast-moving markets could matter.

---

## 3. Step 2: Volatility Estimation (GARCH)

**File:** `em_sde/garch.py`

The goal is to answer: **how volatile is this stock right now?** The answer is a single number, `sigma_1d`, the estimated daily volatility in decimal (e.g., 0.015 = 1.5% per day).

### GARCH(1,1)

The standard model. Volatility tomorrow depends on today's volatility and today's surprise:

$$\sigma_{t+1}^{2} = \omega + \alpha \epsilon_t^{2} + \beta \sigma_t^{2}$$

- **omega**: a small constant floor (prevents vol from collapsing to zero)
- **alpha**: how much today's surprise matters (higher = more reactive)
- **beta**: how much yesterday's vol persists (higher = slower to change)
- **shock**: the unexpected part of today's return

The model is **stationary** (well-behaved long-term) when `alpha + beta < 1`. In that case, volatility eventually settles to a long-run level:

$$\sigma_{\mathrm{long\_run}} = \sqrt{\frac{\omega}{1-\alpha-\beta}}$$

### GJR-GARCH (Leverage Effect)

An extension that lets negative surprises (drops) have more impact than positive ones:

$$\sigma_{t+1}^{2} = \omega + \left(\alpha + \gamma I_{\epsilon_t<0}\right)\epsilon_t^2 + \beta \sigma_t^2$$

where `I[drop]` is 1 when the shock was negative. This captures the empirical fact that markets react more to bad news than good news.

Stationarity condition: $\alpha + \beta + \gamma/2 < 1$.

### EWMA Fallback

If GARCH fitting fails (optimizer doesn't converge), the system falls back to Exponentially Weighted Moving Average volatility with a 252-day span:

$$\sigma_t^2 = \sum_{j=1}^{m} w_j r_{t-j}^2,\quad \sum_{j=1}^{m} w_j = 1$$

This always produces a number but loses the shock-response dynamics of GARCH.

### Fitting Procedure

1. Take the most recent 756 daily returns (configurable)
2. Fit GARCH or GJR-GARCH using the `arch` library
3. Extract the one-step-ahead volatility forecast (`sigma_1d`)
4. Extract the fitted parameters (`omega`, `alpha`, `beta`, `gamma`)
5. Compute diagnostics (persistence, stationarity, half-life)

### Stationarity Projection (U4)

**Problem:** Sometimes the fitted parameters have `alpha + beta >= 1`, meaning volatility would explode over time in simulation.

**Fix:** Scale all the shock-response parameters down proportionally to hit a target persistence (default 0.98), and recompute `omega` so the long-run variance equals the current forecast:

$$s = \frac{0.98}{\phi_{\mathrm{current}}},\quad \alpha_{\mathrm{new}} = s\alpha,\quad \beta_{\mathrm{new}} = s\beta,\quad \omega_{\mathrm{new}} = \sigma_{1d}^2(1-0.98)$$

This preserves the relative importance of each parameter while anchoring the simulation to current volatility.

### Where to look for problems

- If `alpha + beta` is frequently >= 1, the underlying data may have structural breaks.
- If `omega_new` becomes very small, the GARCH simulation becomes essentially a martingale — volatility doesn't mean-revert.
- The EWMA fallback loses all shock-asymmetry information. When it triggers often, GJR features are wasted.

---

## 4. Step 3: Defining "Large Move" (Threshold)

**File:** `em_sde/backtest.py`

Before we can estimate the probability of a "large move," we need to define what that means. The threshold is the minimum absolute return that counts as a large move.

### Four Threshold Modes

| Mode | Formula | When to use |
|------|---------|-------------|
| `fixed_pct` | $\tau = c$ (e.g., $c=5\%$) | Best for genuine discrimination — the goalpost doesn't move |
| `vol_scaled` | $\tau = k\,\sigma_{1d}\sqrt{H}$ | Legacy mode — threshold moves with vol, making AUC ~0.50 |
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

$$
\mathrm{event}_t^{(H)} =
\begin{cases}
1,& \left|\frac{P_{t+H}}{P_t}-1\right|\ge \tau \\
0,& \text{otherwise}
\end{cases}
$$

This is **two-sided**: both large up-moves and large down-moves count.

### Where to look for problems

- `vol_scaled` makes the threshold track current vol, so high-vol periods automatically get higher thresholds. This is circular — the model is essentially asking "will vol stay high?" rather than "will a large move happen?"
- `fixed_pct` can produce very different event rates across regimes. A 5% threshold might fire 30% of the time during a crash but 1% during calm markets. The calibrator has to handle this imbalance.
- `regime_gated` has a 252-day warmup. During warmup it uses the mid-vol mode, which might not be optimal.

---

## 5. Step 4: Simulating Future Prices (Monte Carlo)

**File:** `em_sde/monte_carlo.py`

We generate thousands of possible future price paths (default: 100,000) and look at where they end up. Each path is one possible future.

### Constant-Volatility GBM

The simplest model. Each daily step:

$$\log P_{t+1} = \log P_t + \left(\mu-\frac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\,Z_t$$

where:
- `drift` = annualized expected return (default 0)
- `vol` = annualized volatility = sigma_1d * sqrt(252)
- `dt` = 1/252 (one trading day)
- `Z` = random standard normal draw

After `H` steps, convert back: `price_terminal = exp(log_price_final)`.

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

Adds rare, sudden jumps on top of the diffusion:

On any given day, there is a small probability of a jump:

$$N_{\mathrm{jump}} \sim \mathrm{Poisson}(\lambda/252)$$

If a jump happens:

$$J_t \sim \mathcal{N}(\mu_J,\sigma_J),\quad \log P_t \leftarrow \log P_t + J_t$$

The drift is compensated so that adding jumps doesn't change the expected return:

$$\text{drift\_adjustment} = \lambda\left(e^{\mu_J + \frac{1}{2}\sigma_J^2} - 1\right)$$

### State-Dependent Jumps (U2)

Jump parameters vary with the current volatility regime:

$$t = \mathrm{clip}\!\left(\frac{\sigma_{1d}-\sigma_{25\%}}{\sigma_{75\%}-\sigma_{25\%}},\,0,\,1\right)$$

$$\lambda_t = \lambda_{\mathrm{low}} + t\left(\lambda_{\mathrm{high}}-\lambda_{\mathrm{low}}\right)$$
$$\mu_{J,t} = \mu_{J,\mathrm{low}} + t\left(\mu_{J,\mathrm{high}}-\mu_{J,\mathrm{low}}\right)$$
$$\sigma_{J,t} = \sigma_{J,\mathrm{low}} + t\left(\sigma_{J,\mathrm{high}}-\sigma_{J,\mathrm{low}}\right)$$

In calm markets (low `t`): fewer, smaller jumps. In stressed markets (high `t`): more frequent, larger jumps.

### Fat Tails (Student-t)

Instead of drawing `Z` from a standard normal, draw from a Student-t distribution and rescale to unit variance:

$$Z_{\mathrm{fat}} = Z_t\sqrt{\frac{\nu-2}{\nu}},\quad Z_t \sim t_{\nu}$$

This produces more extreme daily moves than a normal distribution, matching empirical return distributions better.

### Where to look for problems

- **Path count**: 100K paths give Monte Carlo standard error ~0.001 for p=0.05. Reducing to 1K (for speed) gives SE ~0.007 — still usable for ranking but noisy for absolute levels.
- **GARCH-in-sim with non-stationary params**: If stationarity projection fails or is disabled, volatility can explode inside paths. The variance floor (`1e-12`) prevents NaN but produces unrealistic paths.
- **Jump compensator**: If `mu_J` is very negative and `lambda` is high, the compensator is large and the diffusion drift becomes strongly positive, which may look odd.
- **dt = 1/252 assumption**: Ignores weekends, holidays, overnight gaps. These matter more for short horizons (H=5).

---

## 6. Step 5: Computing Raw Probability

**File:** `em_sde/monte_carlo.py` (`compute_move_probability`)

After simulation, we have terminal prices for each path. The raw probability is simply:

```
simulated_returns = terminal_prices / starting_price - 1
p_raw = count(|simulated_returns| >= threshold) / n_paths
```

With Monte Carlo standard error:

```
SE = sqrt(p_raw * (1 - p_raw) / n_paths)
```

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

The simplest calibrator maps raw to calibrated via a logistic function:

```
p_cal = sigmoid(a + b * logit(p_raw))
```

where `sigmoid(x) = 1/(1+exp(-x))` and `logit(p) = log(p/(1-p))`.

**Initialization:** `a=0, b=1` (identity — calibrated = raw).

**Online update** (when an outcome `y` arrives):

```
error = y - p_cal
a += lr * error
b += lr * error * logit(p_raw)
```

Learning rate decays over time: `lr_effective = lr / sqrt(1 + n_updates)`.

The calibrator only starts adjusting after `min_updates` (default 50) outcomes have arrived.

### Multi-Feature Calibrator

Uses 6 features instead of just `logit(p_raw)`:

```
features = [
  1.0,                          (intercept)
  logit(p_raw),                 (MC probability in log-odds)
  sigma_1d * 100,               (current vol level)
  delta_sigma_20d * 100,        (vol change over 20 days)
  realized_vol / sigma_1d,      (forecast error ratio)
  vol_of_vol * 100,             (instability of vol itself)
]

p_cal = sigmoid(weights^T @ features)
```

Updated via SGD with L2 regularization (prevents overfitting) and gradient clipping (prevents explosions). Requires 100 outcomes before activating.

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

After the Platt/logistic mapping, an optional second pass corrects residual bin-level bias:

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

Bayesian shrinkage dampens the correction when bin sample counts are low. At 15 samples (activation threshold), shrinkage = 0.50. At 63 samples (typical per-regime bin), shrinkage ≈ 0.81. At 190 samples, shrinkage ≈ 0.93. This prevents noisy overcorrection in sparse bins without requiring decay (which reduces effective sample size and amplifies noise).

Monotonic enforcement via the Pool Adjacent Violators (PAV) algorithm ensures that higher raw predictions always map to higher corrected predictions. When adjacent bin corrections would violate this ordering, PAV pools them to their average. This provides two benefits: it guarantees AUC cannot be damaged by histogram corrections, and it reduces variance by pooling noisy adjacent bins. Enabled by default via `histogram_post_calibration: true` and `histogram_monotonic: true` in the calibration config.

**File:** `em_sde/calibration.py` (`HistogramCalibrator`)

### Where to look for problems

- **Cold start**: The first 50-100 predictions are uncalibrated. If the evaluation window is short, this dominates the metrics.
- **Learning rate**: Too high causes oscillation. Too low causes the calibrator to never catch up with regime changes. The adaptive decay helps but doesn't solve structural shifts.
- **Multi-feature overfitting**: With 6 features and online learning, the calibrator can chase noise. L2 regularization mitigates this, but the default `1e-4` may be too weak for very noisy data.
- **Gate hysteresis**: Once the safety gate or discrimination gate activates, there's no explicit re-entry logic — it re-evaluates every step. This can cause rapid on/off switching.
- **Histogram calibrator + Platt interaction**: The histogram post-pass learns from the Platt-scaled output at update time (not prediction time). If Platt parameters shift significantly between prediction and resolution, the histogram correction may lag. This is mitigated by adaptive lr decay on the Platt stage.

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
| **Brier Score** | $\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2$ | Average squared error. Lower = better. Range [0, 1]. |
| **Brier Skill Score** | $\mathrm{BSS}=1-\frac{\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2}{\bar{y}(1-\bar{y})}$ | How much better than always predicting the event rate. BSS > 0 = useful. |
| **Log Loss** | $-\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]$ | Penalizes confident wrong predictions heavily. Lower = better. |
| **ECE** | $\sum_{k=1}^{K}\frac{n_k}{N}\lvert\hat{p}_k-\hat{y}_k\rvert$ | Measures systematic over/under-confidence. Lower = better. |

### Secondary: Discrimination Quality

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| **AUC-ROC** | $\int \mathrm{TPR}\,d(\mathrm{FPR})$ | Can the model rank events above non-events? 0.5 = random. 1.0 = perfect. |
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

Splits the data into folds:

```
Fold 1:  Train on [0% - 40%],  Test on [40% - 52%]
Fold 2:  Train on [0% - 52%],  Test on [52% - 64%]
Fold 3:  Train on [0% - 64%],  Test on [64% - 76%]
Fold 4:  Train on [0% - 76%],  Test on [76% - 88%]
Fold 5:  Train on [0% - 88%],  Test on [88% - 100%]
```

Each fold runs a full walk-forward backtest. Metrics are computed on the test portion only.

### Promotion Gates (U5)

Hard go/no-go criteria applied per volatility regime bucket:

```
Pool all out-of-fold (OOF) row-level predictions across CV folds.
Assign each row to low / mid / high vol regime using its own sigma_1d tercile.

In EACH regime bucket, the model must pass ALL of:
  BSS  >= 0.00    (must beat climatology)
  AUC  >= 0.55    (must have some discrimination)
  ECE  <= 0.02    (must be reasonably calibrated)

Minimum sample guards: 30 rows and 5 events per regime required.
Regimes with insufficient data are skipped (not counted as pass or fail).

If any gate fails in any evaluated bucket: model is BLOCKED from promotion.
```

Row-level regime assignment gives ~500-700 OOF rows per regime (vs 1-2 fold-level means in the legacy approach). Bootstrap confidence intervals for ECE are reported for defensibility.

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

1. **Close-to-close returns only.** No intraday path — a stock could breach the threshold intraday and close within it, and the system would label it as a non-event.

2. **No transaction costs or market impact.** This is a probability engine, not a trading simulator.

3. **GARCH assumes a single volatility regime.** Structural breaks (e.g., 2020 COVID crash) can cause parameter instability. The stationarity projection mitigates but doesn't eliminate this.

4. **Overlapping predictions are correlated.** A bad week shows up in H=5, H=10, and H=20 predictions simultaneously. Always check N_eff, not raw N.

5. **Calibration cold start.** The first ~100 predictions are essentially uncalibrated. Early backtest performance is unreliable.

6. **Jump parameters are fixed or linearly interpolated.** Real jump behavior is more complex (clustering, contagion). The Merton model is a first-order approximation.

7. **Monte Carlo variance.** At 100K paths, the standard error for p=0.05 is about 0.0007. At 1K paths (fast mode), it's 0.007. For close comparisons between models, use high path counts.

8. **Student-t tail calibration.** The degrees of freedom `t_df` is a config input, not estimated from data. Misspecification can bias tail probabilities.

---

## 13. File Map

| File | What it does |
|------|-------------|
| `em_sde/data_layer.py` | Loads prices, caches, validates, runs quality checks |
| `em_sde/garch.py` | Fits GARCH/GJR, EWMA fallback, stationarity projection |
| `em_sde/monte_carlo.py` | Simulates price paths (GBM, GARCH-in-sim, jumps), computes p_raw |
| `em_sde/calibration.py` | Online/multi-feature/regime calibrators, histogram post-calibration, safety gates |
| `em_sde/backtest.py` | Walk-forward loop, resolution queues, threshold routing |
| `em_sde/evaluation.py` | Brier, BSS, AUC, ECE, VaR, CRPS, all scoring metrics |
| `em_sde/model_selection.py` | Cross-validation, model comparison, promotion gates |
| `em_sde/config.py` | YAML config loading and validation |
| `em_sde/output.py` | CSV/JSON output, chart generation |
| `em_sde/run.py` | CLI entry point |
| `tests/test_framework.py` | 157 unit tests |
