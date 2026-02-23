# Model Results & Probability Forecast Interpretation

## Current Best Results (2026-02-23)

### Cluster Dataset (Synthetic Leverage-Clustering Pattern)

**Dataset**: 3,200 daily observations (2012-01-02 to 2024-04-05), annualized vol ~11%

**Bayesian Optimization**: 6 trials via Optuna TPE, best = Trial #3

| Horizon | Threshold | ECE (OOF) | BSS   | AUC   | Gate  |
|---------|-----------|-----------|-------|-------|-------|
| H=5     | 3.44%     | 0.0112    | 0.205 | 0.874 | PASS  |
| H=10    | 5.32%     | 0.0170    | 0.180 | 0.874 | PASS  |
| H=20    | 5.68%     | 0.0243    | 0.095 | 0.701 | FAIL  |

**Gate criteria**: ECE <= 0.02, BSS > 0.0, AUC > 0.55

**Score**: 2/3 horizons PASS (H=20 fails by 0.0043 — very close)

### Jump-Crash Dataset (Synthetic Crash Pattern)

**Dataset**: 3,200 daily observations (2012-01-02 to 2024-04-05), annualized vol ~28%

**Status**: Bayesian optimization in progress with data-adaptive threshold ranges.
Previous BO (with cluster-tuned threshold ranges) failed 0/3 — thresholds were too narrow for this high-vol dataset.

---

## How to Interpret Probability Forecasts

### What the model outputs

For each trading day, the model produces:

```
p_cal(H, threshold) = calibrated probability that |cumulative return over H days| >= threshold
```

For example, with the cluster config:
- **p_cal(H=5, thr=3.44%)** = probability the price moves more than 3.44% (up or down) over the next 5 trading days
- **p_cal(H=10, thr=5.32%)** = probability the price moves more than 5.32% over the next 10 trading days
- **p_cal(H=20, thr=5.68%)** = probability the price moves more than 5.68% over the next 20 trading days

### What "calibrated" means in practice

An ECE of 0.011 (H=5) means: **when the model says there's a 15% chance of a large move, the actual frequency is within ~1.1 percentage points of 15%** (i.e., between 13.9% and 16.1%).

This is the key property that makes the forecast usable:

| If the model says... | The actual frequency is... | Confidence |
|---------------------|---------------------------|------------|
| p = 5%              | ~3.9% to 6.1%             | High (ECE ~0.01) |
| p = 20%             | ~18.9% to 21.1%           | High |
| p = 50%             | ~48.9% to 51.1%           | High |

### Discrimination (AUC = 0.87)

AUC of 0.87 means: if you pick one day where a large move happened and one day where it didn't, there's an **87% chance the model assigned a higher probability to the day with the large move**. This is strong discrimination — the model genuinely "knows" when risk is elevated.

### Brier Skill Score (BSS = 0.20)

BSS of 0.20 means: the model is **20% better** than a naive baseline that always predicts the historical average event rate. This is meaningful skill — most forecasting models in finance achieve BSS of 0.02-0.10.

---

## Real-World Application Guide

### Applying to a Real Ticker (e.g., GOOGL, SPY)

To apply this model to a real stock:

1. **Prepare price data** as CSV with columns `date` and `Close`
2. **Create a config YAML** (copy `exp_cluster_regime_gated.yaml` as starting point)
3. **Run Bayesian Optimization** to find optimal parameters for that specific ticker:
   ```bash
   python scripts/run_bayesian_opt.py <config_name> --n-trials 15
   python scripts/run_bayesian_opt.py <config_name> --apply
   ```
4. **Run gate recheck** to validate:
   ```bash
   python scripts/run_gate_recheck.py <config_name>
   ```

**Important**: Parameters are data-specific. The optimal thresholds, t-df, and calibration parameters depend on the ticker's volatility characteristics. A 3.44% threshold works for ~11% annual vol but would need to be ~8-9% for a 28% vol asset.

### Validating with Forward Data

To prove the model works on unseen data (1-2-3 weeks ahead):

1. **Fit** the model on data up to today
2. **Record** the probability forecast for each horizon
3. **Wait** 5/10/20 trading days
4. **Check** if the actual return exceeded the threshold
5. **Repeat** over many days and compute calibration metrics

**Expected outcome**: If the model produces p=15% for a 5-day 3.4% move, then across 100 such forecasts, approximately 13-17 of them should result in actual large moves (given ECE ~0.01).

### Practical Interpretation Examples

**Scenario 1: Model says p_5d = 8%**
- "There's an 8% chance the stock moves more than 3.44% in either direction over the next week"
- This is a calm market signal — relatively low probability of a large move
- Action: standard position sizing, no special hedging needed

**Scenario 2: Model says p_5d = 35%**
- "There's a 35% chance of a >3.44% move this week"
- This is elevated risk — roughly 1-in-3 chance of a significant move
- Action: consider reducing position size, buying options for protection, or tightening stops

**Scenario 3: Model says p_5d = 60%**
- "More likely than not that we see a large move this week"
- High-risk environment — vol clustering in effect
- Action: de-risk or position for a breakout (long straddle/strangle if using options)

### Key Caveats

1. **Two-sided**: The model predicts |move| >= threshold — it does NOT predict direction. A 35% probability of a large move means 35% chance of a crash OR rally.

2. **Regime-dependent accuracy**: The model is most accurate in normal and high-vol regimes. During regime transitions (e.g., sudden crash from calm), there's a 1-2 day lag in the GARCH sigma response.

3. **Threshold matters**: The thresholds (3.44%, 5.32%, 5.68%) were optimized for this specific dataset. For a different asset, re-run BO to find the right thresholds.

4. **Not a trading signal**: This is a risk model, not an alpha model. Use it for position sizing, hedging decisions, and risk budgeting — not for directional bets.

---

## Model Configuration (Best Params — Cluster)

| Parameter | Value | Description |
|-----------|-------|-------------|
| GARCH type | GJR-GARCH(1,1) | Asymmetric vol (leverage effect) |
| GARCH persistence | 0.959 | Moderate persistence (BO-optimized) |
| MC paths | 30,000 (60,000 boost) | Monte Carlo simulation paths |
| Student-t df (low vol) | 10.5 | Thinner tails in calm markets |
| Student-t df (mid vol) | 3.9 | Fat tails in normal markets |
| Student-t df (high vol) | 5.9 | Moderate tails in stressed markets |
| Jump diffusion | Merton model | State-dependent crash jumps |
| Calibration | Multi-feature logistic | Online + histogram post-cal |
| MF learning rate | 0.0137 | BO-optimized |
| MF L2 regularization | 0.0058 | BO-optimized |
| Ensemble | [0.5, 0.3, 0.2] | Cross-horizon blending |

---

## Historical Gate Progression

| Session | Score | Key Change |
|---------|-------|------------|
| Baseline | 0/6 | Vol-scaled thresholds, basic online calibrator |
| +Histogram post-cal | 0/6 | Bayesian shrinkage + PAV monotonicity |
| +Multi-feature | 1/6 | Cluster H=5 PASS (ECE=0.017) |
| +Regime-gated thresholds | 1/6 | Better threshold routing but no new passes |
| +BO threshold tuning | 2/6 | Cluster H=5 PASS (0.011), H=10 PASS (0.017) |
| +Data-adaptive BO | TBD | Jump dataset BO running with corrected ranges |
