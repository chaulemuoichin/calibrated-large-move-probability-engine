# Institutional Review Memo (for Claude)

## Scope
- 3 synthetic market patterns: trend/low-vol, volatility-clustering+leverage, jump-crash.
- 3 model variants per pattern: `legacy`, `dyn_volscaled`, `inst_fixed_multi`.
- 9 full walk-forward runs (no lookahead), horizons H=5/10/20, MC=30k base paths.

## Key Aggregate Metrics (calibrated)

| pattern | variant | avg_brier_cal | avg_bss_cal | avg_auc_cal | avg_ece_cal | avg_abs_bias_cal | avg_std_p_cal | composite_rank_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cluster | dyn_volscaled | 0.050450 | -0.009180 | 0.510819 | 0.010560 | 0.010560 | 0.019357 | 2.300000 |
| cluster | inst_fixed_multi | 0.051463 | 0.138374 | 0.805532 | 0.027390 | 0.018644 | 0.062195 | 2.000000 |
| cluster | legacy | 0.048312 | -0.003204 | 0.472291 | 0.005540 | 0.005540 | 0.008351 | 1.700000 |
| jump | dyn_volscaled | 0.048557 | -0.001889 | 0.515385 | 0.006584 | 0.006584 | 0.013676 | 1.500000 |
| jump | inst_fixed_multi | 0.220369 | -0.021657 | 0.467165 | 0.058547 | 0.041379 | 0.042232 | 2.800000 |
| jump | legacy | 0.048434 | -0.005666 | 0.451151 | 0.002575 | 0.002575 | 0.010792 | 1.700000 |
| trend | dyn_volscaled | 0.056385 | -0.006369 | 0.392130 | 0.004356 | 0.004356 | 0.006801 | 1.700000 |
| trend | inst_fixed_multi | 0.112940 | -0.007829 | 0.456955 | 0.014704 | 0.012406 | 0.010376 | 2.600000 |
| trend | legacy | 0.055739 | -0.007134 | 0.388067 | 0.006248 | 0.006248 | 0.006689 | 1.700000 |

## Best Variant per Pattern (composite rank)

| pattern | variant | composite_rank_score |
| --- | --- | --- |
| cluster | legacy | 1.700000 |
| jump | dyn_volscaled | 1.500000 |
| trend | dyn_volscaled | 1.700000 |

## GARCH Stationarity Stress Check (sampled windows, GJR)

| pattern | sampled_windows | non_stationary_windows | non_stationary_rate | ewma_fallback_windows | ewma_fallback_rate |
| --- | --- | --- | --- | --- | --- |
| trend | 148 | 1 | 0.006757 | 0 | 0.000000 |
| cluster | 148 | 0 | 0.000000 | 0 | 0.000000 |
| jump | 148 | 6 | 0.040541 | 0 | 0.000000 |

## Critical Findings
1. Model behavior is highly pattern-dependent; there is no single robust configuration.
2. `inst_fixed_multi` is strong on cluster pattern (AUC~0.81 avg) but fails badly on jump pattern (avg Brier~0.220, avg AUC~0.467, avg BSS~-0.022).
3. `vol_scaled` variants remain near-random on most patterns (AUC close to 0.5, weak separation).
4. On trend pattern, all variants underperform calibration-wise (negative BSS across the board).
5. Non-stationary GARCH warnings occur in stressed windows (jump sampled non-stationary rate ~4.1%).

## Why Underperformance Happens (code-linked)
- Threshold regime mismatch: fixed threshold (`threshold_mode=fixed_pct`) can over-trigger when event-rate regime shifts, causing calibration collapse on jump pattern.
- Weak base signal under `vol_scaled`: threshold tied to sigma dampens discriminative variation in exceedance probability.
- Calibrator instability risk: online SGD calibrators can degrade when base signal SNR is low (safety gate checks Brier only, not discrimination).
- Simulation mismatch: jump parameters are static, not state-dependent; this misses regime-varying jump intensity/severity.
- GARCH persistence boundary risk: near-unit persistence produces fragile volatility forecasts in stress windows.

## Required Next Upgrades (priority)
P0 (must):
- Dynamic threshold policy (state-conditioned) instead of one fixed threshold globally.
- Regime-gated routing between threshold modes (vol-scaled vs fixed/anchored).
- Calibration guardrail on discrimination (AUC/separation), not only Brier.

P1 (high):
- State-dependent jump model (lambda, mean, vol conditional on realized-vol regime).
- Stationarity-constrained GARCH parameter projection before simulation.
- Cross-validated model selection as a promotion gate.

P2 (medium):
- Add CRPS + conditional coverage acceptance tests per horizon/regime bucket.
- Expand forecasting features beyond volatility level (e.g., realized-vol term structure, downside semivol).

## Go/No-Go Recommendation
- No-go for institutional deployment now: robustness fails under jump and trend patterns.
- Go for next sprint only if P0 is implemented and validated with hard gates (BSS>0, AUC>0.55, ECE<0.02 per regime bucket).