# CLAUDE Institutional Review Memo (Post-Full Validation)

Date: February 21, 2026  
Run set completed from artifacts dated February 20, 2026

## 1) Executive Verdict
- Institutional promotion status: **NO-GO**
- Reason: **0/36 config-horizon combinations passed promotion gates** in CV (`BSS > 0`, `AUC > 0.55`, `ECE < 0.02` per vol-regime bucket).
- The system has meaningful gains in jump-sensitive settings, but calibration reliability is not yet institutional-grade across regimes.

## 2) Scope of Validation Executed
- Unit tests: `137/137` pass (`outputs/diagnostics/00_pytest.log`)
- Pre-gate runs:
  - `outputs/quick_validation_results.csv` (6 configs x 3 horizons)
  - `outputs/stress_suite_results.csv` (12 configs x 3 horizons)
- Stability:
  - `outputs/diagnostics/stage1_seed_stability.csv` (30 runs: 6 configs x 5 seeds)
- CV + gates:
  - `outputs/diagnostics/cv_cluster_*`
  - `outputs/diagnostics/cv_jump_*`
  - `outputs/diagnostics/cv_trend_*`

## 3) Findings (Ranked by Severity)

### F1 - Hard gate failure across all candidates (Critical)
- All families failed promotion gates at every horizon.
- Gate pass-rate by metric (all rows):
  - Cluster: `auc_cal 44.4%`, `bss_cal 19.4%`, `ece_cal 19.4%`
  - Jump: `auc_cal 41.7%`, `bss_cal 19.4%`, `ece_cal 13.9%`
  - Trend: `auc_cal 16.7%`, `bss_cal 11.1%`, `ece_cal 44.4%`
- Interpretation: failure is structural, not noise.

### F2 - Jump regime improves ranking/skill but remains miscalibrated (High)
- Quick validation H=10, `regime_gated` vs `inst_fixed_multi`:
  - `brier_cal`: `-0.047642` (better)
  - `bss_cal`: `+0.103178` (better)
  - `auc_cal`: `+0.145446` (better)
- CV top for jump is `exp_jump_regime_gated` across H=5/10/20.
- However, gate failure is dominated by **ECE in all vol buckets**:
  - H=20 top model fails only ECE (3/9 failures), worst margin `-0.1041`.
- Conclusion: discrimination is improved, but probability level quality is still not promotable.

### F3 - Trend regime still has weak base signal (High)
- CV H=10 best model is `exp_trend_legacy`, but:
  - `bss_cal_mean = -0.0070`
  - `auc_cal_mean = 0.4422`
- Trend gates fail mostly on AUC/BSS (near-random ordering).
- Conclusion: calibrator cannot recover a weak base model; signal layer needs improvement.

### F4 - Cluster regime has good AUC but unstable skill/calibration by sub-regime (Medium)
- Stress H=10 best remains `exp_cluster_inst_fixed_multi`:
  - `brier_cal = 0.0376`, `bss_cal = 0.1108`, `auc_cal = 0.7785`
- CV gate failures for cluster top models are from:
  - negative BSS in low/mid vol buckets
  - high-vol ECE well above 0.02 (e.g., `0.0609`)
- Conclusion: aggregate performance hides sub-regime reliability failure.

### F5 - Frequent non-stationary GARCH boundary events in CV run (Operational Risk)
- `outputs/diagnostics/06_remaining_cv.log` contains `1634` occurrences of:
  - `GARCH non-stationary: persistence=1.0000 >= 1.0`
- U4 projection prevents blow-up, but repeated boundary hits indicate persistent fit instability in stressed windows.

## 4) Stability Check (30-run multi-seed)
- H=10 deltas (`regime_gated - inst_fixed_multi`) are stable in sign for Brier:
  - Cluster: `+0.000241` (consistently slightly worse)
  - Jump: `-0.047014` (consistently better)
  - Trend: `-0.012785` (consistently better)
- Seed variance is low for Brier deltas, so current conclusions are robust to RNG.

## 5) Why We Underperform
1. **Calibration error dominates promotion failure** even when ranking improves (especially jump high-vol states).
2. **Trend base model has insufficient separability**; calibrator cannot create signal from near-random ordering.
3. **Regime behavior is non-uniform**; aggregate metrics mask bucket-level failures.
4. **Frequent near-nonstationary volatility fits** imply unstable volatility dynamics under stress.

## 6) Required Fixes Before Promotion

### P0 (must-have)
1. Add **regime-conditional calibration maps** (separate calibration heads by vol regime and/or threshold regime) with per-bucket monotonic constraints.
2. Add **ECE-targeted training control** (calibration update objective or gate that explicitly minimizes ECE where current failures concentrate).
3. Enforce **bucket-level acceptance as primary** in model ranking (not aggregate Brier alone).

### P1 (high priority)
1. Improve trend signal features (directional + asymmetry terms) before calibration.
2. Add diagnostics to CV outputs for U4 frequency and projected parameter drift so instability can be modeled, not just logged.
3. Re-tune regime routing thresholds to reduce low/mid-vol BSS degradation in cluster family.

### P2 (next)
1. Add conditional coverage and CRPS monitoring by regime/horizon.
2. Add warning-budget policy for repeated non-stationary events to prevent silent degradation.

## 7) Promotion Recommendation to Claude
- Do **not** promote any config to institutional deployment in current state.
- Fastest path forward:
  1. Keep `regime_gated` as jump candidate baseline (best directional progress),
  2. implement P0 calibration redesign,
  3. re-run full CV gates and require first **non-zero pass set** before any rollout.

## 8) Generated Supporting Files
- `outputs/diagnostics/institutional_seed_stability_summary.csv`
- `outputs/diagnostics/institutional_gate_verdict_cluster.csv`
- `outputs/diagnostics/institutional_gate_verdict_jump.csv`
- `outputs/diagnostics/institutional_gate_verdict_trend.csv`
