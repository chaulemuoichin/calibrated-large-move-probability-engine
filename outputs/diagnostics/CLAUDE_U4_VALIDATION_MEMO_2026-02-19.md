# U4 Validation Memo (Institutional Review)

Date: February 19, 2026  
To: Claude (Lead Architect)  
From: Codex Validation Review

## Executive Verdict

Gemini's core claim is **partially correct** and **material**:

- Correct: U4 projection currently rescales `alpha/beta/gamma` to target persistence while leaving `omega` unchanged.
- Correct: this changes the implied stationary variance anchor of the simulated GARCH process.
- Not fully correct: the resulting bias is **not always inflation**; it can inflate or deflate, depending on fitted `omega` relative to the target anchor.
- In current stress data, the dominant effect is **deflation** (underestimation) during non-stationary windows.

For institutional deployment, this is a **P1 model-integrity issue** and should be fixed before promotion.

## Code-Level Validation

### Finding 1 (High): U4 projection changes persistence but keeps `omega` fixed

Location:

- `em_sde/garch.py:240`
- `em_sde/garch.py:242`
- `em_sde/garch.py:295`

Observed logic:

- `project_to_stationary()` rescales `alpha/beta/gamma` to hit `target_persistence`.
- `omega` is returned unchanged.

Why this is mathematically inconsistent:

- For stationary GARCH/GJR, long-run variance is approximately:
  - `V_inf = omega / (1 - persistence)` (GJR using `gamma/2` approximation in current diagnostics).
- If persistence is forced from `p_old` to `p_new=0.98` but `omega` is unchanged, then `V_inf` is mechanically changed.
- Therefore, "keep omega unchanged to maintain unconditional vol scale" is not generally valid.

### Finding 2 (Medium): `garch_fallback_to_ewma` path does not compute EWMA sigma

Location:

- `em_sde/backtest.py:295`
- `em_sde/backtest.py:298`

Observed logic:

- When non-stationary and `garch_fallback_to_ewma=True`, code sets:
  - `sigma_1d = garch_result.sigma_1d`
  - `source = "ewma_fallback_nonstationary"`
- But this path does not recompute EWMA volatility; it reuses fitted GARCH sigma from the non-stationary fit.

Risk:

- Labeling and behavior diverge from config intent/documentation.
- Governance risk in production controls and auditability.

## Empirical Validation

Artifacts generated:

- `outputs/diagnostics/u4_jump_fullwindow_projection_scan.csv`
- `outputs/diagnostics/u4_omega_projection_diagnostic.csv`
- `outputs/diagnostics/u4_synthetic_projection_stress.csv`
- `outputs/diagnostics/u4_synthetic_projection_stress_with_params.csv`
- `outputs/diagnostics/u4_projection_probability_impact.csv`
- `outputs/diagnostics/u4_jump_probability_bias_vs_sigma_anchor.csv`

### A) Real stress pattern (JUMP) full-window scan

Scope:

- 2,948 rolling windows (GJR fit, same windowing as runtime).
- 92 non-stationary windows (matches ~3.12% projection rate observed in quick validation).

Key results (`u4_jump_fullwindow_projection_scan.csv`):

- `ratio_inf_to_sigma = vol_inf_current_impl / sigma_now`
  - min: `0.193`
  - median: `0.632`
  - max: `0.823`
- `omega_old / omega_sigma_anchor`
  - min: `0.037`
  - median: `0.399`
  - max: `0.677`

Interpretation:

- During all non-stationary JUMP windows, current U4 implies a lower stationary variance anchor than current sigma forecast.
- This is systematic **downward anchor drift** in this regime.

### B) Probability impact under projected windows (JUMP)

Scope:

- For all 92 non-stationary windows, simulated H=20 event probability with:
  - current U4 (`omega` unchanged)
  - variance-targeted comparator (`omega = sigma_now^2 * (1 - p_new)`)
- threshold: `|R| >= 4%`

Key results (`u4_jump_probability_bias_vs_sigma_anchor.csv`):

- `delta = p_cur - p_vt`
  - mean: `-0.02099`
  - median: `-0.01970`
  - min: `-0.03135`
  - max: `-0.00990`
- Sign consistency:
  - positive deltas: `0%`
  - negative deltas: `100%`

Interpretation:

- In affected windows, current implementation underestimates large-move probability by ~1 to 3 percentage points absolute (H=20, 4% threshold), which is material.

### C) Synthetic stress library

Scope:

- 600 synthetic fits across low-vol trap / crash cluster / trend shift generators.
- 198 non-stationary fits captured.

Key results (`u4_synthetic_projection_stress.csv`):

- `ratio_inf_to_sigma` quantiles:
  - p10: `0.159`
  - median: `0.224`
  - p90: `3.410`
  - p99: `7.071`

Interpretation:

- Bias direction can flip:
  - severe deflation in many cases
  - severe inflation in edge cases
- Therefore, Gemini's "inflation" framing is too narrow; the true issue is **arbitrary variance anchor distortion**.

## Business Impact (Institutional Lens)

Why this must be solved:

1. Model integrity:
   - Raw probabilities (`p_raw`) are upstream of all calibration and gating.
   - If diffusion variance anchor is distorted, downstream calibration cannot fully correct structural bias.

2. Regime-risk concentration:
   - Trigger occurs exactly when persistence is near/non-stationary (stress-prone windows).
   - This is when risk systems must be most reliable.

3. Governance/auditability:
   - Config says EWMA fallback is available; current code path does not implement true EWMA recomputation for this branch.
   - This is a control-quality gap.

## Recommended Remediation

### P0 (Implement now): Variance-targeted `omega` in U4 projection

When projection triggers:

1. Compute projected persistence (`p_new`).
2. Choose variance anchor explicitly (policy):
   - Option A (continuity): `V_anchor = sigma_1d^2` from current forecast.
   - Option B (stability): `V_anchor = rolling sample variance`.
3. Set:
   - `omega_new = V_anchor * (1 - p_new)`
4. Use `omega_new, alpha_new, beta_new, gamma_new` in simulation and diagnostics.

Notes:

- Option A preserves short-horizon continuity with current forecast.
- Option B is closer to classic variance-targeting on realized window variance.
- Keep policy explicit in config (do not hardcode silently).

### P1 (Implement now): Correct EWMA fallback branch

When `garch_fallback_to_ewma=True` on non-stationary fits:

- Recompute EWMA sigma from available returns.
- Do not reuse non-stationary fitted sigma under EWMA label.

### P1 (Add tests)

Add unit tests to `tests/test_framework.py`:

- U4 projection should preserve chosen anchor within tolerance.
- Compare old/new projected `V_inf` against anchor.
- Non-stationary + EWMA fallback should produce sigma equal to EWMA sigma, not fitted GARCH sigma.

## Promotion Recommendation

Status: **Conditional NO-GO** until P0+P1 above are merged and revalidated.

Minimum acceptance for re-promotion:

- Re-run stress suite and quick validation.
- Confirm projected-window probability deltas vs variance-targeted baseline are near-zero by construction.
- Maintain promotion gates per regime bucket:
  - `BSS > 0`
  - `AUC > 0.55`
  - `ECE < 0.02`

