# Methodology

A short, self-contained description of the pipeline — enough for a collaborator to understand, audit, and extend the system without reading the source.

For deeper context see [paper/main.tex](paper/main.tex) (full derivations, results, citations) and [CLAUDE.md](CLAUDE.md) (project rules, removed-features log, current ticker state). If this document disagrees with the code, the code wins.

---

## 1. Pipeline

```
Prices -> GARCH vol -> Monte Carlo -> p_raw -> Calibration -> p_cal
                                                     ^
                                  realized outcomes -+  (walk-forward only)
```

Every prediction at time `t` is queued and only resolved once `t + H` trading days have passed. No step of the pipeline ever reads data from `> t`.

---

## 2. Data

**File:** [em_sde/data_layer.py](em_sde/data_layer.py)

Daily OHLCV via `yfinance`, local CSV, or synthetic GBM. Minimum 756 rows (~3 years). Quality checks flag outliers beyond 5×IQR, stale closes (5+ identical), gaps >5 business days, split-like jumps, and OHLC inconsistencies. When OHLC is missing, range-based features auto-disable.

Optional inputs: earnings dates (cached from yfinance), implied vol series (VIX CSV or generic IV CSV).

---

## 3. Volatility

**File:** [em_sde/garch.py](em_sde/garch.py)

**GARCH(1,1)** [Bollerslev, 1986]:
$$\sigma_{t+1}^{2} = \omega + \alpha\epsilon_t^{2} + \beta\sigma_t^{2}$$

**GJR-GARCH(1,1)** [Glosten et al., 1993] adds a leverage term:
$$\sigma_{t+1}^{2} = \omega + (\alpha + \gamma I_{\epsilon_t<0})\epsilon_t^{2} + \beta\sigma_t^{2}$$

Fit on the most recent 756 returns using the `arch` library. If fitting fails, fall back to 252-day EWMA.

**Stationarity projection.** If fitted persistence $\phi = \alpha + \beta + \gamma/2 \ge 1$, scale $\{\alpha,\beta,\gamma\}$ by $s = 0.98/\phi$ and reset $\omega = \sigma_{1d}^{2}(1-0.98)$ so the simulation remains stationary while anchored to current vol.

**Term-structure average** (enabled by default). For multi-step horizons, the MC simulation is initialized with the horizon-averaged analytic vol:
$$\sigma_{\mathrm{avg}}^{2}(H) = \frac{1}{H}\sum_{h=1}^{H}\!\left[\sigma_{\mathrm{unc}}^{2} + \phi^{h}(\sigma_t^{2} - \sigma_{\mathrm{unc}}^{2})\right]$$

This corrects the GARCH upward-bias after vol spikes.

---

## 4. Threshold

**File:** [em_sde/backtest.py](em_sde/backtest.py)

The event is two-sided: $\mathrm{event}_t^{(H)} = \mathbf{1}\{|P_{t+H}/P_t - 1| \ge \tau\}$.

Three modes:

| Mode | Threshold $\tau$ |
|------|------------------|
| `fixed_pct` | constant $c$ (typically 5%) |
| `anchored_vol` | $k \cdot \bar{\sigma}_{\mathrm{unc}} \cdot \sqrt{H}$ |
| `regime_gated` | routes to fixed or anchored by current vol percentile (252-day rolling) |

`fixed_pct` is the recommended default — the goalpost does not move with vol, so discrimination is not circular.

---

## 5. Monte Carlo

**File:** [em_sde/monte_carlo.py](em_sde/monte_carlo.py)

Default: 100,000 paths of GARCH-in-sim with Student-t innovations and optional Merton jumps.

**Path dynamics:**
$$\log P_{t+1} = \log P_t + (\mu - \tfrac{1}{2}\sigma_t^{2})\Delta t + \sigma_t\sqrt{\Delta t}\,Z_t,\quad Z_t \sim t_\nu\sqrt{(\nu-2)/\nu}$$

with $\Delta t = 1/252$, per-path $\sigma_t$ updated by the GARCH equation each step. Regime-conditional $\nu$ (low/mid/high vol): default 8/5/4.

**Jumps.** Merton with Poisson rate $\lambda/252$ and Gaussian size $\mathcal{N}(\mu_J,\sigma_J)$, drift-compensated by $\lambda(e^{\mu_J + \sigma_J^2/2}-1)$. Rate, mean, and std interpolate linearly between calm and stressed regime parameters.

**Implied-vol blend** (optional, `implied_vol_enabled`). Per-horizon MC sigma becomes
$$\sigma_{\mathrm{blend}} = (1-w)\sigma_{\mathrm{hist}} + w\sigma_{\mathrm{implied}}$$

with $w=0.3$ by default and horizon-matched IV (VIX9D for $H\!=\!5$, interpolated for $H\!=\!10$, VIX for $H\!=\!20$). Staleness guard skips blending when IV is >5 business days old.

**Raw probability:**
$$p_{\mathrm{raw}} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\{|P_{T}^{(i)}/P_0 - 1| \ge \tau\},\quad \mathrm{SE}_{\mathrm{MC}} = \sqrt{p(1-p)/N} \approx 0.0007\ \text{at}\ N=10^{5}, p=0.05.$$

Path count doubles when the trailing 252-day event rate drops below 2%.

---

## 6. Calibration

**File:** [em_sde/calibration.py](em_sde/calibration.py)

Three stacked layers. Each is online, walk-forward, and receives only outcomes whose resolution time $\le t$.

### 6.1 Platt (basic)
$$p_{\mathrm{cal}} = \sigma(a + b\cdot\mathrm{logit}(p_{\mathrm{raw}}))$$

SGD with decaying rate $\eta_t = \eta_0/\sqrt{1+n}$. Activates after 50 outcomes.

### 6.2 Multi-feature logistic
Feature vector $x_t = [1,\ \mathrm{logit}(p_{\mathrm{raw}}),\ 100\sigma_d,\ 100\Delta\sigma_{20},\ \sigma_r/\sigma_d,\ 100 v_{ov},\ \text{earn}?,\ \text{IV ratio}?]$.

Features 7 and 8 are conditional: earnings proximity activates for $H\!\le\!5$ on single stocks; IV ratio activates when implied vol data is loaded. Update is SGD with L2 penalty and gradient clipping. Activates after 100 outcomes. An optional regime-conditional variant maintains one weight vector per vol tercile.

### 6.3 Histogram post-calibration with PAV
After the logistic layer, residual bin-level bias is corrected:

```
correction_k = (mean_pred_k - mean_obs_k) * count_k / (count_k + prior)
p_final      = clip(p_cal - correction, 0, 1)
```

Apply Pool-Adjacent-Violators to enforce monotonicity so AUC is preserved. 10 bins, prior strength 15, activates per bin at ≥15 samples. Per-horizon overrides allow fewer bins + stronger priors at sparse long horizons.

### 6.4 Safety gates
- **Brier gate**: if rolling Brier of $p_{\mathrm{cal}}$ exceeds that of $p_{\mathrm{raw}}$, emit $p_{\mathrm{raw}}$.
- **Discrimination gate**: if rolling AUC $<0.50$, emit $p_{\mathrm{raw}}$.

---

## 7. Walk-forward backtest

**File:** [em_sde/backtest.py](em_sde/backtest.py)

Each trading day $t$:

1. **Resolve.** Any queued prediction from day $t-H$ is now observable — compute $y$, feed to the calibrators.
2. **Fit.** Refit GARCH on data up to $t$.
3. **Project.** Apply stationarity projection if needed.
4. **Threshold.** Compute $\tau$ per horizon from today's vol regime.
5. **Simulate.** Run MC; compute $p_{\mathrm{raw}}$.
6. **Calibrate.** Apply the three calibration layers; emit $p_{\mathrm{cal}}$.
7. **Queue.** Store prediction for resolution at $t+H$.

H=20 predictions overlap by 19 days — outcomes are correlated. All statistical conclusions use effective sample size, not raw N (Section 9).

---

## 8. Cross-validation

**File:** [em_sde/model_selection.py](em_sde/model_selection.py)

Expanding-window, 5 folds, 40% minimum train:

```
Fold 1:  Train [0%-40%],  Test [40%-52%]
...
Fold 5:  Train [0%-88%],  Test [88%-100%]
```

Metrics are computed on the test portion only. Out-of-fold (OOF) predictions are then pooled row-level across folds for gate evaluation, yielding ~1,900 pooled rows per horizon.

---

## 9. Evaluation

**File:** [em_sde/evaluation.py](em_sde/evaluation.py)

| Metric | Formula |
|--------|---------|
| Brier | $\frac{1}{N}\sum(p_i - y_i)^2$ |
| BSS | $1 - \mathrm{Brier}/\bar{y}(1-\bar{y})$ |
| AUC | area under ROC |
| ECE (adaptive bins) | $\sum_k (n_k/N)\,\lvert \bar{p}_k - \bar{y}_k\rvert$ |
| CRPS skill | $1 - \mathrm{CRPS}/\mathrm{CRPS_{clim}}$ |
| PIT KS | KS statistic on probability integral transform |

**Effective sample size.** ACF is computed on residuals $r_t = p_{\mathrm{cal},t} - y_t$ (not binary labels) using the Bartlett estimator. For overlapping H-horizon predictions $N_{\mathrm{eff}} \ll N$; the ratio governs statistical power.

**Bootstrap CIs.** BCa 95% CIs for BSS, AUC, ECE. For overlapping predictions, circular block bootstrap with `block_size=H` preserves temporal structure. Paired loss-differential p-values use the same block scheme.

**Multiple-testing correction.** Benjamini–Hochberg FDR at $\alpha=0.05$ is applied to every table that reports $>1$ p-value (main results, ablation, baselines).

---

## 10. Promotion gates

A config is **promoted** only when the pooled OOF row-level evaluation clears all of:

```
BSS   >= 0.00
AUC   >= 0.55
ECE   <= 0.02
n >= 100, events >= 30, non-events >= 30 per regime
```

Optional stricter gates (off by default) add CRPS-skill ≥ 0, PIT KS ≤ 0.12, tail-coverage error ≤ 0.05, and worst overfit status ≤ YELLOW.

Each gate row reports `n_eff`, `neff_ratio = n_eff / n_bo_params`, and a color (`GREEN` >100×, `YELLOW` 50–100×, `RED` <50×). ECE rows additionally report a confidence annotation (`solid_pass`, `fragile_pass`, `solid_fail`, `fragile_fail`) derived from the bootstrap CI.

Per-regime rows (low/mid/high vol) remain as diagnostics but do not block promotion when pooled mode is active — their per-bucket ECE noise (~0.02–0.03 at ~200 samples) is comparable to the gate threshold itself.

---

## 11. Bayesian hyperparameter optimization

**File:** [scripts/run_bayesian_opt.py](scripts/run_bayesian_opt.py)

Formulated as a **constrained** minimization:

$$\min_{\theta \in \Theta}\ \overline{\mathrm{ECE}}(\theta)\quad\text{s.t.}\quad \mathrm{ECE}(\theta) \le 0.02,\ \ \mathrm{AUC}(\theta) \ge 0.55,\ \ \mathrm{BSS}(\theta) \ge 0$$

**Solver.** Optuna TPE sampler ([Bergstra et al., 2011]) with `constraints_func` passing the three constraint slacks per trial. Feasible trials dominate infeasible ones regardless of objective. SQLite-backed persistence enables resume. Study keys are hashed over feature flags and the optimization formulation (`constrained_v1`) so flag changes auto-fork to a fresh study.

**Pruning.** A `MedianPruner` (`n_startup_trials=5`, `n_warmup_steps=2`) terminates trials whose intermediate (per-fold) mean ECE falls below the running median. A **fast-fail** short-circuits when fold-0 ECE exceeds 0.10 — unrecoverable regardless of subsequent folds. Together these typically prune 30–50% of trials at 20–40% of their fold budget, giving a 2–4× wall-clock speedup.

**Search space.**
- *Lean mode* (default, 3 params): `multi_feature_lr`, `multi_feature_l2`, `garch_target_persistence`. Thresholds frozen.
- `--tune-thresholds` adds `thr_5`, `thr_10`, `thr_20` with data-adaptive bounds (P80×0.9 to P90 of realized returns).
- `--full` adds HAR-RV on/off, `har_rv_ridge`, `har_rv_refit`, and the three regime-conditional `t_df` parameters.

**Event-rate guard.** Adaptive minimum event rate `max((100 × n_params) / (2 × n_oof), 3%)` — derived from the N_eff/N_params ≥ 100× requirement. Trials falling below are returned as maximally infeasible.

---

## 12. Live prediction

At the end of any walk-forward run, final calibrator weights, GARCH parameters, and metadata are written to `outputs/state/{ticker}/`. [em_sde/predict.py](em_sde/predict.py) loads the checkpoint, fetches the latest price panel, runs MC, applies the calibrators, and returns a probability per horizon in seconds. [em_sde/resolve.py](em_sde/resolve.py) pairs predictions with realized outcomes once $t + H$ has passed and feeds the labels back to the live calibrators (same update rule as the backtest — see CLAUDE.md).

The forecast ledger ([em_sde/ledger.py](em_sde/ledger.py)) is append-only JSONL with SHA-256-derived IDs keyed on `ticker+date+horizon+model_version`, making same-session republishes idempotent and cross-model coexistence safe. [scripts/anchor_ledger.py](scripts/anchor_ledger.py) produces tamper-evident manifests via annotated git tags.

---

## 13. How this differs from alternative approaches

Large-move probability is approached from at least five angles in the literature. Each targets a related but distinct quantity and leaves a different gap that this system is designed to close.

| Approach | What it produces | What it does not deliver |
|----------|------------------|--------------------------|
| **Black–Scholes / implied vol** | Risk-neutral probability from option prices | Risk-neutral ≠ physical; not directly comparable to realized frequencies; no self-correction |
| **GARCH + Gaussian/Student-t CDF** | Analytic tail probability from the fitted conditional distribution | Upward bias after vol spikes (persistence ≈ 0.97); no ECE-level calibration |
| **HMM / regime-switching models** | Regime-conditional probability | Regime estimation is noisy with <2,000 observations; calibration is left ad hoc |
| **Feature-based ML (logistic / GBM)** | Binary probability from hand-crafted features | No structural volatility model; calibration depends on train/test split; rarely audited for overlap-induced correlation |
| **VaR / CVaR models** | A quantile of the loss distribution | Not a calibrated probability of an event; evaluated on coverage, not ECE |
| **This system** | Physical probability of a two-sided move $\ge \tau$ | — |

**Where we differ:**

1. **Generative + calibrated, not one or the other.** GARCH-MC supplies the structural prior; three-layer online calibration fixes the residual bias against realized outcomes. Both pieces are necessary — see the ablation in [paper/main.tex](paper/main.tex).
2. **Strict no-lookahead everywhere.** Outcomes flow back through a resolution queue; calibrators only see labels at $t+H$. Many "walk-forward" papers leak future data via CV or calibration-on-full-sample.
3. **Overlap-corrected statistics.** Residual-based ACF for $N_{\mathrm{eff}}$, circular block bootstrap for CIs and paired tests, Benjamini–Hochberg FDR across tables. Most published large-move work reports raw N and i.i.d. bootstrap.
4. **Hard, auditable promotion gates.** ECE ≤ 0.02, BSS ≥ 0, AUC ≥ 0.55 on pooled OOF rows, with bootstrap-CI-based fragility flags. A config that misses any gate is not released, regardless of in-sample numbers.
5. **Constrained BO, not unconstrained objective.** The hyperparameter search is explicitly constrained on the gate thresholds (Section 11), so feasibility dominates objective — the optimizer cannot buy a lower ECE at the cost of failing AUC or BSS.
6. **Verifiable live operation.** Append-only JSONL ledger with SHA-256 IDs and git-tag anchoring, plus per-version rolling metrics, lets third parties recompute every metric from the raw forecast log. Backtest-only papers cannot be audited this way.

What we do **not** claim better:

- **Point-forecast accuracy** (direction, magnitude): not the target.
- **Option pricing**: implied-vol methods remain the right tool for risk-neutral densities.
- **Regime-change detection speed**: HMMs and change-point methods identify regime shifts faster than a 252-day rolling percentile; we trade speed for stability.

A head-to-head comparison on Brier loss against seven implemented baselines (historical frequency, GARCH-CDF, implied-vol BS, feature logistic, gradient boosting, VIX threshold, market-implied straddle) with FDR-adjusted paired block-bootstrap tests is in [paper/main.tex](paper/main.tex) (Section 4.2).

---

## 14. Known limitations

1. **Overlap-induced correlation.** A bad week propagates into H=5, H=10, and H=20 predictions. Always report N_eff, not N.
2. **Cold start.** The first ~100 predictions are effectively uncalibrated. Evaluation windows shorter than that are dominated by the warm-up.
3. **Jump parameters are not fit from data.** The Merton model is a first-order approximation; clustering/contagion is not modeled.
4. **Student-t df is a config input.** Misspecification biases tail probabilities; regime-conditional df mitigates but does not remove this.
5. **No transaction costs inside the probability pipeline.** Economic-significance analysis adds cost sensitivity externally.
6. **Monte Carlo variance.** 100K paths → SE ≈ 0.001 at p=0.05. Close head-to-head comparisons need high path counts.

---

## 14. References

Papers cited in this document — full list with hyperlinks in [paper/main.tex](paper/main.tex).

- Bollerslev (1986) — GARCH.
- Glosten, Jagannathan & Runkle (1993) — GJR leverage term.
- Corsi (2009) — HAR-RV.
- Merton (1976) — jump diffusion.
- Platt (1999) — probabilistic SVM outputs (Platt scaling).
- Naeini, Cooper & Hauskrecht (2015) — ECE via adaptive binning.
- Brier (1950) — Brier score.
- Bergstra, Bardenet, Bengio & Kégl (2011) — TPE sampler.
- Benjamini & Hochberg (1995) — FDR control.
- Vittinghoff & McCulloch (2007) — events-per-parameter rule.
- Dubinsky, Johannes, Kaeck & Seeger (2019); Savor & Wilson (2016) — earnings and tails.
