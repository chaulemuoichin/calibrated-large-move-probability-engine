# Contributing

This project targets **institutional-grade calibrated probabilities**. Correctness over feature count.

## Ground rules

1. **No lookahead.** Predictions queue at `t`, resolve at `t+H`. If you touch [backtest.py](em_sde/backtest.py), [calibration.py](em_sde/calibration.py), or [predict.py](em_sde/predict.py), add a test that future data cannot influence a past prediction.
2. **Tests must pass.** `python -m pytest tests/` (356+ tests). Features add tests; fixes add a regression test.
3. **Gates are non-negotiable.** ECE ≤ 0.02, BSS ≥ 0, AUC ≥ 0.55 on pooled OOF rows. Do not relax.
4. **N_eff, not N.** Use `effective_sample_size(y, H, p_cal)` from [evaluation.py](em_sde/evaluation.py) when reporting power.
5. **Flags, not rewrites.** New behavior ships disabled by default behind a YAML flag.
6. **Update docs in the same PR.** [CLAUDE.md](CLAUDE.md) (current state), [METHODOLOGY.md](METHODOLOGY.md) (technical), [RESULTS.md](RESULTS.md) (numbers).

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt
python -m pytest tests/
```

## Workflows

**Fix a bug.** Failing test → fix → full suite green → one commit explaining *why*.

**Add a feature.** Open an issue if non-trivial. YAML flag default `false`. Unit tests for main paths. If it affects calibration, run `python scripts/run_gate_recheck.py <ticker>` on at least one ticker. If it affects runtime, document in METHODOLOGY.md.

**Touch calibration / MC.** These are load-bearing. Run the full suite before and after. Run `scripts/run_gate_recheck.py spy` before and after — ECE must not degrade by more than 0.001 on any passing config without a documented reason.

**Reproduce the paper.** `python paper/reproduce.py --all` (~2–4 hours). Required before PRs that change any paper number.

## Style

- Follow existing patterns. Composition and flags, not inheritance.
- No unused imports. No commented-out code.
- Comments explain *why*, not *what*.
- Type hints on public functions.

## Do not re-add

These were tried and removed. See [CLAUDE.md](CLAUDE.md#removed-features-do-not-re-implement) for evidence:

HMM regime detection, neural/MLP calibrators, isotonic post-cal, `vol_scaled` threshold, Filtered Historical Simulation, GARCH ensemble averaging.

Open an issue or start a discussion for anything non-trivial. Keep the bar high, the scope narrow.
