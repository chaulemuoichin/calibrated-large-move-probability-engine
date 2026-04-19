# Contributing

Thanks for considering a contribution. This project targets **institutional-grade calibrated probabilities**, so correctness and honesty take priority over feature count.

## Ground rules

1. **No lookahead bias.** Every change must preserve strict chronological ordering: predictions queue at time `t` and resolve at `t + H`. If you touch [backtest.py](em_sde/backtest.py), [calibration.py](em_sde/calibration.py), or [predict.py](em_sde/predict.py), add a test that a future feature cannot influence a past prediction.
2. **Tests must pass.** `python -m pytest tests/` — currently 345+ tests. New features add tests; bug fixes add a regression test.
3. **Promotion gates are sacred.** The ECE <= 0.02 / BSS > 0 / AUC >= 0.55 triad is the quality bar. Do not relax gates to make a feature "work."
4. **N_eff, not N.** Overlapping prediction windows create correlated samples. Use `effective_sample_size(y, H, p_cal)` from [evaluation.py](em_sde/evaluation.py) when reporting statistical power.
5. **Features behind flags.** New behavior ships disabled by default with a YAML flag, so existing configs and tests do not change.
6. **Update docs.** If you change behavior, update [CLAUDE.md](CLAUDE.md) (current state), [METHODOLOGY.md](METHODOLOGY.md) (technical details), and [RESULTS.md](RESULTS.md) (numbers) in the same PR.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Typical workflows

### Fixing a bug

1. Write a failing test in [tests/](tests/) that reproduces the bug.
2. Fix the code.
3. Confirm the test passes and the full suite still passes.
4. One commit per bug, with a commit message that explains the *why*.

### Adding a feature

1. Open an issue first if the change is non-trivial — alignment before code saves rewrites.
2. Put the feature behind a YAML flag (default `false`).
3. Add unit tests covering the feature's main code paths.
4. If the feature affects calibration, validate with `python scripts/run_gate_recheck.py <ticker>` on at least one ticker.
5. If the feature affects runtime performance, document the expected impact in [METHODOLOGY.md](METHODOLOGY.md).

### Modifying the calibration or MC pipeline

These modules are load-bearing. Before touching them:

- Run the full test suite and note baseline timing.
- Make the change and confirm the same tests still pass.
- Run `scripts/run_gate_recheck.py spy` before and after. ECE must not degrade by more than 0.001 on any passing config without a documented justification.

### Running the full paper reproduction

```bash
python paper/reproduce.py --all
```

This regenerates all tables and figures (~2-4 hours). Run before submitting changes that affect any paper number.

## Style

- Follow existing patterns. This codebase prefers composition and flags over inheritance.
- No unused imports, no commented-out code.
- Comments explain *why*, not *what*. Well-named variables carry the "what."
- Type hints on public functions; optional on internal helpers.

## Things that have been tried and do not work

Do **not** re-add these without new evidence. They are documented in [CLAUDE.md](CLAUDE.md#removed-features-do-not-re-implement):

- HMM regime detection
- Neural / MLP calibrators
- Isotonic post-calibration
- `vol_scaled` threshold mode
- Filtered Historical Simulation (FHS)
- GARCH ensemble averaging

## Questions

Open an issue or start a discussion. The goal is to keep the bar high and the scope narrow.
