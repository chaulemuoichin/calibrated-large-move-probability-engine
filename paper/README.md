# Paper: Calibrated Large-Move Probability Estimation

## Reproducing Results

### Prerequisites

```bash
pip install -r requirements.txt
```

Required data files in `data/`:
- `spy_daily.csv`, `googl_daily.csv`, `amzn_daily.csv`, `nvda_daily.csv`
- `vix_history.csv` (VIX/VIX9D/VIX3M)

### Full Reproduction

```bash
python paper/reproduce.py --all
```

Expected runtime: 2-4 hours. Outputs go to `outputs/paper/`.

### Individual Components

```bash
# Table 1: Main CV results
python paper/reproduce.py --main-results

# Table 2: Baseline comparison
python paper/reproduce.py --baselines

# Table 3: Ablation study
python paper/reproduce.py --ablation

# Table 4: Temporal hold-out
python paper/reproduce.py --holdout

# Table 5: Economic significance
python paper/reproduce.py --economic

# All figures
python paper/reproduce.py --figures
```

### Compiling the Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Output Structure

```
outputs/paper/
  tables/
    main_results.csv          Table 1
    main_results.tex          LaTeX version
    baseline_comparison.csv   Table 2
    baseline_comparison.tex   LaTeX version
  figures/
    reliability_diagrams.pdf  Figure 1
    ablation_heatmap.pdf      Figure 2
    baseline_comparison.pdf   Figure 3
    rolling_ece.pdf           Figure 4
  ablation_all.csv            Full ablation data
  holdout_all.csv             Full hold-out data
  economic_all.csv            Full economic analysis data
  reproduction.log            Execution log
```
