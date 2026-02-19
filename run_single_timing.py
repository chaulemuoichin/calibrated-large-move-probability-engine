"""Quick timing test: run one config with minimal MC paths on subset of data."""
import logging, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from em_sde.config import load_config
from em_sde.data_layer import load_data
from em_sde.backtest import run_walkforward
from em_sde.evaluation import compute_metrics

cfg = load_config("configs/exp_suite/exp_cluster_inst_fixed_multi.yaml")
cfg.model.mc_base_paths = 1000
cfg.model.mc_boost_paths = 2000

# Limit to last 5 years (2020-2025) to reduce walk-forward steps
cfg.data.start = "2020-01-01"
cfg.data.min_rows = 252

df, meta = load_data(cfg)
print(f"Data: {len(df)} rows")

t0 = time.perf_counter()
results = run_walkforward(df, cfg)
elapsed = time.perf_counter() - t0
print(f"Walk-forward: {len(results)} rows in {elapsed:.1f}s")

metrics = compute_metrics(results, cfg.model.horizons)
for H in cfg.model.horizons:
    m = metrics["overlapping"].get(H, {})
    print(f"  H={H}: Brier={m.get('brier_cal',0):.4f} BSS={m.get('bss_cal',0):+.4f} AUC={m.get('auc_cal',0):.4f}")
