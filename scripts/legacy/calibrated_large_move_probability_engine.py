"""
Calibrated Large-Move Probability Engine
=========================================

Convenience wrapper. Run via:

    python -m em_sde.run --config configs/spy.yaml

Or import components directly:

    from em_sde.data_layer import load_data
    from em_sde.garch import fit_garch
    from em_sde.monte_carlo import simulate_gbm_terminal
    from em_sde.calibration import OnlineCalibrator
    from em_sde.backtest import run_walkforward
"""

import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from em_sde.run import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
