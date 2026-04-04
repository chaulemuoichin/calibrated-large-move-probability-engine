"""
Economic significance analysis: does calibrated probability improve decisions?

Two simple demonstrations:
  1. Risk-Managed Portfolio: reduce position size when p > threshold
     vs. unconditional buy-and-hold
  2. Selective Hedging: buy protective puts when p > threshold
     vs. unhedged and always-hedged strategies

Metrics: max drawdown reduction, Sharpe ratio, tail risk (CVaR)

This is NOT a trading strategy backtest — it demonstrates that calibrated
probabilities contain actionable risk information.

Usage:
    python scripts/run_economic_significance.py spy
    python scripts/run_economic_significance.py all
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from em_sde.config import load_config
from em_sde.data_layer import load_data
from em_sde.backtest import run_walkforward
from em_sde.evaluation import max_drawdown, value_at_risk, conditional_var

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TICKER_CONFIGS = {
    "spy": "configs/exp_suite/exp_spy_regime_gated.yaml",
    "googl": "configs/exp_suite/exp_googl_regime_gated.yaml",
    "amzn": "configs/exp_suite/exp_amzn_regime_gated.yaml",
    "nvda": "configs/exp_suite/exp_nvda_regime_gated.yaml",
}


def _annualized_sharpe(daily_returns: np.ndarray) -> float:
    """Annualized Sharpe ratio (assuming 0 risk-free rate)."""
    if len(daily_returns) < 20:
        return np.nan
    mu = float(np.mean(daily_returns)) * 252
    sigma = float(np.std(daily_returns)) * np.sqrt(252)
    return mu / sigma if sigma > 1e-10 else np.nan


def _max_drawdown_pct(equity_curve: np.ndarray) -> float:
    """Max drawdown as a fraction."""
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    return float(np.min(drawdowns))


def risk_managed_portfolio(
    results: pd.DataFrame,
    prices: np.ndarray,
    dates: pd.DatetimeIndex,
    H: int = 5,
    p_threshold: float = 0.20,
    reduced_weight: float = 0.5,
    cost_bps: float = 0.0,
) -> dict:
    """
    Risk-managed portfolio: reduce exposure when P(large move) > threshold.

    Strategy:
    - Full weight (1.0) when p_cal < p_threshold
    - Reduced weight when p_cal >= p_threshold
    - Daily rebalancing (using H=5 predictions)

    Comparison: buy-and-hold at full weight.
    """
    p_col = f"p_cal_{H}"
    if p_col not in results.columns:
        return {}

    # Align results with price data
    result_dates = pd.to_datetime(results["date"])
    p_cal = results[p_col].to_numpy(dtype=float)

    # Build daily return series
    daily_returns = np.diff(prices) / prices[:-1]
    n = len(daily_returns)

    # Map predictions to daily dates
    p_by_date = {}
    for i, d in enumerate(result_dates):
        if np.isfinite(p_cal[i]):
            p_by_date[d] = p_cal[i]

    # Simulate strategies
    bh_equity = [1.0]  # Buy-and-hold
    rm_equity = [1.0]  # Risk-managed
    bh_returns = []
    rm_returns = []
    last_p = 0.0
    n_risk_on = 0
    n_risk_off = 0

    for i in range(len(daily_returns)):
        date = dates[i + 1] if i + 1 < len(dates) else dates[-1]
        ret = daily_returns[i]

        # Update probability estimate (use most recent prediction)
        if date in p_by_date:
            last_p = p_by_date[date]

        # Position sizing
        if last_p >= p_threshold:
            weight = reduced_weight
            n_risk_off += 1
        else:
            weight = 1.0
            n_risk_on += 1

        bh_ret = ret
        rm_ret = weight * ret

        # Transaction cost when weight changes
        if i > 0:
            prev_date = dates[i] if i < len(dates) else dates[-1]
            prev_p = p_by_date.get(prev_date, last_p)
            prev_weight = reduced_weight if prev_p >= p_threshold else 1.0
            if abs(weight - prev_weight) > 0.01 and cost_bps > 0:
                rm_ret -= abs(weight - prev_weight) * cost_bps / 10000.0

        bh_returns.append(bh_ret)
        rm_returns.append(rm_ret)
        bh_equity.append(bh_equity[-1] * (1 + bh_ret))
        rm_equity.append(rm_equity[-1] * (1 + rm_ret))

    bh_returns = np.array(bh_returns)
    rm_returns = np.array(rm_returns)
    bh_equity = np.array(bh_equity)
    rm_equity = np.array(rm_equity)

    return {
        "strategy": "Risk-Managed",
        "horizon": H,
        "p_threshold": p_threshold,
        "reduced_weight": reduced_weight,
        "cost_bps": cost_bps,
        "bh_sharpe": _annualized_sharpe(bh_returns),
        "rm_sharpe": _annualized_sharpe(rm_returns),
        "sharpe_improvement": _annualized_sharpe(rm_returns) - _annualized_sharpe(bh_returns),
        "bh_max_dd": _max_drawdown_pct(bh_equity),
        "rm_max_dd": _max_drawdown_pct(rm_equity),
        "dd_reduction": _max_drawdown_pct(bh_equity) - _max_drawdown_pct(rm_equity),
        "bh_cvar95": conditional_var(bh_returns),
        "rm_cvar95": conditional_var(rm_returns),
        "bh_annual_return": float(np.mean(bh_returns)) * 252,
        "rm_annual_return": float(np.mean(rm_returns)) * 252,
        "pct_risk_off": n_risk_off / max(n_risk_on + n_risk_off, 1),
        "n_days": len(bh_returns),
    }


def selective_hedging_analysis(
    results: pd.DataFrame,
    prices: np.ndarray,
    dates: pd.DatetimeIndex,
    H: int = 5,
    p_threshold: float = 0.20,
    hedge_cost_daily: float = 0.0002,  # ~5% annual cost of protective puts
    hedge_protection: float = 0.5,     # hedges capture 50% of downside
) -> dict:
    """
    Selective hedging: buy protection only when model signals elevated risk.

    Three strategies compared:
    - Unhedged: pure buy-and-hold
    - Always-hedged: always pay hedge cost, always protected
    - Selective: hedge only when p_cal > threshold
    """
    p_col = f"p_cal_{H}"
    if p_col not in results.columns:
        return {}

    result_dates = pd.to_datetime(results["date"])
    p_cal = results[p_col].to_numpy(dtype=float)
    daily_returns = np.diff(prices) / prices[:-1]

    p_by_date = {}
    for i, d in enumerate(result_dates):
        if np.isfinite(p_cal[i]):
            p_by_date[d] = p_cal[i]

    unhedged = []
    always_hedged = []
    selective = []
    last_p = 0.0
    n_hedged_days = 0

    for i in range(len(daily_returns)):
        date = dates[i + 1] if i + 1 < len(dates) else dates[-1]
        ret = daily_returns[i]

        if date in p_by_date:
            last_p = p_by_date[date]

        # Unhedged
        unhedged.append(ret)

        # Always hedged
        hedged_ret = ret - hedge_cost_daily
        if ret < 0:
            hedged_ret = ret * (1 - hedge_protection) - hedge_cost_daily
        always_hedged.append(hedged_ret)

        # Selective hedging
        if last_p >= p_threshold:
            sel_ret = ret - hedge_cost_daily
            if ret < 0:
                sel_ret = ret * (1 - hedge_protection) - hedge_cost_daily
            n_hedged_days += 1
        else:
            sel_ret = ret
        selective.append(sel_ret)

    unhedged = np.array(unhedged)
    always_hedged = np.array(always_hedged)
    selective = np.array(selective)

    return {
        "strategy": "Selective Hedging",
        "horizon": H,
        "p_threshold": p_threshold,
        "unhedged_sharpe": _annualized_sharpe(unhedged),
        "always_hedged_sharpe": _annualized_sharpe(always_hedged),
        "selective_sharpe": _annualized_sharpe(selective),
        "unhedged_cvar95": conditional_var(unhedged),
        "always_hedged_cvar95": conditional_var(always_hedged),
        "selective_cvar95": conditional_var(selective),
        "unhedged_max_dd": _max_drawdown_pct(np.cumprod(1 + unhedged)),
        "always_hedged_max_dd": _max_drawdown_pct(np.cumprod(1 + always_hedged)),
        "selective_max_dd": _max_drawdown_pct(np.cumprod(1 + selective)),
        "pct_hedged_days": n_hedged_days / max(len(unhedged), 1),
        "hedge_cost_saved": (1 - n_hedged_days / max(len(unhedged), 1)) * hedge_cost_daily * 252,
    }


def run_economic_analysis(ticker: str) -> pd.DataFrame:
    """Run full economic significance analysis for a ticker."""
    config_path = TICKER_CONFIGS.get(ticker.lower())
    if not config_path:
        raise ValueError(f"Unknown ticker: {ticker}")

    cfg = load_config(config_path)
    df, _meta = load_data(cfg)
    prices = df["price"].to_numpy(dtype=float)

    logger.info("Running walk-forward for %s...", ticker.upper())
    results = run_walkforward(df, cfg)

    all_results = []

    for H in cfg.model.horizons:
        for p_thr in [0.15, 0.20, 0.25, 0.30]:
            # Risk management with transaction cost sensitivity
            for cost in [0, 5, 10, 20]:
                rm = risk_managed_portfolio(
                    results, prices, df.index, H=H, p_threshold=p_thr,
                    cost_bps=cost,
                )
                if rm:
                    rm["ticker"] = ticker.upper()
                    all_results.append(rm)

            # Selective hedging (no cost sweep needed -- hedge cost is built in)
            sh = selective_hedging_analysis(
                results, prices, df.index, H=H, p_threshold=p_thr,
            )
            if sh:
                sh["ticker"] = ticker.upper()
                all_results.append(sh)

    results_df = pd.DataFrame(all_results)

    # Save
    os.makedirs("outputs/paper", exist_ok=True)
    outpath = f"outputs/paper/economic_{ticker.lower()}.csv"
    results_df.to_csv(outpath, index=False)

    # Print summary
    print(f"\n{'='*80}")
    print(f"ECONOMIC SIGNIFICANCE: {ticker.upper()}")
    print(f"{'='*80}")

    rm_results = results_df[results_df["strategy"] == "Risk-Managed"]
    if len(rm_results) > 0:
        print("\n--- Risk-Managed Portfolio ---")
        cols = ["horizon", "p_threshold", "bh_sharpe", "rm_sharpe",
                "sharpe_improvement", "bh_max_dd", "rm_max_dd", "pct_risk_off"]
        available = [c for c in cols if c in rm_results.columns]
        print(rm_results[available].to_string(index=False))

    sh_results = results_df[results_df["strategy"] == "Selective Hedging"]
    if len(sh_results) > 0:
        print("\n--- Selective Hedging ---")
        cols = ["horizon", "p_threshold", "unhedged_sharpe", "selective_sharpe",
                "unhedged_cvar95", "selective_cvar95", "pct_hedged_days"]
        available = [c for c in cols if c in sh_results.columns]
        print(sh_results[available].to_string(index=False))

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Economic significance analysis")
    parser.add_argument("ticker", help="Ticker or 'all'")
    args = parser.parse_args()

    if args.ticker.lower() == "all":
        all_results = []
        for ticker in TICKER_CONFIGS:
            try:
                result = run_economic_analysis(ticker)
                all_results.append(result)
            except Exception as e:
                logger.error("Economic analysis failed for %s: %s", ticker, e)
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            combined.to_csv("outputs/paper/economic_all.csv", index=False)
    else:
        run_economic_analysis(args.ticker)


if __name__ == "__main__":
    main()
