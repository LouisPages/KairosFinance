"""Ratios de Sharpe annualisés pour benchmarks marché (SPY, CMKT, etc.)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_sharpe(
    returns: pd.Series | np.ndarray,
    rf_annual: float,
    periods_per_year: int = 252,
) -> float:
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 2:
        return 0.0
    ann_mean = float(r.mean()) * periods_per_year
    ann_vol = float(r.std(ddof=1)) * (periods_per_year**0.5)
    if ann_vol < 1e-10:
        return 0.0
    return round((ann_mean - rf_annual) / ann_vol, 4)


def market_sharpe_pair(
    market_returns: pd.Series,
    train_index: pd.Index,
    test_index: pd.Index,
    rf_annual: float,
    periods_per_year: int = 12,
) -> tuple[float, float]:
    train_idx = market_returns.index.intersection(train_index)
    test_idx = market_returns.index.intersection(test_index)
    return (
        annualized_sharpe(market_returns.loc[train_idx], rf_annual, periods_per_year),
        annualized_sharpe(market_returns.loc[test_idx], rf_annual, periods_per_year),
    )


def market_sharpe_triplet(
    market_returns: pd.Series,
    train_index: pd.Index,
    test_index: pd.Index,
    rf_annual: float,
    periods_per_year: int = 12,
) -> tuple[float, float, float]:
    train_s, test_s = market_sharpe_pair(
        market_returns, train_index, test_index, rf_annual, periods_per_year
    )
    total_s = annualized_sharpe(market_returns, rf_annual, periods_per_year)
    return train_s, test_s, total_s
