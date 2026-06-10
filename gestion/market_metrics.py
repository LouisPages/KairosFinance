"""Ratios de Sharpe annualisés pour benchmarks marché (SPY, CMKT, etc.)."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from yahoo_prices import yf_single_ticker_adj_series


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


def rf_annual_from_irx(start: str, end: str, default: float = 0.02) -> float:
    """Taux sans risque annualisé (moyenne ^IRX) sur la période demandée."""
    start_d = datetime.fromisoformat(start[:10])
    end_d = datetime.fromisoformat(end[:10]) + timedelta(days=1)
    try:
        irx_s = yf_single_ticker_adj_series("^IRX", start_d, end_d, "1d")
        if irx_s is not None and not irx_s.empty:
            return float((irx_s / 100).mean())
    except Exception:
        pass
    return default


def spy_sharpe_triplet_for_period(
    start: str,
    end: str,
    rf_annual: float,
    periods_per_year: int = 252,
    use_log_returns: bool = False,
    min_train: int = 10,
) -> tuple[float, float, float]:
    """
    Sharpe SPY sur la période utilisateur (indépendant du portefeuille).
    Découpe train/test en 80/20 sur le calendrier du benchmark.
    """
    start_d = datetime.fromisoformat(start[:10])
    end_d = datetime.fromisoformat(end[:10]) + timedelta(days=1)
    spy_s = yf_single_ticker_adj_series("SPY", start_d, end_d, "1d")
    if spy_s is None or spy_s.empty:
        return 0.0, 0.0, 0.0

    if periods_per_year == 12:
        prices = spy_s.resample("ME").last()
    else:
        prices = spy_s

    if use_log_returns:
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        returns = prices.pct_change().dropna()

    if len(returns) < min_train + 2:
        return 0.0, 0.0, 0.0

    if periods_per_year == 12:
        split = max(int(len(returns) * 0.8), min_train)
    else:
        split = max(int(len(returns) * 0.8), min_train)
    if split >= len(returns):
        return 0.0, 0.0, 0.0

    train_r = returns.iloc[:split]
    test_r = returns.iloc[split:]
    return (
        annualized_sharpe(train_r, rf_annual, periods_per_year),
        annualized_sharpe(test_r, rf_annual, periods_per_year),
        annualized_sharpe(returns, rf_annual, periods_per_year),
    )
