import numpy as np
import pandas as pd
import yfinance as yf
from typing import Any


def run(tickers: list[str], start: str, end: str, risk_free_rate: float = 0.03, num_portfolios: int = 10000) -> dict[str, Any]:
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False, group_by="column")
    if data.empty:
        return {"error": "Données insuffisantes"}
    if len(tickers) == 1:
        pc = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
        prices = pc.to_frame(name=tickers[0]) if isinstance(pc, pd.Series) else pc
    else:
        prices = data["Adj Close"].copy() if "Adj Close" in data.columns else data["Close"].copy()
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = [c[-1] if isinstance(c, tuple) else c for c in prices.columns]
    prices = prices.dropna(axis=1, how="all")
    valid = [c for c in tickers if c in prices.columns]
    if len(valid) < 2:
        return {"error": "Données insuffisantes"}
    prices = prices[valid]
    n = len(prices)
    split = int(n * 0.8)
    if split < 10:
        return {"error": "Pas assez de données pour la période"}
    train_prices = prices.iloc[:split]
    test_prices = prices.iloc[split:]
    returns = np.log(train_prices / train_prices.shift(1)).dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    n_assets = len(valid)
    all_weights = np.zeros((num_portfolios, n_assets))
    ret_arr = np.zeros(num_portfolios)
    vol_arr = np.zeros(num_portfolios)
    sharpe_arr = np.zeros(num_portfolios)
    for i in range(num_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        all_weights[i, :] = weights
        ret_arr[i] = np.sum(mean_returns * weights)
        vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_arr[i] = (ret_arr[i] - risk_free_rate) / vol_arr[i] if vol_arr[i] > 1e-10 else 0
    max_idx = sharpe_arr.argmax()
    best_weights = all_weights[max_idx, :]
    weights_dict = {valid[i]: float(best_weights[i]) for i in range(n_assets)}
    test_returns = np.log(test_prices / test_prices.shift(1)).dropna()
    portfolio_returns = (test_returns * best_weights).sum(axis=1)
    cum = (1 + portfolio_returns).cumprod()
    portfolio_series = 100 * cum / cum.iloc[0]
    spy = yf.download("SPY", start=test_prices.index[0], end=test_prices.index[-1], auto_adjust=False, progress=False)
    if not spy.empty and "Adj Close" in spy.columns:
        spy_ret = np.log(spy["Adj Close"] / spy["Adj Close"].shift(1)).dropna()
        spy_cum = (1 + spy_ret).cumprod()
        market_series = 100 * spy_cum / spy_cum.iloc[0]
        common_idx = portfolio_series.index.intersection(market_series.index)
        portfolio_series = portfolio_series.reindex(common_idx).ffill().bfill()
        market_series = market_series.reindex(common_idx).ffill().bfill()
    else:
        market_series = portfolio_series.copy()
    comparison_data = [
        {"date": d.strftime("%Y-%m-%d"), "portfolio": round(float(portfolio_series.loc[d]), 2), "market": round(float(market_series.loc[d]), 2)}
        for d in portfolio_series.index
    ]
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_drawdown = float(-drawdown.min() * 100) if len(drawdown) > 0 else 0
    return {
        "weights": weights_dict,
        "sharpe": round(float(sharpe_arr[max_idx]), 4),
        "expectedReturn": round(float(ret_arr[max_idx]) * 100, 2),
        "volatility": round(float(vol_arr[max_idx]) * 100, 2),
        "maxDrawdown": round(max_drawdown, 2),
        "comparisonData": comparison_data,
    }
