"""
Markowitz 1 facteur : CAPM. Les espérances de rendement viennent d’une régression
(Ri - Rf) = alpha + beta * (Rm - Rf). On utilise seulement le facteur marché (Mkt-RF).
"""
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import statsmodels.api as sm
from typing import Any


def run(tickers: list[str], start: str, end: str, num_portfolios: int = 10000) -> dict[str, Any]:
    stock_data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False, group_by="column")
    if stock_data.empty:
        return {"error": "Données insuffisantes"}
    if len(tickers) == 1:
        pc = stock_data["Adj Close"] if "Adj Close" in stock_data.columns else stock_data["Close"]
        prices = pc.to_frame(name=tickers[0]) if isinstance(pc, pd.Series) else pc
    else:
        prices = stock_data["Adj Close"].copy() if "Adj Close" in stock_data.columns else stock_data["Close"].copy()
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = [c[-1] if isinstance(c, tuple) else c for c in prices.columns]
        prices = prices.dropna(axis=1, how="all")
    valid = [c for c in tickers if c in prices.columns]
    if len(valid) < 2:
        return {"error": "Données insuffisantes"}
    prices = prices[valid]

    # Passage en mensuel pour matcher Fama-French
    monthly_prices = prices.resample("ME").last()
    stock_returns = monthly_prices.pct_change().dropna()
    stock_returns.index = stock_returns.index.to_period("M")

    # Facteurs Fama-French (on n’utilise que Mkt-RF et RF ici)
    try:
        ff_data = web.DataReader("F-F_Research_Data_Factors", "famafrench", start=start, end=end)[0]
    except Exception:
        return {"error": "Impossible de charger les facteurs Fama-French"}
    ff_data.index = pd.to_datetime(ff_data.index.to_timestamp())
    ff_data = ff_data.resample("ME").last()
    ff_data.columns = ["Mkt-RF", "SMB", "HML", "RF"]
    merged = pd.merge(stock_returns, ff_data, left_index=True, right_index=True, how="inner")
    if len(merged) < 24:
        return {"error": "Pas assez de données mensuelles"}

    n = len(merged)
    split = int(n * 0.8)
    train = merged.iloc[:split]
    stocks_train = train[valid]
    rf_train = train["RF"] / 100
    mkt_train = train["Mkt-RF"] / 100

    # CAPM : E[Ri] = Rf + beta * E[Rm - Rf], une régression OLS par actif
    X = sm.add_constant(mkt_train)
    expected_returns = {}
    for t in valid:
        Y = stocks_train[t] - rf_train
        model = sm.OLS(Y, X).fit()
        b = model.params["Mkt-RF"]
        avg_mkt = mkt_train.mean()
        avg_rf = rf_train.mean()
        expected_returns[t] = (avg_rf + b * avg_mkt) * 12
    mu = pd.Series(expected_returns)
    cov_matrix = stocks_train.cov() * 12
    n_assets = len(valid)
    all_weights = np.zeros((num_portfolios, n_assets))
    ret_arr = np.zeros(num_portfolios)
    vol_arr = np.zeros(num_portfolios)
    sharpe_arr = np.zeros(num_portfolios)
    current_rf = rf_train.mean() * 12
    # Même idée que le classique : Monte Carlo sur les poids, on garde le meilleur Sharpe
    for i in range(num_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        all_weights[i, :] = weights
        ret_arr[i] = np.sum(weights * mu)
        vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_arr[i] = (ret_arr[i] - current_rf) / vol_arr[i] if vol_arr[i] > 1e-10 else 0
    max_idx = sharpe_arr.argmax()
    best_weights = all_weights[max_idx, :]
    weights_dict = {valid[i]: float(best_weights[i]) for i in range(n_assets)}
    # Backtest sur la partie test
    test = merged.iloc[split:]
    test_returns = test[valid]
    portfolio_returns = (test_returns * best_weights).sum(axis=1)
    cum = (1 + portfolio_returns).cumprod()
    portfolio_series = 100 * cum / cum.iloc[0]
    spy = yf.download("SPY", start=test.index[0], end=test.index[-1], auto_adjust=False, progress=False, interval="1mo")
    if not spy.empty and "Adj Close" in spy.columns:
        spy_ret = spy["Adj Close"].pct_change().dropna()
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
    # Sortie pour l’API
    return {
        "weights": weights_dict,
        "sharpe": round(float(sharpe_arr[max_idx]), 4),
        "expectedReturn": round(float(ret_arr[max_idx]) * 100, 2),
        "volatility": round(float(vol_arr[max_idx]) * 100, 2),
        "maxDrawdown": round(max_drawdown, 2),
        "comparisonData": comparison_data,
    }
