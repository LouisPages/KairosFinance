"""
Markowitz 3 facteurs Fama-French : espérances de rendement via régression
sur Mkt-RF, SMB, HML. E[Ri] = Rf + b*(Mkt-Rf) + s*SMB + h*HML.
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

    # Mensuel pour Fama-French
    monthly_prices = prices.resample("ME").last()
    stock_returns = monthly_prices.pct_change().dropna()
    stock_returns.index = stock_returns.index.to_period("M")

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
    factors = train[["Mkt-RF", "SMB", "HML"]].apply(lambda x: x / 100)
    rf_train = train["RF"] / 100

    # Régression multi-facteurs : Ri - Rf = alpha + b*MktRF + s*SMB + h*HML, puis E[Ri] = Rf + b*E[MktRF] + ...
    X = sm.add_constant(factors)
    expected_returns = {}
    for t in valid:
        Y = stocks_train[t] - rf_train
        model = sm.OLS(Y, X).fit()
        b = model.params["Mkt-RF"]
        s = model.params["SMB"]
        h = model.params["HML"]
        avg_mkt = factors["Mkt-RF"].mean()
        avg_smb = factors["SMB"].mean()
        avg_hml = factors["HML"].mean()
        current_rf = rf_train.mean()
        expected_ret = current_rf + (b * avg_mkt) + (s * avg_smb) + (h * avg_hml)
        expected_returns[t] = expected_ret * 12
    mu = pd.Series(expected_returns)
    cov_matrix = stocks_train.cov() * 12
    n_assets = len(valid)
    all_weights = np.zeros((num_portfolios, n_assets))
    ret_arr = np.zeros(num_portfolios)
    vol_arr = np.zeros(num_portfolios)
    sharpe_arr = np.zeros(num_portfolios)
    current_rf = rf_train.mean() * 12
    # Monte Carlo des poids, max Sharpe
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
    # Série portefeuille et marché sur 100 % de la période (base 100 au premier jour du backtest)
    test = merged.iloc[split:]
    full_returns = merged[valid]
    portfolio_returns_full = (full_returns * best_weights).sum(axis=1)
    cum_full = (1 + portfolio_returns_full).cumprod()
    portfolio_series = 100 * cum_full / cum_full.iloc[0]
    spy = yf.download("SPY", start=merged.index[0], end=merged.index[-1], auto_adjust=False, progress=False, interval="1mo")
    if not spy.empty and "Adj Close" in spy.columns:
        spy_ret = spy["Adj Close"].pct_change().dropna()
        spy_cum = (1 + spy_ret).cumprod()
        market_series = 100 * spy_cum / spy_cum.iloc[0]
        common_idx = portfolio_series.index.intersection(market_series.index)
        portfolio_series = portfolio_series.reindex(common_idx).ffill().bfill()
        market_series = market_series.reindex(common_idx).ffill().bfill()
    else:
        market_series = portfolio_series.copy()
        common_idx = portfolio_series.index

    # Renormaliser base 100 au premier jour de la période de backtest
    test_start_date = test.index[0]
    idx_at_or_after = common_idx[common_idx >= test_start_date]
    base_date = idx_at_or_after[0] if len(idx_at_or_after) > 0 else common_idx[-1]
    base_p = float(portfolio_series.loc[base_date])
    base_m = float(market_series.loc[base_date])
    if base_p > 1e-12 and base_m > 1e-12:
        portfolio_series = 100 * portfolio_series / base_p
        market_series = 100 * market_series / base_m
    def _date_str(d):
        return d.to_timestamp().strftime("%Y-%m-%d") if hasattr(d, "to_timestamp") else d.strftime("%Y-%m-%d")
    comparison_data = [
        {"date": _date_str(d), "portfolio": round(float(portfolio_series.loc[d]), 2), "market": round(float(market_series.loc[d]), 2)}
        for d in portfolio_series.index
    ]
    # Max drawdown sur période test
    test_returns = test[valid]
    portfolio_returns_test = (test_returns * best_weights).sum(axis=1)
    cum_test = (1 + portfolio_returns_test).cumprod()
    peak = cum_test.cummax()
    drawdown = (cum_test - peak) / peak
    max_drawdown = float(-drawdown.min() * 100) if len(drawdown) > 0 else 0

    # Frontière efficiente : points (vol, rendement) Pareto-optimaux
    vol_pct = vol_arr * 100
    ret_pct = ret_arr * 100
    order = np.argsort(vol_pct)
    frontier_vol, frontier_ret, frontier_sharpe, frontier_backtest_ret = [], [], [], []
    max_ret_so_far = -np.inf
    for i in order:
        if ret_pct[i] >= max_ret_so_far:
            max_ret_so_far = ret_pct[i]
            port_ret_test = (test_returns * all_weights[i]).sum(axis=1)
            total_ret = ((1 + port_ret_test).prod() - 1) * 100
            frontier_vol.append(round(float(vol_pct[i]), 2))
            frontier_ret.append(round(float(ret_pct[i]), 2))
            frontier_sharpe.append(round(float(sharpe_arr[i]), 4))
            frontier_backtest_ret.append(round(float(total_ret), 2))
    # Sortie pour l’API
    def _fmt(d):
        return d.to_timestamp().strftime("%Y-%m-%d") if hasattr(d, "to_timestamp") else d.strftime("%Y-%m-%d")
    return {
        "weights": weights_dict,
        "sharpe": round(float(sharpe_arr[max_idx]), 4),
        "expectedReturn": round(float(ret_arr[max_idx]) * 100, 2),
        "volatility": round(float(vol_arr[max_idx]) * 100, 2),
        "maxDrawdown": round(max_drawdown, 2),
        "comparisonData": comparison_data,
        "numPortfolios": num_portfolios,
        "trainPeriodStart": _fmt(train.index[0]),
        "trainPeriodEnd": _fmt(train.index[-1]),
        "testPeriodStart": _fmt(test.index[0]),
        "testPeriodEnd": _fmt(test.index[-1]),
        "efficientFrontier": [{"volatility": v, "expectedReturn": r, "sharpe": s, "backtestReturn": b} for v, r, s, b in zip(frontier_vol, frontier_ret, frontier_sharpe, frontier_backtest_ret)],
    }
