
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Any
from Methodes_de_descente.gradient_pas_optimal import opt_sharpe_gradient_optimal

def run(tickers: list[str], start: str, end: str, risk_free_rate: float = 0.03, num_portfolios: int = 10000) -> dict[str, Any]:
    # Récup des prix (yfinance peut renvoyer MultiIndex selon le nombre de tickers)
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

    # Train 80% / test 20%, en quotidien
    n = len(prices)
    split = int(n * 0.8)
    if split < 10:
        return {"error": "Pas assez de données pour la période"}
    train_prices = prices.iloc[:split]
    test_prices = prices.iloc[split:]

    # Rendements log, annualisés (252 jours)
    returns = np.log(train_prices / train_prices.shift(1)).dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    n_assets = len(valid)

    best_weights = opt_sharpe_gradient_optimal(mean_returns.values, cov_matrix.values, risk_free_rate)

    
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Any
from Methodes_de_descente.gradient_pas_fixe import opt_sharpe_gradient


def run(tickers: list[str], start: str, end: str, risk_free_rate: float = 0.03, num_portfolios: int = 10000) -> dict[str, Any]:
    # Récup des prix (yfinance peut renvoyer MultiIndex selon le nombre de tickers)
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

    # Train 80% / test 20%, en quotidien
    n = len(prices)
    split = int(n * 0.8)
    if split < 10:
        return {"error": "Pas assez de données pour la période"}
    train_prices = prices.iloc[:split]
    test_prices = prices.iloc[split:]

    # Rendements log, annualisés (252 jours)
    returns = np.log(train_prices / train_prices.shift(1)).dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    n_assets = len(valid)

    best_weights = opt_sharpe_gradient_optimal(mean_returns.values, cov_matrix.values, risk_free_rate)

    # --- CALCUL DES MÉTRIQUES OPTIMALES ---
    opt_ret_val = np.sum(mean_returns * best_weights)
    opt_vol_val = np.sqrt(best_weights.T @ cov_matrix @ best_weights)
    opt_sharpe_val = (opt_ret_val - risk_free_rate) / opt_vol_val if opt_vol_val > 1e-10 else 0
    
    weights_dict = {valid[i]: float(best_weights[i]) for i in range(len(valid))}

    # --- SÉRIE PORTEFEUILLE ET MARCHÉ ---
    full_returns = np.log(prices / prices.shift(1)).dropna()
    portfolio_returns_full = (full_returns * best_weights).sum(axis=1)
    cum_full = (1 + portfolio_returns_full).cumprod()
    portfolio_series = 100 * cum_full / cum_full.iloc[0]

    spy = yf.download("SPY", start=prices.index[0], end=prices.index[-1], auto_adjust=False, progress=False)
    if not spy.empty and "Adj Close" in spy.columns:
        spy_ret = np.log(spy["Adj Close"] / spy["Adj Close"].shift(1)).dropna()
        spy_cum = (1 + spy_ret).cumprod()
        market_series = 100 * spy_cum / spy_cum.iloc[0]
        common_idx = portfolio_series.index.intersection(market_series.index)
        portfolio_series = portfolio_series.reindex(common_idx).ffill().bfill()
        market_series = market_series.reindex(common_idx).ffill().bfill()
    else:
        market_series = portfolio_series.copy()
        common_idx = portfolio_series.index

    # Renormalisation base 100 au début de la période de TEST
    test_start_date = test_prices.index[0]
    idx_at_or_after = common_idx[common_idx >= test_start_date]
    base_date = idx_at_or_after[0] if len(idx_at_or_after) > 0 else common_idx[-1]
    
    def _to_scalar(s, key):
        val = s.loc[key]
        return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)

    base_p = _to_scalar(portfolio_series, base_date)
    base_m = _to_scalar(market_series, base_date)
    if base_p > 1e-12 and base_m > 1e-12:
        portfolio_series = 100 * portfolio_series / base_p
        market_series = 100 * market_series / base_m

    comparison_data = [
        {"date": d.strftime("%Y-%m-%d"), "portfolio": round(_to_scalar(portfolio_series, d), 2), "market": round(_to_scalar(market_series, d), 2)}
        for d in portfolio_series.index
    ]

    # --- MAX DRAWDOWN ---
    test_returns = np.log(test_prices / test_prices.shift(1)).dropna()
    portfolio_returns_test = (test_returns * best_weights).sum(axis=1)
    # On repasse en rendements arithmétiques pour le cumul réel
    cum_test = np.exp(portfolio_returns_test.cumsum()) 
    peak = cum_test.cummax()
    drawdown = (cum_test - peak) / peak
    _dd_min = drawdown.min()
    max_drawdown = float(-_dd_min * 100) if not pd.isna(_dd_min) else 0

    # --- RETOUR API ---
    opt_backtest_ret = round(float((cum_test.iloc[-1] - 1) * 100), 2) if len(cum_test) > 0 else 0

    return {
        "weights": weights_dict,
        "sharpe": round(float(opt_sharpe_val), 4),
        "expectedReturn": round(float(opt_ret_val * 100), 2),
        "volatility": round(float(opt_vol_val * 100), 2),
        "maxDrawdown": round(max_drawdown, 2),
        "comparisonData": comparison_data,
        "numPortfolios": 1,
        "trainPeriodStart": train_prices.index[0].strftime("%Y-%m-%d"),
        "trainPeriodEnd": train_prices.index[-1].strftime("%Y-%m-%d"),
        "testPeriodStart": test_prices.index[0].strftime("%Y-%m-%d"),
        "testPeriodEnd": test_prices.index[-1].strftime("%Y-%m-%d"),
        "efficientFrontier": [{
            "volatility": round(float(opt_vol_val * 100), 2),
            "expectedReturn": round(float(opt_ret_val * 100), 2),
            "sharpe": round(float(opt_sharpe_val), 4),
            "backtestReturn": opt_backtest_ret
        }]
    }