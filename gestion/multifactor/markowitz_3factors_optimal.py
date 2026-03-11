
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from get_facteurs import load_famafrench_factors
from Methodes_de_descente.gradient_pas_optimal import opt_sharpe_gradient_optimal

FACTORS = ["Mkt-RF", "SMB", "HML"]


def run(tickers: list[str], start: str, end: str, num_portfolios: int = 10000) -> dict[str, Any]:
    # --- Collecte des prix et rééchantillonnage mensuel ---
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

    monthly_prices = prices.resample("ME").last()
    returns_monthly = monthly_prices.pct_change().dropna()
    returns_monthly.index = returns_monthly.index.to_period("M")

    n = len(returns_monthly)
    split = max(int(n * 0.8), 24)
    if split >= n:
        return {"error": "Pas assez de données pour la période"}

    train_returns = returns_monthly.iloc[:split]
    test_returns = returns_monthly.iloc[split:]

    # --- Facteurs Fama-French 3 ---
    try:
        ff = load_famafrench_factors(start=start, end=end)
    except Exception as e:
        return {"error": f"Impossible de charger les facteurs Fama-French : {e}"}

    # Les facteurs sont en pourcentage dans le fichier de Ken French
    ff_pct = ff[FACTORS + ["RF"]] / 100.0

    # Aligner sur la période d'entraînement
    common_train = train_returns.index.intersection(ff_pct.index)
    if len(common_train) < 24:
        return {"error": "Pas assez d'observations alignées pour l'entraînement"}

    train_r = train_returns.loc[common_train]
    ff_train = ff_pct.loc[common_train]

    rf_train = ff_train["RF"]
    factors_train = ff_train[FACTORS]

    rf_mean = float(rf_train.mean())
    factor_means = factors_train.mean().values  # [mean(Mkt-RF), mean(SMB), mean(HML)]

    # --- Régression OLS à 3 facteurs pour chaque actif ---
    X = np.column_stack([np.ones(len(common_train)), factors_train.values])
    mu_monthly = np.zeros(len(valid))

    for idx, ticker in enumerate(train_r.columns):
        y = (train_r[ticker] - rf_train).values
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            # coeffs[0] = alpha (ignoré), coeffs[1:] = betas des 3 facteurs
            betas = coeffs[1:]
        except Exception:
            betas = np.zeros(len(FACTORS))
        mu_monthly[idx] = rf_mean + float(np.dot(betas, factor_means))

    mu = mu_monthly * 12  # annualiser
    cov_matrix = train_r.cov().values * 12
    n_assets = len(valid)
    rf_annual = rf_mean * 12

    best_weights = opt_sharpe_gradient_optimal(mu, cov_matrix, rf_annual)

    # Calcul des métriques optimales à partir des poids trouvés
    opt_ret_val = np.dot(best_weights, mu)
    opt_vol_val = np.sqrt(best_weights @ cov_matrix @ best_weights)
    opt_sharpe_val = (opt_ret_val - rf_annual) / opt_vol_val if opt_vol_val > 1e-10 else 0
    
    weights_dict = {valid[i]: float(best_weights[i]) for i in range(n_assets)}

    # --- Backtest : rendements cumulés ---
    full_common = returns_monthly.index
    portfolio_returns_full = (returns_monthly * best_weights).sum(axis=1)
    cum_full = (1 + portfolio_returns_full).cumprod()
    portfolio_series = 100 * cum_full / cum_full.iloc[0]

    # SPY mensuel de référence
    spy_data = yf.download("SPY", start=start, end=end, auto_adjust=False, progress=False)
    if not spy_data.empty:
        spy_col = spy_data["Adj Close"] if "Adj Close" in spy_data.columns else spy_data["Close"]
        spy_monthly = spy_col.resample("ME").last().pct_change().dropna()
        spy_monthly.index = spy_monthly.index.to_period("M")
        spy_common = full_common.intersection(spy_monthly.index)
        spy_ret = spy_monthly.reindex(spy_common).fillna(0)
        market_series = 100 * (1 + spy_ret).cumprod()
        market_series = market_series.reindex(full_common).ffill().bfill()
    else:
        market_series = portfolio_series.copy()

    # Rebase au début de la période de test
    test_start_date = test_returns.index[0]
    idx_test = full_common[full_common >= test_start_date]
    base_date = idx_test[0] if len(idx_test) > 0 else full_common[-1]

    def _scalar(s, key):
        val = s.loc[key]
        return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)

    bp, bm = _scalar(portfolio_series, base_date), _scalar(market_series, base_date)
    portfolio_series = 100 * portfolio_series / bp
    market_series = 100 * market_series / bm

    comparison_data = [
        {"date": str(d), "portfolio": round(_scalar(portfolio_series, d), 2), "market": round(_scalar(market_series, d), 2)}
        for d in portfolio_series.index
    ]

    # --- Max drawdown sur période test ---
    test_common = test_returns.index.intersection(full_common)
    test_ret_series = (returns_monthly.loc[test_common] * best_weights).sum(axis=1)
    cum_test = (1 + test_ret_series).cumprod()
    max_drawdown = float(-((cum_test - cum_test.cummax()) / cum_test.cummax()).min() * 100)
    opt_backtest_ret = round(float((cum_test.iloc[-1] - 1) * 100), 2) if len(cum_test) > 0 else 0

    return {
        "weights": weights_dict,
        "sharpe": round(float(opt_sharpe_val), 4),
        "expectedReturn": round(float(opt_ret_val) * 100, 2),
        "volatility": round(float(opt_vol_val) * 100, 2),
        "maxDrawdown": round(max_drawdown, 2),
        "comparisonData": comparison_data,
        "numPortfolios": 1,
        "trainPeriodStart": str(train_returns.index[0]),
        "trainPeriodEnd": str(train_returns.index[-1]),
        "testPeriodStart": str(test_returns.index[0]),
        "testPeriodEnd": str(test_returns.index[-1]),
        "efficientFrontier": [{
            "volatility": round(float(opt_vol_val * 100), 2),
            "expectedReturn": round(float(opt_ret_val * 100), 2),
            "sharpe": round(float(opt_sharpe_val), 4),
            "backtestReturn": opt_backtest_ret
        }],
    }