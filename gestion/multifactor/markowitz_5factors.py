"""
Markowitz Fama-French 5 facteurs : les rendements espérés sont estimés par régression OLS
sur les facteurs Mkt-RF, SMB, HML, RMW, CMA chargés depuis get_facteurs.py.
Fallback ridge (lambda=1e-4) en cas de quasi-singularité de X'X.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from get_facteurs import load_famafrench_5factors
from Methodes_de_descente.gradient_pas_optimal import opt_sharpe_gradient_optimal
from Methodes_de_descente.gradient_pas_fixe import opt_sharpe_gradient

"""
Choisir une methode entre : "monte_carlo", "gradient_fixe" et "gradient_optimal" 
"""

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
RIDGE_LAMBDA = 1e-4


def _ols_or_ridge(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS avec fallback ridge si X'X est quasi-singulière."""
    XtX = X.T @ X
    Xty = X.T @ y
    cond = np.linalg.cond(XtX)
    if cond > 1e12:
        # Ridge : (X'X + lambda*I) beta = X'y
        reg = XtX + RIDGE_LAMBDA * np.eye(XtX.shape[0])
        return np.linalg.solve(reg, Xty)
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return coeffs
    except np.linalg.LinAlgError:
        reg = XtX + RIDGE_LAMBDA * np.eye(XtX.shape[0])
        return np.linalg.solve(reg, Xty)


def run(tickers: list[str], start: str, end: str, method: str = "gradient",num_portfolios: int = 10000) -> dict[str, Any]:
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

    # --- Facteurs Fama-French 5 ---
    try:
        ff = load_famafrench_5factors(start=start, end=end)
    except Exception as e:
        return {"error": f"Impossible de charger les facteurs Fama-French 5 : {e}"}

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
    factor_means = factors_train.mean().values  # [Mkt-RF, SMB, HML, RMW, CMA] moyennes

    # --- Régression OLS (ou ridge) à 5 facteurs pour chaque actif ---
    X = np.column_stack([np.ones(len(common_train)), factors_train.values])
    mu_monthly = np.zeros(len(valid))

    for idx, ticker in enumerate(train_r.columns):
        y = (train_r[ticker] - rf_train).values
        try:
            coeffs = _ols_or_ridge(X, y)
            betas = coeffs[1:]  # coeffs[0] = alpha, ignoré
        except Exception:
            betas = np.zeros(len(FACTORS))
        mu_monthly[idx] = rf_mean + float(np.dot(betas, factor_means))

    mu = mu_monthly * 12  # annualiser
    cov_matrix = train_r.cov().values * 12
    n_assets = len(valid)
    rf_annual = rf_mean * 12

    if method == "monte_carlo":
        # --- Monte-Carlo ---
        all_weights = np.zeros((num_portfolios, n_assets))
        ret_arr = np.zeros(num_portfolios)
        vol_arr = np.zeros(num_portfolios)
        sharpe_arr = np.zeros(num_portfolios)
        for i in range(num_portfolios):
            w = np.random.random(n_assets)
            w /= w.sum()
            all_weights[i] = w
            ret_arr[i] = np.dot(w, mu)
            vol_arr[i] = np.sqrt(w @ cov_matrix @ w)
            sharpe_arr[i] = (ret_arr[i] - rf_annual) / vol_arr[i] if vol_arr[i] > 1e-10 else 0

        max_idx = sharpe_arr.argmax()
        best_weights = all_weights[max_idx]
        weights_dict = {valid[i]: float(best_weights[i]) for i in range(n_assets)}

        # --- Backtest : rendements cumulés sur toute la période ---
        full_common = returns_monthly.index
        portfolio_returns_full = (returns_monthly * best_weights).sum(axis=1)
        cum_full = (1 + portfolio_returns_full).cumprod()
        portfolio_series = 100 * cum_full / cum_full.iloc[0]

        # SPY mensuel de référence
        spy_data = yf.download("SPY", start=start, end=end, auto_adjust=False, progress=False)
        if not spy_data.empty and ("Adj Close" in spy_data.columns or "Close" in spy_data.columns):
            spy_col = spy_data["Adj Close"] if "Adj Close" in spy_data.columns else spy_data["Close"]
            if isinstance(spy_col, pd.DataFrame):
                spy_col = spy_col.iloc[:, 0]
            spy_monthly = spy_col.resample("ME").last().pct_change().dropna()
            spy_monthly.index = spy_monthly.index.to_period("M")
            spy_common = full_common.intersection(spy_monthly.index)
            spy_ret = spy_monthly.reindex(spy_common).fillna(0)
            spy_cum = (1 + spy_ret).cumprod()
            market_series = 100 * spy_cum / spy_cum.iloc[0]
            market_series = market_series.reindex(full_common).ffill().bfill()
        else:
            market_series = portfolio_series.copy()

        # Rebase à 100 au premier mois de la période de test
        test_start_date = test_returns.index[0]
        idx_at_or_after = full_common[full_common >= test_start_date]
        base_date = idx_at_or_after[0] if len(idx_at_or_after) > 0 else full_common[-1]

        def _scalar(s, key):
            val = s.loc[key]
            return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)

        base_p = _scalar(portfolio_series, base_date)
        base_m = _scalar(market_series, base_date)
        if base_p > 1e-12:
            portfolio_series = 100 * portfolio_series / base_p
        if base_m > 1e-12:
            market_series = 100 * market_series / base_m

        comparison_data = [
            {"date": str(d), "portfolio": round(_scalar(portfolio_series, d), 2), "market": round(_scalar(market_series, d), 2)}
            for d in portfolio_series.index
        ]

        # --- Max drawdown sur période test ---
        test_common = test_returns.index.intersection(full_common)
        test_ret_series = (returns_monthly.loc[test_common] * best_weights).sum(axis=1)
        cum_test = (1 + test_ret_series).cumprod()
        peak = cum_test.cummax()
        drawdown = (cum_test - peak) / peak
        dd_min = drawdown.min()
        dd_min = float(dd_min.iloc[0]) if isinstance(dd_min, pd.Series) else float(dd_min)
        max_drawdown = float(-dd_min * 100) if len(drawdown) > 0 else 0.0

        # --- Frontière efficiente ---
        vol_pct = vol_arr * 100
        ret_pct = ret_arr * 100
        order = np.argsort(vol_pct)
        frontier_vol, frontier_ret, frontier_sharpe, frontier_backtest_ret = [], [], [], []
        max_ret_so_far = -np.inf
        for i in order:
            if ret_pct[i] >= max_ret_so_far:
                max_ret_so_far = ret_pct[i]
                port_test = (returns_monthly.loc[test_common] * np.asarray(all_weights[i])).sum(axis=1)
                total_ret = (np.prod(1 + port_test) - 1) * 100
                frontier_vol.append(round(float(vol_pct[i]), 2))
                frontier_ret.append(round(float(ret_pct[i]), 2))
                frontier_sharpe.append(round(float(sharpe_arr[i]), 4))
                frontier_backtest_ret.append(round(float(total_ret), 2))

        opt_vol = round(float(vol_pct[max_idx]), 2)
        opt_ret = round(float(ret_pct[max_idx]), 2)
        opt_sharpe = round(float(sharpe_arr[max_idx]), 4)
        opt_test = (returns_monthly.loc[test_common] * best_weights).sum(axis=1)
        opt_backtest_ret = round(float((np.prod(1 + opt_test) - 1) * 100), 2)
        if (opt_vol, opt_ret) not in list(zip(frontier_vol, frontier_ret)):
            frontier_vol.append(opt_vol)
            frontier_ret.append(opt_ret)
            frontier_sharpe.append(opt_sharpe)
            frontier_backtest_ret.append(opt_backtest_ret)

        return {
            "weights": weights_dict,
            "sharpe": round(float(sharpe_arr[max_idx]), 4),
            "expectedReturn": round(float(ret_arr[max_idx]) * 100, 2),
            "volatility": round(float(vol_arr[max_idx]) * 100, 2),
            "maxDrawdown": round(max_drawdown, 2),
            "comparisonData": comparison_data,
            "numPortfolios": num_portfolios,
            "trainPeriodStart": str(train_returns.index[0]),
            "trainPeriodEnd": str(train_returns.index[-1]),
            "testPeriodStart": str(test_returns.index[0]),
            "testPeriodEnd": str(test_returns.index[-1]),
            "efficientFrontier": [
                {"volatility": v, "expectedReturn": r, "sharpe": s, "backtestReturn": b}
                for v, r, s, b in zip(frontier_vol, frontier_ret, frontier_sharpe, frontier_backtest_ret)
            ],
        }
    elif method == "gradient_fixe":
        best_weights = opt_sharpe_gradient(mu, cov_matrix, rf_annual)
        
        # Métriques optimales
        opt_ret_val = np.dot(best_weights, mu)
        opt_vol_val = np.sqrt(best_weights.T @ cov_matrix @ best_weights)
        opt_sharpe_val = (opt_ret_val - rf_annual) / opt_vol_val if opt_vol_val > 1e-10 else 0
        weights_dict = {valid[i]: float(best_weights[i]) for i in range(n_assets)}

        # --- Backtest ---
        full_common = returns_monthly.index
        portfolio_returns_full = (returns_monthly * best_weights).sum(axis=1)
        cum_full = (1 + portfolio_returns_full).cumprod()
        portfolio_series = 100 * cum_full / cum_full.iloc[0]

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

        # Rebase
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

        # Max Drawdown
        test_common = test_returns.index.intersection(full_common)
        test_ret_series = (returns_monthly.loc[test_common] * best_weights).sum(axis=1)
        cum_test = (1 + test_ret_series).cumprod()
        max_drawdown = float(-((cum_test - cum_test.cummax()) / cum_test.cummax()).min() * 100)
        opt_backtest_ret = round(float((cum_test.iloc[-1] - 1) * 100), 2)

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
    else:  # gradient_optimal
        best_weights = opt_sharpe_gradient_optimal(mu, cov_matrix, rf_annual)
    
        # Métriques optimales
        opt_ret_val = np.dot(best_weights, mu)
        opt_vol_val = np.sqrt(best_weights.T @ cov_matrix @ best_weights)
        opt_sharpe_val = (opt_ret_val - rf_annual) / opt_vol_val if opt_vol_val > 1e-10 else 0
        weights_dict = {valid[i]: float(best_weights[i]) for i in range(n_assets)}

        # --- Backtest ---
        full_common = returns_monthly.index
        portfolio_returns_full = (returns_monthly * best_weights).sum(axis=1)
        cum_full = (1 + portfolio_returns_full).cumprod()
        portfolio_series = 100 * cum_full / cum_full.iloc[0]

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

        # Rebase
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

        # Max Drawdown
        test_common = test_returns.index.intersection(full_common)
        test_ret_series = (returns_monthly.loc[test_common] * best_weights).sum(axis=1)
        cum_test = (1 + test_ret_series).cumprod()
        max_drawdown = float(-((cum_test - cum_test.cummax()) / cum_test.cummax()).min() * 100)
        opt_backtest_ret = round(float((cum_test.iloc[-1] - 1) * 100), 2)

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

