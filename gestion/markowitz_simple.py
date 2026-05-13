"""
Markowitz classique : moyenne-variance sur rendements historiques uniquement.
Pas de facteurs, pas de Fama-French. On estime mu et Sigma sur l’historique, puis on maximise le Sharpe.
"""
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from Methodes_de_descente.gradient_pas_fixe import opt_sharpe_gradient
from Methodes_de_descente.gradient_pas_optimal import opt_sharpe_gradient_optimal
from yahoo_prices import yf_adj_close_wide, yf_single_ticker_adj_series

"""
Choisir une methode entre : "monte_carlo", "gradient_fixe" et "gradient_optimal" 
"""


def _spy_adj_series(index_start, index_end) -> pd.Series | None:
    start_d = pd.Timestamp(index_start).to_pydatetime()
    end_d = pd.Timestamp(index_end).to_pydatetime() + timedelta(days=1)
    return yf_single_ticker_adj_series("SPY", start_d, end_d, "1d")


def run(tickers: list[str], start: str, end: str, risk_free_rate: float = 0.03, method: str = "gradient", num_portfolios: int = 10000) -> dict[str, Any]:
    start_d = datetime.fromisoformat(start[:10])
    end_d = datetime.fromisoformat(end[:10]) + timedelta(days=1)
    prices, _missing = yf_adj_close_wide(tickers, start_d, end_d, "1d")
    valid = [c for c in tickers if c in prices.columns]
    if prices.empty or len(valid) < 2:
        return {"error": "Données insuffisantes"}
    prices = prices[valid].dropna(how="any")
    if len(prices) < 15:
        return {"error": "Données insuffisantes"}

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

    if method == "monte_carlo":

        # Monte Carlo : on tire des poids aléatoires et on garde le portefeuille avec le meilleur Sharpe
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
        weights_dict = {valid[i]: float(np.asarray(best_weights)[i]) for i in range(n_assets)}

        # Série portefeuille et marché sur 100 % de la période (base 100 au premier jour du backtest)
        full_returns = np.log(prices / prices.shift(1)).dropna()
        portfolio_returns_full = (full_returns * best_weights).sum(axis=1)
        cum_full = (1 + portfolio_returns_full).cumprod()
        portfolio_series = 100 * cum_full / cum_full.iloc[0]

        spy_s = _spy_adj_series(prices.index[0], prices.index[-1])
        if spy_s is not None and not spy_s.empty:
            spy_ret = np.log(spy_s / spy_s.shift(1)).dropna()
            spy_cum = (1 + spy_ret).cumprod()
            market_series = 100 * spy_cum / spy_cum.iloc[0]
            common_idx = portfolio_series.index.intersection(market_series.index)
            portfolio_series = portfolio_series.reindex(common_idx).ffill().bfill()
            market_series = market_series.reindex(common_idx).ffill().bfill()
        else:
            market_series = portfolio_series.copy()
            common_idx = portfolio_series.index

        # Renormaliser base 100 au premier jour de la période de backtest
        test_start_date = test_prices.index[0]
        idx_at_or_after = common_idx[common_idx >= test_start_date]
        base_date = idx_at_or_after[0] if len(idx_at_or_after) > 0 else common_idx[-1]
        _val_p = portfolio_series.loc[base_date]
        _val_m = market_series.loc[base_date]
        base_p = float(_val_p.iloc[0]) if isinstance(_val_p, pd.Series) else float(_val_p)
        base_m = float(_val_m.iloc[0]) if isinstance(_val_m, pd.Series) else float(_val_m)
        if base_p > 1e-12 and base_m > 1e-12:
            portfolio_series = 100 * portfolio_series / base_p
            market_series = 100 * market_series / base_m

        def _to_scalar(s, key):
            val = s.loc[key]
            return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)

        comparison_data = [
            {"date": d.strftime("%Y-%m-%d"), "portfolio": round(_to_scalar(portfolio_series, d), 2), "market": round(_to_scalar(market_series, d), 2)}
            for d in portfolio_series.index
        ]

        # Max drawdown sur période test pour cohérence avec les métriques
        test_returns = np.log(test_prices / test_prices.shift(1)).dropna()
        portfolio_returns_test = (test_returns * best_weights).sum(axis=1)
        cum_test = (1 + portfolio_returns_test).cumprod()
        peak = cum_test.cummax()
        drawdown = (cum_test - peak) / peak
        _dd_min = drawdown.min()
        _dd_min = _dd_min.iloc[0] if isinstance(_dd_min, pd.Series) else _dd_min
        max_drawdown = float(-_dd_min * 100) if len(drawdown) > 0 else 0

        # Frontière efficiente : points (vol, rendement) Pareto-optimaux (pour chaque vol, rendement max)
        vol_pct = vol_arr * 100
        ret_pct = ret_arr * 100
        order = np.argsort(vol_pct)
        frontier_vol, frontier_ret, frontier_sharpe, frontier_backtest_ret = [], [], [], []
        max_ret_so_far = -np.inf
        for i in order:
            if ret_pct[i] >= max_ret_so_far:
                max_ret_so_far = ret_pct[i]
                # Rendement réel sur la période de test (backtest)
                port_ret_test = (test_returns * np.asarray(all_weights[i])).sum(axis=1)
                total_ret = (np.exp(port_ret_test.sum()) - 1) * 100  # log -> géométrique total
                frontier_vol.append(round(float(vol_pct[i]), 2))
                frontier_ret.append(round(float(ret_pct[i]), 2))
                frontier_sharpe.append(round(float(sharpe_arr[i]), 4))
                frontier_backtest_ret.append(round(float(total_ret), 2))

        # Toujours inclure le portefeuille optimal pour que son backtestReturn corresponde à la courbe
        opt_vol = round(float(vol_pct[max_idx]), 2)
        opt_ret = round(float(ret_pct[max_idx]), 2)
        opt_sharpe = round(float(sharpe_arr[max_idx]), 4)
        port_ret_test_opt = (test_returns * np.asarray(best_weights)).sum(axis=1)
        opt_backtest_ret = round(float((np.exp(port_ret_test_opt.sum()) - 1) * 100), 2)
        if (opt_vol, opt_ret) not in list(zip(frontier_vol, frontier_ret)):
            frontier_vol.append(opt_vol)
            frontier_ret.append(opt_ret)
            frontier_sharpe.append(opt_sharpe)
            frontier_backtest_ret.append(opt_backtest_ret)

        # Sortie pour l’API
        train_start = train_prices.index[0].strftime("%Y-%m-%d")
        train_end = train_prices.index[-1].strftime("%Y-%m-%d")
        test_start = test_prices.index[0].strftime("%Y-%m-%d")
        test_end = test_prices.index[-1].strftime("%Y-%m-%d")
        return {
            "weights": weights_dict,
            "sharpe": round(float(sharpe_arr[max_idx]), 4),
            "expectedReturn": round(float(ret_arr[max_idx]) * 100, 2),
            "volatility": round(float(vol_arr[max_idx]) * 100, 2),
            "maxDrawdown": round(max_drawdown, 2),
            "comparisonData": comparison_data,
            "numPortfolios": num_portfolios,
            "trainPeriodStart": train_start,
            "trainPeriodEnd": train_end,
            "testPeriodStart": test_start,
            "testPeriodEnd": test_end,
            "efficientFrontier": [{"volatility": v, "expectedReturn": r, "sharpe": s, "backtestReturn": b} for v, r, s, b in zip(frontier_vol, frontier_ret, frontier_sharpe, frontier_backtest_ret)],
        }
    elif method == "gradient_fixe":
        best_weights = opt_sharpe_gradient(mean_returns.values, cov_matrix.values, risk_free_rate)
        best_weights = np.asarray(best_weights).ravel()

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

        spy_s = _spy_adj_series(prices.index[0], prices.index[-1])
        if spy_s is not None and not spy_s.empty:
            spy_ret = np.log(spy_s / spy_s.shift(1)).dropna()
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

        # --- MAX DRAWDOWN et métriques backtest ---
        test_returns = np.log(test_prices / test_prices.shift(1)).dropna()
        portfolio_returns_test = (test_returns * best_weights).sum(axis=1)
        cum_test = np.exp(portfolio_returns_test.cumsum())
        peak = cum_test.cummax()
        drawdown = (cum_test - peak) / peak
        _dd_min = drawdown.min()
        max_drawdown = float(-_dd_min * 100) if not pd.isna(_dd_min) else 0
        opt_backtest_ret = round(float((cum_test.iloc[-1] - 1) * 100), 2) if len(cum_test) > 0 else 0
        ann_mean = float(portfolio_returns_test.mean()) * 252
        ann_vol = float(portfolio_returns_test.std()) * (252 ** 0.5) if portfolio_returns_test.std() > 1e-12 else 1e-10
        backtest_sharpe = round((ann_mean - risk_free_rate) / ann_vol, 4) if ann_vol > 1e-10 else 0.0

        return {
            "weights": weights_dict,
            "sharpe": round(float(opt_sharpe_val), 4),
            "expectedReturn": round(float(opt_ret_val * 100), 2),
            "volatility": round(float(opt_vol_val * 100), 2),
            "maxDrawdown": round(max_drawdown, 2),
            "backtestReturn": opt_backtest_ret,
            "backtestSharpe": backtest_sharpe,
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
    else: #method = "gradient_optimal"
        best_weights = opt_sharpe_gradient_optimal(mean_returns.values, cov_matrix.values, risk_free_rate)
        best_weights = np.asarray(best_weights).ravel()

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

        spy_s = _spy_adj_series(prices.index[0], prices.index[-1])
        if spy_s is not None and not spy_s.empty:
            spy_ret = np.log(spy_s / spy_s.shift(1)).dropna()
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

        # --- MAX DRAWDOWN et métriques backtest ---
        test_returns = np.log(test_prices / test_prices.shift(1)).dropna()
        portfolio_returns_test = (test_returns * best_weights).sum(axis=1)
        cum_test = np.exp(portfolio_returns_test.cumsum())
        peak = cum_test.cummax()
        drawdown = (cum_test - peak) / peak
        _dd_min = drawdown.min()
        max_drawdown = float(-_dd_min * 100) if not pd.isna(_dd_min) else 0
        opt_backtest_ret = round(float((cum_test.iloc[-1] - 1) * 100), 2) if len(cum_test) > 0 else 0
        ann_mean = float(portfolio_returns_test.mean()) * 252
        ann_vol = float(portfolio_returns_test.std()) * (252 ** 0.5) if portfolio_returns_test.std() > 1e-12 else 1e-10
        backtest_sharpe = round((ann_mean - risk_free_rate) / ann_vol, 4) if ann_vol > 1e-10 else 0.0

        return {
            "weights": weights_dict,
            "sharpe": round(float(opt_sharpe_val), 4),
            "expectedReturn": round(float(opt_ret_val * 100), 2),
            "volatility": round(float(opt_vol_val * 100), 2),
            "maxDrawdown": round(max_drawdown, 2),
            "backtestReturn": opt_backtest_ret,
            "backtestSharpe": backtest_sharpe,
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


