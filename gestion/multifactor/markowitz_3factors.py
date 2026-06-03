"""
Markowitz Fama-French 3 facteurs : les rendements espérés sont estimés par régression OLS
sur les facteurs Mkt-RF, SMB, HML chargés depuis get_facteurs.py.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from yahoo_prices import yf_adj_close_wide, yf_single_ticker_adj_series
from get_facteurs import load_famafrench_factors
from Methodes_de_descente.gradient_pas_fixe import opt_sharpe_gradient
from Methodes_de_descente.gradient_pas_optimal import opt_sharpe_gradient_optimal
from ols_with_stats import ols_factor_regression
from market_metrics import market_sharpe_triplet

"""
Choisir une methode entre : "monte_carlo", "gradient_fixe" et "gradient_optimal" 
"""

FACTORS = ["Mkt-RF", "SMB", "HML"]


def _spy_monthly_pct_change_period_index(start: str, end: str) -> pd.Series | None:
    start_d = datetime.fromisoformat(start[:10])
    end_d = datetime.fromisoformat(end[:10]) + timedelta(days=1)
    spy_s = yf_single_ticker_adj_series("SPY", start_d, end_d, "1d")
    if spy_s is None or spy_s.empty:
        return None
    spy_monthly = spy_s.resample("ME").last().pct_change().dropna()
    spy_monthly.index = spy_monthly.index.to_period("M")
    return spy_monthly


def run(tickers: list[str], start: str, end: str, method: str = "gradient",num_portfolios: int = 10000) -> dict[str, Any]:
    # --- Collecte des prix et rééchantillonnage mensuel ---
    start_d = datetime.fromisoformat(start[:10])
    end_d = datetime.fromisoformat(end[:10]) + timedelta(days=1)
    prices, _missing = yf_adj_close_wide(tickers, start_d, end_d, "1d")
    valid = [c for c in tickers if c in prices.columns]
    if prices.empty or len(valid) < 2:
        return {"error": "Données insuffisantes"}
    prices = prices[valid].dropna(how="any")
    if len(prices) < 60:
        return {"error": "Données insuffisantes"}

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

    # --- Régression OLS à 3 facteurs pour chaque actif (avec tests statistiques) --
    X_factors = factors_train.values  # (n, 3)
    betas_matrix = []  # Va stocker les lignes de bêtas [b_mkt, b_smb, b_hml]
    residual_variances = []
    factor_tests_by_ticker = {}

    for ticker in train_r.columns:
        y = (train_r[ticker] - rf_train).values
        result = ols_factor_regression(y, X_factors, FACTORS)
        
        if result is not None:
            b_mkt = result["coeffs"].get("Mkt-RF", 0.0)
            b_smb = result["coeffs"].get("SMB", 0.0)
            b_hml = result["coeffs"].get("HML", 0.0)
            betas_row = [b_mkt, b_smb, b_hml]
            betas_matrix.append(betas_row)
            
            # Calcul de la variance des résidus
            y_pred = np.dot(X_factors, betas_row)
            residuals = y - y_pred
            residual_variances.append(np.var(residuals))
        else:
            betas_matrix.append([1.0, 0.0, 0.0])
            residual_variances.append(0.01)

    betas_np = np.array(betas_matrix)  # Taille (N_actifs, 3)
    
    # Vrai vecteur mu unique (annualisé)
    mu = np.array([(rf_mean + np.dot(b_row, factor_means)) * 12 for b_row in betas_np])

    # Vrai matrice de covariance de Fama-French
    # Covariance des 3 facteurs entre eux (Mkt, SMB, HML)
    cov_facteurs_annual = factors_train.cov().values * 12
    
    # Produit matriciel : B * Omega * B^T
    cov_factoriel = betas_np @ cov_facteurs_annual @ betas_np.T
    
    # Ajout du risque spécifique (diagonale)
    cov_residual = np.diag(residual_variances) * 12
    
    cov_matrix = cov_factoriel + cov_residual
    rf_annual = rf_mean * 12
    market_sharpe_train, market_sharpe_test, market_sharpe_total = 0.0, 0.0, 0.0
    spy_benchmark = _spy_monthly_pct_change_period_index(start, end)
    if spy_benchmark is not None:
        market_sharpe_train, market_sharpe_test, market_sharpe_total = market_sharpe_triplet(
            spy_benchmark, train_returns.index, test_returns.index, rf_annual, 12
        )

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
        spy_monthly = _spy_monthly_pct_change_period_index(start, end)
        if spy_monthly is not None:
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
        test_mean_ann = float(opt_test.mean()) * 12
        test_vol_ann = float(opt_test.std()) * (12 ** 0.5) if opt_test.std() > 1e-12 else 1e-10
        backtest_sharpe = round((test_mean_ann - rf_annual) / test_vol_ann, 4) if test_vol_ann > 1e-10 else 0.0
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
            "backtestSharpe": backtest_sharpe,
            "marketSharpe": market_sharpe_train,
            "marketBacktestSharpe": market_sharpe_test,
            "marketTotalSharpe": market_sharpe_total,
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
            "factor_tests": factor_tests_by_ticker,
        }
    elif method == "gradient_fixe":
        
        weights_calculated = opt_sharpe_gradient(mu, cov_matrix, rf_annual)
        best_weights = np.asarray(weights_calculated).flatten()

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
        spy_monthly = _spy_monthly_pct_change_period_index(start, end)
        if spy_monthly is not None:
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
        test_mean_ann = float(test_ret_series.mean()) * 12
        test_vol_ann = float(test_ret_series.std()) * (12 ** 0.5) if test_ret_series.std() > 1e-12 else 1e-10
        backtest_sharpe = round((test_mean_ann - rf_annual) / test_vol_ann, 4) if test_vol_ann > 1e-10 else 0.0

        return {
            "weights": weights_dict,
            "sharpe": round(float(opt_sharpe_val), 4),
            "expectedReturn": round(float(opt_ret_val) * 100, 2),
            "volatility": round(float(opt_vol_val) * 100, 2),
            "maxDrawdown": round(max_drawdown, 2),
            "backtestSharpe": backtest_sharpe,
            "marketSharpe": market_sharpe_train,
            "marketBacktestSharpe": market_sharpe_test,
            "marketTotalSharpe": market_sharpe_total,
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
            "factor_tests": factor_tests_by_ticker,
        }
    else:  # gradient_optimal
        mu_pure = np.array(mu).flatten()
        cov_pure = np.array(cov_matrix)
        rf_ann = float(rf_annual)
        
        # Si le taux sans risque passé est resté en échelle % (ex: 4.0), on le repasse en décimal
        if rf_ann > 1.0:
            rf_ann /= 100.0
            
        # Appel de l'algorithme de descente à pas optimal (Uzawa/Robbins-Monro corrigé)
        weights_calculated = opt_sharpe_gradient_optimal(mu_pure, cov_pure, rf_ann)
        best_weights = np.asarray(weights_calculated).flatten()

        # Calcul propre des métriques théoriques annualisées pour la sortie de l'API
        opt_ret_val = float(np.dot(best_weights, mu_pure))
        opt_vol_val = float(np.sqrt(max(best_weights.T @ cov_pure @ best_weights, 1e-10)))
        opt_sharpe_val = (opt_ret_val - rf_ann) / opt_vol_val if opt_vol_val > 1e-10 else 0.0
        
        # Synchronisation stricte de l'index des poids avec le dictionnaire de sortie
        weights_dict = {train_r.columns[i]: float(best_weights[i]) for i in range(len(train_r.columns))}

        # --- Backtest : rendements cumulés ---
        full_common = returns_monthly.index
        portfolio_returns_full = (returns_monthly[train_r.columns] * best_weights).sum(axis=1)
        cum_full = (1 + portfolio_returns_full).cumprod()
        portfolio_series = 100 * cum_full / cum_full.iloc[0]

        # SPY mensuel de référence
        spy_monthly = _spy_monthly_pct_change_period_index(start, end)
        if spy_monthly is not None:
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
        if bp > 1e-12: portfolio_series = 100 * portfolio_series / bp
        if bm > 1e-12: market_series = 100 * market_series / bm

        comparison_data = [
            {"date": str(d), "portfolio": round(_scalar(portfolio_series, d), 2), "market": round(_scalar(market_series, d), 2)}
            for d in portfolio_series.index
        ]

        # --- Max drawdown sur période test ---
        test_common = test_returns.index.intersection(full_common)
        test_ret_series = (returns_monthly.loc[test_common, train_r.columns] * best_weights).sum(axis=1)
        cum_test = (1 + test_ret_series).cumprod()
        max_drawdown = float(-((cum_test - cum_test.cummax()) / cum_test.cummax()).min() * 100) if len(cum_test) > 0 else 0.0
        opt_backtest_ret = round(float((cum_test.iloc[-1] - 1) * 100), 2) if len(cum_test) > 0 else 0
        test_mean_ann = float(test_ret_series.mean()) * 12
        test_vol_ann = float(test_ret_series.std()) * (12 ** 0.5) if test_ret_series.std() > 1e-12 else 1e-10
        backtest_sharpe = round((test_mean_ann - rf_ann) / test_vol_ann, 4) if test_vol_ann > 1e-10 else 0.0

        return {
            "weights": weights_dict,
            "sharpe": round(float(opt_sharpe_val), 4),
            "expectedReturn": round(float(opt_ret_val) * 100, 2),
            "volatility": round(float(opt_vol_val) * 100, 2),
            "maxDrawdown": round(max_drawdown, 2),
            "backtestSharpe": backtest_sharpe,
            "marketSharpe": market_sharpe_train,
            "marketBacktestSharpe": market_sharpe_test,
            "marketTotalSharpe": market_sharpe_total,
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
            "factor_tests": factor_tests_by_ticker,
        }