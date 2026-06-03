"""
Markowitz CAPM (1 facteur) : les rendements espérés sont estimés par la définition 
analytique exacte de la covariance avec le marché (MEDAF / CAPM).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any
from yahoo_prices import yf_adj_close_wide, yf_single_ticker_adj_series
from Methodes_de_descente.gradient_pas_fixe import opt_sharpe_gradient
from Methodes_de_descente.gradient_pas_optimal import opt_sharpe_gradient_optimal
from ols_with_stats import ols_factor_regression
from market_metrics import market_sharpe_triplet

"""
Choisir une methode entre : "monte_carlo", "gradient_fixe" et "gradient_optimal" 
"""

def run(tickers: list[str], start: str, end: str, method: str = "gradient", num_portfolios: int = 10000) -> dict[str, Any]:
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

    # Rééchantillonnage mensuel (dernier jour ouvré du mois)
    monthly_prices = prices.resample("ME").last()
    returns_monthly = monthly_prices.pct_change().dropna()

    n = len(returns_monthly)
    split = max(int(n * 0.8), 24)
    if split >= n:
        return {"error": "Pas assez de données pour la période"}

    train_returns = returns_monthly.iloc[:split]
    test_returns = returns_monthly.iloc[split:]

    # --- Facteur de marché : SPY mensuel ---
    spy_s = yf_single_ticker_adj_series("SPY", start_d, end_d, "1d")
    if spy_s is None or spy_s.empty:
        return {"error": "Impossible de télécharger SPY"}
    spy_monthly = spy_s.resample("ME").last()
    rm = spy_monthly.pct_change().dropna()

    # Taux sans risque mensuel depuis ^IRX
    try:
        irx_s = yf_single_ticker_adj_series("^IRX", start_d, end_d, "1d")
        if irx_s is not None and not irx_s.empty:
            rf_monthly = (irx_s / 100 / 12).resample("ME").last()
        else:
            rf_monthly = pd.Series(0.02 / 12, index=rm.index)
    except Exception:
        rf_monthly = pd.Series(0.02 / 12, index=rm.index)

    # Aligner les index sur les rendements mensuels des actifs (train)
    common_train = train_returns.index.intersection(rm.index).intersection(rf_monthly.index)
    if len(common_train) < 24:
        return {"error": "Pas assez d'observations alignées pour l'entraînement"}

    train_r = train_returns.loc[common_train]
    rm_train = rm.loc[common_train]
    rf_train = rf_monthly.reindex(common_train).ffill().fillna(0.02 / 12)

    mkt_excess = (rm_train - rf_train).values
    rf_mean = float(rf_train.mean())
    mkt_excess_mean = float(mkt_excess.mean())
    
    # Variance du marché de référence (échantillon ddof=1)
    mkt_var = np.var(mkt_excess, ddof=1)

    # --- CALCUL FINANCIER ROBUSTE : CAPM ET MATRICE DE COVARIANCE ---
    betas_dict = {}
    residual_variances = []
    factor_tests_by_ticker = {}
    
    X_mkt = mkt_excess.reshape(-1, 1)
    
    for ticker in train_r.columns:
        y = (train_r[ticker] - rf_train).values
        
        # 1. Calcul du Bêta purement mathématique (Insensible aux bugs externes)
        cov_matrix_2x2 = np.cov(y, mkt_excess)
        beta_math = cov_matrix_2x2[0, 1] / mkt_var
        betas_dict[ticker] = beta_math
        
        # 2. Calcul du risque spécifique (Variance des résidus exacts avec Alpha)
        alpha = np.mean(y) - beta_math * mkt_excess_mean
        residuals = y - (alpha + beta_math * mkt_excess)
        residual_variances.append(np.var(residuals, ddof=1))
        
        # 3. Appel à OLS uniquement pour peupler l'API d'affichage du frontend
        result = ols_factor_regression(y, X_mkt, ["Mkt-RF"])
        if result is not None:
            factor_tests_by_ticker[ticker] = {
                "factor_stats": result["factor_tests"],
                "model_stats": result["model_stats"],
            }
        else:
            factor_tests_by_ticker[ticker] = {"factor_stats": {}, "model_stats": None}

    # Reconstruction de mu (Rendements Attendus) 
    mu = np.array([
        (rf_mean + betas_dict[ticker] * mkt_excess_mean) * 12 
        for ticker in train_r.columns
    ])

    # Reconstruction de la vraie matrice de covariance à un facteur (Théorème de Sharpe)
    betas_vector = np.array([betas_dict[ticker] for ticker in train_r.columns]).reshape(-1, 1)
    
    # Produit vectoriel des Bêtas * Variance du marché annualisée
    cov_factoriel = (betas_vector @ betas_vector.T) * (mkt_var * 12)
    
    # Ajout du risque idiosyncratique sur la diagonale
    cov_residual = np.diag(residual_variances) * 12
    cov_matrix = cov_factoriel + cov_residual
    
    n_assets = len(train_r.columns)
    rf_annual = rf_mean * 12
    
    market_sharpe_train, market_sharpe_test, market_sharpe_total = market_sharpe_triplet(
        rm, train_returns.index, test_returns.index, rf_annual, 12
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
        weights_dict = {train_r.columns[i]: float(best_weights[i]) for i in range(n_assets)}

        # --- Backtest : rendements cumulés sur toute la période ---
        full_common = returns_monthly.index.intersection(rm.index)
        full_returns = returns_monthly.reindex(full_common)
        portfolio_returns_full = (full_returns[train_r.columns] * best_weights).sum(axis=1)
        cum_full = (1 + portfolio_returns_full).cumprod()
        portfolio_series = 100 * cum_full / cum_full.iloc[0]

        # SPY de référence sur la même plage
        spy_full = spy_monthly.reindex(full_common).pct_change()
        spy_cum = (1 + spy_full.fillna(0)).cumprod()
        market_series = 100 * spy_cum / spy_cum.iloc[0]

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
            {"date": d.strftime("%Y-%m-%d"), "portfolio": round(_scalar(portfolio_series, d), 2), "market": round(_scalar(market_series, d), 2)}
            for d in portfolio_series.index
        ]

        # --- Max drawdown et métriques backtest sur période test ---
        test_common = test_returns.index.intersection(full_common)
        test_ret_series = (returns_monthly.loc[test_common, train_r.columns] * best_weights).sum(axis=1)
        cum_test = (1 + test_ret_series).cumprod()
        peak = cum_test.cummax()
        drawdown = (cum_test - peak) / peak
        dd_min = drawdown.min()
        dd_min = float(dd_min.iloc[0]) if isinstance(dd_min, pd.Series) else float(dd_min)
        max_drawdown = float(-dd_min * 100) if len(drawdown) > 0 else 0.0
        opt_backtest_ret = round(float((np.prod(1 + test_ret_series) - 1) * 100), 2)
        test_mean_ann = float(test_ret_series.mean()) * 12
        test_vol_ann = float(test_ret_series.std()) * (12 ** 0.5) if test_ret_series.std() > 1e-12 else 1e-10
        backtest_sharpe = round((test_mean_ann - rf_annual) / test_vol_ann, 4) if test_vol_ann > 1e-10 else 0.0

        # --- Frontière efficiente ---
        vol_pct = vol_arr * 100
        ret_pct = ret_arr * 100
        order = np.argsort(vol_pct)
        frontier_vol, frontier_ret, frontier_sharpe, frontier_backtest_ret = [], [], [], []
        max_ret_so_far = -np.inf
        for i in order:
            if ret_pct[i] >= max_ret_so_far:
                max_ret_so_far = ret_pct[i]
                port_test = (returns_monthly.loc[test_common, train_r.columns] * np.asarray(all_weights[i])).sum(axis=1)
                total_ret = (np.prod(1 + port_test) - 1) * 100
                frontier_vol.append(round(float(vol_pct[i]), 2))
                frontier_ret.append(round(float(ret_pct[i]), 2))
                frontier_sharpe.append(round(float(sharpe_arr[i]), 4))
                frontier_backtest_ret.append(round(float(total_ret), 2))

        opt_vol = round(float(vol_pct[max_idx]), 2)
        opt_ret = round(float(ret_pct[max_idx]), 2)
        opt_sharpe = round(float(sharpe_arr[max_idx]), 4)
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
            "backtestReturn": opt_backtest_ret,
            "backtestSharpe": backtest_sharpe,
            "marketSharpe": market_sharpe_train,
            "marketBacktestSharpe": market_sharpe_test,
            "marketTotalSharpe": market_sharpe_total,
            "comparisonData": comparison_data,
            "numPortfolios": num_portfolios,
            "trainPeriodStart": train_returns.index[0].strftime("%Y-%m-%d"),
            "trainPeriodEnd": train_returns.index[-1].strftime("%Y-%m-%d"),
            "testPeriodStart": test_returns.index[0].strftime("%Y-%m-%d"),
            "testPeriodEnd": test_returns.index[-1].strftime("%Y-%m-%d"),
            "efficientFrontier": [
                {"volatility": v, "expectedReturn": r, "sharpe": s, "backtestReturn": b}
                for v, r, s, b in zip(frontier_vol, frontier_ret, frontier_sharpe, frontier_backtest_ret)
            ],
            "factor_tests": factor_tests_by_ticker,
        }
    elif method == "gradient_fixe":
        
        weights_calculated = opt_sharpe_gradient(mu, cov_matrix, rf_annual)
        best_weights = np.asarray(weights_calculated).flatten()

        # Calcul des métriques optimales pour le retour API
        opt_ret_val = np.dot(best_weights, mu)
        opt_vol_val = np.sqrt(max(best_weights.T @ cov_matrix @ best_weights, 1e-10))
        opt_sharpe_val = (opt_ret_val - rf_annual) / opt_vol_val if opt_vol_val > 1e-10 else 0
        weights_dict = {train_r.columns[i]: float(best_weights[i]) for i in range(len(train_r.columns))}

        # --- Backtest ---
        full_common = returns_monthly.index.intersection(rm.index)
        full_returns = returns_monthly.reindex(full_common)
        portfolio_returns_full = (full_returns[train_r.columns] * best_weights).sum(axis=1)
        cum_full = (1 + portfolio_returns_full).cumprod()
        portfolio_series = 100 * cum_full / cum_full.iloc[0]

        spy_full = spy_monthly.reindex(full_common).pct_change()
        spy_cum = (1 + spy_full.fillna(0)).cumprod()
        market_series = 100 * spy_cum / spy_cum.iloc[0]

        # Rebase au début de la période de test
        test_start_date = test_returns.index[0]
        idx_at_or_after = full_common[full_common >= test_start_date]
        base_date = idx_at_or_after[0] if len(idx_at_or_after) > 0 else full_common[-1]

        def _scalar(s, key):
            val = s.loc[key]
            return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)

        base_p, base_m = _scalar(portfolio_series, base_date), _scalar(market_series, base_date)
        if base_p > 1e-12: portfolio_series = 100 * portfolio_series / base_p
        if base_m > 1e-12: market_series = 100 * market_series / base_m

        comparison_data = [
            {"date": d.strftime("%Y-%m-%d"), "portfolio": round(_scalar(portfolio_series, d), 2), "market": round(_scalar(market_series, d), 2)}
            for d in portfolio_series.index
        ]

        # --- Max drawdown et métriques backtest ---
        test_common = test_returns.index.intersection(full_common)
        test_ret_series = (returns_monthly.loc[test_common, train_r.columns] * best_weights).sum(axis=1)
        cum_test = (1 + test_ret_series).cumprod()
        max_drawdown = float(-((cum_test - cum_test.cummax()) / cum_test.cummax()).min() * 100) if len(cum_test) > 0 else 0.0
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
            "backtestReturn": opt_backtest_ret,
            "backtestSharpe": backtest_sharpe,
            "marketSharpe": market_sharpe_train,
            "marketBacktestSharpe": market_sharpe_test,
            "marketTotalSharpe": market_sharpe_total,
            "comparisonData": comparison_data,
            "numPortfolios": 1,
            "trainPeriodStart": train_returns.index[0].strftime("%Y-%m-%d"),
            "trainPeriodEnd": train_returns.index[-1].strftime("%Y-%m-%d"),
            "testPeriodStart": test_returns.index[0].strftime("%Y-%m-%d"),
            "testPeriodEnd": test_returns.index[-1].strftime("%Y-%m-%d"),
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
        
        # Sécurisation de l'échelle du taux sans risque pour l'optimiseur
        rf_ann = float(rf_annual)
        if rf_ann > 1.0:  # Si exprimé en % (ex: 4.0), on repasse en décimal
            rf_ann = rf_ann / 100.0
            
        weights_calculated = opt_sharpe_gradient_optimal(mu_pure, cov_pure, rf_ann)
        best_weights = np.asarray(weights_calculated).flatten()
        
        # CALCULS FINAUX STRICTEMENT DÉCIMAUX POUR L'API
        opt_ret_val = float(np.dot(best_weights, mu_pure))
        opt_vol_val = float(np.sqrt(max(best_weights.T @ cov_pure @ best_weights, 1e-10)))
        
        # Formule propre du Sharpe
        opt_sharpe_val = (opt_ret_val - rf_ann) / opt_vol_val if opt_vol_val > 1e-10 else 0.0
        
        weights_dict = {train_r.columns[i]: float(best_weights[i]) for i in range(len(train_r.columns))}

        # --- Backtest ---
        full_common = returns_monthly.index.intersection(rm.index)
        full_returns = returns_monthly.reindex(full_common)
        portfolio_returns_full = (full_returns[train_r.columns] * best_weights).sum(axis=1)
        cum_full = (1 + portfolio_returns_full).cumprod()
        portfolio_series = 100 * cum_full / cum_full.iloc[0]

        spy_full = spy_monthly.reindex(full_common).pct_change()
        spy_cum = (1 + spy_full.fillna(0)).cumprod()
        market_series = 100 * spy_cum / spy_cum.iloc[0]

        test_start_date = test_returns.index[0]
        idx_at_or_after = full_common[full_common >= test_start_date]
        base_date = idx_at_or_after[0] if len(idx_at_or_after) > 0 else full_common[-1]

        def _scalar(s, key):
            val = s.loc[key]
            return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)

        base_p, base_m = _scalar(portfolio_series, base_date), _scalar(market_series, base_date)
        if base_p > 1e-12: portfolio_series = 100 * portfolio_series / base_p
        if base_m > 1e-12: market_series = 100 * market_series / base_m

        comparison_data = [
            {"date": d.strftime("%Y-%m-%d"), "portfolio": round(_scalar(portfolio_series, d), 2), "market": round(_scalar(market_series, d), 2)}
            for d in portfolio_series.index
        ]

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
            "sharpe": round(opt_sharpe_val, 4),
            "expectedReturn": round(opt_ret_val * 100, 2),
            "volatility": round(opt_vol_val * 100, 2),
            "maxDrawdown": round(max_drawdown, 2),
            "backtestReturn": opt_backtest_ret,
            "backtestSharpe": backtest_sharpe,
            "marketSharpe": market_sharpe_train,
            "marketBacktestSharpe": market_sharpe_test,
            "marketTotalSharpe": market_sharpe_total,
            "comparisonData": comparison_data,
            "numPortfolios": 1,
            "trainPeriodStart": train_returns.index[0].strftime("%Y-%m-%d"),
            "trainPeriodEnd": train_returns.index[-1].strftime("%Y-%m-%d"),
            "testPeriodStart": test_returns.index[0].strftime("%Y-%m-%d"),
            "testPeriodEnd": test_returns.index[-1].strftime("%Y-%m-%d"),
            "efficientFrontier": [{
                "volatility": round(opt_vol_val * 100, 2),
                "expectedReturn": round(opt_ret_val * 100, 2),
                "sharpe": round(opt_sharpe_val, 4),
                "backtestReturn": opt_backtest_ret
            }],
            "factor_tests": factor_tests_by_ticker,
        }