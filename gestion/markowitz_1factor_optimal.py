"""
Markowitz CAPM (1 facteur) : les rendements espérés sont estimés par régression OLS
sur le facteur de marché (MEDAF / CAPM), à fréquence mensuelle.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Any
from Methodes_de_descente.gradient_pas_optimal import opt_sharpe_gradient_optimal
from gestion.Methodes_de_descente.gradient_pas_fixe import opt_sharpe_gradient


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
    spy_data = yf.download("SPY", start=start, end=end, auto_adjust=False, progress=False)
    if spy_data.empty or ("Adj Close" not in spy_data.columns and "Close" not in spy_data.columns):
        return {"error": "Impossible de télécharger SPY"}
    spy_col = spy_data["Adj Close"] if "Adj Close" in spy_data.columns else spy_data["Close"]
    if isinstance(spy_col, pd.DataFrame):
        spy_col = spy_col.iloc[:, 0]
    spy_monthly = spy_col.resample("ME").last()
    rm = spy_monthly.pct_change().dropna()

    # Taux sans risque mensuel depuis ^IRX
    try:
        irx = yf.download("^IRX", start=start, end=end, auto_adjust=False, progress=False)
        irx_col = irx["Adj Close"] if "Adj Close" in irx.columns else irx["Close"]
        if isinstance(irx_col, pd.DataFrame):
            irx_col = irx_col.iloc[:, 0]
        rf_monthly = (irx_col / 100 / 12).resample("ME").last()
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

    # --- Régression CAPM pour chaque actif ---
    betas = {}
    for ticker in train_r.columns:
        y = (train_r[ticker] - rf_train).values
        X = np.column_stack([np.ones(len(mkt_excess)), mkt_excess])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            betas[ticker] = float(coeffs[1])
        except Exception:
            betas[ticker] = 1.0

    # Rendement espéré annualisé via CAPM (alpha ignoré conformément à la théorie)
    mu = np.array([
        (rf_mean + betas[t] * mkt_excess_mean) * 12
        for t in train_r.columns
    ])

    # Matrice de covariance annualisée
    cov_matrix = train_r.cov().values * 12
    n_assets = len(valid)

    # Taux sans risque annualisé pour le ratio de Sharpe
    rf_annual = rf_mean * 12

    best_weights = opt_sharpe_gradient_optimal(mu, cov_matrix, rf_annual)

    # Calcul des métriques optimales pour le retour API
    opt_ret_val = np.dot(best_weights, mu)
    opt_vol_val = np.sqrt(best_weights.T @ cov_matrix @ best_weights)
    opt_sharpe_val = (opt_ret_val - rf_annual) / opt_vol_val if opt_vol_val > 1e-10 else 0
    weights_dict = {valid[i]: float(best_weights[i]) for i in range(len(valid))}

    # --- Backtest ---
    full_common = returns_monthly.index.intersection(rm.index)
    full_returns = returns_monthly.reindex(full_common)
    portfolio_returns_full = (full_returns * best_weights).sum(axis=1)
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

    # --- Max drawdown ---
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
    }