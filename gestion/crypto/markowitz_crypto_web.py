"""
Fama-French crypto (CMKT, SIZE, MOM) pour l'API web — sortie alignée sur SimulateResult.
Données : CSV locaux (CoinGecko), rééchantillonnage mensuel, split train/test 80/20.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pandas as pd

_gestion_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _gestion_dir not in sys.path:
    sys.path.insert(0, _gestion_dir)

from Methodes_de_descente.gradient_pas_fixe import opt_sharpe_gradient
from Methodes_de_descente.gradient_pas_optimal import opt_sharpe_gradient_optimal
from ols_with_stats import ols_factor_regression
from market_metrics import annualized_sharpe

from .crypto_fama_french import (
    CSV_FILES,
    RF_ANNUAL,
    STABLECOINS,
    build_factors,
    load_prices_for_symbols,
)

FACTOR_NAMES = ["CMKT", "SIZE", "MOM"]

CRYPTO_LABELS: dict[str, str] = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "USDT": "Tether USDt",
    "BNB": "BNB",
    "XRP": "XRP",
    "USDC": "USDC",
    "SOL": "Solana",
    "TRX": "TRON",
    "DOGE": "Dogecoin",
    "ADA": "Cardano",
    "BCH": "Bitcoin Cash",
    "HYPE": "Hyperliquid",
    "LEO": "UNUS SED LEO",
    "XMR": "Monero",
    "LINK": "Chainlink",
    "USDe": "Ethena USDe",
    "CC": "Canton",
    "DAI": "Dai",
    "XLM": "Stellar",
    "USD1": "World Liberty Financial USD",
    "LTC": "Litecoin",
    "HBAR": "Hedera",
    "AVAX": "Avalanche",
    "PYUSD": "PayPal USD",
    "SUI": "Sui",
    "ZEC": "Zcash",
    "SHIB": "Shiba Inu",
    "TON": "Toncoin",
    "CRO": "Cronos",
    "XAUt": "Tether Gold",
    "WLFI": "World Liberty Financial",
    "PAXG": "PAX Gold",
    "DOT": "Polkadot",
    "UNI": "Uniswap",
    "MNT": "Mantle",
    "PI": "Pi",
    "TAO": "Bittensor",
    "OKB": "OKB",
    "M": "MemeCore",
    "SKY": "Sky",
    "ASTER": "Aster",
    "AAVE": "Aave",
    "USDG": "Global Dollar",
    "RAIN": "Rain",
    "CIRCLEUSYC": "Circle USYC",
    "BUIDL": "Buidl",
    "WBT": "WhiteBIT Coin",
    "USDS": "USDS",
    "FIGR": "Figure HELOC",
}


def list_crypto_assets() -> list[dict[str, str]]:
    out = []
    for sym in CSV_FILES:
        out.append({"symbol": sym, "name": CRYPTO_LABELS.get(sym, sym)})
    return out


def yahoo_ticker_for_news(code: str) -> str:
    c = code.strip().upper()
    if c == "USDe":
        return "USDE-USD"
    if c == "XAUt":
        return "XAUT-USD"
    return f"{c}-USD"


def crypto_portfolio_common_bounds(symbols: list[str]) -> dict[str, Any]:
    """
    Intersection calendaire des plages où chaque crypto a au moins un prix (CSV).
    Retourne commonStart / commonEnd au format YYYY-MM-DD, ou error.
    """
    from .crypto_fama_french import CSV_FOLDER

    syms = [s.strip().upper() for s in symbols if s.strip()]
    syms = [s for s in syms if s in CSV_FILES]
    if len(syms) < 2:
        return {
            "error": "Sélectionnez au moins 2 cryptos reconnues.",
            "commonStart": None,
            "commonEnd": None,
        }

    firsts: list[pd.Timestamp] = []
    lasts: list[pd.Timestamp] = []

    for sym in syms:
        path = os.path.join(CSV_FOLDER, CSV_FILES[sym])
        if not os.path.isfile(path):
            return {
                "error": f"Fichier de données absent pour {sym}.",
                "commonStart": None,
                "commonEnd": None,
            }
        try:
            df = pd.read_csv(path, parse_dates=["snapped_at"])
            dates = pd.to_datetime(df["snapped_at"], utc=True).dt.tz_localize(None)
            close = pd.to_numeric(df["price"], errors="coerce")
            ok = close.notna()
            if not ok.any():
                return {
                    "error": f"Aucun prix exploitable pour {sym}.",
                    "commonStart": None,
                    "commonEnd": None,
                }
            firsts.append(pd.Timestamp(dates[ok].min()))
            lasts.append(pd.Timestamp(dates[ok].max()))
        except Exception as e:
            return {
                "error": f"Lecture CSV impossible pour {sym}: {e}",
                "commonStart": None,
                "commonEnd": None,
            }

    common_start = max(firsts)
    common_end = min(lasts)
    if common_start >= common_end:
        return {
            "error": "Les cryptos sélectionnées n'ont pas de période commune avec des données.",
            "commonStart": None,
            "commonEnd": None,
        }

    return {
        "commonStart": common_start.strftime("%Y-%m-%d"),
        "commonEnd": common_end.strftime("%Y-%m-%d"),
    }


def run(
    tickers: list[str],
    method: str = "gradient_optimal",
    num_portfolios: int = 10000,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    syms = [s.strip().upper() for s in tickers if s.strip()]
    syms = [s for s in syms if s in CSV_FILES]
    if len(syms) < 2:
        return {"error": "Sélectionnez au moins 2 cryptos reconnues (fichiers CSV présents)."}

    try:
        returns, mcaps = load_prices_for_symbols(syms, freq="ME", start=start, end=end)
    except RuntimeError as e:
        return {"error": str(e)}

    if returns.shape[1] < 2:
        return {"error": "Données insuffisantes pour les cryptos sélectionnées."}

    factors, _size_large = build_factors(returns, mcaps)
    common = returns.index.intersection(factors.index)
    ret = returns.loc[common].astype(float)
    F = factors.loc[common].astype(float)

    n = len(ret)
    split = max(int(n * 0.8), 12)
    if split >= n:
        return {"error": "Pas assez d'observations mensuelles pour entraîner le modèle."}

    train_r = ret.iloc[:split]
    test_r = ret.iloc[split:]
    Fi = F.iloc[:split]

    tradeable = [c for c in train_r.columns if c not in STABLECOINS]
    if len(tradeable) < 2:
        return {
            "error": "Au moins 2 actifs hors stablecoin sont nécessaires pour l'optimisation.",
        }

    factor_means = Fi[FACTOR_NAMES].mean().values
    factor_tests_by_ticker: dict[str, dict[str, Any]] = {}
    mu_monthly: list[float] = []

    for col in tradeable:
        y = train_r[col].values
        X = Fi[FACTOR_NAMES].values
        res = ols_factor_regression(y, X, FACTOR_NAMES, add_constant=True)
        if res is not None:
            coeffs = res["coeffs"]
            b = np.array([coeffs.get(f, 0.0) for f in FACTOR_NAMES])
            alpha = float(coeffs.get("alpha", 0.0))
            mu_m = alpha + float(np.dot(b, factor_means))
            mu_monthly.append(mu_m)
            factor_tests_by_ticker[col] = {
                "factor_stats": res["factor_tests"],
                "model_stats": res["model_stats"],
            }
        else:
            mu_monthly.append(float(train_r[col].mean()))
            factor_tests_by_ticker[col] = {"factor_stats": {}, "model_stats": None}

    mu = np.asarray(mu_monthly, dtype=float) * 12.0
    cov_matrix = train_r[tradeable].cov().values * 12.0
    n_assets = len(tradeable)
    rf_annual = float(RF_ANNUAL)

    full_r = ret[tradeable].astype(float)
    cmkt_full = full_r.mean(axis=1)
    market_sharpe_train = annualized_sharpe(cmkt_full.loc[train_r.index], rf_annual, 12)
    market_sharpe_test = annualized_sharpe(cmkt_full.loc[test_r.index], rf_annual, 12)

    def _scalar_series(s: pd.Series, key) -> float:
        val = s.loc[key]
        return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)

    def _build_comparison(best_weights: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        port_rets = (full_r.values @ best_weights).ravel()
        cum_p = np.cumprod(1.0 + port_rets)
        mkt_rets = cmkt_full.values
        cum_m = np.cumprod(1.0 + mkt_rets)

        portfolio_series = pd.Series(100.0 * cum_p / cum_p[0], index=full_r.index)
        market_series = pd.Series(100.0 * cum_m / cum_m[0], index=full_r.index)

        test_start = test_r.index[0]
        idx_after = full_r.index[full_r.index >= test_start]
        base_date = idx_after[0] if len(idx_after) else full_r.index[-1]

        bp = _scalar_series(portfolio_series, base_date)
        bm = _scalar_series(market_series, base_date)
        portfolio_series = 100.0 * portfolio_series / bp
        market_series = 100.0 * market_series / bm

        comparison_data = [
            {
                "date": str(d),
                "portfolio": round(_scalar_series(portfolio_series, d), 2),
                "market": round(_scalar_series(market_series, d), 2),
            }
            for d in portfolio_series.index
        ]

        test_idx = full_r.index.intersection(test_r.index)
        test_port = pd.Series(
            (full_r.loc[test_idx].values @ best_weights).ravel(),
            index=test_idx,
        )
        cum_test = (1.0 + test_port).cumprod()
        dd = (cum_test - cum_test.cummax()) / cum_test.cummax()
        max_drawdown = float(-dd.min() * 100) if len(dd) else 0.0

        return comparison_data, max_drawdown

    if method == "monte_carlo":
        all_weights = np.zeros((num_portfolios, n_assets))
        ret_arr = np.zeros(num_portfolios)
        vol_arr = np.zeros(num_portfolios)
        sharpe_arr = np.zeros(num_portfolios)
        for i in range(num_portfolios):
            w = np.random.random(n_assets)
            w /= w.sum()
            all_weights[i] = w
            ret_arr[i] = float(np.dot(w, mu))
            vol_arr[i] = float(np.sqrt(max(w @ cov_matrix @ w, 0.0)))
            sharpe_arr[i] = (
                (ret_arr[i] - rf_annual) / vol_arr[i] if vol_arr[i] > 1e-10 else 0.0
            )
        max_idx = int(np.argmax(sharpe_arr))
        best_weights = all_weights[max_idx]
        weights_dict = {tradeable[i]: float(best_weights[i]) for i in range(n_assets)}

        comparison_data, max_drawdown = _build_comparison(best_weights)

        test_common = test_r.index.intersection(full_r.index)
        opt_test = (full_r.loc[test_common].values @ best_weights).ravel()
        opt_backtest_ret = (
            round(float((np.prod(1.0 + opt_test) - 1.0) * 100), 2) if len(opt_test) else 0.0
        )

        vol_pct = vol_arr * 100.0
        ret_pct = ret_arr * 100.0
        order = np.argsort(vol_pct)
        frontier_vol, frontier_ret, frontier_sharpe, frontier_backtest_ret = [], [], [], []
        max_ret_so_far = -np.inf
        for i in order:
            if ret_pct[i] >= max_ret_so_far:
                max_ret_so_far = ret_pct[i]
                port_test = (full_r.loc[test_common].values @ np.asarray(all_weights[i])).ravel()
                total_ret = (np.prod(1.0 + port_test) - 1.0) * 100 if len(port_test) else 0.0
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
            "sharpe": opt_sharpe,
            "expectedReturn": opt_ret,
            "volatility": opt_vol,
            "maxDrawdown": round(max_drawdown, 2),
            "marketSharpe": market_sharpe_train,
            "marketBacktestSharpe": market_sharpe_test,
            "comparisonData": comparison_data,
            "numPortfolios": num_portfolios,
            "trainPeriodStart": str(train_r.index[0]),
            "trainPeriodEnd": str(train_r.index[-1]),
            "testPeriodStart": str(test_r.index[0]),
            "testPeriodEnd": str(test_r.index[-1]),
            "efficientFrontier": [
                {"volatility": v, "expectedReturn": r, "sharpe": s, "backtestReturn": b}
                for v, r, s, b in zip(frontier_vol, frontier_ret, frontier_sharpe, frontier_backtest_ret)
            ],
            "factor_tests": factor_tests_by_ticker,
        }

    if method == "gradient_fixe":
        best_weights = np.asarray(opt_sharpe_gradient(mu, cov_matrix, rf_annual)).ravel()
    else:
        best_weights = np.asarray(opt_sharpe_gradient_optimal(mu, cov_matrix, rf_annual)).ravel()

    opt_ret_val = float(np.dot(best_weights, mu))
    opt_vol_val = float(np.sqrt(max(best_weights @ cov_matrix @ best_weights, 0.0)))
    opt_sharpe_val = (
        (opt_ret_val - rf_annual) / opt_vol_val if opt_vol_val > 1e-10 else 0.0
    )
    weights_dict = {tradeable[i]: float(best_weights[i]) for i in range(n_assets)}

    comparison_data, max_drawdown = _build_comparison(best_weights)

    test_common = test_r.index.intersection(full_r.index)
    test_ret_series = (full_r.loc[test_common].values @ best_weights).ravel()
    cum_test = pd.Series(np.cumprod(1.0 + test_ret_series))
    max_drawdown = float(
        -((cum_test - cum_test.cummax()) / cum_test.cummax()).min() * 100
    ) if len(cum_test) else max_drawdown
    opt_backtest_ret = (
        round(float((cum_test.iloc[-1] - 1.0) * 100), 2) if len(cum_test) else 0.0
    )

    return {
        "weights": weights_dict,
        "sharpe": round(float(opt_sharpe_val), 4),
        "expectedReturn": round(float(opt_ret_val * 100), 2),
        "volatility": round(float(opt_vol_val * 100), 2),
        "maxDrawdown": round(max_drawdown, 2),
        "marketSharpe": market_sharpe_train,
        "marketBacktestSharpe": market_sharpe_test,
        "comparisonData": comparison_data,
        "numPortfolios": 1,
        "trainPeriodStart": str(train_r.index[0]),
        "trainPeriodEnd": str(train_r.index[-1]),
        "testPeriodStart": str(test_r.index[0]),
        "testPeriodEnd": str(test_r.index[-1]),
        "efficientFrontier": [
            {
                "volatility": round(float(opt_vol_val * 100), 2),
                "expectedReturn": round(float(opt_ret_val * 100), 2),
                "sharpe": round(float(opt_sharpe_val), 4),
                "backtestReturn": opt_backtest_ret,
            }
        ],
        "factor_tests": factor_tests_by_ticker,
    }
