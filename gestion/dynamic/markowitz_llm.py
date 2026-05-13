"""
Pipeline LLM complet : sélection dynamique des facteurs Fama-French + optimisation Markowitz.

Mode backtest glissant mensuel :
  Pour chaque mois t de la période de test (20% des données) :
    1. LLM sélectionne les facteurs actifs (masque booléen)
    2. Régression OLS sur la fenêtre d'entraînement disponible jusqu'à t
       (uniquement sur les facteurs actifs sélectionnés)
    3. Optimisation Markowitz Monte-Carlo → poids pour le mois t+1
    4. On applique ces poids au rendement réalisé de t+1
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from gestion.get_facteurs import load_famafrench_5factors, load_momentum_factor
from gestion.yahoo_prices import yf_adj_close_wide, yf_single_ticker_adj_series
from gestion.ols_with_stats import ols_factor_regression
from gestion.dynamic.llm_news_fetcher import fetch_news
from gestion.dynamic.llm_factor_selector import select_factors, get_all_prompt_examples, reset_prompt_log, FactorMask
from gestion.dynamic.llm_config import ALL_FACTORS
from gestion.dynamic.fred_loader import load_fred_factors

logger = logging.getLogger(__name__)


def _json_sanitize(obj: Any) -> Any:
    """Remplace nan/inf par None pour que la réponse soit JSON-sérialisable."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(x) for x in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ---------------------------------------------------------------------------
# Helpers prix
# ---------------------------------------------------------------------------

def _download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    start_d = datetime.fromisoformat(start[:10])
    end_d = datetime.fromisoformat(end[:10]) + timedelta(days=1)
    prices, _missing = yf_adj_close_wide(tickers, start_d, end_d, "1d")
    if prices.empty:
        return pd.DataFrame()
    tickers_clean = [t.strip() for t in tickers if t.strip()]
    cols = [c for c in tickers_clean if c in prices.columns]
    if not cols:
        return pd.DataFrame()
    out = prices[cols].dropna(how="any")
    return out


def _monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convertit des prix quotidiens en rendements mensuels log."""
    monthly = prices.resample("ME").last()
    return np.log(monthly / monthly.shift(1)).dropna()


# ---------------------------------------------------------------------------
# Régression OLS multi-facteurs (avec tests statistiques)
# ---------------------------------------------------------------------------

def _ols_regression_with_stats(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    active_factors: list[str],
    rf: pd.Series,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    """
    OLS sur les facteurs actifs via statsmodels. Retourne (coeffs, factor_tests).
    factor_tests est None si la régression échoue.
    """
    X_cols = factor_returns[active_factors]
    common_idx = asset_returns.index.intersection(X_cols.index).intersection(rf.index)
    if len(common_idx) < max(len(active_factors) + 2, 5):
        return {"alpha": 0.0, **{f: 0.0 for f in active_factors}}, None

    y = (asset_returns.loc[common_idx] - rf.loc[common_idx]).values
    X_raw = X_cols.loc[common_idx].values / 100.0  # décimal
    result = ols_factor_regression(y, X_raw, active_factors)
    if result is not None:
        return result["coeffs"], {
            "factor_stats": result["factor_tests"],
            "model_stats": result["model_stats"],
        }
    # Fallback : coefficients à 0, pas de stats
    return {"alpha": 0.0, **{f: 0.0 for f in active_factors}}, {"factor_stats": {}, "model_stats": None}


def _expected_return_annualized(
    betas: dict[str, float],
    factor_means: pd.Series,
    rf_mean: float,
    active_factors: list[str],
) -> float:
    """Rendement attendu annualisé à partir des betas OLS sur les facteurs actifs."""
    mu = rf_mean / 100.0
    for f in active_factors:
        mu += betas.get(f, 0.0) * (factor_means[f] / 100.0)
    return mu * 12.0


# ---------------------------------------------------------------------------
# Optimisation Markowitz Monte-Carlo
# ---------------------------------------------------------------------------

def _optimize_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.03,
    num_portfolios: int = 5_000,
) -> tuple[np.ndarray, float, float, float]:
    n = len(mean_returns)
    best_sharpe = -np.inf
    best_w = np.ones(n) / n
    best_ret = 0.0
    best_vol = 0.0

    # Éviter NaN si matrice de covariance dégénérée (p. ex. peu de données)
    if not np.isfinite(cov_matrix).all() or not np.isfinite(mean_returns).all():
        return np.ones(n) / n, 0.0, 0.0, 0.0

    for _ in range(num_portfolios):
        w = np.random.random(n)
        w /= w.sum()
        ret = float(np.dot(mean_returns, w))
        vol = float(np.sqrt(np.maximum(w @ cov_matrix @ w, 0.0)))
        sharpe = (ret - risk_free_rate) / vol if vol > 1e-10 else 0.0
        if np.isfinite(sharpe) and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_w = w.copy()
            best_ret = ret
            best_vol = vol

    if not np.isfinite(best_sharpe):
        best_sharpe = 0.0
    return best_w, best_ret, best_vol, best_sharpe


# ---------------------------------------------------------------------------
# Un pas du backtest glissant
# ---------------------------------------------------------------------------

def _step(
    tickers: list[str],
    monthly_returns: pd.DataFrame,
    ff5: pd.DataFrame,
    year: int,
    month: int,
    risk_free_rate: float,
    num_portfolios: int,
) -> dict[str, Any] | None:
    """
    Calcule les poids optimaux pour le mois suivant (year, month) en utilisant
    toutes les données disponibles jusqu'à ce mois.
    """
    cutoff = pd.Period(f"{year}-{month:02d}", freq="M")
    train = monthly_returns.loc[monthly_returns.index <= cutoff]

    if len(train) < 12:
        return None

    train_ff = ff5.loc[ff5.index.isin(train.index)]
    if len(train_ff) < 6:
        return None

    # Phase 1 : news
    news = fetch_news(tickers, year, month)

    # Phase 2 : sélection des facteurs (masque booléen LLM)
    per_ticker_masks, global_factors = select_factors(tickers, year, month, news)

    # Phase 3 : régression OLS sur les facteurs actifs
    rf_train = train_ff["RF"]

    # Exclure les facteurs sans données suffisantes dans la fenêtre d'entraînement
    # (cas des variables FRED absentes ou NaN en début de série)
    available_factors = [
        f for f in global_factors
        if f in train_ff.columns and train_ff[f].notna().sum() >= max(len(global_factors) + 2, 5)
    ]
    if not available_factors:
        available_factors = ["Mkt-RF"]

    factor_means = train_ff[available_factors].mean()
    rf_mean = float(rf_train.mean())

    mu_vec = []
    valid_tickers = []
    factor_tests_by_ticker: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        if ticker not in train.columns:
            continue
        ticker_mask: FactorMask = per_ticker_masks.get(ticker, {f: True for f in ALL_FACTORS})
        # Facteurs actifs pour ce ticker : intersection du masque LLM et des facteurs disponibles
        active = [f for f in available_factors if ticker_mask.get(f, True)]
        if not active:
            active = available_factors[:1]
        betas, factor_tests = _ols_regression_with_stats(train[ticker], train_ff, active, rf_train)
        mu = _expected_return_annualized(betas, factor_means, rf_mean, active)
        mu_vec.append(mu)
        valid_tickers.append(ticker)
        factor_tests_by_ticker[ticker] = factor_tests if factor_tests else {"factor_stats": {}, "model_stats": None}

    if len(valid_tickers) < 2:
        return None

    mu_arr = np.array(mu_vec)
    cov_matrix = train[valid_tickers].cov().values * 12

    # Phase 4 : optimisation
    best_w, best_ret, best_vol, best_sharpe = _optimize_portfolio(
        mu_arr, cov_matrix, risk_free_rate, num_portfolios
    )

    weights_dict = {valid_tickers[i]: float(best_w[i]) for i in range(len(valid_tickers))}

    return {
        "weights": weights_dict,
        "expectedReturn": round(float(best_ret) * 100, 2),
        "volatility": round(float(best_vol) * 100, 2),
        "sharpe": round(float(best_sharpe), 4),
        # Masque booléen par ticker — ex: {"AAPL": {"Mkt-RF": true, "SMB": false, ...}}
        "selectedFactors": {
            t: {f: bool(per_ticker_masks.get(t, {}).get(f, True)) for f in ALL_FACTORS}
            for t in valid_tickers
        },
        "newsSummaries": {
            t: {
                "summary": news.get(t, {}).get("summary", ""),
                "sentiment": news.get(t, {}).get("sentiment", "neutre"),
            }
            for t in valid_tickers
        },
        "factor_tests": factor_tests_by_ticker,
    }


# ---------------------------------------------------------------------------
# Point d'entrée public
# ---------------------------------------------------------------------------

def run(
    tickers: list[str],
    start: str,
    end: str,
    risk_free_rate: float = 0.03,
    num_portfolios: int = 5_000,
    progress_callback=None,
) -> dict[str, Any]:
    """
    Backtest glissant mensuel LLM.

    Pour chaque mois t de la période de test :
      - Le LLM sélectionne les facteurs avec l'actu disponible jusqu'à t
      - Les betas sont ré-estimés sur toutes les données jusqu'à t
      - Les poids sont optimisés → appliqués au rendement réalisé de t+1

    Sortie :
      comparisonData  : courbe $10 000 du portefeuille LLM mois par mois
      monthlyHistory  : par mois → weights, selectedFactors, newsSummaries, sharpe, etc.
    """
    def _emit(event_type: str, **kwargs):
        if progress_callback is not None:
            try:
                progress_callback({"type": event_type, **kwargs})
            except Exception:
                pass

    reset_prompt_log()

    if len(tickers) < 2:
        return {"error": "Sélectionnez au moins 2 actions."}

    _emit("status", step="init", message="Téléchargement des données de prix…")
    prices = _download_prices(tickers, start, end)
    if prices.empty or len(prices.columns) < 2:
        return {"error": "Données de prix insuffisantes."}

    valid = [c for c in tickers if c in prices.columns]
    if len(valid) < 2:
        return {"error": "Données insuffisantes pour les tickers demandés."}
    prices = prices[valid]

    _emit("status", step="factors", message="Chargement des facteurs Fama-French…")
    try:
        ff5 = load_famafrench_5factors(start, end)
    except Exception as e:
        return {"error": f"Impossible de charger les facteurs Fama-French 5 : {e}"}

    # Ajout du facteur Momentum (UMD) de Ken French
    try:
        umd = load_momentum_factor(start, end)
        ff5 = ff5.join(umd.rename("UMD"), how="left")
    except Exception as e:
        logger.warning("Momentum (UMD) indisponible : %s — colonne absente.", e)
        ff5["UMD"] = float("nan")

    _emit("status", step="fred", message="Chargement des données macro (FRED)…")
    # Ajout des variables macro FRED (HY_SPREAD, TERM_SPREAD, VIX)
    try:
        fred = load_fred_factors(start, end)
        ff5 = ff5.join(fred, how="left")
    except Exception as e:
        logger.warning("Facteurs FRED indisponibles : %s — colonnes absentes.", e)
        for col in ("HY_SPREAD", "TERM_SPREAD", "VIX"):
            ff5[col] = float("nan")

    monthly_ret = _monthly_returns(prices)
    monthly_ret.index = monthly_ret.index.to_period("M")

    n_months = len(monthly_ret)

    # Backtest = 1 an (12 mois), part test ≈ 20 % si assez de données ; min 24 mois (12 train + 12 test)
    if n_months < 24:
        _first = str(monthly_ret.index[0]) if n_months > 0 else "?"
        _last = str(monthly_ret.index[-1]) if n_months > 0 else "?"
        return {"error": "Pas assez de données pour la période. Il faut au moins 24 mois (12 d'entraînement + 12 de backtest). Période disponible : {} à {} ({} mois). Un ou plusieurs titres n'ont peut-être pas de données sur toute la plage.".format(_first, _last, n_months)}
    # Derniers 12 mois = backtest (1 an), le reste = entraînement (≈ 80 % quand n_months ~60)
    split = n_months - 12

    train_periods = monthly_ret.index[:split]
    test_periods = monthly_ret.index[split:]

    if len(test_periods) == 0:
        return {"error": "Période de test vide."}

    train_start_s = str(train_periods[0])
    train_end_s = str(train_periods[-1])
    test_start_s = str(test_periods[0])
    test_end_s = str(test_periods[-1])

    num_test_months = len(test_periods)
    _emit("status", step="backtest_start", message=f"Démarrage du backtest sur {num_test_months} mois…", total=num_test_months)

    # -------------------------------------------------------------------
    # Backtest glissant : pour chaque mois t de la période de test,
    # on calcule les poids à partir des données jusqu'à t,
    # puis on applique ces poids au rendement réalisé de t+1 (si disponible)
    # -------------------------------------------------------------------
    INITIAL_VALUE = 10_000.0
    portfolio_value = INITIAL_VALUE
    portfolio_curve: list[dict] = []   # [{ date, value }]
    monthly_history: list[dict] = []   # [{ month, weights, selectedFactors, ... }]

    # Pour le premier point de la courbe (début de la période de test)
    portfolio_curve.append({"date": test_start_s, "value": round(portfolio_value, 2)})

    for i, t in enumerate(test_periods):
        year_t, month_t = t.year, t.month

        _emit(
            "month",
            step="month",
            current=i + 1,
            total=num_test_months,
            month=str(t),
            message=f"Analyse du mois {t} ({i + 1}/{num_test_months})…",
        )

        step_result = _step(valid, monthly_ret, ff5, year_t, month_t, risk_free_rate, num_portfolios)

        if step_result is None:
            # Fallback : poids équipondérés
            weights = {ticker: 1.0 / len(valid) for ticker in valid}
            logger.warning("Étape %s échouée — poids équipondérés utilisés.", t)
        else:
            weights = step_result["weights"]

        # Enregistrement de l'historique du mois t
        monthly_history.append({
            "month": str(t),
            "weights": weights,
            "selectedFactors": step_result["selectedFactors"] if step_result else {tk: {f: True for f in ALL_FACTORS} for tk in valid},
            "newsSummaries": step_result["newsSummaries"] if step_result else {tk: {"summary": "", "sentiment": "neutre"} for tk in valid},
            "sharpe": step_result["sharpe"] if step_result else None,
            "expectedReturn": step_result["expectedReturn"] if step_result else None,
            "volatility": step_result["volatility"] if step_result else None,
            "factor_tests": step_result["factor_tests"] if step_result else {},
        })

        # Appliquer les poids au rendement du mois suivant (t+1)
        next_period = t + 1
        if next_period in monthly_ret.index:
            next_returns = monthly_ret.loc[next_period]
            port_ret = sum(
                weights.get(tk, 0.0) * float(next_returns.get(tk, 0.0))
                for tk in valid
            )
            portfolio_value *= np.exp(port_ret)  # rendements log → cumul multiplicatif
            next_date = str(next_period)
        else:
            next_date = str(t)

        portfolio_curve.append({"date": next_date, "value": round(portfolio_value, 2)})

    # -------------------------------------------------------------------
    # SPY (marché) sur la même période, base $10 000
    # -------------------------------------------------------------------
    spy_curve: list[dict] = []
    try:
        ts = pd.Period(test_start_s, freq="M").to_timestamp()
        start_s = ts.strftime("%Y-%m-%d")
        end_ts = pd.Timestamp(prices.index[-1])
        end_s = end_ts.strftime("%Y-%m-%d")
        start_d = datetime.fromisoformat(start_s[:10])
        end_d = datetime.fromisoformat(end_s[:10]) + timedelta(days=1)
        adj = yf_single_ticker_adj_series("SPY", start_d, end_d, "1d")
        if adj is not None and not adj.empty:
            spy_monthly = adj.resample("ME").last()
            spy_monthly.index = spy_monthly.index.to_period("M")
            test_spy = spy_monthly.loc[spy_monthly.index >= test_periods[0]]
            if len(test_spy) >= 2:
                spy_log_ret = np.log(test_spy / test_spy.shift(1)).dropna()
                spy_value = INITIAL_VALUE
                spy_curve.append({"date": str(test_spy.index[0]), "value": round(spy_value, 2)})
                for period, lr in spy_log_ret.items():
                    spy_value *= np.exp(float(lr))
                    spy_curve.append({"date": str(period), "value": round(spy_value, 2)})
    except Exception as e:
        logger.warning("SPY indisponible : %s", e)

    # -------------------------------------------------------------------
    # Fusion des courbes sur les dates communes
    # -------------------------------------------------------------------
    spy_map = {d["date"]: d["value"] for d in spy_curve}
    comparison_data = [
        {
            "date": pt["date"],
            "portfolio": pt["value"],
            "market": spy_map.get(pt["date"]),
        }
        for pt in portfolio_curve
    ]

    # Résumé global sur la période de test
    final_value = portfolio_curve[-1]["value"] if portfolio_curve else INITIAL_VALUE
    total_return_pct = round((final_value / INITIAL_VALUE - 1) * 100, 2)

    # Rendements mensuels réalisés du portefeuille (pour max drawdown)
    values = [pt["value"] for pt in portfolio_curve]
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    max_drawdown = round(max_dd * 100, 2)

    _emit("status", step="done", message="Backtest terminé. Calcul des métriques…")

    result = {
        # Métriques globales (période de test)
        "totalReturn": total_return_pct,
        "maxDrawdown": max_drawdown,
        "initialValue": INITIAL_VALUE,
        "finalValue": round(final_value, 2),
        # Courbe de performance ($)
        "comparisonData": comparison_data,
        # Historique mensuel détaillé
        "monthlyHistory": monthly_history,
        # Metadata
        "numMonths": len(test_periods),
        "trainPeriodStart": train_start_s,
        "trainPeriodEnd": train_end_s,
        "testPeriodStart": test_start_s,
        "testPeriodEnd": test_end_s,
        # Tous les prompts LLM capturés (un par ticker × mois)
        "promptExamples": get_all_prompt_examples(),
    }
    return _json_sanitize(result)
