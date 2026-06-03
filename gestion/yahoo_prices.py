"""
Téléchargement Yahoo Finance robuste : extraction groupée native sécurisée.
Élimine les bugs de clonage liés au multithreading.
"""
from __future__ import annotations

from datetime import datetime
import pandas as pd
import yfinance as yf


def yf_single_ticker_adj_series(
    ticker: str,
    start: datetime,
    end: datetime,
    interval: str = "1d",
) -> pd.Series | None:
    """Téléchargement sécurisé pour un symbole isolé (ex: SPY ou ^IRX)."""
    t = ticker.strip()
    if not t:
        return None
    try:
        data = yf.download(
            t,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return None
        
    if data is None or data.empty:
        return None
        
    # Gestion de la structure MultiIndex de yfinance (versions récentes)
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.levels[0]:
            s = data["Adj Close"]
        elif "Close" in data.columns.levels[0]:
            s = data["Close"]
        else:
            return None
    else:
        if "Adj Close" in data.columns:
            s = data["Adj Close"]
        elif "Close" in data.columns:
            s = data["Close"]
        else:
            return None
            
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
        
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
        
    if hasattr(s.index, "tz") and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
        
    return s


def yf_adj_close_wide(
    tickers: list[str],
    start: datetime,
    end: datetime,
    interval: str = "1d",
) -> tuple[pd.DataFrame, list[str]]:
    """Historique aligné : UN SEUL appel Yahoo groupé et sécurisé."""
    valid_tickers = [t.strip() for t in tickers if t.strip()]
    if not valid_tickers:
        return pd.DataFrame(), []

    # Appel natif groupé : sécurisé, rapide et évite le clonage
    data = yf.download(
        valid_tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        return pd.DataFrame(), valid_tickers

    # Extraction des prix ajustés depuis le MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.levels[0]:
            prices = data["Adj Close"].copy()
        elif "Close" in data.columns.levels[0]:
            prices = data["Close"].copy()
        else:
            return pd.DataFrame(), valid_tickers
    else:
        # Sécurité si un seul ticker est valide
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        if col in data.columns:
            prices = pd.DataFrame(data[col])
            prices.columns = [valid_tickers[0]]
        else:
            return pd.DataFrame(), valid_tickers

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
        if len(valid_tickers) == 1:
            prices.columns = valid_tickers

    # Retrait des Timezones pour l'alignement
    if hasattr(prices.index, "tz") and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    prices = prices.sort_index()

    # Vérification des manquants
    missing = [t for t in valid_tickers if t not in prices.columns]
    found_tickers = [t for t in valid_tickers if t in prices.columns]
    
    return prices[found_tickers], missing