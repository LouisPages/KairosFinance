"""
Téléchargement Yahoo Finance robuste : un symbole à la fois (évite MultiIndex / KeyError),
puis alignement en DataFrame commun.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import yfinance as yf


def yf_single_ticker_adj_series(
    ticker: str,
    start: datetime,
    end: datetime,
    interval: str = "1d",
) -> pd.Series | None:
    """Un symbole à la fois : contourne les plantages yfinance du téléchargement groupé."""
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
    """Historique aligné : un appel Yahoo par symbole (parallèle), puis jointure des séries."""
    tickers = [t.strip() for t in tickers if t.strip()]
    if not tickers:
        return pd.DataFrame(), []

    workers = min(8, max(1, len(tickers)))
    series_map: dict[str, pd.Series] = {}
    missing: list[str] = []

    def fetch_one(t: str) -> tuple[str, pd.Series | None]:
        return t, yf_single_ticker_adj_series(t, start, end, interval)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for t, s in ex.map(fetch_one, tickers):
            if s is None or s.empty:
                missing.append(t)
            else:
                series_map[t] = s

    if not series_map:
        return pd.DataFrame(), missing or tickers[:]
    prices = pd.DataFrame(series_map).sort_index()
    return prices, missing
