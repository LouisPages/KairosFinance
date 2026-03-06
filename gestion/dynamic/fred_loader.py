"""
Charge 3 variables macro depuis FRED (Federal Reserve Bank of St. Louis).
Accès public via CSV, sans clé API.

Séries chargées :
  - HY_SPREAD  : spread HY - IG (BAMLH0A0HYM2EY - BAMLC0A0CMEY), proxy risque crédit
  - TERM_SPREAD: pente courbe taux 10Y - 3M (T10Y3M), proxy cycle économique
  - VIX        : volatilité implicite S&P 500 (VIXCLS), proxy aversion au risque

Les séries sont rééchantillonnées en mensuel (dernière valeur disponible du mois)
puis converties en variations mensuelles (diff()) pour la stationnarité.

Un décalage d'un mois (.shift(1)) est appliqué pour éviter tout look-ahead bias :
on n'utilise que les données publiées avant le début du mois de prédiction.
"""
import io
import logging

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

_SERIES = {
    "HY_SPREAD_RAW": "BAMLH0A0HYM2EY",
    "IG_SPREAD_RAW": "BAMLC0A0CMEY",
    "TERM_SPREAD": "T10Y3M",
    "VIX": "VIXCLS",
}


def _fetch_series(series_id: str) -> pd.Series:
    url = _FRED_CSV_URL.format(series_id=series_id)
    with httpx.Client(timeout=30) as client:
        resp = client.get(url)
        resp.raise_for_status()
    # FRED renvoie un CSV à 2 colonnes (date, valeur) dont le nom de la colonne date
    # peut varier. On utilise index_col=0 + parse_dates=True pour ne pas dépendre du nom.
    df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
    series = df.iloc[:, 0]
    series = pd.to_numeric(series, errors="coerce")
    return series


def load_fred_factors(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Retourne un DataFrame mensuel avec colonnes HY_SPREAD, TERM_SPREAD, VIX,
    indexé en Period("M").

    Les valeurs sont des variations mensuelles (diff) décalées d'un mois (.shift(1))
    pour éviter le look-ahead bias.
    """
    raw: dict[str, pd.Series] = {}
    for name, series_id in _SERIES.items():
        try:
            raw[name] = _fetch_series(series_id)
        except Exception as e:
            logger.warning("FRED : impossible de charger %s (%s) — colonne absente.", series_id, e)
            raw[name] = pd.Series(dtype=float)

    # Rééchantillonnage mensuel (dernière valeur du mois)
    def to_monthly(s: pd.Series) -> pd.Series:
        if s.empty:
            return s
        return s.resample("ME").last()

    hy_monthly = to_monthly(raw.get("HY_SPREAD_RAW", pd.Series(dtype=float)))
    ig_monthly = to_monthly(raw.get("IG_SPREAD_RAW", pd.Series(dtype=float)))
    term_monthly = to_monthly(raw.get("TERM_SPREAD", pd.Series(dtype=float)))
    vix_monthly = to_monthly(raw.get("VIX", pd.Series(dtype=float)))

    # HY_SPREAD = écart entre HY et IG (spread net)
    hy_spread = hy_monthly - ig_monthly

    df = pd.DataFrame({
        "HY_SPREAD": hy_spread,
        "TERM_SPREAD": term_monthly,
        "VIX": vix_monthly,
    })

    # Variations mensuelles pour stationnarité
    df = df.diff()

    # Décalage d'un mois : on utilise la variation connue au début du mois t,
    # c'est-à-dire la variation calculée sur t-1 → t-2
    df = df.shift(1)

    # Conversion en Period("M")
    df.index = df.index.to_period("M")

    # Filtrage sur la plage demandée
    try:
        if start:
            start_p = pd.Period(pd.Timestamp(start).strftime("%Y-%m"), freq="M")
            df = df.loc[df.index >= start_p]
        if end:
            end_p = pd.Period(pd.Timestamp(end).strftime("%Y-%m"), freq="M")
            df = df.loc[df.index <= end_p]
    except Exception:
        pass

    return df
