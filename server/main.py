import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
# Pour que les modules gestion.* trouvent Methodes_de_descente (sous-dossier de gestion)
sys.path.insert(1, str(_root / "gestion"))

import json
import os
import re
import time
import asyncio
import threading
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Any, Optional
import yfinance as yf
import pandas as pd

from .tickers_data import get_all_stocks
from gestion.yahoo_prices import yf_adj_close_wide
from gestion.market_metrics import rf_annual_from_irx, spy_sharpe_triplet_for_period

HISTORY_FILE = Path(__file__).parent / "simulation_history.json"
MARKET_SHARPE_VERSION = 2
CRYPTO_RF_ANNUAL = 0.04
CLASSIC_RF_ANNUAL = 0.03
_history_lock = threading.Lock()
_HISTORY_SAMPLE_MODEL_IDS = (
    "markowitz-classic",
    "markowitz-1factor",
    "markowitz-3factors",
    "markowitz-5factors",
    "markowitz-llm",
    "markowitz-crypto-ff3",
)

# Cache bornes simulation (actions) : évite double appel React StrictMode + répétitions navigation
_sim_bounds_cache: dict[tuple[str, ...], tuple[float, dict[str, Optional[str]]]] = {}
_sim_bounds_lock = threading.Lock()
_SIM_BOUNDS_CACHE_TTL_SEC = 120.0


def _infer_periods_per_year(dates: list[pd.Timestamp]) -> int:
    if len(dates) < 3:
        return 252
    deltas = pd.Series(dates).diff().dropna().dt.days
    if deltas.empty:
        return 252
    median_days = float(deltas.median())
    # Courbes mensuelles (modèles multifactoriels / crypto) vs quotidiennes (Markowitz classique)
    return 12 if median_days >= 20 else 252


def _annualized_sharpe_from_levels(
    levels: pd.Series, periods_per_year: int, rf_annual: float = 0.0
) -> float:
    r = levels.pct_change().dropna()
    if len(r) < 2:
        return 0.0
    ann_mean = float(r.mean()) * periods_per_year
    ann_vol = float(r.std(ddof=1)) * (periods_per_year ** 0.5)
    if ann_vol < 1e-10:
        return 0.0
    return round((ann_mean - rf_annual) / ann_vol, 4)


def _normalize_history_date(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().replace(" 00:00:00", "")
    if len(s) >= 10:
        return s[:10]
    if len(s) == 7 and s[4] == "-":
        return f"{s}-01"
    return None


def _infer_simulation_period(entry: dict[str, Any], result: Optional[dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
    start = _normalize_history_date(entry.get("simulationStartDate"))
    end = _normalize_history_date(entry.get("simulationEndDate"))
    if start and end:
        return start, end
    if isinstance(result, dict):
        start = start or _normalize_history_date(result.get("trainPeriodStart"))
        end = end or _normalize_history_date(result.get("testPeriodEnd"))
    return start, end


def _market_sharpes_for_stock_model(
    model_id: str,
    start: str,
    end: str,
    cache: dict[tuple[str, str, str], tuple[float, float, float]],
) -> tuple[float, float, float]:
    key = (model_id, start, end)
    if key in cache:
        return cache[key]
    if model_id == "markowitz-classic":
        vals = spy_sharpe_triplet_for_period(
            start, end, CLASSIC_RF_ANNUAL, periods_per_year=252, use_log_returns=True
        )
    else:
        benchmark_rf = rf_annual_from_irx(start, end)
        vals = spy_sharpe_triplet_for_period(
            start, end, benchmark_rf, periods_per_year=12, use_log_returns=False, min_train=24
        )
    cache[key] = vals
    return vals


def _market_sharpes_from_comparison_levels(
    result: dict[str, Any], rf_annual: float = 0.0
) -> Optional[tuple[float, float, float]]:
    comp = result.get("comparisonData")
    if not isinstance(comp, list) or len(comp) < 3:
        return None

    df = pd.DataFrame(comp)
    if not {"date", "market"}.issubset(df.columns):
        return None
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        return None
    df = df.dropna(subset=["date", "market"]).sort_values("date")
    if len(df) < 3:
        return None

    periods_per_year = _infer_periods_per_year(list(df["date"]))
    market = pd.Series(df["market"].astype(float).values, index=df["date"])

    def _slice(series: pd.Series, start: Optional[str], end: Optional[str]) -> pd.Series:
        out = series
        if start:
            out = out[out.index >= pd.to_datetime(start)]
        if end:
            out = out[out.index <= pd.to_datetime(end)]
        return out

    train_mkt = _slice(market, result.get("trainPeriodStart"), result.get("trainPeriodEnd"))
    test_mkt = _slice(market, result.get("testPeriodStart"), result.get("testPeriodEnd"))
    train_target = train_mkt if len(train_mkt) >= 3 else market
    test_target = test_mkt if len(test_mkt) >= 3 else market
    return (
        _annualized_sharpe_from_levels(train_target, periods_per_year, rf_annual),
        _annualized_sharpe_from_levels(test_target, periods_per_year, rf_annual),
        _annualized_sharpe_from_levels(market, periods_per_year, rf_annual),
    )


def _apply_market_sharpes(result: dict[str, Any], sharpes: tuple[float, float, float]) -> bool:
    changed = False
    for key, value in zip(
        ("marketSharpe", "marketBacktestSharpe", "marketTotalSharpe"),
        sharpes,
    ):
        if result.get(key) != value:
            result[key] = value
            changed = True
    return changed


def _migrate_market_sharpes_in_result(
    result: dict[str, Any],
    entry: dict[str, Any],
    cache: dict[tuple[str, str, str], tuple[float, float, float]],
) -> bool:
    if not isinstance(result, dict):
        return False
    model_id = str(entry.get("modelId") or "")
    if model_id == "markowitz-crypto-ff3":
        sharpes = _market_sharpes_from_comparison_levels(result, CRYPTO_RF_ANNUAL)
    else:
        start, end = _infer_simulation_period(entry, result)
        if not start or not end:
            return False
        sharpes = _market_sharpes_for_stock_model(model_id, start, end, cache)
    if sharpes is None:
        return False
    return _apply_market_sharpes(result, sharpes)


def _migrate_entry_market_sharpes(
    entry: dict[str, Any],
    cache: dict[tuple[str, str, str], tuple[float, float, float]],
) -> bool:
    if entry.get("marketSharpeVersion", 0) >= MARKET_SHARPE_VERSION:
        return False

    changed = False
    for key in ("result", "classicResult"):
        payload = entry.get(key)
        if isinstance(payload, dict):
            changed = _migrate_market_sharpes_in_result(payload, entry, cache) or changed

    comparison_payload = entry.get("comparisonData")
    if isinstance(comparison_payload, dict):
        for key in ("monteCarlo", "bestGradient"):
            payload = comparison_payload.get(key)
            if isinstance(payload, dict):
                changed = _migrate_market_sharpes_in_result(payload, entry, cache) or changed

    entry["marketSharpeVersion"] = MARKET_SHARPE_VERSION
    return True


def _backfill_sharpes_from_comparison(result: dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    comp = result.get("comparisonData")
    if not isinstance(comp, list) or len(comp) < 3:
        return False

    df = pd.DataFrame(comp)
    if not {"date", "portfolio", "market"}.issubset(df.columns):
        return False
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        return False
    df = df.dropna(subset=["date", "portfolio", "market"]).sort_values("date")
    if len(df) < 3:
        return False

    periods_per_year = _infer_periods_per_year(list(df["date"]))
    portfolio = pd.Series(df["portfolio"].astype(float).values, index=df["date"])
    market = pd.Series(df["market"].astype(float).values, index=df["date"])

    changed = False
    test_start = result.get("testPeriodStart")
    test_end = result.get("testPeriodEnd")
    train_start = result.get("trainPeriodStart")
    train_end = result.get("trainPeriodEnd")

    def _slice(series: pd.Series, start: Optional[str], end: Optional[str]) -> pd.Series:
        out = series
        if start:
            out = out[out.index >= pd.to_datetime(start)]
        if end:
            out = out[out.index <= pd.to_datetime(end)]
        return out

    train_port = _slice(portfolio, train_start, train_end)
    test_port = _slice(portfolio, test_start, test_end)
    train_mkt = _slice(market, train_start, train_end)
    test_mkt = _slice(market, test_start, test_end)

    if result.get("backtestSharpe") is None:
        target = test_port if len(test_port) >= 3 else portfolio
        result["backtestSharpe"] = _annualized_sharpe_from_levels(target, periods_per_year)
        changed = True
    if result.get("marketSharpe") is None:
        target = train_mkt if len(train_mkt) >= 3 else market
        result["marketSharpe"] = _annualized_sharpe_from_levels(target, periods_per_year)
        changed = True
    if result.get("marketBacktestSharpe") is None:
        target = test_mkt if len(test_mkt) >= 3 else market
        result["marketBacktestSharpe"] = _annualized_sharpe_from_levels(target, periods_per_year)
        changed = True
    if result.get("marketTotalSharpe") is None:
        result["marketTotalSharpe"] = _annualized_sharpe_from_levels(market, periods_per_year)
        changed = True

    return changed


_REMOVED_LLM_FACTORS = frozenset({"HY_SPREAD", "TERM_SPREAD", "VIX"})


def _strip_removed_factors_from_dict(d: dict) -> bool:
    changed = False
    for factor in _REMOVED_LLM_FACTORS:
        if factor in d:
            del d[factor]
            changed = True
    return changed


def _clean_explication_text(text: str) -> str:
    if not text:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept: list[str] = []
    for sentence in sentences:
        if any(factor in sentence for factor in _REMOVED_LLM_FACTORS):
            continue
        if re.search(
            r"\b(VIX|spread de crédit|spread crédit|pente de la courbe|volatilité implicite)\b",
            sentence,
            flags=re.IGNORECASE,
        ):
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


def _strip_removed_factors_from_prompt(text: str) -> str:
    if not text:
        return text
    cleaned = text
    for factor in _REMOVED_LLM_FACTORS:
        cleaned = re.sub(
            rf"-\s*{re.escape(factor)}\s*:.*?(?=\n- |\n\nRéponds|\Z)",
            "",
            cleaned,
            flags=re.DOTALL,
        )
        cleaned = re.sub(rf',?\s*"{re.escape(factor)}"\s*:\s*_', "", cleaned)
        cleaned = re.sub(
            rf'"{re.escape(factor)}"\s*:\s*(true|false)\s*,?\s*',
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
    cleaned = cleaned.replace("9 clés", "6 clés")
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return cleaned
    if not isinstance(obj, dict):
        return cleaned
    changed = False
    for factor in _REMOVED_LLM_FACTORS:
        if factor in obj:
            del obj[factor]
            changed = True
    expl = obj.get("explication")
    if isinstance(expl, str):
        cleaned_expl = _clean_explication_text(expl)
        if cleaned_expl != expl:
            obj["explication"] = cleaned_expl
            changed = True
    return json.dumps(obj, ensure_ascii=False) if changed else cleaned


def _sanitize_llm_result(llm_result: dict) -> bool:
    if not isinstance(llm_result, dict):
        return False
    changed = False

    for month in llm_result.get("monthlyHistory") or []:
        if not isinstance(month, dict):
            continue
        selected = month.get("selectedFactors")
        if isinstance(selected, dict):
            for ticker, mask in selected.items():
                if isinstance(mask, dict) and _strip_removed_factors_from_dict(mask):
                    changed = True
        factor_tests = month.get("factor_tests")
        if isinstance(factor_tests, dict):
            for ticker_data in factor_tests.values():
                if not isinstance(ticker_data, dict):
                    continue
                factor_stats = ticker_data.get("factor_stats")
                if isinstance(factor_stats, dict) and _strip_removed_factors_from_dict(factor_stats):
                    changed = True

    for example in llm_result.get("promptExamples") or []:
        if not isinstance(example, dict):
            continue
        for field in ("system", "user", "response"):
            raw = example.get(field)
            if not isinstance(raw, str):
                continue
            cleaned = _strip_removed_factors_from_prompt(raw)
            if cleaned != raw:
                example[field] = cleaned
                changed = True

    return changed


def _read_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        return []
    # Migration douce: historique sans tag → libellé neutre (gris côté UI).
    dirty = False
    market_sharpe_cache: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not entry.get("personTag"):
            entry["personTag"] = "Simulation de Test"
            dirty = True
        elif entry.get("personTag") == "test système simulation":
            entry["personTag"] = "Simulation de Test"
            dirty = True
        if _migrate_entry_market_sharpes(entry, market_sharpe_cache):
            dirty = True
        if isinstance(entry.get("result"), dict):
            dirty = _backfill_sharpes_from_comparison(entry["result"]) or dirty
        if isinstance(entry.get("classicResult"), dict):
            dirty = _backfill_sharpes_from_comparison(entry["classicResult"]) or dirty
        comparison_payload = entry.get("comparisonData")
        if isinstance(comparison_payload, dict):
            mc = comparison_payload.get("monteCarlo")
            bg = comparison_payload.get("bestGradient")
            if isinstance(mc, dict):
                dirty = _backfill_sharpes_from_comparison(mc) or dirty
            if isinstance(bg, dict):
                dirty = _backfill_sharpes_from_comparison(bg) or dirty
        llm_result = entry.get("llmResult")
        if isinstance(llm_result, dict):
            dirty = _sanitize_llm_result(llm_result) or dirty
    if dirty:
        _write_history(entries)
    return entries


def _write_history(entries: list) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _pick_one_history_entry_per_model(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("modelId") or "")
        if model_id and model_id not in by_model:
            by_model[model_id] = entry
            if len(by_model) >= len(_HISTORY_SAMPLE_MODEL_IDS):
                break
    return [by_model[m] for m in _HISTORY_SAMPLE_MODEL_IDS if m in by_model]


def _cors_allow_origins() -> list[str]:
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]
    extra = (os.environ.get("ALLOW_ORIGINS") or "").strip()
    if extra:
        for o in extra.split(","):
            u = o.strip().rstrip("/")
            if u and u not in origins:
                origins.append(u)
    return origins


app = FastAPI(title="Kairos Finance API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/stocks")
def list_stocks():
    try:
        return get_all_stocks()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Cryptos (CSV locaux, format CoinGecko) ───────────────────────────────────

CRYPTO_DATA_ROOT = _root / "gestion" / "crypto" / "données"


@app.get("/api/crypto/list")
def crypto_list():
    try:
        from gestion.crypto.markowitz_crypto_web import list_crypto_assets

        return list_crypto_assets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/crypto/history")
def crypto_history(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
):
    """Série de prix USD depuis les CSV « données » (dernier prix par jour)."""
    from gestion.crypto.crypto_fama_french import CSV_FILES

    sym = (symbol or "").strip().upper()
    if not sym or sym not in CSV_FILES:
        raise HTTPException(status_code=400, detail="Symbole crypto inconnu")
    path = CRYPTO_DATA_ROOT / CSV_FILES[sym]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Fichier de données absent")
    try:
        df = pd.read_csv(path, parse_dates=["snapped_at"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    df = df.rename(columns={"snapped_at": "Date", "price": "Close"})
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    if start:
        try:
            df = df[df["Date"] >= pd.Timestamp(start)]
        except Exception:
            pass
    if end:
        try:
            df = df[df["Date"] <= pd.Timestamp(end)]
        except Exception:
            pass
    dates = [d.strftime("%Y-%m-%d") for d in df["Date"]]
    series = {sym: [round(float(x), 6) for x in df["Close"].values]}
    return {"dates": dates, "series": series}


@app.get("/api/crypto/news-symbol")
def crypto_news_symbol(code: str):
    """Symbole Yahoo Finance pour les actualités (ex. BTC → BTC-USD)."""
    from gestion.crypto.markowitz_crypto_web import yahoo_ticker_for_news

    c = (code or "").strip().upper()
    if not c:
        raise HTTPException(status_code=400, detail="code requis")
    return {"code": c, "yahooSymbol": yahoo_ticker_for_news(c)}


# ── News (actualités) par symbole, avec cache ───────────────────────────────

NEWS_CACHE: dict[str, tuple[list[dict], datetime]] = {}
NEWS_CACHE_TTL_MINUTES = 25
_news_lock = threading.Lock()


def _get_news_for_symbol(symbol: str, limit: int = 12) -> list[dict]:
    """Récupère les actualités yfinance pour un symbole. Normalise title, url, publisher, date, thumbnail."""
    try:
        ticker = yf.Ticker(symbol)
        raw = ticker.get_news(count=min(limit, 50), tab="news")
    except Exception:
        return []
    out = []
    for item in raw or []:
        content = item.get("content")
        if not isinstance(content, dict):
            continue
        title = (content.get("title") or "").strip()
        if not title:
            continue
        # URL: canonicalUrl ou clickThroughUrl
        url = None
        for key in ("canonicalUrl", "clickThroughUrl"):
            u = content.get(key)
            if isinstance(u, dict) and u.get("url"):
                url = u["url"]
                break
        if not url:
            continue
        # Publisher
        provider = content.get("provider") or {}
        publisher = provider.get("displayName", "") if isinstance(provider, dict) else ""
        # Date
        pub_date = content.get("pubDate") or content.get("displayTime") or ""
        # Thumbnail (yfinance fournit parfois content.thumbnail.originalUrl)
        thumbnail = None
        thumb_obj = content.get("thumbnail")
        if isinstance(thumb_obj, dict) and thumb_obj.get("originalUrl"):
            thumbnail = thumb_obj["originalUrl"]
        summary = (content.get("summary") or content.get("description") or "").strip()
        out.append({
            "title": title,
            "url": url,
            "publisher": publisher,
            "publishedAt": pub_date,
            "thumbnail": thumbnail,
            "summary": summary[:300] if summary else None,
        })
        if len(out) >= limit:
            break
    return out


@app.get("/api/news")
def get_news(symbol: str, limit: int = 12):
    """Actualités récentes pour un symbole. Cache 25 min par symbole."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol requis")
    limit = max(1, min(30, limit))
    now = datetime.now()
    with _news_lock:
        cached = NEWS_CACHE.get(symbol)
        if cached:
            articles, fetched_at = cached
            if (now - fetched_at).total_seconds() < NEWS_CACHE_TTL_MINUTES * 60:
                return {"symbol": symbol, "articles": articles[:limit]}
        articles = _get_news_for_symbol(symbol, limit=limit)
        NEWS_CACHE[symbol] = (articles, now)
    return {"symbol": symbol, "articles": articles}


# ── Simulation history endpoints ──────────────────────────────────────────────

class SimulationEntry(BaseModel):
    id: str
    date: str
    modelId: str
    symbols: list[str]
    result: Any = None
    llmResult: Any = None
    classicResult: Any = None
    comparisonData: Optional[Any] = None  # { monteCarlo, bestGradient, bestGradientLabel }
    description: Optional[str] = None
    personTag: Optional[str] = None
    observedInterpretation: Optional[str] = None
    # Mode d'actifs au moment de la simulation : actions (défaut) ou crypto
    assetMode: Optional[str] = "actions"
    # Plage historique choisie (ajustement + backtest) au moment de la simulation
    simulationStartDate: Optional[str] = None
    simulationEndDate: Optional[str] = None


class DescriptionUpdate(BaseModel):
    description: str


class AnalysisUpdate(BaseModel):
    observedInterpretation: str


@app.get("/api/history/list")
def history_list():
    with _history_lock:
        return _read_history()


@app.get("/api/history/samples")
def history_samples():
    with _history_lock:
        entries = _read_history()
    return _pick_one_history_entry_per_model(entries)


@app.post("/api/history/save")
def history_save(entry: SimulationEntry):
    with _history_lock:
        entries = _read_history()
        entries.insert(0, entry.model_dump())
        _write_history(entries)
    return {"ok": True}


@app.patch("/api/history/{entry_id}/description")
def history_update_description(entry_id: str, body: DescriptionUpdate):
    with _history_lock:
        entries = _read_history()
        found = False
        for e in entries:
            if e.get("id") == entry_id:
                e["description"] = body.description
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail="Entrée introuvable")
        _write_history(entries)
    return {"ok": True}


@app.patch("/api/history/{entry_id}/analysis")
def history_update_analysis(entry_id: str, body: AnalysisUpdate):
    with _history_lock:
        entries = _read_history()
        found = False
        for e in entries:
            if e.get("id") == entry_id:
                e["observedInterpretation"] = body.observedInterpretation
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail="Entrée introuvable")
        _write_history(entries)
    return {"ok": True}


@app.delete("/api/history/{entry_id}")
def history_delete(entry_id: str):
    with _history_lock:
        entries = _read_history()
        new_entries = [e for e in entries if e.get("id") != entry_id]
        if len(new_entries) == len(entries):
            raise HTTPException(status_code=404, detail="Entrée introuvable")
        _write_history(new_entries)
    return {"ok": True}


# ── Stock price history ───────────────────────────────────────────────────────


def _equity_common_date_bounds(tickers: list[str]) -> dict[str, Optional[str]]:
    """Premier / dernier jour où tous les titres ont une cotation quotidienne (Adj Close) alignée.

    Évite les artefacts Yahoo (points isolés avant IPO) : on ne retient que les lignes complètes
    du DataFrame commun, comme en simulation après dropna(how='any').
    """
    tickers = [t.strip() for t in tickers if t.strip()]
    if not tickers:
        return {"commonStart": None, "commonEnd": None, "error": "Aucun symbole."}

    end_d = datetime.now() + timedelta(days=1)
    start_d = datetime(1990, 1, 1)
    prices, missing = yf_adj_close_wide(tickers, start_d, end_d, "1d")
    if missing:
        return {
            "commonStart": None,
            "commonEnd": None,
            "error": f"Pas de série Yahoo pour : {', '.join(missing)}",
        }
    cols = [t for t in tickers if t in prices.columns]
    if len(cols) < len(tickers):
        absent = [t for t in tickers if t not in prices.columns]
        return {
            "commonStart": None,
            "commonEnd": None,
            "error": f"Pas de série Yahoo pour : {', '.join(absent)}",
        }
    if prices.empty:
        return {
            "commonStart": None,
            "commonEnd": None,
            "error": "Téléchargement vide (symboles invalides ?)",
        }
    aligned = prices[cols].dropna(how="any")
    if aligned.empty:
        return {
            "commonStart": None,
            "commonEnd": None,
            "error": "Aucune date où tous les symboles ont une cotation simultanée.",
        }
    common_start = pd.Timestamp(aligned.index.min())
    common_end = pd.Timestamp(aligned.index.max())
    if common_start >= common_end:
        return {
            "commonStart": None,
            "commonEnd": None,
            "error": "Aucune fenêtre commune (cotations disjointes).",
        }
    return {
        "commonStart": common_start.strftime("%Y-%m-%d"),
        "commonEnd": common_end.strftime("%Y-%m-%d"),
    }


def _cached_equity_simulation_bounds(tickers: list[str]) -> dict[str, Optional[str]]:
    """Même résultat que _equity_common_date_bounds avec cache court (page Simulation plus réactive)."""
    key = tuple(sorted({t.strip() for t in tickers if t.strip()}))
    if not key:
        return {"commonStart": None, "commonEnd": None, "error": "Aucun symbole."}
    now = time.time()
    with _sim_bounds_lock:
        hit = _sim_bounds_cache.get(key)
        if hit and hit[0] > now:
            return dict(hit[1])
    out = _equity_common_date_bounds(list(key))
    with _sim_bounds_lock:
        _sim_bounds_cache[key] = (now + _SIM_BOUNDS_CACHE_TTL_SEC, dict(out))
    return dict(out)


def _parse_simulation_dates(start_date: Optional[str], end_date: Optional[str]) -> tuple[str, str]:
    if end_date:
        end_dt = datetime.fromisoformat(end_date[:10])
    else:
        end_dt = datetime.now()
    end_s = end_dt.strftime("%Y-%m-%d")
    if start_date:
        start_s = datetime.fromisoformat(start_date[:10]).strftime("%Y-%m-%d")
    else:
        start_s = "2005-01-01"
    if start_s >= end_s:
        raise HTTPException(
            status_code=400,
            detail="La date de début doit être strictement antérieure à la date de fin.",
        )
    return start_s, end_s


@app.get("/api/simulation-data-bounds")
def simulation_data_bounds(symbols: str, asset_mode: str = "actions"):
    """Borne min/max calendaires communes à tout le portefeuille (actions : Yahoo, cryptos : CSV)."""
    tickers = [s.strip() for s in (symbols or "").split(",") if s.strip()]
    if len(tickers) < 2:
        raise HTTPException(status_code=400, detail="Indiquez au moins 2 symboles.")
    mode = (asset_mode or "actions").strip().lower()
    if mode == "crypto":
        from gestion.crypto.markowitz_crypto_web import crypto_portfolio_common_bounds

        out = crypto_portfolio_common_bounds(tickers)
        if out.get("error"):
            raise HTTPException(status_code=400, detail=out["error"])
        return {
            "commonStart": out["commonStart"],
            "commonEnd": out["commonEnd"],
            "assetMode": "crypto",
        }
    out = _cached_equity_simulation_bounds(tickers)
    if out.get("error"):
        raise HTTPException(status_code=400, detail=out["error"])
    if not out.get("commonStart") or not out.get("commonEnd"):
        raise HTTPException(status_code=400, detail="Impossible de déterminer les bornes.")
    return {
        "commonStart": out["commonStart"],
        "commonEnd": out["commonEnd"],
        "assetMode": "actions",
    }


@app.get("/api/history")
def get_history(
    symbols: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
):
    if not symbols:
        raise HTTPException(status_code=400, detail="symbols requis")
    tickers = [s.strip() for s in symbols.split(",") if s.strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="symbols requis")
    if end is None:
        end_d = datetime.now()
    else:
        try:
            end_d = datetime.fromisoformat(end.replace("Z", ""))
        except ValueError:
            end_d = datetime.now()
    if start is None:
        start_d = datetime(2020, 3, 1)
    else:
        try:
            start_d = datetime.fromisoformat(start.replace("Z", ""))
        except ValueError:
            start_d = datetime(2005, 1, 1)
    interval_map = {"daily": "1d", "monthly": "1mo", "annual": "1y", "1d": "1d", "1mo": "1mo", "1y": "1y"}
    yf_interval = interval_map.get(interval, "1d")
    prices, _missing = yf_adj_close_wide(tickers, start_d, end_d, yf_interval)
    if prices.empty:
        return {"dates": [], "series": {}}
    dates = [d.strftime("%Y-%m-%d") for d in prices.index]
    series: dict[str, list[float]] = {}
    for t in tickers:
        if t in prices.columns:
            series[t] = [round(float(x), 2) for x in prices[t].values]
    return {"dates": dates, "series": series}


class SimulateRequest(BaseModel):
    model: str
    symbols: list[str]
    method: Optional[str] = None  # "monte_carlo" | "gradient_fixe" | "gradient_optimal" (ignoré pour markowitz-llm)
    start_date: Optional[str] = None  # YYYY-MM-DD — plage complète (ajustement + backtest)
    end_date: Optional[str] = None
    # Tirages Monte-Carlo (même défaut que gestion/*.py : 10 000). Ignoré si method n’est pas monte_carlo.
    monte_carlo_simulations: Optional[int] = Field(default=None, ge=100, le=500_000)


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    if len(req.symbols) < 2:
        raise HTTPException(status_code=400, detail="Sélectionnez au moins 2 actions pour lancer une simulation.")
    from gestion.config import OPTIMIZATION_METHOD
    method = req.method if req.method in ("monte_carlo", "gradient_fixe", "gradient_optimal") else OPTIMIZATION_METHOD
    mc_n = req.monte_carlo_simulations if req.monte_carlo_simulations is not None else 10_000
    try:
        if req.model == "markowitz-crypto-ff3":
            import gestion.crypto.markowitz_crypto_web as markowitz_crypto_web

            m = req.method if req.method in ("monte_carlo", "gradient_fixe", "gradient_optimal") else "gradient_optimal"
            if req.start_date and req.end_date:
                start_s, end_s = _parse_simulation_dates(req.start_date, req.end_date)
                result = markowitz_crypto_web.run(req.symbols, method=m, num_portfolios=mc_n, start=start_s, end=end_s)
            else:
                result = markowitz_crypto_web.run(req.symbols, method=m, num_portfolios=mc_n)
        else:
            start_s, end_s = _parse_simulation_dates(req.start_date, req.end_date)
            if req.model == "markowitz-classic":
                import gestion.markowitz_simple as markowitz_simple
                result = markowitz_simple.run(req.symbols, start_s, end_s, method=method, num_portfolios=mc_n)
            elif req.model == "markowitz-1factor":
                import gestion.multifactor.markowitz_1factor as markowitz_1factor
                result = markowitz_1factor.run(req.symbols, start_s, end_s, method=method, num_portfolios=mc_n)
            elif req.model == "markowitz-3factors":
                import gestion.multifactor.markowitz_3factors as markowitz_3factors
                result = markowitz_3factors.run(req.symbols, start_s, end_s, method=method, num_portfolios=mc_n)
            elif req.model == "markowitz-5factors":
                import gestion.multifactor.markowitz_5factors as markowitz_5factors
                result = markowitz_5factors.run(req.symbols, start_s, end_s, method=method, num_portfolios=mc_n)
            elif req.model == "markowitz-llm":
                import gestion.dynamic.markowitz_llm as markowitz_llm
                result = markowitz_llm.run(req.symbols, start_s, end_s)
            else:
                raise HTTPException(status_code=400, detail="Modèle inconnu")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/simulate-llm-stream")
async def simulate_llm_stream(req: SimulateRequest):
    """SSE endpoint streaming la progression du backtest LLM mois par mois."""
    if len(req.symbols) < 2:
        raise HTTPException(status_code=400, detail="Sélectionnez au moins 2 actions.")

    start_s, end_s = _parse_simulation_dates(req.start_date, req.end_date)

    async def event_generator():
        import gestion.dynamic.markowitz_llm as markowitz_llm

        loop = asyncio.get_running_loop()

        def send(event: str, data: dict):
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        try:
            progress_queue: asyncio.Queue = asyncio.Queue()

            import concurrent.futures
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = loop.run_in_executor(executor, lambda: markowitz_llm.run(
                req.symbols,
                start_s,
                end_s,
                progress_callback=lambda ev: asyncio.run_coroutine_threadsafe(
                    progress_queue.put(ev), loop
                ),
            ))

            while True:
                try:
                    evt = await asyncio.wait_for(progress_queue.get(), timeout=0.2)
                    yield send(evt["type"], evt)
                except asyncio.TimeoutError:
                    if future.done():
                        break
                    yield ": keepalive\n\n"
                    continue

            result = future.result()
            if "error" in result:
                yield send("error", {"message": result["error"]})
            else:
                yield send("result", result)

        except Exception as e:
            yield send("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


_dist_dir = _root / "dist"
_assets_dir = _dist_dir / "assets"


def _safe_dist_path(rel: str) -> Path | None:
    """rel sans slash initial ; None si traversal hors dist/."""
    if not rel or rel.startswith("/"):
        return None
    if any(p == ".." for p in rel.split("/")):
        return None
    p = (_dist_dir / rel).resolve()
    try:
        p.relative_to(_dist_dir.resolve())
    except ValueError:
        return None
    return p


if _dist_dir.is_dir() and _assets_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")

    @app.get("/")
    def spa_index():
        index = _dist_dir / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(index)

    @app.get("/{full_path:path}")
    def spa_or_static(full_path: str):
        if full_path.startswith("api/") or full_path in ("docs", "openapi.json", "redoc"):
            raise HTTPException(status_code=404)
        p = _safe_dist_path(full_path)
        if p is None:
            raise HTTPException(status_code=404)
        if p.is_file():
            return FileResponse(p)
        index = _dist_dir / "index.html"
        if index.is_file():
            return FileResponse(index)
        raise HTTPException(status_code=404)
