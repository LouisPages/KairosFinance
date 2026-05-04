import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
# Pour que les modules gestion.* trouvent Methodes_de_descente (sous-dossier de gestion)
sys.path.insert(1, str(_root / "gestion"))

import json
import os
import asyncio
import threading
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Any, Optional
import yfinance as yf
import pandas as pd

from .tickers_data import get_all_stocks

HISTORY_FILE = Path(__file__).parent / "simulation_history.json"
_history_lock = threading.Lock()


def _read_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        return []
    # Migration douce: historique sans tag → libellé neutre (gris côté UI).
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not entry.get("personTag"):
            entry["personTag"] = "Simulation de Test"
        elif entry.get("personTag") == "test système simulation":
            entry["personTag"] = "Simulation de Test"
    return entries


def _write_history(entries: list) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


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


class DescriptionUpdate(BaseModel):
    description: str


class AnalysisUpdate(BaseModel):
    observedInterpretation: str


@app.get("/api/history/list")
def history_list():
    with _history_lock:
        return _read_history()


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
    data = yf.download(tickers, start=start_d, end=end_d, auto_adjust=False, progress=False, group_by="column")
    if data.empty:
        return {"dates": [], "series": {}}
    if len(tickers) == 1:
        if "Adj Close" in data.columns:
            series = data["Adj Close"]
        else:
            series = data["Close"]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        series = series.dropna()
        out = {
            "dates": [d.strftime("%Y-%m-%d") for d in series.index],
            "series": {tickers[0]: [round(float(x), 2) for x in series.values]},
        }
        return out
    if isinstance(data.columns, pd.MultiIndex):
        if data.columns.names[0] in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
            prices = data["Adj Close"].copy() if "Adj Close" in data.columns else data["Close"].copy()
        else:
            prices = data.xs("Adj Close", axis=1, level=1).copy() if "Adj Close" in data.columns.get_level_values(1) else data.xs("Close", axis=1, level=1).copy()
    else:
        prices = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = [c[-1] if isinstance(c, tuple) else c for c in prices.columns]
    dates = [d.strftime("%Y-%m-%d") for d in prices.index]
    series = {}
    for t in tickers:
        if t in prices.columns:
            series[t] = [round(float(x), 2) for x in prices[t].values]
    return {"dates": dates, "series": series}


class SimulateRequest(BaseModel):
    model: str
    symbols: list[str]
    method: Optional[str] = None  # "monte_carlo" | "gradient_fixe" | "gradient_optimal" (ignoré pour markowitz-llm)


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    if len(req.symbols) < 2:
        raise HTTPException(status_code=400, detail="Sélectionnez au moins 2 actions pour lancer une simulation.")
    end_d = datetime.now()
    start_d = datetime(2005, 1, 1)
    start_s = start_d.strftime("%Y-%m-%d")
    end_s = end_d.strftime("%Y-%m-%d")
    from gestion.config import OPTIMIZATION_METHOD
    method = req.method if req.method in ("monte_carlo", "gradient_fixe", "gradient_optimal") else OPTIMIZATION_METHOD
    try:
        if req.model == "markowitz-classic":
            import gestion.markowitz_simple as markowitz_simple
            result = markowitz_simple.run(req.symbols, start_s, end_s, method=method)
        elif req.model == "markowitz-1factor":
            import gestion.multifactor.markowitz_1factor as markowitz_1factor
            result = markowitz_1factor.run(req.symbols, start_s, end_s, method=method)
        elif req.model == "markowitz-3factors":
            import gestion.multifactor.markowitz_3factors as markowitz_3factors
            result = markowitz_3factors.run(req.symbols, start_s, end_s, method=method)
        elif req.model == "markowitz-5factors":
            import gestion.multifactor.markowitz_5factors as markowitz_5factors
            result = markowitz_5factors.run(req.symbols, start_s, end_s, method=method)
        elif req.model == "markowitz-llm":
            import gestion.dynamic.markowitz_llm as markowitz_llm
            result = markowitz_llm.run(req.symbols, start_s, end_s)
        elif req.model == "markowitz-crypto-ff3":
            import gestion.crypto.markowitz_crypto_web as markowitz_crypto_web
            m = req.method if req.method in ("monte_carlo", "gradient_fixe", "gradient_optimal") else "gradient_optimal"
            result = markowitz_crypto_web.run(req.symbols, method=m)
        else:
            raise HTTPException(status_code=400, detail="Modèle inconnu")
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

    # Backtest LLM : 1 an de backtest (12 mois) ≈ 20 % → plage large pour atteindre 59+ mois (déc. 2020 → janv. 2026)
    start_d = datetime(2005, 1, 1)
    end_d = datetime(2026, 1, 1)
    start_s = start_d.strftime("%Y-%m-%d")
    end_s = end_d.strftime("%Y-%m-%d")

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
