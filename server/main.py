import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import asyncio
import threading
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Any, Optional
import yfinance as yf
import pandas as pd

from .tickers_data import get_all_stocks

HISTORY_FILE = Path(__file__).parent / "simulation_history.json"
MAX_ENTRIES = 20
_history_lock = threading.Lock()


def _read_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_history(entries: list) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


app = FastAPI(title="PE25 Portfolio API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/stocks")
def list_stocks():
    try:
        return get_all_stocks()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Simulation history endpoints ──────────────────────────────────────────────

class SimulationEntry(BaseModel):
    id: str
    date: str
    modelId: str
    symbols: list[str]
    result: Any = None
    llmResult: Any = None
    classicResult: Any = None


@app.get("/api/history/list")
def history_list():
    with _history_lock:
        return _read_history()


@app.post("/api/history/save")
def history_save(entry: SimulationEntry):
    with _history_lock:
        entries = _read_history()
        entries.insert(0, entry.model_dump())
        if len(entries) > MAX_ENTRIES:
            entries = entries[:MAX_ENTRIES]
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


@app.delete("/api/history")
def history_clear():
    with _history_lock:
        _write_history([])
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


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    if len(req.symbols) < 2:
        raise HTTPException(status_code=400, detail="Sélectionnez au moins 2 actions pour lancer une simulation.")
    end_d = datetime.now()
    start_d = datetime(2005, 1, 1)
    start_s = start_d.strftime("%Y-%m-%d")
    end_s = end_d.strftime("%Y-%m-%d")
    try:
        if req.model == "markowitz-classic":
            import gestion.markowitz_simple as markowitz_simple
            result = markowitz_simple.run(req.symbols, start_s, end_s)
        elif req.model == "markowitz-1factor":
            import gestion.markowitz_1factor as markowitz_1factor
            result = markowitz_1factor.run(req.symbols, start_s, end_s)
        elif req.model == "markowitz-3factors":
            import gestion.markowitz_3factors as markowitz_3factors
            result = markowitz_3factors.run(req.symbols, start_s, end_s)
        elif req.model == "markowitz-llm":
            import gestion.dynamic.markowitz_llm as markowitz_llm
            result = markowitz_llm.run(req.symbols, start_s, end_s)
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

    end_d = datetime.now()
    start_d = datetime(2005, 1, 1)
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
