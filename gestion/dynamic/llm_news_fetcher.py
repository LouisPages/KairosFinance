"""
Phase 1 — Collecte des news via Mistral Le Chat (accès AFP).
Pour chaque ticker et chaque mois cible, demande un résumé économique
sur une fenêtre glissante de NEWS_WINDOW_MONTHS mois.
Les résultats sont mis en cache dans llm_cache/news_cache.json.
"""
import asyncio
import json
import logging
from datetime import date
from dateutil.relativedelta import relativedelta

import httpx

from .llm_config import (
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    TEMPERATURE,
    MAX_TOKENS_NEWS,
    NEWS_WINDOW_MONTHS,
    ALL_FACTORS,
    NEWS_CACHE_FILE,
)

logger = logging.getLogger(__name__)

MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

NewsResult = dict  # { "summary": str, "key_events": list[str], "sentiment": str }


def _cache_key(ticker: str, year: int, month: int) -> str:
    return f"{ticker}_{year:04d}{month:02d}"


def _load_cache() -> dict:
    if NEWS_CACHE_FILE.exists():
        try:
            with open(NEWS_CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(NEWS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.warning("Impossible d'écrire le cache news : %s", e)


def _build_prompt(ticker: str, window_start: date, window_end: date) -> tuple[str, str]:
    system_msg = (
        "Tu es un analyste financier expert. Résume les événements économiques, "
        "sectoriels et macroéconomiques importants concernant l'action ou la société indiquée. "
        "Sois factuel, concis et objectif. "
        "Réponds UNIQUEMENT en JSON valide avec exactement ces trois champs : "
        '{"summary": "<string>", "key_events": ["<string>", ...], "sentiment": "<positif|neutre|négatif>"}'
    )
    user_msg = (
        f"Donne-moi les news importantes pour {ticker} "
        f"entre le {window_start.strftime('%d/%m/%Y')} et le {window_end.strftime('%d/%m/%Y')}."
    )
    return system_msg, user_msg


async def _fetch_single(
    client: httpx.AsyncClient,
    ticker: str,
    year: int,
    month: int,
) -> tuple[str, NewsResult]:
    """Appel asynchrone à l'API Mistral pour un (ticker, mois)."""
    window_end = date(year, month, 1) + relativedelta(months=1) - relativedelta(days=1)
    window_start = date(year, month, 1) - relativedelta(months=NEWS_WINDOW_MONTHS)
    system_msg, user_msg = _build_prompt(ticker, window_start, window_end)

    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS_NEWS,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    result = None
    for attempt in range(4):
        try:
            resp = await client.post(MISTRAL_ENDPOINT, json=payload, headers=headers, timeout=60)
            if resp.status_code == 429:
                wait = min(15 * (2 ** attempt), 60)
                logger.warning("Mistral rate limit pour %s (%d-%02d), tentative %d — attente %ds…", ticker, year, month, attempt + 1, wait)
                await asyncio.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = min(10 * (2 ** attempt), 60)
                logger.warning("Mistral erreur serveur %d pour %s (%d-%02d), tentative %d — attente %ds…", resp.status_code, ticker, year, month, attempt + 1, wait)
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            # Résoudre le JSON tronqué : extraire proprement si nécessaire
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                # Tenter de réparer le JSON tronqué
                import re as _re
                fixed = raw.strip()
                if not fixed.endswith("}"):
                    fixed = _re.sub(r',\s*"[^"]*"\s*:\s*[^\]}]*$', '', fixed)
                    fixed = fixed.rstrip(",").rstrip() + "}"
                result = json.loads(fixed)
            if not isinstance(result.get("key_events"), list):
                result["key_events"] = []
            if result.get("sentiment") not in ("positif", "neutre", "négatif"):
                result["sentiment"] = "neutre"
            # Tronquer les key_events si trop nombreux pour économiser le cache
            result["key_events"] = result["key_events"][:10]
            break
        except Exception as e:
            logger.warning("Erreur Mistral pour %s (%d-%02d) : %s", ticker, year, month, e)
            if attempt < 3:
                await asyncio.sleep(5)
            # Ne pas break sur la dernière tentative, laisser le fallback s'appliquer
    if result is None:
        result = {
            "summary": f"Résumé indisponible pour {ticker}.",
            "key_events": [],
            "sentiment": "neutre",
        }

    return ticker, result


async def _fetch_all_async(
    tickers: list[str],
    year: int,
    month: int,
    cache: dict,
) -> dict[str, NewsResult]:
    """Lance les requêtes en parallèle pour tous les tickers, en utilisant le cache."""
    to_fetch = []
    results: dict[str, NewsResult] = {}

    for ticker in tickers:
        key = _cache_key(ticker, year, month)
        if key in cache:
            results[ticker] = cache[key]
        else:
            to_fetch.append(ticker)

    if not to_fetch:
        return results

    if not MISTRAL_API_KEY:
        logger.warning("MISTRAL_API_KEY non définie — résumés factices utilisés.")
        for ticker in to_fetch:
            results[ticker] = {
                "summary": f"[Clé API manquante] Résumé indisponible pour {ticker}.",
                "key_events": [],
                "sentiment": "neutre",
            }
        return results

    async with httpx.AsyncClient() as client:
        fetched = []
        for t in to_fetch:
            pair = await _fetch_single(client, t, year, month)
            fetched.append(pair)
            # Respecter la limite du tier gratuit Mistral (~1 req/s)
            await asyncio.sleep(1.5)

    for ticker, result in fetched:
        results[ticker] = result
        cache[_cache_key(ticker, year, month)] = result

    return results


def fetch_news(
    tickers: list[str],
    year: int,
    month: int,
) -> dict[str, NewsResult]:
    """
    Point d'entrée synchrone.
    Retourne un dict { ticker: { summary, key_events, sentiment } }.
    """
    cache = _load_cache()
    results = asyncio.run(_fetch_all_async(tickers, year, month, cache))
    _save_cache(cache)
    return results
