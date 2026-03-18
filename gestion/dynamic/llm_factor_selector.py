"""
Phase 2 — Sélection binaire des facteurs Fama-French via LLM.

Le LLM retourne un masque booléen par facteur (actif / inactif) selon
l'actualité économique du mois.

Providers supportés : "anthropic" (défaut), "mistral", "gemini", "openai".
"""
import json
import logging
import time
from typing import Any

from .llm_config import (
    SELECTOR_PROVIDER,
    MISTRAL_MODEL,
    MISTRAL_API_KEY,
    OPENAI_MODEL,
    OPENAI_API_KEY,
    ANTHROPIC_MODEL,
    ANTHROPIC_API_KEY,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    TEMPERATURE,
    MAX_TOKENS_SELECTOR,
    ALL_FACTORS,
    MIN_FACTORS_REQUIRED,
    FACTORS_CACHE_FILE,
    SELECTOR_MAX_RETRIES,
)

logger = logging.getLogger(__name__)

# Masque booléen par facteur
FactorMask = dict[str, bool]

_FALLBACK_MASK: FactorMask = {
    "Mkt-RF": True, "SMB": False, "HML": False, "RMW": False, "CMA": False,
    "UMD": False, "HY_SPREAD": False, "TERM_SPREAD": False, "VIX": False,
}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
Tu es un expert en finance quantitative spécialisé dans les modèles factoriels.

Ta mission : pour l'action {ticker}, détermine si chaque facteur est \
PERTINENT (true) ou NON PERTINENT (false) ce mois-ci, compte tenu de l'actualité économique.

Définitions des facteurs :
- Mkt-RF    : prime de risque de marché (exposition au risque systématique global). \
               Pertinent uniquement si un choc systémique ou un mouvement de marché majeur \
               domine l'actualité du mois. Non pertinent en période calme où les mouvements \
               sont idiosyncratiques.
- SMB       : Small Minus Big (prime de taille — favorise les petites capitalisations)
- HML       : High Minus Low (prime de valeur — favorise les entreprises avec P/B faible)
- RMW       : Robust Minus Weak (prime de profitabilité — favorise les entreprises rentables)
- CMA       : Conservative Minus Aggressive (prime d'investissement — favorise les entreprises \
               qui investissent peu / rachètent des actions)
- UMD       : Up Minus Down (momentum — les titres récemment performants continuent \
               de surperformer à 1-12 mois). Pertinent si la tendance du titre est claire \
               (hausse ou baisse soutenue récente). Non pertinent si le titre est en retournement \
               ou en phase de consolidation sans tendance.
- HY_SPREAD : variation mensuelle du spread crédit HY vs IG. Pertinent si l'environnement \
               de crédit est sous tension ou se détend fortement (stress financier, resserrement \
               monétaire, récession, rally obligataire). Non pertinent en période calme.
- TERM_SPREAD: variation mensuelle de la pente de la courbe des taux (10Y - 3M). Pertinent \
               si la politique monétaire ou les anticipations de croissance évoluent fortement \
               ce mois (hausse/baisse de taux, inversion/normalisation de la courbe). \
               Non pertinent si la courbe est stable.
- VIX       : variation mensuelle de la volatilité implicite (indice VIX). Pertinent si \
               l'aversion au risque change brutalement (choc de marché, crise, rebond après \
               sell-off). Non pertinent en période de volatilité stable et faible.


Réponds UNIQUEMENT avec ce JSON compact (9 clés, valeurs booléennes true/false) :
{{"Mkt-RF": _, "SMB": _, "HML": _, "RMW": _, "CMA": _, "UMD": _, "HY_SPREAD": _, "TERM_SPREAD": _, "VIX": _, "explication": "<string>"}}

IMPORTANT : EXACTEMENT DEUX FACTEURS DOIVENT VALOIR TRUE UNIQUEMENT ET TU DOIS AJOUTER UNE EXPLICATION DE TON CHOIX"""


def _build_system_prompt(ticker: str) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(
        ticker=ticker,
    )


def _build_user_message(ticker: str, year: int, month: int, news: dict[str, Any]) -> str:
    summary = news.get("summary", "Aucun résumé disponible.")
    sentiment = news.get("sentiment", "neutre")
    key_events = news.get("key_events", [])
    events_str = "\n".join(f"  • {e}" for e in key_events) if key_events else "  Aucun événement clé identifié."
    return (
        f"Actualité pour {ticker} — {year}-{month:02d} :\n\n"
        f"Résumé : {summary}\n\n"
        f"Événements clés :\n{events_str}\n\n"
        f"Sentiment global : {sentiment}\n\n"
        "Pour chaque facteur, réponds true si l'actualité de ce mois le rend pertinent "
        "pour ce titre, false sinon. Retourne UNIQUEMENT le JSON compact. "
        "IMPORTANT : EXACTEMENT DEUX FACTEURS DOIVENT VALOIR TRUE UNIQUEMENT ET TU DOIS AJOUTER UNE EXPLICATION DE TON CHOIX"
    )


# ---------------------------------------------------------------------------
# Parsing du masque booléen
# ---------------------------------------------------------------------------

def _parse_mask(raw: str) -> FactorMask | None:
    import re

    obj = None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        fixed = raw.strip()
        if not fixed.endswith("}"):
            fixed = re.sub(r',\s*"[^"]*"\s*:\s*[^,}]*$', '', fixed)
            fixed = re.sub(r',\s*"[^"]*"\s*$', '', fixed)
            fixed = fixed.rstrip(",").rstrip() + "}"
        try:
            obj = json.loads(fixed)
        except json.JSONDecodeError:
            pass

    if obj is not None:
        mask: FactorMask = {}
        for f in ALL_FACTORS:
            val = obj.get(f)
            if isinstance(val, bool):
                mask[f] = val
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                mask[f] = float(val) >= 0.5
            else:
                mask[f] = True

        active = sum(1 for v in mask.values() if v)
        if active < MIN_FACTORS_REQUIRED:
            logger.warning("Moins de %d facteurs actifs — fallback.", MIN_FACTORS_REQUIRED)
            return None
        return mask

    # Extraction regex de secours
    logger.warning("JSON non réparable, tentative regex | raw=%r", raw[:300])
    mask = {}
    for f in ALL_FACTORS:
        m = re.search(rf'"{re.escape(f)}"\s*:\s*(true|false|[01])', raw, re.IGNORECASE)
        if m:
            v = m.group(1).lower()
            mask[f] = v in ("true", "1")
    if len(mask) >= MIN_FACTORS_REQUIRED:
        for f in ALL_FACTORS:
            if f not in mask:
                mask[f] = True
        return mask

    logger.warning("Parsing totalement échoué — fallback complet.")
    return None


# ---------------------------------------------------------------------------
# Appels LLM
# ---------------------------------------------------------------------------

def _call_openai(system_prompt: str, user_msg: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS_SELECTOR,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


# Throttle global pour l'API gratuite Anthropic : 5 req/min max
_ANTHROPIC_MIN_INTERVAL = 13.0  # secondes entre deux appels (60s / 5 req + marge)
_anthropic_last_call: float = 0.0


def _call_anthropic(system_prompt: str, user_msg: str, max_retries: int | None = None) -> str:
    if max_retries is None:
        max_retries = SELECTOR_MAX_RETRIES
    import anthropic

    global _anthropic_last_call
    elapsed = time.monotonic() - _anthropic_last_call
    if elapsed < _ANTHROPIC_MIN_INTERVAL:
        wait = _ANTHROPIC_MIN_INTERVAL - elapsed
        logger.debug("Anthropic throttle — attente %.1fs…", wait)
        time.sleep(wait)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    for attempt in range(max_retries):
        _anthropic_last_call = time.monotonic()
        try:
            resp = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=MAX_TOKENS_SELECTOR,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
                temperature=TEMPERATURE,
            )
            return resp.content[0].text if resp.content else ""
        except Exception as e:
            err_str = str(e)
            if "529" in err_str or "overloaded" in err_str.lower():
                wait = min(30 * (2 ** attempt), 120)
                logger.warning("Anthropic surchargé (tentative %d/%d) — attente %ds…", attempt + 1, max_retries, wait)
                time.sleep(wait)
            elif "429" in err_str or "rate_limit" in err_str.lower():
                wait = min(15 * (2 ** attempt), 120)
                logger.warning("Anthropic rate limit (tentative %d/%d) — attente %ds…", attempt + 1, max_retries, wait)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Anthropic : échec après %d tentatives." % max_retries)


def _call_gemini(system_prompt: str, user_msg: str, max_retries: int | None = None) -> str:
    if max_retries is None:
        max_retries = SELECTOR_MAX_RETRIES
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GOOGLE_API_KEY)
    full_prompt = f"{system_prompt}\n\n{user_msg}"
    time.sleep(4)
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_TOKENS_SELECTOR,
                    response_mime_type="application/json",
                ),
            )
            return resp.text if resp.text else ""
        except Exception as e:
            err_str = str(e)
            if "PerDay" in err_str or "per_day" in err_str.lower():
                raise RuntimeError(
                    "Quota journalier Gemini épuisé. "
                    "Ajoutez SELECTOR_PROVIDER=anthropic dans .env."
                ) from e
            if "429" in err_str or "quota" in err_str.lower():
                wait = min(30 * (2 ** attempt), 120)
                logger.warning("Gemini rate limit (tentative %d/%d) — attente %ds…", attempt + 1, max_retries, wait)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini : quota dépassé après %d tentatives." % max_retries)


def _call_mistral_selector(system_prompt: str, user_msg: str, max_retries: int | None = None) -> str:
    if max_retries is None:
        max_retries = SELECTOR_MAX_RETRIES
    import httpx
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS_SELECTOR,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    time.sleep(1)
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post("https://api.mistral.ai/v1/chat/completions", json=payload, headers=headers)
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = min(10 * (2 ** attempt), 60)
                logger.warning("Mistral %d pour tentative %d/%d — attente %ds…", resp.status_code, attempt + 1, max_retries, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"] or ""
        except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            wait = min(10 * (2 ** attempt), 60)
            logger.warning("Mistral timeout tentative %d/%d — attente %ds… (%s)", attempt + 1, max_retries, wait, e)
            time.sleep(wait)
        except Exception:
            raise
    raise RuntimeError("Mistral sélection : échec après %d tentatives." % max_retries)


# ---------------------------------------------------------------------------
# Cache disque : choix et explications des facteurs (ticker × mois)
# ---------------------------------------------------------------------------

def _factors_cache_key(ticker: str, year: int, month: int) -> str:
    return f"{ticker}_{year:04d}{month:02d}"


def _load_factors_cache() -> dict:
    """Charge le cache des réponses LLM (masque + explication) depuis le fichier."""
    if FACTORS_CACHE_FILE.exists():
        try:
            with open(FACTORS_CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_factors_cache(cache: dict) -> None:
    """Enregistre le cache des choix et explications des facteurs dans llm_cache/factors_cache.json."""
    try:
        FACTORS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FACTORS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.warning("Impossible d'écrire le cache facteurs : %s", e)


# ---------------------------------------------------------------------------
# Log de tous les prompts capturés (un par ticker × mois)
# ---------------------------------------------------------------------------

_prompt_log: list[dict[str, str]] = []


def reset_prompt_log() -> None:
    """Réinitialise le log au début de chaque run."""
    global _prompt_log
    _prompt_log = []


def get_prompt_example() -> dict[str, str] | None:
    """Rétrocompatibilité : retourne le premier prompt capturé."""
    return _prompt_log[0] if _prompt_log else None


def get_all_prompt_examples() -> list[dict[str, str]]:
    """Retourne tous les prompts capturés (triés par mois puis ticker)."""
    return sorted(_prompt_log, key=lambda x: (x["month"], x["ticker"]))


# ---------------------------------------------------------------------------
# Interface publique
# ---------------------------------------------------------------------------

def select_factors_for_ticker(
    ticker: str,
    year: int,
    month: int,
    news: dict[str, Any],
    cache: dict | None = None,
) -> FactorMask:
    """
    Retourne un dict { facteur: bool } pour un ticker et un mois donnés.
    En cas d'échec, retourne le fallback (tous les facteurs actifs).
    Si cache est fourni, lit/écrit les choix et explications dans le cache disque.
    """
    provider = SELECTOR_PROVIDER.lower()
    key = _factors_cache_key(ticker, year, month)

    # Lecture depuis le cache (choix + explication dans "response")
    if cache is not None and key in cache:
        entry = cache[key]
        raw = entry.get("response", "")
        mask = _parse_mask(raw)
        if mask is not None:
            _prompt_log.append({
                "ticker": ticker,
                "month": f"{year}-{month:02d}",
                "provider": entry.get("provider", provider),
                "system": entry.get("system", ""),
                "user": entry.get("user", ""),
                "response": raw.strip(),
            })
            logger.info(
                "%s %d-%02d (cache) — facteurs actifs : %s",
                ticker, year, month,
                [f for f, v in mask.items() if v],
            )
            return mask

    api_key_map = {
        "openai": OPENAI_API_KEY,
        "anthropic": ANTHROPIC_API_KEY,
        "gemini": GOOGLE_API_KEY,
        "mistral": MISTRAL_API_KEY,
    }
    api_key = api_key_map.get(provider, "")
    if not api_key:
        logger.warning("Clé API %s manquante pour %s — fallback.", provider.upper(), ticker)
        return _FALLBACK_MASK.copy()

    system_prompt = _build_system_prompt(ticker)
    user_msg = _build_user_message(ticker, year, month, news)

    try:
        if provider == "openai":
            raw = _call_openai(system_prompt, user_msg)
        elif provider == "anthropic":
            raw = _call_anthropic(system_prompt, user_msg)
        elif provider == "gemini":
            raw = _call_gemini(system_prompt, user_msg)
        elif provider == "mistral":
            raw = _call_mistral_selector(system_prompt, user_msg)
        else:
            logger.warning("Provider inconnu '%s' — fallback.", provider)
            return _FALLBACK_MASK.copy()

        mask = _parse_mask(raw)
        if mask is None:
            logger.warning("Parsing échoué pour %s — fallback.", ticker)
            return _FALLBACK_MASK.copy()

        # Enregistrement dans le log et dans le cache disque (choix + explication)
        log_entry = {
            "ticker": ticker,
            "month": f"{year}-{month:02d}",
            "provider": provider,
            "system": system_prompt,
            "user": user_msg,
            "response": raw.strip(),
        }
        _prompt_log.append(log_entry)
        if cache is not None:
            cache[key] = log_entry

        logger.info(
            "%s %d-%02d — facteurs actifs : %s",
            ticker, year, month,
            [f for f, v in mask.items() if v],
        )
        return mask

    except Exception as e:
        logger.warning("Erreur LLM sélection pour %s : %s — fallback.", ticker, e)
        return _FALLBACK_MASK.copy()


def select_factors(
    tickers: list[str],
    year: int,
    month: int,
    news_results: dict[str, dict],
) -> tuple[dict[str, FactorMask], list[str]]:
    """
    Lance la sélection pour tous les tickers.
    Charge le cache des facteurs (choix + explications), le met à jour, puis le sauvegarde.
    Retourne :
      - per_ticker  : { ticker: { facteur: bool } }
      - active_factors : facteurs dont au moins un ticker les a activés
    """
    cache = _load_factors_cache()
    per_ticker: dict[str, FactorMask] = {}
    for ticker in tickers:
        news = news_results.get(ticker, {})
        per_ticker[ticker] = select_factors_for_ticker(ticker, year, month, news, cache=cache)
    _save_factors_cache(cache)

    active_factors = [f for f in ALL_FACTORS if any(per_ticker[t].get(f, True) for t in tickers)]
    if len(active_factors) < MIN_FACTORS_REQUIRED:
        active_factors = ALL_FACTORS[:]

    return per_ticker, active_factors
