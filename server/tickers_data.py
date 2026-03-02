"""
Liste des actions supportées (S&P 500, NASDAQ-100, Dow Jones).
Chargement depuis un fichier JSON local pour éviter tout fetch à chaque démarrage.
Les cours sont récupérés via l'API Yahoo Finance (yfinance) dans l'app.
"""
import json
from pathlib import Path
from typing import Optional

_CACHE: Optional[list[dict[str, str]]] = None

_SERVER_DIR = Path(__file__).resolve().parent
_STOCKS_JSON = _SERVER_DIR / "stocks_data.json"
_STOCKS_DEFAULT_JSON = _SERVER_DIR / "stocks_data.default.json"


def _load_from_file(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
        out = []
        for item in data:
            if isinstance(item, dict) and item.get("symbol"):
                out.append({
                    "symbol": str(item["symbol"]).strip(),
                    "name": str(item.get("name") or item["symbol"]).strip(),
                    "index": str(item.get("index") or "S&P 500").strip(),
                })
        return out
    except (OSError, json.JSONDecodeError):
        return []


def get_all_stocks() -> list[dict[str, str]]:
    """
    Retourne toutes les actions des indices S&P 500, NASDAQ-100 et Dow Jones.
    Charge depuis server/stocks_data.json si présent, sinon server/stocks_data.default.json.
    Pour regénérer les données : `cd server && python update_stocks_data.py`
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    data = _load_from_file(_STOCKS_JSON)
    if not data:
        data = _load_from_file(_STOCKS_DEFAULT_JSON)
    _CACHE = data
    return _CACHE
