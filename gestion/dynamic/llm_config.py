"""
Configuration centralisée pour le pipeline LLM.
Les clés API sont lues depuis les variables d'environnement ou un fichier .env.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

# --- Modèles ---
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-medium-latest")
SELECTOR_PROVIDER = os.getenv("SELECTOR_PROVIDER", "gemini")  # "openai", "anthropic", "gemini" ou "mistral"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# --- Clés API ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# --- Paramètres LLM ---
TEMPERATURE = 0.0
MAX_TOKENS_NEWS = 1500
MAX_TOKENS_SELECTOR = 800
# Nombre max de tentatives en cas de rate limit / erreur temporaire (sélection facteurs)
SELECTOR_MAX_RETRIES = 1

# --- Fenêtre de news ---
NEWS_WINDOW_MONTHS = 3

# --- Facteurs (FF5 + Momentum + Macro FRED) ---
ALL_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "HY_SPREAD", "TERM_SPREAD", "VIX"]
MIN_FACTORS_REQUIRED = 1

# --- Cache ---
CACHE_DIR = Path(__file__).resolve().parent / "llm_cache"
CACHE_DIR.mkdir(exist_ok=True)
NEWS_CACHE_FILE = CACHE_DIR / "news_cache.json"
FACTORS_CACHE_FILE = CACHE_DIR / "factors_cache.json"