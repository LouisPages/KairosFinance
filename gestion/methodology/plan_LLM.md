# Plan technique — Sélection dynamique des facteurs Fama-French par LLM

## Contexte

Les modèles existants (`markowitz_1factor.py`, et les futurs `markowitz_3factors.py` / `markowitz_5factors.py`) utilisent des facteurs Fama-French **fixés statiquement**. L'objectif est d'introduire un pipeline LLM qui, chaque mois, détermine quels facteurs parmi les 5 (Mkt-RF, SMB, HML, RMW, CMA) sont pertinents pour chaque action du portefeuille, en s'appuyant sur l'actualité récente. Les poids du portefeuille sont ensuite recalculés sur une fenêtre glissante avec les seuls facteurs retenus.

---

## Vue d'ensemble du pipeline

```
[Utilisateur : liste de tickers] 
        ↓
[Pour chaque mois n]
        ↓
[LLM 1 — Mistral Le Chat] ← news AFP pour chaque ticker jusqu'au mois n
        ↓ résumé économique par ticker
[LLM 2 — Agent de sélection] ← résumé + définitions des 5 facteurs
        ↓ masque binaire des facteurs retenus par ticker
[Régression multi-facteurs sur fenêtre glissante (80% des données jusqu'au mois n)]
        ↓
[Optimisation Markowitz (Monte-Carlo) avec facteurs filtrés]
        ↓
[Poids du portefeuille pour le mois n+1]
```

---

## Phase 1 — Collecte des news (LLM 1 : Mistral Le Chat)

### Choix technique
Mistral dispose d'un accès direct aux dépêches AFP via son produit **Le Chat**. On utilise l'API de Mistral (endpoint `/v1/chat/completions`, modèle `mistral-large-latest` ou `mistral-medium`) avec une requête de type RAG implicite : on demande au modèle de synthétiser les événements économiques récents pour un ticker donné.

### Structure de la requête
Pour chaque ticker et pour le mois `n` :

```
System : "Tu es un analyste financier. Résume les événements économiques, 
sectoriels et macroéconomiques importants concernant [TICKER / NOM SOCIÉTÉ] 
survenus entre [n-3 mois] et [fin du mois n]. Sois factuel et concis. 
Retourne un JSON avec les champs : summary (str), key_events (list[str]), 
sentiment (str parmi positif/neutre/négatif)."

User : "Donne-moi les news importantes pour [TICKER]."
```

- **Fenêtre de news** : 3 mois glissants avant le mois `n` (paramétrable).
- **Format de sortie** : JSON strict (utiliser `response_format: { type: "json_object" }` de l'API Mistral).
- **Requêtes parallèles** : une requête par ticker, exécutées en parallèle via `asyncio` + `httpx.AsyncClient`.
- **Cache** : les résumés sont mis en cache dans un fichier JSON indexé par `(ticker, année_mois)` pour éviter de reconsommer l'API sur les mois déjà traités.

### Fichier concerné
`gestion/dynamic/llm_news_fetcher.py` (à créer)

---

## Phase 2 — Sélection des facteurs (LLM 2 : agent de sélection)

### Choix technique
Modèle recommandé : **`gpt-4o`** (OpenAI) ou **`claude-3-5-sonnet`** (Anthropic) — modèles à haut niveau de raisonnement structuré. Le choix final dépend des coûts, mais GPT-4o est préféré pour sa fiabilité sur les tâches de classification structurée avec JSON.

### Structure de la requête
Pour chaque ticker, à partir du résumé produit en Phase 1 :

```
System : "Tu es un expert en finance quantitative. En te basant sur le contexte 
économique d'une entreprise, tu dois décider quels facteurs du modèle 
Fama-French 5 facteurs sont pertinents pour prédire son rendement le mois suivant.

Définitions :
- Mkt-RF : prime de risque de marché (excès de rendement du marché sur le taux sans risque)
- SMB : Small Minus Big (prime de taille)
- HML : High Minus Low (prime de valeur/book-to-market)
- RMW : Robust Minus Weak (prime de profitabilité)
- CMA : Conservative Minus Aggressive (prime d'investissement)

Réponds uniquement en JSON : { 'Mkt-RF': bool, 'SMB': bool, 'HML': bool, 'RMW': bool, 'CMA': bool }"

User : "Contexte pour [TICKER] au mois [n] : [résumé Phase 1]"
```

- **Format de sortie** : JSON strict avec 5 clés booléennes.
- **Fallback** : si le parsing JSON échoue ou si moins de 2 facteurs sont sélectionnés, on revient au jeu complet des 5 facteurs pour ce ticker (comportement sûr).
- **Agrégation** : pour l'optimisation du portefeuille, on construit un **masque de facteurs global** = union des facteurs sélectionnés sur l'ensemble des tickers (un facteur est retenu si au moins un ticker le requiert).

### Fichier concerné
`gestion/dynamic/llm_factor_selector.py` (à créer)

---

## Phase 3 — Régression multi-facteurs sur fenêtre glissante

### Données de facteurs
Les 5 facteurs Fama-French sont chargés via `gestion/famafrench_data.py` (existant, à étendre pour le dataset 5 facteurs — fichier `F-F_Research_Data_5_Factors_2x3_CSV.zip` sur le site de Ken French, même logique de parsing, et à renommer en 'get_facteurs.py').

### Fenêtre glissante
Pour prédire le mois `n+1` :
- **Fenêtre d'entraînement** : tous les mois disponibles depuis le début de l'historique jusqu'au mois `n`, dont on prend les **80% les plus anciens** comme ensemble d'entraînement (cohérent avec le split existant dans `markowitz_1factor.py`).
- Formule : `split = int(total_mois_disponibles_jusqu_à_n * 0.8)`

### Régression
Pour chaque actif `i`, on estime sur la fenêtre d'entraînement :

$$R_i - R_f = \alpha_i + \sum_{k \in \mathcal{F}_n} \beta_{i,k} \cdot F_k + \varepsilon_i$$

où $\mathcal{F}_n$ est l'ensemble des facteurs sélectionnés par le LLM pour le mois `n`. Les $\hat{\beta}_{i,k}$ sont estimés par OLS (`numpy.linalg.lstsq` ou `statsmodels`).

Le rendement espéré annualisé :

$$\hat{\mu}_i = \left(\bar{R}_f + \sum_{k \in \mathcal{F}_n} \hat{\beta}_{i,k} \cdot \bar{F}_k \right) \times 12$$

La matrice de covariance $\Sigma$ est estimée sur les rendements mensuels bruts, puis annualisée (×12).

### Fichier concerné
`gestion/dynamic/markowitz_llm.py` (à créer) — s'inspire de `markowitz_1factor.py` mais avec facteurs dynamiques.

---

## Phase 4 — Optimisation et sortie

### Optimisation
Identique aux modèles existants : simulation Monte-Carlo (10 000 portefeuilles), maximisation du ratio de Sharpe. Les poids obtenus sont les **poids pour le mois `n+1`**.

### Mode de fonctionnement
Deux modes :
1. **Mode prédiction (un seul mois)** : étant donné une date cible `n`, calculer les poids pour `n+1`.
2. **Mode backtest glissant** : pour chaque mois de la période de test, recalculer les poids en faisant avancer la fenêtre — permet de mesurer la performance historique du pipeline complet.

### Sortie API
Le dictionnaire de sortie reprend la structure des modèles existants (champs `weights`, `sharpe`, `expectedReturn`, `volatility`, `maxDrawdown`, `comparisonData`, `efficientFrontier`) enrichi de :
- `selected_factors` : `{ ticker: [facteurs retenus], ... }` — un champ par mois pour le mode backtest
- `news_summaries` : `{ ticker: { summary, sentiment }, ... }`

---

## Architecture des fichiers

```
gestion/
├── famafrench_data.py          # existant — à étendre pour les 5 facteurs, à renommer en 'get_facteurs.py'
gestion/dynamic
├── markowitz_simple.py         # existant
├── markowitz_1factor.py        # existant
├── markowitz_3factors.py       # à créer (Fama-French statique 3 facteurs)
├── markowitz_5factors.py       # à créer (Fama-French statique 5 facteurs)
├── markowitz_llm.py            # à créer — pipeline LLM complet
├── llm_news_fetcher.py         # à créer — Phase 1 (Mistral Le Chat)
├── llm_factor_selector.py      # à créer — Phase 2 (GPT-4o ou Claude)
└── llm_cache/                  # répertoire de cache JSON (gitignore)
    └── news_cache.json
```

---

## Gestion des clés API et configuration

Les clés API (`MISTRAL_API_KEY`, `OPENAI_API_KEY` ou `ANTHROPIC_API_KEY`) sont lues depuis les variables d'environnement ou un fichier `.env` (via `python-dotenv`). Elles ne sont jamais hardcodées.

Un fichier `gestion/dynamic/llm_config.py` (à créer) centralise :
- les noms de modèles utilisés
- les paramètres (température = 0 pour reproductibilité, `max_tokens`)
- la fenêtre de news (défaut : 3 mois)
- le fallback en cas d'échec LLM

---

## Dépendances supplémentaires

| Package | Usage |
|---|---|
| `mistralai` | SDK officiel Mistral pour Le Chat |
| `openai` ou `anthropic` | SDK pour l'agent de sélection |
| `python-dotenv` | Chargement des clés API |
| `asyncio` + `httpx` (déjà présent) | Requêtes LLM parallèles |
| `statsmodels` (optionnel) | OLS avec statistiques de régression |
