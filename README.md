<p align="center">
  <img src="public/logo.png" alt="École Centrale de Lyon" width="120" />
</p>

<h1 align="center">Kairos Finance</h1>

<p align="center">
  <strong>PE25 — Projet d’études</strong> · École Centrale de Lyon<br/>
  Application web de <strong>gestion et simulation de portefeuille</strong> : sélection d’actifs (actions des grands indices US, cryptomonnaies), historiques de prix, actualités, modèles d’optimisation type Markowitz (classique, multi-facteurs, pipeline LLM, crypto Fama-French) et suivi des simulations.<br/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white" alt="Vite" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white" alt="TypeScript" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-3-06B6D4?logo=tailwindcss&logoColor=white" alt="Tailwind CSS" />
  <img src="https://img.shields.io/badge/React_Router-6-CA4245?logo=react-router&logoColor=white" alt="React Router" />
  <img src="https://img.shields.io/badge/Zod-3-3E67B3?logo=zod&logoColor=white" alt="Zod" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white" alt="Python" />
</p>

---

## Fonctionnalités

- **Accueil** — présentation du projet, liens vers les modules principaux, indicateurs (nombre d’actions / cryptos chargées depuis l’API).
- **Mode actifs** — bascule **actions** / **crypto** (navbar) : adapte le portefeuille sauvegardé, les modèles disponibles en simulation et les libellés côté interface.
- **Mon portefeuille** (`/portfolio`) — composition du portefeuille à partir du catalogue d’actions (S&P 500, NASDAQ-100, Dow Jones) ou des cryptos listées par le backend ; graphiques de performance et métriques de risque sur données historiques (**yfinance** côté API).
- **Simulation** (`/simulation`) — choix du modèle d’optimisation et de la méthode numérique (Monte-Carlo, gradient à pas fixe / optimal, mode comparaison) ; pour **Markowitz LLM**, progression en **Server-Sent Events** (backtest mois par mois) ; mode **crypto** : modèle **Fama-French à 3 facteurs** sur séries locales.
- **Historique** (`/history`) — enregistrement des dernières simulations (fichier JSON côté serveur), édition de description, suppression d’entrées.
- **Architecture** (`/architecture`) — page de documentation / schéma du système pour le rapport de projet.
- **Actualités** — flux d’articles liés à un symbole (Yahoo Finance), avec cache côté API pour limiter les appels répétés.
- **Expérience desktop** — sur petit écran, un écran **MobileBlock** invite à utiliser un affichage plus large (l’UI cible le bureau).
- **Interface** — composants **shadcn/ui** (Radix), graphiques **Recharts**, animations **Framer Motion**, notifications **Sonner**, formulaires compatibles **React Hook Form** + **Zod** où pertinent.

> **Note :** le dépôt contient aussi des notebooks / scripts de recherche sous `gestion/` (méthodologies Markowitz, Fama-French, module **VIX** en Python, etc.). Tout n’est pas forcément exposé par l’API FastAPI : les routes listées plus bas reflètent ce que le front consomme aujourd’hui.

---

## Stack technique

| Couche | Technologie |
|--------|-------------|
| Build & dev (front) | [Vite 5](https://vitejs.dev) |
| UI | [React 18](https://react.dev) |
| Langage (front) | TypeScript 5 |
| Routage | [React Router 6](https://reactrouter.com) |
| Style | [Tailwind CSS 3](https://tailwindcss.com) + thème applicatif |
| Composants | [Radix UI](https://www.radix-ui.com), [shadcn/ui](https://ui.shadcn.com), [Lucide React](https://lucide.dev) |
| Données client | [TanStack Query](https://tanstack.com/query) |
| Validation | [Zod](https://zod.dev), [React Hook Form](https://react-hook-form.com) |
| Graphiques | [Recharts](https://recharts.org) |
| API | [FastAPI](https://fastapi.tiangolo.com) + [Uvicorn](https://www.uvicorn.org) |
| Données marché | [yfinance](https://github.com/ranaroussi/yfinance), [pandas](https://pandas.pydata.org) |
| Stats / facteurs | [statsmodels](https://www.statsmodels.org), [pandas-datareader](https://pandas-datareader.readthedocs.io) |
| LLM (sélection de facteurs, etc.) | SDK **Mistral**, **OpenAI**, **Anthropic**, **Google GenAI** (selon configuration) |
| Tests (front) | [Vitest](https://vitest.dev), Testing Library |

---

## Prérequis

- **Node.js** ≥ 18 (LTS recommandé)
- **Python** ≥ 3.10 avec **pip**
- Connexion Internet pour **yfinance** (téléchargement des cours) et, si vous utilisez le modèle LLM ou les actualités, pour les appels réseau correspondants
- Pour le pipeline **Markowitz LLM** : clé API du fournisseur choisi (`SELECTOR_PROVIDER` dans `.env`, voir ci-dessous)

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/<org>/PE25.git
cd PE25
```

### 2. Installer les dépendances

**Frontend**

```bash
npm install
```

**Backend**

```bash
pip install -r server/requirements.txt
```

### 3. Configurer les variables d’environnement

À la racine du projet, créer un fichier **`.env`** (le chargeur Python lit ce fichier pour le module `gestion.dynamic.llm_config`). Un exemple minimal est fourni dans **`.env.example`**.

```bash
cp .env.example .env
```

Exemple de contenu (adapter selon les besoins) :

```env
# LLM — pipeline « choix dynamique des facteurs » (Markowitz LLM)
MISTRAL_API_KEY=
MISTRAL_MODEL=mistral-medium-latest
SELECTOR_PROVIDER=mistral

# Optionnel — autres fournisseurs si SELECTOR_PROVIDER change
# OPENAI_API_KEY=
# OPENAI_MODEL=gpt-4o
# ANTHROPIC_API_KEY=
# ANTHROPIC_MODEL=claude-sonnet-4-5
# GOOGLE_API_KEY=
# GEMINI_MODEL=gemini-2.0-flash
```

**Frontend** : si l’API n’est pas sur `http://localhost:8000`, définir par exemple :

```env
VITE_API_URL=http://127.0.0.1:8000
```

### 4. Données actions (optionnel)

La liste des tickers est lue depuis `server/stocks_data.json` si présent, sinon **`server/stocks_data.default.json`**. Pour régénérer le fichier à jour :

```bash
cd server && python update_stocks_data.py
```

### 5. Données crypto

Les séries historiques crypto attendues par l’API sont des **CSV** sous `gestion/crypto/données/` (format attendu : colonnes dont `snapped_at`, `price`). Sans ces fichiers, la liste ou l’historique crypto peut être vide ou renvoyer une erreur « fichier absent ».

### 6. Lancer l’application en développement

Le script **`npm run start`** lance en parallèle **Uvicorn** (port **8000**) et **Vite** (port **8080**).

```bash
npm run start
```

- Interface : [http://localhost:8080](http://localhost:8080)  
- API seule : `npm run api` → [http://127.0.0.1:8000](http://127.0.0.1:8000)  
- Front seul : `npm run dev` (sans backend, les appels `/api/...` échoueront sauf proxy ou `VITE_API_URL` vers une API déjà démarrée)

### 7. Build de production (frontend)

```bash
npm run build
```

Le résultat est dans **`dist/`**. En local, l’API et le front peuvent rester sur deux ports (voir `VITE_API_URL` et CORS dans `server/main.py`). Pour une **mise en ligne sur une seule URL** (recommandé), utiliser le **Dockerfile** à la racine : build Vite + Uvicorn qui sert l’UI et `/api/*` (même origine, pas de CORS à configurer).

### 8. Déploiement sur [Render](https://render.com)

Le fichier **`render.yaml`** décrit un **Web Service** Docker (plan **Free** par défaut). L’application répond sur une URL du type `https://pe25-portfolio.onrender.com` : page d’accueil, routes React (`/portfolio`, etc.) et API sous `/api/...`.

**Étapes**

1. Créer un compte Render et connecter le dépôt **GitHub** (ou GitLab) contenant ce projet.
2. **Dashboard** → **New** → **Blueprint** → sélectionner le dépôt → Render détecte `render.yaml` et propose de créer le service. Confirmer le déploiement.  
   *Alternative : **New** → **Web Service** → même dépôt → **Runtime** : **Docker** → répertoire racine du repo, **Dockerfile path** : `Dockerfile`.*
3. Dans **Environment** du service, renseigner au besoin les secrets (voir `.env.example`) : **`MISTRAL_API_KEY`** si vous utilisez Markowitz LLM ; laisser **`ALLOW_ORIGINS`** vide pour l’image monolithique. Les variables `SELECTOR_PROVIDER` / `MISTRAL_MODEL` sont préremplies dans le blueprint ; adaptez-les si vous changez de fournisseur.
4. Attendre la fin du **premier build** (plusieurs minutes : `npm ci`, `npm run build`, `pip install`). Le **health check** utilise `GET /api/health`.
5. Ouvrir l’URL **HTTPS** affichée par Render : c’est l’unique origine à utiliser pour la démo.

**Comportement du plan gratuit**

Le service **se met en veille** après une période sans trafic ; la **première requête** après veille peut prendre **une minute ou plus** (cold start + build image si redéploiement). Pour une démo sans veille, passer à un plan payant Render (**Starter** ou supérieur) sur ce même Web Service.

**Fichiers utiles**

| Fichier | Rôle |
|--------|------|
| `Dockerfile` | Image : Node (build `dist/`) + Python 3.12 + `uvicorn` sur le port **`PORT`** (injecté par Render). |
| `render.yaml` | Blueprint : nom du service, `healthCheckPath`, variables d’environnement. |
| `.dockerignore` | Réduit le contexte Docker ; l’historique `server/simulation_history.json` n’est pas copié (fichier recréé à l’usage sur l’instance). |

---

## Commandes disponibles

| Commande | Description |
|----------|-------------|
| `npm run start` | **Uvicorn** (reload) sur le port 8000 + **Vite** sur le port 8080 (recommandé en local) |
| `npm run dev` | Vite seul (port 8080) |
| `npm run api` | API FastAPI seule (`uvicorn server.main:app --reload --host 0.0.0.0 --port 8000`) |
| `npm run build` | Build de production dans `dist/` |
| `npm run build:dev` | Build en mode development |
| `npm run preview` | Prévisualisation du build Vite |
| `npm run lint` | ESLint |
| `npm test` | Vitest (une fois) |
| `npm run test:watch` | Vitest en mode watch |

---

## Structure du projet

```
├── gestion/                      # Logique métier Python (optimisation, facteurs, crypto, LLM…)
│   ├── Methodes_de_descente/     # Descentes de gradient (pas fixe / optimal)
│   ├── multifactor/              # Markowitz 1, 3 et 5 facteurs
│   ├── dynamic/                  # Pipeline LLM, config, cache, loaders FRED, etc.
│   ├── crypto/                   # Markowitz crypto + CSV « données » + helpers FF
│   ├── vix/                      # Travaux autour du VIX (recherche / extension)
│   ├── methodology/              # Notes méthodologiques (.md)
│   ├── config.py                 # Méthode d’optimisation par défaut (ex. gradient_optimal)
│   └── markowitz_simple.py       # Markowitz classique
├── server/
│   ├── main.py                   # Application FastAPI : routes /api/*
│   ├── tickers_data.py           # Chargement de la liste d’actions depuis JSON
│   ├── stocks_data.json          # Liste d’actions (optionnel, sinon .default.json)
│   ├── stocks_data.default.json
│   ├── simulation_history.json   # Historique des simulations (créé à l’usage)
│   ├── update_stocks_data.py     # Script de régénération des tickers
│   └── requirements.txt
├── src/
│   ├── App.tsx                   # Routes, providers, garde mobile
│   ├── main.tsx
│   ├── context/                  # Mode actifs actions / crypto
│   ├── components/               # Navbar, Footer, résultats de simulation, ui/ (shadcn)
│   ├── hooks/
│   ├── lib/                      # Client API, historique, stockage local portefeuille
│   ├── pages/                    # Home, Portfolio, Simulation, History, Architecture, NotFound
│   └── index.css
├── index.html
├── vite.config.ts                # Port 8080, alias `@` → `./src`
├── Dockerfile                    # Build front + API pour prod (ex. Render)
├── render.yaml                   # Blueprint Render (Web Service Docker)
├── structure.md                  # Guide pédagogique détaillé du front
├── .env.example
└── package.json
```

---

## API (FastAPI)

Base URL côté front : en dev, `VITE_API_URL` ou `http://localhost:8000` ; en build prod **sans** `VITE_API_URL`, requêtes relatives vers la même origine (Docker / Render monolithique).

| Méthode | Chemin | Rôle |
|---------|--------|------|
| `GET` | `/api/stocks` | Liste des actions (symbole, nom, indice) |
| `GET` | `/api/crypto/list` | Liste des cryptos disponibles (données locales) |
| `GET` | `/api/crypto/history` | Historique de prix (`symbol`, `start`, `end` optionnels) |
| `GET` | `/api/crypto/news-symbol` | Symbole Yahoo pour les news à partir du code crypto |
| `GET` | `/api/history` | Historique de cours pour un ou plusieurs tickers (`symbols`, dates, `interval`) |
| `GET` | `/api/news` | Actualités pour un symbole Yahoo (`symbol`, `limit`) |
| `POST` | `/api/simulate` | Lance une optimisation (`model`, `symbols`, `method` optionnel) |
| `POST` | `/api/simulate-llm-stream` | SSE : progression + résultat Markowitz LLM |
| `GET` | `/api/history/list` | Liste l’historique des simulations enregistrées |
| `POST` | `/api/history/save` | Enregistre une entrée d’historique |
| `PATCH` | `/api/history/{id}/description` | Met à jour la description d’une entrée |
| `DELETE` | `/api/history/{id}` | Supprime une entrée |

**Modèles** attendus par `POST /api/simulate` (champ `model`) : `markowitz-classic`, `markowitz-1factor`, `markowitz-3factors`, `markowitz-5factors`, `markowitz-llm`, `markowitz-crypto-ff3`.

---

## Crédits

**Projet d’études (PE25)** — **École Centrale de Lyon** : application de recherche et de démonstration autour de l’optimisation de portefeuille et des modèles factoriels, sans conseil en investissement.

Pour une présentation pas à pas du code React / Vite du dépôt, voir aussi **`structure.md`**.
