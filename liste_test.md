# Plan de tests

## Méthodologie (suivant le message de Bruno)
On compare les modèles d’optimisation de portefeuille sur plusieurs **contextes de marché** et plusieurs **univers d’actions**, le tout limité aux **titres du S&P 500** (pas de cryptomonnaies dans ce plan).

**Portefeuilles**

- **Grandes capitalisations** : entreprises très larges (type « mega cap »), souvent tech ou leaders sectoriels.
- **Moyennes / petites (dans l’indice)** : entreprises du S&P 500 typiquement moins massives en capitalisation boursière que le premier groupe (toujours des constituants de l’indice, pas des micro-caps hors indice).
- **Diversifié** : mélange des deux approches pour un univers hétérogène.

**Périodes**

| Période   | Lecture macro (indicative)                          |
| --------- | --------------------------------------------------- |
| 2005–2015 | Inclut la crise financière de 2008–2009             |
| 2010–2019 | Longue phase de hausse après la crise               |
| 2015–2022 | Fin de cycle long, inflation, choc Covid            |
| 2018–2024 | Post-Covid, remontée des taux, régimes variés       |

**Après chaque simulation**

- Comparer les **performances** : rendement, volatilité, ratio de Sharpe (et indicateurs retenus dans le projet).
- **Tests de régression** sur les modèles à **1, 3 et 5 facteurs** : examiner les **p-valeurs** des variables / facteurs ajoutés pour juger si les extensions du modèle apportent une information statistiquement significative.

---

## Liste des simulations (actions uniquement)

Chaque ligne = une simulation à lancer : **portefeuille** + **fenêtre temporelle**. Les symboles sont des **tickers S&P 500** (à ajuster si l’indice est rééquilibré et qu’un titre en sort).

### Portefeuille A — Grandes entreprises

| ID | Période   | Tickers (exemple S&P 500) |
| -- | --------- | ------------------------- |
| A1 | 2005–2015 | AAPL, MSFT, GOOGL, AMZN, NVDA, ORCL |
| A2 | 2010–2019 | AAPL, MSFT, GOOGL, AMZN, NVDA, ORCL |
| A3 | 2015–2022 | AAPL, MSFT, GOOGL, AMZN, NVDA, ORCL |
| A4 | 2018–2024 | AAPL, MSFT, GOOGL, AMZN, NVDA, ORCL |

### Portefeuille B — Moyennes / petites (constituants S&P 500)

| ID | Période   | Tickers (exemple S&P 500) |
| -- | --------- | ------------------------- |
| B1 | 2005–2015 | CAG, CPB, HRL, CHD, IPG, KMX, AOS, TAP |
| B2 | 2010–2019 | CAG, CPB, HRL, CHD, IPG, KMX, AOS, TAP |
| B3 | 2015–2022 | CAG, CPB, HRL, CHD, IPG, KMX, AOS, TAP |
| B4 | 2018–2024 | CAG, CPB, HRL, CHD, IPG, KMX, AOS, TAP |

### Portefeuille C — Diversifié (grandes + moyennes/petites)

| ID | Période   | Tickers (exemple S&P 500) |
| -- | --------- | ------------------------- |
| C1 | 2005–2015 | AAPL, MSFT, GOOGL, AMZN, NVDA, ORCL, CAG, CPB, HRL, CHD, IPG, KMX |
| C2 | 2010–2019 | AAPL, MSFT, GOOGL, AMZN, NVDA, ORCL, CAG, CPB, HRL, CHD, IPG, KMX |
| C3 | 2015–2022 | AAPL, MSFT, GOOGL, AMZN, NVDA, ORCL, CAG, CPB, HRL, CHD, IPG, KMX |
| C4 | 2018–2024 | AAPL, MSFT, GOOGL, AMZN, NVDA, ORCL, CAG, CPB, HRL, CHD, IPG, KMX |
