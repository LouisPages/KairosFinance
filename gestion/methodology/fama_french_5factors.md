# Plan technique — Modèle de Fama-French à cinq facteurs

## Description

Ajout d'un modèle d'optimisation de portefeuille basé sur le modèle à cinq facteurs de Fama et French (2015). Les rendements espérés sont estimés par régression OLS sur cinq facteurs (Mkt-RF, SMB, HML, RMW, CMA) chargés depuis `get_facteurs.py`. L'allocation optimale est ensuite déterminée par simulation Monte-Carlo. Ce modèle est une extension directe du modèle à trois facteurs : les deux facteurs supplémentaires RMW (profitabilité) et CMA (investissement) sont incorporés à la régression sans modifier l'architecture générale.

## Fichiers concernés

### À créer
- `gestion/markowitz_5factors.py` — module principal exposant la fonction `run()`

### À utiliser (sans modification)
- `gestion/get_facteurs.py` — fonction `load_famafrench_5factors(start, end)` qui retourne un `DataFrame` mensuel avec les colonnes `Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`, `RF`

### Référence structurelle
- `gestion/markowitz_3factors.py` (à créer en parallèle) — identique à ce modèle, sans RMW et CMA

---

## Algorithme pas à pas

### 1. Collecte et prétraitement des données

- Télécharger les prix de clôture ajustés via `yfinance` pour la liste de tickers sur la période `[start, end]`.
- Rééchantillonner les prix au **dernier jour ouvré de chaque mois** (`resample("ME").last()`).
- Calculer les rendements arithmétiques mensuels : $R_{i,t} = P_{i,t}/P_{i,t-1} - 1$.
- Écarter les actifs dont l'historique est entièrement manquant.
- Découper en **train (80 %)** / **test (20 %)** avec un minimum de 24 observations en train.
- Charger les facteurs Fama-French 5 via `load_famafrench_5factors(start, end)` depuis `get_facteurs.py`.
- Aligner l'index mensuel (`Period("M")`) des facteurs sur celui des rendements par jointure interne.

### 2. Estimation des rendements espérés par régression OLS à cinq facteurs

Pour chaque actif $i$, estimer sur l'ensemble d'entraînement :

$$R_{i,t} - R_{f,t} = \alpha_i + \beta_i^{Mkt}(R_{m,t} - R_{f,t}) + \beta_i^{SMB} \cdot SMB_t + \beta_i^{HML} \cdot HML_t + \beta_i^{RMW} \cdot RMW_t + \beta_i^{CMA} \cdot CMA_t + \varepsilon_{i,t}$$

- La matrice de régression $X$ comprend une constante et les cinq colonnes de facteurs (`Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`).
- Les coefficients sont estimés par OLS : $\hat{\boldsymbol{\beta}}_i = (X^\top X)^{-1} X^\top y_i$.
- Le rendement espéré mensuel est reconstitué à partir des **moyennes empiriques** des cinq facteurs sur la période d'entraînement :

$$\hat{\mu}_i^{\text{mensuel}} = \bar{R}_f + \hat{\beta}_i^{Mkt} \cdot \overline{(R_m - R_f)} + \hat{\beta}_i^{SMB} \cdot \overline{SMB} + \hat{\beta}_i^{HML} \cdot \overline{HML} + \hat{\beta}_i^{RMW} \cdot \overline{RMW} + \hat{\beta}_i^{CMA} \cdot \overline{CMA}$$

- Annualiser : $\hat{\mu}_i = \hat{\mu}_i^{\text{mensuel}} \times 12$.
- La matrice de covariance $\Sigma$ est estimée empiriquement sur les rendements mensuels d'entraînement, puis annualisée ($\times 12$).
- En cas de multicolinéarité entre facteurs (détectable par un VIF élevé ou une matrice $X^\top X$ quasi-singulière), appliquer une régression ridge légère avec $\lambda = 10^{-4}$ comme fallback.

### 3. Optimisation par simulation Monte-Carlo

- Tirer `num_portfolios` (10 000 par défaut) jeux de poids aléatoires normalisés (somme = 1, pas de vente à découvert).
- Pour chaque portefeuille : calculer rendement espéré $\mathbf{w}^\top\hat{\mu}$, volatilité $\sqrt{\mathbf{w}^\top\Sigma\mathbf{w}}$ et ratio de Sharpe $(\mathbf{w}^\top\hat{\mu} - r_f^{\text{annuel}}) / \text{vol}$.
- Retenir le portefeuille maximisant le ratio de Sharpe.

### 4. Backtesting et comparaison au marché

- Calculer les rendements cumulés sur l'ensemble de la période (train + test) avec les poids optimaux.
- Rebaser à 100 au premier mois de la période de test.
- Télécharger SPY sur la même période et appliquer le même rebasage pour comparaison.
- Calculer le **drawdown maximum** exclusivement sur la période de test.

### 5. Frontière efficiente

- Trier les portefeuilles simulés par volatilité croissante.
- Conserver uniquement les points Pareto-optimaux (rendement maximal pour chaque niveau de vol).
- Calculer pour chaque point le rendement total réalisé sur la période de test : $(\prod_t (1 + r_t) - 1) \times 100$.
- Toujours inclure le portefeuille optimal dans la sortie.

### 6. Sortie

La fonction `run()` retourne un dictionnaire avec les mêmes clés que les autres modèles : `weights`, `sharpe`, `expectedReturn`, `volatility`, `maxDrawdown`, `comparisonData`, `trainPeriodStart`, `trainPeriodEnd`, `testPeriodStart`, `testPeriodEnd`, `efficientFrontier`.
