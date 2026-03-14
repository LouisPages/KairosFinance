# Plan technique — Modèle de Fama-French à trois facteurs

## Description

Ajout d'un modèle d'optimisation de portefeuille basé sur le modèle à trois facteurs de Fama et French (1993). Les rendements espérés de chaque actif sont estimés par régression OLS sur les trois facteurs (Mkt-RF, SMB, HML) chargés depuis `get_facteurs.py`, puis l'allocation optimale est déterminée par simulation Monte-Carlo sur la frontière moyenne-variance.

## Fichiers concernés

### À créer
- `gestion/multifactor/markowitz_3factors.py` — module principal exposant la fonction `run()`

### À utiliser (sans modification)
- `gestion/get_facteurs.py` — fonction `load_famafrench_factors(start, end)` qui retourne un `DataFrame` mensuel avec les colonnes `Mkt-RF`, `SMB`, `HML`, `RF`

### Référence structurelle
- `gestion/multifactor/markowitz_1factor.py` — architecture identique (CAPM à un seul facteur de marché), à étendre avec deux facteurs supplémentaires (SMB, HML)

---

## Algorithme pas à pas

### 1. Collecte et prétraitement des données

- Télécharger les prix de clôture ajustés via `yfinance` pour la liste de tickers sur la période `[start, end]`.
- Rééchantillonner les prix au **dernier jour ouvré de chaque mois** (`resample("ME").last()`).
- Calculer les rendements arithmétiques mensuels : $R_{i,t} = P_{i,t}/P_{i,t-1} - 1$.
- Écarter les actifs dont l'historique est entièrement manquant.
- Découper en **train (80 %)** / **test (20 %)** avec un minimum de 24 observations en train.
- Charger les facteurs Fama-French 3 via `load_famafrench_factors(start, end)` depuis `get_facteurs.py`.
- Aligner l'index mensuel (`Period("M")`) des facteurs sur celui des rendements par jointure interne.

### 2. Estimation des rendements espérés par régression OLS à trois facteurs

Pour chaque actif $i$, estimer sur l'ensemble d'entraînement :

$$R_{i,t} - R_{f,t} = \alpha_i + \beta_i^{Mkt}(R_{m,t} - R_{f,t}) + \beta_i^{SMB} \cdot SMB_t + \beta_i^{HML} \cdot HML_t + \varepsilon_{i,t}$$

- La matrice de régression $X$ comprend une constante (pour $\alpha_i$) et les trois colonnes de facteurs (`Mkt-RF`, `SMB`, `HML`).
- Les coefficients sont estimés par OLS : $\hat{\boldsymbol{\beta}}_i = (X^\top X)^{-1} X^\top y_i$.
- Le rendement espéré mensuel est reconstitué à partir des **moyennes empiriques** des facteurs sur la période d'entraînement :

$$\hat{\mu}_i^{\text{mensuel}} = \bar{R}_f + \hat{\beta}_i^{Mkt} \cdot \overline{(R_m - R_f)} + \hat{\beta}_i^{SMB} \cdot \overline{SMB} + \hat{\beta}_i^{HML} \cdot \overline{HML}$$

- Annualiser : $\hat{\mu}_i = \hat{\mu}_i^{\text{mensuel}} \times 12$.
- La matrice de covariance $\Sigma$ est estimée empiriquement sur les rendements mensuels d'entraînement, puis annualisée ($\times 12$).

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
