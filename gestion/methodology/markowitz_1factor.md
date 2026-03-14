## Fichiers concernés

### À créer
- `gestion/multifactor/markowitz_1factor.py` — module principal exposant la fonction `run()` (CAPM à un seul facteur de marché)

### À utiliser (sans modification)
- `gestion/markowitz_simple.py` — référence structurelle : même pipeline Monte-Carlo, remplacer l'estimation de $\mu$ par la régression CAPM

---

## Collecte et prétraitement des données

Les prix de clôture ajustés sont récupérés via `yfinance` pour chaque ticker sur la période demandée. Contrairement à la version simple, les rendements sont calculés à fréquence **mensuelle** plutôt que journalière, afin de s'aligner sur la granularité du facteur de marché. Les prix sont donc rééchantillonnés au dernier jour de chaque mois avant le calcul des rendements arithmétiques. Les actifs dont l'historique est entièrement manquant sont écartés, et la série résultante est divisée en un **ensemble d'entraînement** (les 80 premiers pourcents des mois disponibles, avec un minimum de 24 observations) et un **ensemble de test** (les 20 derniers pourcents).

Le facteur de marché est construit à partir du SPY : son rendement mensuel est calculé comme la variation de prix entre deux fins de mois consécutives. Le taux sans risque mensuel est obtenu via le taux des bons du Trésor à 13 semaines (`^IRX`), converti en taux mensuel par la formule $r_f^{\text{mensuel}} = r_f^{\text{annuel}} / 12$. En cas d'indisponibilité, un taux de 2 % annualisé est utilisé par défaut.

## Estimation des rendements espérés par régression CAPM

C'est ici que ce modèle se distingue fondamentalement de la version simple. Plutôt qu'utiliser la moyenne historique des rendements comme estimateur de $\mu$, le modèle contraint les rendements espérés via le **Modèle d'Évaluation des Actifs Financiers (MEDAF)**.

Pour chaque actif $i$, une régression OLS est estimée sur l'ensemble d'entraînement :

$$R_i - R_f = \alpha_i + \beta_i (R_m - R_f) + \varepsilon_i$$

où $R_i$ est le rendement mensuel de l'actif, $R_f$ le taux sans risque, et $R_m$ le rendement du marché (SPY). Le coefficient $\beta_i$ mesure la sensibilité de l'actif aux mouvements du marché. L'ordonnée à l'origine $\alpha_i$ (l'alpha de Jensen) n'est pas utilisée dans l'estimation du rendement espéré — conformément à la théorie du MEDAF, qui postule que $\alpha_i = 0$ à l'équilibre.

Le rendement espéré annualisé de chaque actif est alors calculé comme :

$$\hat{\mu}_i = \left(\bar{R}_f + \hat{\beta}_i \cdot \overline{(R_m - R_f)}\right) \times 12$$

où les barres désignent les moyennes empiriques sur la période d'entraînement. La matrice de covariance $\Sigma$ reste estimée empiriquement à partir des rendements mensuels historiques, puis annualisée en multipliant par 12.

## Optimisation par simulation de Monte-Carlo

La procédure d'optimisation est identique à celle de la version simple : `num_portfolios` (10 000 par défaut) jeux de poids aléatoires sont tirés, normalisés pour sommer à un, et évalués selon leur ratio de Sharpe $(\mathbf{w}^\top \hat{\mu} - r_f) / \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}$. Le taux sans risque utilisé pour le calcul du Sharpe est la moyenne du taux mensuel sur la période d'entraînement, annualisée. Le portefeuille maximisant ce ratio est retenu comme allocation optimale.

L'apport du MEDAF se situe donc exclusivement dans la construction de $\hat{\mu}$ : en ancrant les rendements espérés sur le bêta de marché plutôt que sur les moyennes historiques brutes, le modèle produit des estimations plus stables et économiquement plus cohérentes, moins sujettes au bruit d'estimation qui affecte les moyennes empiriques sur des fenêtres courtes.

## Backtesting et comparaison au marché

Une fois les poids optimaux fixés, la performance est évaluée sur l'ensemble de la période disponible à fréquence mensuelle. Les rendements cumulés sont rebasés à 100 au premier mois de la période de test, et le même rebasage est appliqué à une série de référence SPY téléchargée à fréquence mensuelle sur la même plage de dates. Cela produit une série temporelle comparable entre le portefeuille et le marché, restituée dans le champ `comparisonData` de la sortie.

Le drawdown maximum est calculé exclusivement sur la période de test, comme la baisse maximale de la valeur cumulée depuis son dernier sommet :

$$\max_{t} \left(1 - \frac{C_t}{\max_{s \leq t} C_s}\right)$$

Cette métrique capture la perte maximale qu'un investisseur aurait subie pendant la fenêtre d'évaluation hors échantillon.

## Frontière efficiente

La frontière efficiente est construite de la même manière que dans la version simple : les 10 000 portefeuilles simulés sont triés par volatilité croissante, et seuls les points Pareto-optimaux sont conservés. Pour chaque point de la frontière, le rendement de backtest réalisé est calculé sur la période de test comme le rendement total géométrique $\left(\prod_t (1 + r_t) - 1\right) \times 100$. Le portefeuille optimal est toujours inclus dans la sortie pour garantir la cohérence entre les métriques rapportées et la visualisation de la frontière.
