## Collecte et prétraitement des données

Les prix de clôture ajustés sont récupérés via `yfinance` pour chaque ticker sur la période demandée. Les actifs dont l'historique de prix est entièrement manquant sont écartés avant tout calcul. Les séries de prix restantes sont ensuite divisées en un **ensemble d'entraînement** (les 80 premiers pourcents des jours de bourse) et un **ensemble de test** (les 20 derniers pourcents), ce qui permet de calibrer le modèle sur une première période et de l'évaluer hors échantillon sur une seconde.

## Estimation des paramètres

Sur l'ensemble d'entraînement, les rendements logarithmiques journaliers sont calculés comme $\ln(P_t / P_{t-1})$. Le vecteur des rendements espérés $\mu$ et la matrice de covariance $\Sigma$ sont estimés empiriquement à partir de cette série de rendements, puis annualisés en les multipliant par 252 (le nombre conventionnel de jours de bourse par an). Ces deux grandeurs constituent les seules entrées de l'optimisation — il n'y a ni rétrécissement, ni régularisation, ni décomposition factorielle.

## Optimisation par simulation de Monte-Carlo

Plutôt que de résoudre le programme quadratique de manière analytique, l'algorithme adopte une approche par Monte-Carlo : il tire `num_portfolios` (10 000 par défaut) jeux de poids aléatoires, chacun normalisé pour sommer à un, de sorte que le portefeuille est toujours entièrement investi et sans vente à découvert. Pour chaque portefeuille simulé, il calcule le rendement espéré annualisé $\mathbf{w}^\top \mu$, la volatilité annualisée $\sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}$ et le ratio de Sharpe $(\text{rendement} - r_f) / \text{volatilité}$, où $r_f$ est le taux sans risque paramétrable (3 % par défaut). Le portefeuille présentant le ratio de Sharpe le plus élevé parmi toutes les simulations est retenu comme allocation optimale.

Cette approche par simulation est moins précise qu'un solveur analytique, mais elle reste interprétable et robuste en pratique pour un nombre modéré d'actifs, à condition que le budget de simulation soit suffisamment grand pour couvrir l'espace des poids de manière satisfaisante.

## Backtesting et comparaison au marché

Une fois les poids optimaux fixés, la performance est évaluée sur l'ensemble de la période disponible. Les rendements cumulés sont rebasés à 100 au premier jour de la période de test, et le même rebasage est appliqué à une série de référence SPY téléchargée sur la même plage de dates. Cela produit une série temporelle comparable entre le portefeuille et le marché, restituée dans le champ `comparisonData` de la sortie.

Le drawdown maximum est calculé exclusivement sur la période de test, comme la baisse maximale de la valeur cumulée depuis son dernier sommet :

$$\max_{t} \left(1 - \frac{C_t}{\max_{s \leq t} C_s}\right)$$

Cette métrique capture la perte maximale qu'un investisseur aurait subie pendant la fenêtre d'évaluation hors échantillon.

## Frontière efficiente

Pour visualiser l'arbitrage entre risque et rendement, l'algorithme construit une frontière efficiente approchée à partir du même échantillon Monte-Carlo. Les portefeuilles simulés sont triés par volatilité croissante, et seuls les points Pareto-optimaux sont conservés — c'est-à-dire que, pour chaque niveau de volatilité, seul le portefeuille offrant le rendement espéré le plus élevé est retenu. Pour chaque point de la frontière, un rendement de backtest réalisé est également calculé sur la période de test en convertissant la somme des log-rendements en rendement total.