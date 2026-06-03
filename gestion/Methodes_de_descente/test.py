# Test avec des rendements très différents
from gradient_pas_optimal import opt_sharpe_gradient_optimal

rendements_test = [0.25, 0.05, 0.12]  # L'actif 0 est excellent, l'actif 1 est mauvais
matrice_cov_test = [
    [0.04, 0.002, 0.001],
    [0.002, 0.04, 0.002],
    [0.001, 0.002, 0.04]
]
taux_sans_risque = 0.02

poids_optimaux = opt_sharpe_gradient_optimal(rendements_test, matrice_cov_test, taux_sans_risque)
print(poids_optimaux)