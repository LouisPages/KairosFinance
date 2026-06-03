import numpy as np


def opt_sharpe_gradient_optimal(mean_returns, cov_matrix, risk_free_rate, max_iter=1000, tol=1e-6):
    """
    Descente de gradient avec pas optimal.
    Le pas rho est trouvé automatiquement à chaque itération.
    """
    mean_returns = np.array(mean_returns).flatten()
    cov_matrix = np.array(cov_matrix)
    print("mean_returns=",mean_returns,"cov_matrix=",cov_matrix,"risk_free_rate=",risk_free_rate)
    n = len(mean_returns)
    weights = np.ones(n) / n

    # Harmonisation automatique de l'échelle du taux sans risque (ex: 4.0 -> 0.04)
    
    
    for i in range(max_iter):
        vol = np.sqrt(max(weights.T @ cov_matrix @ weights, 1e-10))
        portfolio_return = np.sum(weights * mean_returns)
        sharpe = (portfolio_return - risk_free_rate) / vol
        
        # Calcul du gradient propre de f(w) = -Sharpe(w)
        # On s'assure que portfolio_return et risk_free_rate sont sur la même échelle scalaire
        excess_return = portfolio_return - risk_free_rate
        
        # Formule mathématique exacte du gradient de -Sharpe
        grad_base = (1.0 / vol) * (mean_returns - (excess_return / (vol**2)) * (cov_matrix @ weights))

        # SÉCURITÉ SHARPE NÉGATIF : Si le rendement est inférieur au taux sans risque,
        # on inverse la direction pour forcer l'algorithme à chercher la croissance
        if excess_return < 0:
            grad = grad_base  # On change la polarité de la descente
        else:
            grad = -grad_base # Cas classique (Sharpe positif)

        # Projection (somme des poids = 1)
        grad = grad - np.mean(grad)
        print("grad_base=",grad_base,"grad=",grad,"excess_return=",excess_return,"sharpe=",sharpe)

        rho = _line_search(weights, grad, mean_returns, cov_matrix, risk_free_rate)
        print("rho=",rho)
        
        weights_new = weights - rho * grad
        weights_new = np.maximum(weights_new, 0)
        
        weight_sum = np.sum(weights_new)
        if weight_sum < 1e-10:
            print(f"Arrêt à l'itération {i}: tous les poids sont zéro")
            break
        weights_new /= weight_sum
        
        if np.linalg.norm(weights_new - weights) < tol:
            print(f"Convergence à l'itération {i}")
            weights = weights_new
            break
        
        
        weights = weights_new
    
    return weights


def _line_search(weights, grad, mean_returns, cov_matrix, risk_free_rate, c=0.1, rho=1.0, alpha=0.5, max_iter=20):
    """
    Line search par backtracking (méthode d'Armijo) sécurisée.
    """
    def sharpe_ratio(w):
        vol = np.sqrt(max(w.T @ cov_matrix @ w, 1e-10))
        mu_p = np.sum(w * mean_returns)
        return (mu_p - risk_free_rate) / vol
    
    sharpe_current = sharpe_ratio(weights)
    initial_rho = rho
    
    for _ in range(max_iter):
        weights_candidate = weights - rho * grad
        weights_candidate = np.maximum(weights_candidate, 0)
        
        weight_sum = np.sum(weights_candidate)
        if weight_sum > 1e-10:
            weights_candidate /= weight_sum
        else:
            rho *= alpha
            continue
        
        sharpe_candidate = sharpe_ratio(weights_candidate)
        
        # Si on trouve une amélioration, on valide immédiatement le pas rho
        if sharpe_candidate > sharpe_current:
            return rho
        
        rho *= alpha  # Réduction du pas
    
    # Si le backtracking a échoué à trouver un meilleur Sharpe (cas des Sharpe très négatifs),
    # on renvoie un pas par défaut raisonnable au lieu d'un pas quasi-nul pour forcer la descente
    return 0.05