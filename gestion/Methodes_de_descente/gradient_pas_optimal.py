import numpy as np


def opt_sharpe_gradient_optimal(mean_returns, cov_matrix, risk_free_rate, max_iter=1000, tol=1e-6):
    """
    Descente de gradient avec pas optimal.
    Le pas rho est trouvé automatiquement à chaque itération.
    """
    n = len(mean_returns)
    weights = np.ones((n, 1)) / n
    mu = np.array(mean_returns).reshape(n, 1)
    
    for i in range(max_iter):
        vol = np.sqrt((weights.T @ cov_matrix @ weights).item())
        portfolio_return = (weights.T @ mu).item()
        sharpe = (portfolio_return - risk_free_rate) / vol
        
        grad = 1/vol * (mu - (portfolio_return - risk_free_rate) * np.dot(cov_matrix, weights) / vol**2)
        
        rho = _line_search(weights, grad, mean_returns, cov_matrix, risk_free_rate)
        
        weights_new = weights + rho * grad
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


def _line_search(weights, grad, mean_returns, cov_matrix, risk_free_rate, c=0.1, rho=1.0,alpha=0.5, max_iter=20):
    """
    Line search par backtracking (méthode d'Armijo).
    Trouve le meilleur pas rho qui maximise le Sharpe.
    """
    
    def sharpe_ratio(w):
        """Calcule le ratio de Sharpe pour des poids w"""
        n = len(mean_returns)
        vol = np.sqrt((w.T @ cov_matrix @ w).item())
        mu = np.array(mean_returns).reshape(n, 1)
        mu_p = (w.T @ mu).item()
        if vol < 1e-10:
            return -np.inf
        return (mu_p - risk_free_rate) / vol
    
    # Sharpe initial
    sharpe_current = sharpe_ratio(weights)
    
    # Backtracking : réduire rho jusqu'à amélioration significative
    for _ in range(max_iter):
        weights_candidate = weights + rho * grad
        weights_candidate = np.maximum(weights_candidate, 0)
        
        weight_sum = np.sum(weights_candidate)
        if weight_sum > 1e-10:
            weights_candidate /= weight_sum
        else:
            continue
        
        sharpe_candidate = sharpe_ratio(weights_candidate)
        
        # Condition d'Armijo : amélioration suffisante
        if sharpe_candidate >= sharpe_current + c * rho * (grad.T @ grad).item():
            return rho
        
        rho *= alpha  # Réduire le pas
    
    return rho 