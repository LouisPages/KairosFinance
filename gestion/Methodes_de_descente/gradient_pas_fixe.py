import numpy as np

def opt_sharpe_gradient(mean_returns, cov_matrix, risk_free_rate, max_iter=1000, rho=0.01):
    """
    Descente de gradient à pas fixe pour minimiser f(w) = -Sharpe(w).
    """
    # 1. Sécurisation des entrées en vecteurs plats 1D
    mean_returns = np.array(mean_returns).flatten()
    cov_matrix = np.array(cov_matrix)
    n = len(mean_returns)
    
    # Initialisation uniforme plate
    weights = np.ones(n) / n
    
    for i in range(max_iter):
        # 2. Calculs scalaires stables
        vol = np.sqrt(max(weights.T @ cov_matrix @ weights, 1e-10))
        portfolio_return = np.sum(weights * mean_returns)
        
        # 3. Calcul du gradient exact de -Sharpe
        grad = -1/vol * (mean_returns - (portfolio_return - risk_free_rate) * (cov_matrix @ weights) / vol**2)
        
        # 4. Projection du gradient (somme des variations = 0)
        grad = grad - np.mean(grad)
        
        # 5. Mise à jour (Vraie descente)
        weights_new = weights - rho * grad
        weights_new = np.maximum(weights_new, 0)
        
        # 6. Normalisation de sécurité
        weight_sum = np.sum(weights_new)
        if weight_sum < 1e-10:
            print(f"Arrêt à l'itération {i}: tous les poids sont zéro")
            break
            
        weights_new /= weight_sum
        
        # Optionnel : petit critère d'arrêt hâtif si les poids ne bougent plus
        if np.linalg.norm(weights_new - weights) < 1e-6:
            weights = weights_new
            break
            
        weights = weights_new
        
    return weights