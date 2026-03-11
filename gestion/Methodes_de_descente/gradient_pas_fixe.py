import numpy as np

def opt_sharpe_gradient(mean_returns, cov_matrix, risk_free_rate, max_iter=1000, rho=0.01):
    n = len(mean_returns)
    weights = np.ones((n, 1)) / n
    mu = np.array(mean_returns).reshape(n, 1)
    
    for i in range(max_iter):
        vol = np.sqrt((weights.T @ cov_matrix @ weights).item())
        grad = 1/vol * (mu - ((weights.T @ mu).item() - risk_free_rate) * (cov_matrix @ weights) / vol**2)
        
        weights += rho * grad
        
        weights = np.maximum(weights, 0)
        weight_sum = np.sum(weights)
        
        if weight_sum < 1e-10:
            print(f"Arrêt à l'itération {i}: tous les poids sont zéro")
            break
            
        weights /= weight_sum
        
    return weights

