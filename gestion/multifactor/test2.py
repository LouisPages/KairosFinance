import numpy as np
import pandas as pd
import yfinance as yf
import os
import sys

# Ajoute le dossier parent 'gestion' au chemin de recherche de Python
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Maintenant l'import va fonctionner sans erreur !
from Methodes_de_descente.gradient_pas_optimal import opt_sharpe_gradient_optimal

def test_gradient_with_capm():
    print("=" * 60)
    print("TEST DU MODÈLE À 1 FACTEUR (CAPM) AVEC GRADIENT OPTIMAL")
    print("=" * 60)
    
    # 1. Définition des actifs (Tickers) et de la période
    tickers = ["AAPL", "MSFT", "GOOGL"]
    market_ticker = "SPY"
    start_date = "2023-01-01"
    end_date = "2025-12-31"
    risk_free_rate = 0.03  # Taux sans risque annuel (3%)
    
    print(f"Téléchargement des cours pour {tickers} et {market_ticker}...")
    
    # 2. Téléchargement des prix via yfinance
    all_tickers = tickers + [market_ticker]
    # Correction robuste pour contourner le MultiIndex de yfinance :
    raw_data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
    # Si les données ont un double niveau de colonnes, on extrait proprement "Adj Close"
    data = raw_data["Adj Close"] if "Adj Close" in raw_data.columns else raw_data["Close"]
    if data.empty:
        print("Erreur : Impossible de récupérer les données de Yahoo Finance.")
        return
        
    # 3. Calcul des rendements quotidiens log-normalisés
    returns = np.log(data / data.shift(1)).dropna()
    
    # Séparation des actifs et du marché
    asset_returns = returns[tickers]
    market_returns = returns[market_ticker]
    
    # Rendement du marché en excès (quotidien)
    rf_daily = risk_free_rate / 252
    market_excess = market_returns - rf_daily
    
    # 4. Estimation des Bêtas (Régression linéaire simple Y = alpha + beta * X)
    betas = {}
    print("\nEstimation des bêtas (Modèle à 1 facteur) :")
    
    X = market_excess.values
    X_matrix = np.column_stack((np.ones(len(X)), X))  # Ajout de la constante pour l'intercept
    
    for ticker in tickers:
        y = (asset_returns[ticker] - rf_daily).values
        # Moindres carrés ordinaires (OLS)
        coeffs, _, _, _ = np.linalg.lstsq(X_matrix, y, rcond=None)
        alpha, beta = coeffs[0], coeffs[1]
        betas[ticker] = beta
        print(f"  - Béta de {ticker} : {beta:.4f} (Alpha quotidien : {alpha:.6f})")
        
    # 5. Calcul des rendements attendus annualisés (Formule CAPM)
    # Rendement moyen annuel du marché en excès
    market_excess_annual_mean = market_excess.mean() * 252
    
    mu_capm = []
    for ticker in tickers:
        # Formule du CAPM : E(Ri) = Rf + Beta_i * E(Rm - Rf)
        expected_return = risk_free_rate + betas[ticker] * market_excess_annual_mean
        mu_capm.append(expected_return)
        
    mu_capm = np.array(mu_capm)
    
    # 6. Calcul de la matrice de covariance annualisée (252 jours de bourse)
    cov_matrix_annual = asset_returns.cov().values * 252
    
    print("\nDonnées injectées dans le Gradient Optimal :")
    print(f"  - Rendements attendus annualisés (mu) : {mu_capm}")
    print(f"  - Matrice de covariance (Sigma) :\n{cov_matrix_annual}")
    
    # 7. Exécution de l'algorithme du Gradient Optimal
    print("\nLancement de l'optimisation par descente de gradient...")
    poids_optimaux = opt_sharpe_gradient_optimal(mu_capm, cov_matrix_annual, risk_free_rate)
    
    # 8. Affichage des résultats d'allocation
    print("\nAllocation finale du portefeuille :")
    for i, ticker in enumerate(tickers):
        print(f"  - {ticker} : {poids_optimaux[i].item() * 100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    test_gradient_with_capm()