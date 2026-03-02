import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt



# ==============================================================================
tickers = ["TTE.PA",   # TotalEnergies (Énergie / Value)
    "MC.PA",    # LVMH (Luxe / Growth)
    "BNP.PA",   # BNP Paribas (Banque / Finance)
    "SAN.PA",   # Sanofi (Santé / Défensif)
    "AIR.PA",   # Airbus (Industrie / Aéronautique)
    "OR.PA",    # L'Oréal (Cosmétique / Qualité)
    "DG.PA",    # Vinci (Construction / Infrastructures)
    "SU.PA",    # Schneider Electric (Technologie industrielle)
    #"CS.PA",    # AXA (Assurance)
    #"STMPA.PA"  # STMicroelectronics (Semi-conducteurs / Tech volatile)
]
print(f"Téléchargement des données pour {tickers}...")

# On télécharge 5 ans de données
data = yf.download(tickers, start="2018-01-01", end="2023-12-31", auto_adjust=False)

if data.empty:
    print("ERREUR CRITIQUE : Yahoo bloque le téléchargement.")
    exit() # On arrête le programme proprement
else:
    print("Données reçues !")


prices = data["Adj Close"]

# Calcul des rendements journaliers (Log-returns est souvent préféré en optimisation)
returns = np.log(prices / prices.shift(1)).dropna()

# On annualise les rendements (x 252 jours de bourse)
mean_returns = returns.mean() * 252
# On annualise la matrice de covariance
cov_matrix = returns.cov() * 252

print("\nMatrice de Covariance calculée (Sigma) :")
print(cov_matrix)


# ==============================================================================
num_portfolios = 10000
all_weights = np.zeros((num_portfolios, len(tickers)))
ret_arr = np.zeros(num_portfolios)
vol_arr = np.zeros(num_portfolios)
sharpe_arr = np.zeros(num_portfolios)
risk_free_rate = 0.03 # Hypothèse 3% sans risque

print(f"\nLancement de la simulation sur {num_portfolios} portefeuilles...")

for i in range(num_portfolios):
    # Génération de poids aléatoires
    weights = np.random.random(len(tickers))
    weights = weights / np.sum(weights) # On normalise pour que la somme = 1 (100%)
    all_weights[i,:] = weights # On garde en mémoire pour retrouver le meilleur

    # Calcul du Rendement du Portefeuille (Moyenne pondérée)
    ret_arr[i] = np.sum(mean_returns * weights)

    # Calcul du Risque (La fameuse formule Matricielle !)
    # Sigma_p = sqrt( w.T * Cov * w )
    vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Ratio de Sharpe (Rendement / Risque)
    sharpe_arr[i] = (ret_arr[i] - risk_free_rate) / vol_arr[i]


# ==============================================================================
# On cherche le portefeuille avec le meilleur Ratio de Sharpe (Le point optimal)
max_sharpe_idx = sharpe_arr.argmax()
max_sharpe_vol = vol_arr[max_sharpe_idx]
max_sharpe_ret = ret_arr[max_sharpe_idx]
best_weights = all_weights[max_sharpe_idx,:]

print("\n" + "="*40)
print("   PORTEFEUILLE OPTIMAL (Le point Étoile)")
print("="*40)
print(f"Rendement Espéré : {max_sharpe_ret*100:.2f}%")
print(f"Volatilité       : {max_sharpe_vol*100:.2f}%")
print("-" * 20)
print("Allocation idéale :")
for i, ticker in enumerate(tickers):
    print(f"  {ticker} : {best_weights[i]*100:.2f} %")

# ==============================================================================
# TRACÉ DE LA FRONTIÈRE (Graphique)

plt.figure(figsize=(10, 6))

# Le nuage de points (Tous les portefeuilles testés)
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', s=10, alpha=0.5)
plt.colorbar(label='Ratio de Sharpe (Qualité)')

# Le point Optimal (Étoile Rouge)
plt.scatter(max_sharpe_vol, max_sharpe_ret, c='red', s=300, marker='*', label='Optimal (Max Sharpe)')

# Ton portefeuille actuel (si tu veux comparer, supposons équiréparti)
# Tu peux ajouter ton point ici pour voir où tu te situes
plt.xlabel('Risque (Volatilité Annualisée)')
plt.ylabel('Rendement Espéré (Annualisé)')
plt.title(f'Frontière Efficiente de Markowitz ({num_portfolios} simulations)')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.savefig("frontiere_efficiente.png")
print("\nGraphique généré : frontiere_efficiente.png")
plt.show()