
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt



# ==============================================================================
tickers = [
    # --- CAC 40 (Les Géants) ---
    "AC.PA", "ACA.PA", "AI.PA", "AIR.PA", "ALO.PA","ATO.PA","BN.PA","BNP.PA","CA.PA","CAP.PA",
    "CS.PA", "DG.PA", "DSY.PA", "EL.PA", "EN.PA","ENGI.PA","ERF.PA","GLE.PA","HO.PA","KER.PA",
    "LR.PA", "MC.PA", "ML.PA", "OR.PA", "ORA.PA", "PUB.PA", "RI.PA", "RNO.PA", "SAF.PA", "SAN.PA",
    "SGO.PA", "STMPA.PA", "SU.PA", "SW.PA", "TEP.PA", "TTE.PA", "URW.PA", "VIE.PA", "VIV.PA",

    # --- Next 20 & SBF 120 (Les Grandes Entreprises) ---
    "ADP.PA", "AF.PA", "AKE.PA", "AMUN.PA", "ATE.PA", "BEN.PA", "BIM.PA", "BOL.PA", "BVI.PA",
    "CDI.PA", "CO.PA", "COFA.PA", "COV.PA", "DBV.PA", "DEC.PA", "DSY.PA", "EDEN.PA",
    "EKI.PA", "ELIOR.PA", "ELIS.PA", "ERF.PA", "ES.PA", "ETL.PA", "FGR.PA", "FII.PA", "FNAC.PA",
    "FR.PA", "GFC.PA", "GTT.PA", "ICAD.PA", "IMDA.PA", "IPN.PA", "IPS.PA", "KOF.PA",
    "LI.PA", "LTA.PA", "MDM.PA", "MF.PA", "MMB.PA", "MERY.PA", "MMT.PA", "MRN.PA", "NEX.PA",
    "NK.PA", "NOKIA.PA", "OSE.PA", "OVH.PA", "RCO.PA", "RF.PA", "RUI.PA", "SCR.PA", "SK.PA", "SOI.PA", "SOP.PA", "SPIE.PA", "TRI.PA", "UBI.PA", "VIRP.PA"
]

tickers = list(set(tickers))

print(f"Téléchargement des données pour {tickers}...")

# On télécharge 5 ans de données
data = yf.download(tickers, start="2018-01-01", end="2023-12-31", auto_adjust=False)


prices = data["Adj Close"].dropna(axis=1, how='all')

returns = np.log(prices / prices.shift(1)).dropna()

valid_tickers = returns.columns
n_assets = len(valid_tickers)
print(f"\nDonnées valides pour {n_assets} actions (sur {len(tickers)} demandées).")

# On annualise les rendements (x 252 jours de bourse)
mean_returns = returns.mean() * 252
# On annualise la matrice de covariance
cov_matrix = returns.cov() * 252

print("\nMatrice de Covariance calculée (Sigma) :")
print(cov_matrix)


# ==============================================================================
num_portfolios = 100000
all_weights = np.zeros((num_portfolios, n_assets))
ret_arr = np.zeros(num_portfolios)
vol_arr = np.zeros(num_portfolios)
sharpe_arr = np.zeros(num_portfolios)
risk_free_rate = 0.03 # Hypothèse 3% sans risque

print(f"\nLancement de la simulation sur {num_portfolios} portefeuilles...")

for i in range(num_portfolios):
    # Génération de poids aléatoires
    weights = np.random.random(n_assets)
    weights = weights / np.sum(weights)
    all_weights[i,:] = weights

    ret_arr[i] = np.sum(mean_returns * weights)

    # Sigma_p = sqrt( w.T * Cov * w )
    vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Ratio de Sharpe
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

sorted_indices = np.argsort(best_weights)[::-1] # Tri décroissant
for i in sorted_indices:
    #if best_weights[i] > 0.01: # Seuil de 1%
        print(f"  {valid_tickers[i]} : {best_weights[i]*100:.2f} %")
# ==============================================================================
# TRACÉ DE LA FRONTIÈRE (Graphique)

plt.figure(figsize=(10, 6))

# Le nuage de points (Tous les portefeuilles testés)
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', s=10, alpha=0.5)
plt.colorbar(label='Ratio de Sharpe (Qualité)')

# Le point Optimal (Étoile Rouge)
plt.scatter(max_sharpe_vol, max_sharpe_ret, c='red', s=300, marker='*', label='Optimal (Max Sharpe)')

plt.xlabel('Risque (Volatilité Annualisée)')
plt.ylabel('Rendement Espéré (Annualisé)')
plt.title(f'Frontière Efficiente de Markowitz ({num_portfolios} simulations)')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.savefig("frontiere_efficiente.png")
print("\nGraphique généré : frontiere_efficiente.png")
plt.show()