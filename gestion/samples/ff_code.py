import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import statsmodels.api as sm
import matplotlib.pyplot as plt


tickers = [
    "TTE.PA", "MC.PA", "BNP.PA", "SAN.PA", "AIR.PA",
    "OR.PA", "SU.PA", "CS.PA", "STMPA.PA", "AI.PA"
]

print("1. Téléchargement des Actions...")
stock_data = yf.download(tickers, start="2018-01-01", end="2023-12-31", auto_adjust=False)
prices = stock_data["Adj Close"].dropna()

monthly_prices = prices.resample('ME').last()
stock_returns = monthly_prices.pct_change().dropna()
stock_returns.index = stock_returns.index.to_period('M')

print("2. Téléchargement des Facteurs Fama-French (Europe)...")

try:
    ff_data = web.DataReader("Europe_3_Factors", "famafrench", start="2018-01-01", end="2023-12-31")[0]
except:
    print("Erreur")
    exit()

# ff_data = ff_data / 100

ff_data.columns = ['Mkt-RF', 'SMB', 'HML', 'RF']


merged_data = pd.merge(stock_returns, ff_data, left_index=True, right_index=True, how='inner')


stocks_aligned = merged_data[tickers]
factors = merged_data[['Mkt-RF', 'SMB', 'HML']]
rf = merged_data['RF']


expected_returns_ff = {} # Dictionnaire pour stocker nos E(R) théoriques


avg_mkt_premium = factors['Mkt-RF'].mean()
avg_smb_premium = factors['SMB'].mean()
avg_hml_premium = factors['HML'].mean()
current_rf = rf.mean() # Taux sans risque moyen

print("\n3. Exécution des Régressions OLS...")
X = sm.add_constant(factors) # On ajoute la constante (Alpha)

for ticker in tickers:

    Y = stocks_aligned[ticker] - rf


    model = sm.OLS(Y, X).fit()

    b = model.params['Mkt-RF']
    s = model.params['SMB']
    h = model.params['HML']

    # E(R) = Rf + b*Prime_Mkt + s*Prime_SMB + h*Prime_HML
    expected_ret = current_rf + (b * avg_mkt_premium) + (s * avg_smb_premium) + (h * avg_hml_premium)

    expected_returns_ff[ticker] = expected_ret * 12


mu_ff = pd.Series(expected_returns_ff)

print("\n=== RENDEMENTS ESPÉRÉS (Modèle Fama-French) ===")
print(mu_ff.sort_values(ascending=False))


cov_matrix = stocks_aligned.cov() * 12

num_portfolios = 10000
#results = np.zeros((3, num_portfolios))

ret_arr = np.zeros(num_portfolios)
vol_arr = np.zeros(num_portfolios)
sharpe_arr = np.zeros(num_portfolios)

print(f"\n4. Simulation de {num_portfolios} portefeuilles...")

for i in range(num_portfolios):

    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)

    ret_arr[i] = np.sum(weights * mu_ff)

    # Sigma_p = sqrt( w.T * Cov * w )
    vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Ratio de Sharpe
    sharpe_arr[i] = (ret_arr[i] - current_rf*12) / vol_arr[i]

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

plt.show()
