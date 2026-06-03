import os
import sys
import yfinance as yf

# 1. On récupère le chemin absolu du dossier racine PE25
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. On injecte de force la racine et le dossier gestion dans le PATH de Python
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# On s'assure que Python trouve aussi les modules internes à gestion
gestion_dir = os.path.join(root_dir, "gestion")
if gestion_dir not in sys.path:
    sys.path.insert(1, gestion_dir)

# 3. Importation du module 3 facteurs (Fama-French)
# Ajuste le nom exact du fichier s'il est différent (ex: markowitz_3factors)
import gestion.multifactor.markowitz_3factors as markowitz_3factors

def run_local_simulation():
    print("=" * 60)
    print("TEST DU VRAI MODÈLE 3 FACTEURS (FAMA-FRENCH) AVEC GRADIENT OPTIMAL")
    print("=" * 60)
    
    # Paramètres de test réalistes
    tickers = ["AMD", "META", "AMZN", "MSFT", "NVDA"]
    start_date = "2022-01-01"
    end_date = "2025-12-31"
    
    print(f"Lancement de la simulation historique pour : {tickers}")
    print("Calcul des bêtas (Mkt-RF, SMB, HML) et de la matrice de covariance...")

    # Audit rapide des données Yahoo Finance
    print("\nVérification des inputs de ton modèle :")
    raw_data = yf.download(tickers + ["SPY"], start=start_date, end=end_date, progress=False)
    print(f"Derniers prix téléchargés :\n{raw_data['Close'].tail(2)}\n")
    
    # Appel de la fonction run de ton fichier 3 facteurs
    # On force la méthode 'gradient_optimal'
    result = markowitz_3factors.run(
        tickers=tickers, 
        start=start_date, 
        end=end_date, 
        method="gradient_optimal"
    )
    
    # Analyse du résultat retourné par ton code
    if "error" in result:
        print(f"\n[ERREUR] Le modèle a renvoyé : {result['error']}")
    else:
        print("\n[SUCCÈS] Simulation 3 facteurs terminée avec succès ! ")
        print(f"Période d'entraînement : {result['trainPeriodStart']} au {result['trainPeriodEnd']}")
        print(f"Ratio de Sharpe théorique : {result['sharpe']:.4f}")
        print(f"Rendement attendu : {result['expectedReturn']}%")
        print(f"Volatilité du portefeuille : {result['volatility']}%")
        
        print("\nAllocations d'actifs calculées par le Gradient Optimal (3 facteurs) :")
        print("-" * 55)
        for ticker, weight in result["weights"].items():
            print(f"  - {ticker:<6} : {weight * 100:.2f}%")
        print("-" * 55)
    print("=" * 60)

if __name__ == "__main__":
    run_local_simulation()