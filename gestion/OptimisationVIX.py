import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

def fetch_sp500_tickers():
    print("Récupération de la liste du S&P 500 depuis Wikipedia...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    options = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    table = pd.read_html(url, storage_options=options)[0]
    return [t.replace('.', '-') for t in table['Symbol'].tolist()]

def fetch_data_sp500(tickers, start_date, end_date):
    print(f"Téléchargement de l'historique pour {len(tickers)} actions. Patientez...")
    raw_data = yf.download(tickers, start=start_date, end=end_date, interval='1mo', progress=False, auto_adjust=True)
    
    if isinstance(raw_data.columns, pd.MultiIndex):
        prices = raw_data['Close'] if 'Close' in raw_data.columns.get_level_values(0) else raw_data.swaplevel(0, 1, axis=1)['Close']
    else:
        prices = raw_data['Close']
        
    prices = prices.dropna(axis=1)
    returns = prices.pct_change().dropna()
    
    if returns.index.tz is not None:
        returns.index = returns.index.tz_localize(None)
    returns.index = returns.index.to_period('M')

    print("Téléchargement des facteurs Fama-French...")
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    ff5 = pd.read_csv(url, compression='zip', skiprows=3)
    ff5.rename(columns={ff5.columns[0]: 'Date', 'RF': 'RiskFree'}, inplace=True)
    ff5 = ff5.dropna()
    ff5 = ff5[ff5['Date'].astype(str).str.strip().str.len() == 6]
    ff5['Date'] = pd.to_datetime(ff5['Date'].astype(str), format='%Y%m').dt.to_period('M')
    ff5.set_index('Date', inplace=True)
    ff5 = ff5.astype(float) / 100.0
    
    return returns.join(ff5).dropna()

def fetch_vix(start_date, end_date):
    """NOUVEAU : Télécharge l'indice de la peur (VIX) pour analyser la psychologie."""
    print("Téléchargement de l'Indice de la Peur (VIX)...")
    vix = yf.download('^VIX', start=start_date, end=end_date, interval='1mo', progress=False, auto_adjust=True)
    vix_close = vix['Close'] if not isinstance(vix.columns, pd.MultiIndex) else vix.xs('Close', level=1, axis=1) if 'Close' in vix.columns.get_level_values(1) else vix['Close']
    if vix_close.index.tz is not None:
        vix_close.index = vix_close.index.tz_localize(None)
    vix_close.index = vix_close.index.to_period('M')
    # On gère le cas où yfinance renvoie un DataFrame (nouvelle version)
    if isinstance(vix_close, pd.DataFrame):
        vix_close = vix_close.iloc[:, 0]
    return vix_close

def calculate_expected_returns_ff5(data, tickers):
    expected_returns = {}
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    factor_means = data[factor_cols].mean().values
    rf_mean = data['RiskFree'].mean()

    for ticker in tickers:
        y = (data[ticker] - data['RiskFree']).values
        X_raw = data[factor_cols].values
        X = np.column_stack((np.ones(len(X_raw)), X_raw))
        betas = np.linalg.lstsq(X, y, rcond=None)[0][1:] 
        expected_returns[ticker] = (rf_mean + np.dot(betas, factor_means)) * 12 
    return pd.Series(expected_returns)

def optimize_robust_numpy(expected_returns, cov_matrix, max_weight=0.05, threshold=0.01, shrinkage_factor=0.5):
    er = expected_returns.values
    cov = cov_matrix.values
    num_assets = len(er)
    
    diag_cov = np.diag(np.diag(cov))
    cov_shrunk = (1 - shrinkage_factor) * cov + shrinkage_factor * diag_cov
    
    try:
        inv_cov = np.linalg.inv(cov_shrunk)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_shrunk)
        
    raw_weights = np.dot(inv_cov, er)
    weights = np.maximum(0, raw_weights)
    
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
        
    for _ in range(10): 
        excess = 0.0
        mask_over = weights > max_weight
        excess = np.sum(weights[mask_over] - max_weight)
        weights[mask_over] = max_weight
                
        if excess <= 1e-5:
            break
            
        mask_under = weights < max_weight
        sum_under = np.sum(weights[mask_under])
        if sum_under > 0:
            weights[mask_under] += excess * (weights[mask_under] / sum_under)
        else:
            break
            
    weights[weights < threshold] = 0
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
        
    return pd.Series(weights, index=expected_returns.index)

# ==========================================
# POINT D'ENTRÉE PRINCIPAL
# ==========================================
if __name__ == "__main__":
    TICKERS = fetch_sp500_tickers()
    START = '2015-01-01'
    END = '2023-12-31'
    REBALANCE_FREQ = 6 
    WINDOW_SIZE = 60   

    data = fetch_data_sp500(TICKERS, START, END)
    valid_tickers = [t for t in TICKERS if t in data.columns]
    
    # Récupération de la psychologie du marché
    vix_data = fetch_vix(START, END)
    
    print(f"\n[Backtest Dynamique] S&P 500 ({len(valid_tickers)} actions) | Modèle Psycho-Quantitatif")

    portfolio_returns_list = []
    market_returns_list = []
    selections_history = {}

    for i in range(WINDOW_SIZE, len(data), REBALANCE_FREQ):
        train_data = data.iloc[i - WINDOW_SIZE : i]
        end_idx = min(i + REBALANCE_FREQ, len(data))
        test_chunk = data.iloc[i : end_idx]
        current_date = data.index[i]
        
        # --- NOUVEAU : LECTURE DE LA PSYCHOLOGIE ---
        # On regarde le niveau du VIX au moment de prendre la décision
        try:
            current_vix = vix_data.loc[:current_date].iloc[-1]
        except:
            current_vix = 20 # Valeur neutre par défaut en cas de données manquantes
            
        if current_vix > 25:
            regime = "PEUR (Panique)"
            dynamic_max_weight = 0.03 # 3% max : Force une extrême diversification pour se protéger
        elif current_vix < 15:
            regime = "EUPHORIE (Avidité)"
            dynamic_max_weight = 0.08 # 8% max : Permet de concentrer sur les pépites de croissance
        else:
            regime = "NEUTRE"
            dynamic_max_weight = 0.05 # 5% max : Comportement normal
            
        print(f"-> {current_date} | VIX: {current_vix:.1f} | Humeur: {regime} | Plafond: {dynamic_max_weight*100}%")
        
        momentum_6m = train_data[valid_tickers].iloc[-6:].sum()
        top_momentum_tickers = momentum_6m.nlargest(80).index.tolist()
        
        er_ff5 = calculate_expected_returns_ff5(train_data, top_momentum_tickers)
        cov_matrix = train_data[top_momentum_tickers].cov() * 12 
        
        # On injecte notre plafond psychologique (dynamic_max_weight)
        optimal_weights = optimize_robust_numpy(er_ff5, cov_matrix, max_weight=dynamic_max_weight, threshold=0.01)
        
        selected_assets = optimal_weights[optimal_weights > 0].sort_values(ascending=False)
        selections_history[current_date] = selected_assets
        
        chunk_p_ret = test_chunk[top_momentum_tickers].dot(optimal_weights)
        chunk_m_ret = test_chunk['Mkt-RF'] + test_chunk['RiskFree']
        
        portfolio_returns_list.append(chunk_p_ret)
        market_returns_list.append(chunk_m_ret)

    portfolio_returns = pd.concat(portfolio_returns_list)
    market_returns = pd.concat(market_returns_list)

    cum_portfolio = (1 + portfolio_returns).cumprod()
    cum_market = (1 + market_returns).cumprod()
    
    vol_portfolio = portfolio_returns.std() * np.sqrt(12)
    vol_market = market_returns.std() * np.sqrt(12)

    max_dd_port = ((cum_portfolio - cum_portfolio.cummax()) / cum_portfolio.cummax()).min()
    max_dd_mkt = ((cum_market - cum_market.cummax()) / cum_market.cummax()).min()

    plt.figure(figsize=(12, 7))
    dates_plot = cum_portfolio.index.to_timestamp()
    plt.plot(dates_plot, cum_portfolio, label='Portefeuille Psycho-Quantitatif', color='blue', linewidth=2.5)
    plt.plot(dates_plot, cum_market, label='Marché Boursier', color='red', linestyle='--', linewidth=2)
    
    for date in selections_history.keys():
        plt.axvline(x=date.to_timestamp(), color='grey', linestyle=':', alpha=0.4)
        
    plt.title('Backtest Dynamique : Modèle Fama-French avec Ajustement Psychologique (VIX)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Croissance du Capital (Base 1.0)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.savefig('backtest_sp500_psycho.png', bbox_inches='tight', dpi=300)
    
    print("\n" + "="*50)
    print("RÉSULTAT DU BACKTEST (MODÈLE PSYCHO-QUANTITATIF)")
    print("="*50)
    print(f"Performance Nette Portefeuille : {(cum_portfolio.iloc[-1] - 1)*100:+.2f}%")
    print(f"Performance Nette Marché       : {(cum_market.iloc[-1] - 1)*100:+.2f}%")
    print("-" * 50)
    print(f"Volatilité Annuelle Portefeuille : {vol_portfolio:.2%}")
    print(f"Max Drawdown Portefeuille        : {max_dd_port:.2%}")
    print(f"Max Drawdown Marché              : {max_dd_mkt:.2%}")
    print("="*50)