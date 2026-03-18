"""
============================================================
  MODELE FAMA-FRENCH CRYPTO  (CMKT, SIZE, MOM)
  Lecture directe de fichiers CSV locaux (CoinGecko format)

  FORMAT CSV ATTENDU :
    snapped_at, price, market_cap, total_volume

  USAGE :
    1. Placez vos CSV dans le même dossier que ce script
    2. Adaptez CSV_FILES, START, END si besoin
    3. python crypto_fama_french.py

  SORTIES (8 fichiers PNG séparés) :
    ff_1_betas.png          — Betas factoriels
    ff_2_alpha_r2.png       — Alpha de Jensen & R²
    ff_3_frontiere.png      — Frontière efficiente
    ff_4_portefeuilles.png  — Poids optimaux
    ff_5_facteurs.png       — Évolution des facteurs
    ff_6_correlations.png   — Heatmap des corrélations
    ff_7_endogeneite.png    — Test d'endogénéité
    ff_8_portfolio_vs.png   — Équiréparti vs Sharpe Optimal  ← NOUVEAU
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.optimize import minimize

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[WARN] statsmodels non installé — OLS numpy utilisé.")


# ============================================================
#  CONFIGURATION
# ============================================================

# Dossier des CSV  (par défaut : même dossier que ce script)
CSV_FOLDER = "/Users/syumalouette/Documents/PE_25/crypto/données"

# Correspondance nom affiché -> nom de fichier CSV
CSV_FILES = {
    "BTC":   "btc-usd-max.csv",        # 1  - Bitcoin
    "ETH":   "eth-usd-max.csv",        # 2  - Ethereum
    "USDT":  "usdt-usd-max.csv",       # 3  - Tether USDt
    "BNB":   "bnb-usd-max.csv",        # 4  - BNB
    "XRP":   "xrp-usd-max.csv",        # 5  - XRP
    "USDC":  "usdc-usd-max.csv",       # 6  - USDC
    "SOL":   "sol-usd-max.csv",        # 7  - Solana
    "TRX":   "trx-usd-max.csv",        # 8  - TRON
    "DOGE":  "doge-usd-max.csv",       # 9  - Dogecoin
    "ADA":   "ada-usd-max.csv",        # 10 - Cardano
    "BCH":   "bch-usd-max.csv",        # 11 - Bitcoin Cash
    "HYPE":  "hype-usd-max.csv",       # 12 - Hyperliquid
    "LEO":   "leo-usd-max.csv",        # 13 - UNUS SED LEO
    "XMR":   "xmr-usd-max.csv",        # 14 - Monero
    "LINK":  "link-usd-max.csv",       # 15 - Chainlink
    "USDe":  "usde-usd-max.csv",       # 16 - Ethena USDe
    "CC":    "cc-usd-max.csv",         # 17 - Canton
    "DAI":   "dai-usd-max.csv",        # 18 - Dai
    "XLM":   "xlm-usd-max.csv",        # 19 - Stellar
    "USD1":  "usd1-usd-max.csv",       # 20 - World Liberty Financial USD
    "LTC":   "ltc-usd-max.csv",        # 21 - Litecoin
    "HBAR":  "hbar-usd-max.csv",       # 22 - Hedera
    "AVAX":  "avax-usd-max.csv",       # 23 - Avalanche
    "PYUSD": "pyusd-usd-max.csv",      # 24 - PayPal USD
    "SUI":   "sui-usd-max.csv",        # 25 - Sui
    "ZEC":   "zec-usd-max.csv",        # 26 - Zcash
    "SHIB":  "shib-usd-max.csv",       # 27 - Shiba Inu
    "TON":   "ton-usd-max.csv",        # 28 - Toncoin
    "CRO":   "cro-usd-max.csv",        # 29 - Cronos
    "XAUt":  "xaut-usd-max.csv",       # 30 - Tether Gold
    "WLFI":  "wlfi-usd-max.csv",       # 31 - World Liberty Financial
    "PAXG":  "paxg-usd-max.csv",       # 32 - PAX Gold
    "DOT":   "dot-usd-max.csv",        # 33 - Polkadot
    "UNI":   "uni-usd-max.csv",        # 34 - Uniswap
    "MNT":   "mnt-usd-max.csv",        # 35 - Mantle
    "PI":    "pi-usd-max.csv",         # 36 - Pi
    "TAO":   "tao-usd-max.csv",        # 37 - Bittensor
    "OKB":   "okb-usd-max.csv",        # 38 - OKB
    "M":     "m-usd-max.csv",          # 39 - MemeCore
    "SKY":   "sky-usd-max.csv",        # 40 - Sky
    "ASTER": "aster-usd-max.csv",      # 41 - Aster
    "AAVE":  "aave-usd-max.csv",       # 42 - Aave
    "USDG":  "usdg-usd-max.csv",       # 43 - Global Dollar
    "RAIN":  "rain-usd-max.csv",       # 44 - Rain
    "CIRCLEUSYC": "usyc-usd-max.csv",  # 45 - Circle USYC
    "BUIDL":  "BUIDL-usd-max.csv",     # 47 - Buidl
    "WBT":    "wbt-usd-max.csv",       # 48 - WhiteBIT Coin
    "USDS":  "usds-usd-max.csv",       # 49 - USDS
    "FIGR":  "figr_heloc-usd-max.csv", # 50 - Figure HELOC
}

# Stablecoins à exclure de l'optimisation
STABLECOINS = ["USDT", "USDC", "USDS", "USDe", "DAI", "PYUSD", "PAXG",
               "USDG", "CIRCLEUSYC"]

# Période d'analyse
START = "2015-01-01"
END   = "2025-01-01"

# Fréquence : "W"=hebdo, "ME"=mensuel
FREQ = "W"

# Paramètres du modèle
RF_ANNUAL  = 0.04   # taux sans risque annuel
MOM_WINDOW = 12     # fenêtre momentum (périodes)
MIN_R2     = 0   # R² minimum pour sélection portefeuille
N_SIMUL    = 3000   # simulations Monte Carlo (Fama-French)
N_SIMUL_VS = 50000 # simulations Monte Carlo (comparaison portefeuilles)

# Taille de chaque graphique individuel
FIG_W, FIG_H = 14, 8

# Nombre de grandes caps détectées automatiquement (via market cap)
N_LARGE = 20


# ============================================================
#  STYLE GLOBAL
# ============================================================
def set_style():
    matplotlib.rcParams.update({
        "figure.dpi":       120,
        "savefig.dpi":      180,
        "font.size":        13,
        "axes.titlesize":   16,
        "axes.labelsize":   13,
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
        "legend.fontsize":  11,
        "axes.titlepad":    14,
        "figure.facecolor": "white",
        "axes.facecolor":   "#f7f7f7",
    })
    sns.set_theme(style="darkgrid", palette="muted")


def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


# ============================================================
#  1. CHARGEMENT DES DONNÉES
# ============================================================

def load_prices():
    """Retourne (returns, market_caps) à la fréquence FREQ."""
    print("[1/5] Chargement des CSV...")
    prices_d = {}
    mcaps_d  = {}

    for name, filename in CSV_FILES.items():
        path = os.path.join(CSV_FOLDER, filename)
        if not os.path.exists(path):
            print(f"  [SKIP] {name}: fichier introuvable -> {path}")
            continue
        try:
            df = pd.read_csv(path, parse_dates=["snapped_at"])
            df = df.rename(columns={"snapped_at": "Date", "price": "Close",
                                    "market_cap": "MCap"})
            df = df.set_index("Date")
            df.index = df.index.tz_localize(None)
            df = df[(df.index >= START) & (df.index <= END)]
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.dropna(subset=["Close"])
            if len(df) < 10:
                print(f"  [SKIP] {name}: trop peu de données ({len(df)} lignes)")
                continue
            prices_d[name] = df["Close"]
            if "MCap" in df.columns:
                mcaps_d[name] = pd.to_numeric(df["MCap"], errors="coerce")
            print(f"  [OK]   {name}: {len(df)} jours")
        except Exception as e:
            print(f"  [ERREUR] {name}: {e}")

    if not prices_d:
        raise RuntimeError("Aucun fichier CSV chargé. Vérifiez CSV_FOLDER et CSV_FILES.")

    prices = pd.DataFrame(prices_d).resample(FREQ).last().ffill().bfill()
    mcaps  = (pd.DataFrame(mcaps_d).resample(FREQ).last().ffill().bfill()
              if mcaps_d else pd.DataFrame())

    returns = prices.pct_change().iloc[1:].clip(-0.99, 5.0).fillna(0.0)
    print(f"\n  -> {returns.shape[0]} périodes x {returns.shape[1]} actifs")
    return returns, mcaps


# ============================================================
#  2. CONSTRUCTION DES FACTEURS  (SIZE_LARGE automatique)
# ============================================================

def build_factors(returns, mcaps):
    print("\n[2/5] Construction des facteurs CMKT, SIZE, MOM...")

    if not mcaps.empty:
        avg_mcap   = mcaps.reindex(columns=returns.columns).mean()
        size_large = avg_mcap.nlargest(N_LARGE).index.tolist()
    else:
        size_large = list(returns.columns)[:N_LARGE]

    size_small = [c for c in returns.columns if c not in size_large] or list(returns.columns)

    print(f"  Grandes caps (auto, N={N_LARGE}) : {size_large}")
    print(f"  Petites caps                     : {size_small}")

    CMKT = returns.mean(axis=1);                              CMKT.name = "CMKT"
    SIZE = (returns[size_small].mean(axis=1)
            - returns[size_large].mean(axis=1));              SIZE.name = "SIZE"

    mom_vals = {}
    for t in range(MOM_WINDOW, len(returns)):
        past    = returns.iloc[t - MOM_WINDOW:t].mean()
        winners = past[past >= past.quantile(0.70)].index.tolist()
        losers  = past[past <= past.quantile(0.30)].index.tolist()
        cur     = returns.iloc[t]
        val     = float(cur[winners].mean() - cur[losers].mean()) if (winners and losers) else 0.0
        mom_vals[returns.index[t]] = val
    MOM = pd.Series(mom_vals, name="MOM", dtype=float)

    factors = pd.concat([CMKT, SIZE, MOM], axis=1).dropna()
    print(f"  -> {len(factors)} périodes de facteurs construites")
    return factors, size_large


# ============================================================
#  3. RÉGRESSIONS OLS
# ============================================================

def run_ols(returns, factors):
    print("\n[3/5] Régressions OLS...")
    common = returns.index.intersection(factors.index)
    ret = returns.loc[common].astype(float)
    F   = factors.loc[common].astype(float)
    results = {}

    for asset in ret.columns:
        y = ret[asset].values
        if np.isnan(y).mean() > 0.5 or np.all(y == 0):
            continue
        try:
            if HAS_STATSMODELS:
                m = sm.OLS(y, sm.add_constant(F)).fit(cov_type="HC3")
                results[asset] = {
                    "alpha":     float(m.params["const"]),
                    "beta_CMKT": float(m.params.get("CMKT", np.nan)),
                    "beta_SIZE": float(m.params.get("SIZE", np.nan)),
                    "beta_MOM":  float(m.params.get("MOM",  np.nan)),
                    "R2":        float(m.rsquared),
                    "t_alpha":   float(m.tvalues["const"]),
                }
            else:
                X = np.column_stack([np.ones(len(y)), F.values])
                b, *_ = np.linalg.lstsq(X, y, rcond=None)
                yhat  = X @ b
                ss_res = ((y - yhat)**2).sum()
                ss_tot = ((y - y.mean())**2).sum()
                results[asset] = {
                    "alpha":     float(b[0]),
                    "beta_CMKT": float(b[1]),
                    "beta_SIZE": float(b[2]),
                    "beta_MOM":  float(b[3]),
                    "R2":        float(1 - ss_res / (ss_tot + 1e-12)),
                    "t_alpha":   float(b[0] / (np.std(y - yhat) + 1e-9)),
                }
        except Exception as e:
            print(f"  [SKIP] {asset}: {e}")

    ols_df = pd.DataFrame.from_dict(results, orient="index")
    print(ols_df[["alpha","beta_CMKT","beta_SIZE","beta_MOM","R2","t_alpha"]].round(3).to_string())
    return ols_df


# ============================================================
#  3b. TEST D'ENDOGÉNÉITÉ  (Durbin-Wu-Hausman)
# ============================================================

def test_endogeneity(returns, factors, ols_df):
    print("\n[3b] Test d'endogénéité (Durbin-Wu-Hausman)...")

    common = returns.index.intersection(factors.index)
    ret = returns.loc[common].astype(float)
    F   = factors.loc[common].astype(float)
    factor_names = F.columns.tolist()

    v_resids = {}
    for fj in factor_names:
        others = [f for f in factor_names if f != fj]
        X_oth  = np.column_stack([np.ones(len(F)), F[others].values])
        y_fj   = F[fj].values
        b, *_  = np.linalg.lstsq(X_oth, y_fj, rcond=None)
        v_resids[fj] = y_fj - X_oth @ b

    results = {}
    for asset in ols_df.index:
        if asset not in ret.columns:
            continue
        y = ret[asset].values
        if np.isnan(y).mean() > 0.5 or np.all(y == 0):
            continue

        row = {}
        for fj in factor_names:
            v   = v_resids[fj]
            Xc  = np.column_stack([np.ones(len(y)), F.values, v])
            b, *_ = np.linalg.lstsq(Xc, y, rcond=None)
            yhat   = Xc @ b
            resid  = y - yhat
            T, k   = len(y), Xc.shape[1]
            s2     = (resid**2).sum() / max(T - k, 1)
            XtXinv = np.linalg.pinv(Xc.T @ Xc)
            se_gamma = float(np.sqrt(max(s2 * XtXinv[-1, -1], 0)))
            gamma    = float(b[-1])
            t_stat   = gamma / (se_gamma + 1e-12)

            row[f"t_{fj}"]    = round(t_stat, 3)
            row[f"endo_{fj}"] = abs(t_stat) > 1.96

        results[asset] = row

    endo_df = pd.DataFrame.from_dict(results, orient="index")

    print(f"\n  {'Actif':<8}", end="")
    for fj in factor_names:
        print(f"  t_{fj:<6}  Endogène?", end="")
    print()
    print("  " + "-" * (8 + len(factor_names) * 20))

    for asset, row in endo_df.iterrows():
        print(f"  {asset:<8}", end="")
        for fj in factor_names:
            t   = row.get(f"t_{fj}", np.nan)
            end = row.get(f"endo_{fj}", False)
            flag = "⚠ OUI" if end else "  non"
            print(f"  {t:+7.3f}        {flag}", end="")
        print()

    print("\n  Résumé :")
    for fj in factor_names:
        col   = f"endo_{fj}"
        n_end = endo_df[col].sum() if col in endo_df.columns else 0
        total = len(endo_df)
        print(f"  {fj:<6} : endogène pour {n_end}/{total} actifs")

    print("\n  Note : un facteur endogène biaise les betas OLS.")
    print("         Solution recommandée : utiliser un indice externe")
    print("         comme proxy de CMKT (ex: BTC seul, ou indice CoinMarketCap).")

    return endo_df


# ============================================================
#  4. OPTIMISATION DE PORTEFEUILLE (Fama-French)
# ============================================================

def optimize(returns, ols_df):
    print(f"\n[4/5] Optimisation (R² >= {MIN_R2})...")
    PER = {"W": 52, "ME": 12, "D": 252}.get(FREQ, 52)

    selected = [c for c in ols_df[ols_df["R2"] >= MIN_R2].index
                if c in returns.columns and c not in STABLECOINS]
    if len(selected) < 2:
        selected = [c for c in ols_df.index
                    if c in returns.columns and c not in STABLECOINS]
    print(f"  Actifs : {selected}")

    ret = returns[selected].astype(float)
    mu  = ret.mean().values
    cov = ret.cov().values
    n   = len(selected)

    def stats(w):
        w = np.array(w)
        r = float(w @ mu) * PER
        v = float(np.sqrt(max(w @ cov @ w, 0))) * np.sqrt(PER)
        return r, v, (r - RF_ANNUAL) / (v + 1e-9)

    mc = [dict(zip(["ret","vol","sharpe"], stats(np.random.dirichlet(np.ones(n)))))
          for _ in range(N_SIMUL)]
    mc_df = pd.DataFrame(mc)

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bnds = [(0, 1)] * n
    x0   = np.ones(n) / n
    res_mv = minimize(lambda w:  stats(w)[1],  x0, bounds=bnds, constraints=cons, method="SLSQP")
    res_ms = minimize(lambda w: -stats(w)[2],  x0, bounds=bnds, constraints=cons, method="SLSQP")

    opt = {
        "min_var":    {"w": dict(zip(selected, res_mv.x)), "stats": stats(res_mv.x)},
        "max_sharpe": {"w": dict(zip(selected, res_ms.x)), "stats": stats(res_ms.x)},
    }
    for label, key in [("Minimum Variance","min_var"),("Maximum Sharpe","max_sharpe")]:
        r, v, s = opt[key]["stats"]
        top = sorted(opt[key]["w"].items(), key=lambda x: -x[1])[:5]
        print(f"\n  [{label}]  Rdt={r*100:.1f}%  Vol={v*100:.1f}%  Sharpe={s:.3f}")
        print("  " + "  ".join(f"{k}={vv*100:.1f}%" for k, vv in top))

    return mc_df, opt, selected


# ============================================================
#  5. GRAPHIQUES  — figures 1 à 7 existantes
# ============================================================

def plot_all(ols_df, mc_df, opt, selected, returns, factors, size_large):
    print("\n[5/5] Génération des graphiques...")
    set_style()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # ── 1. Betas factoriels
    betas_df = ols_df[["beta_CMKT","beta_SIZE","beta_MOM"]].astype(float).dropna(how="all")
    if betas_df.empty:
        print("  [WARN] Aucun beta à afficher — figure 1 ignorée.")
    else:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        betas_df.plot(kind="bar", ax=ax, width=0.72, edgecolor="white")
        ax.axhline(0, color="k", lw=1, ls="--")
        ax.set_title("Betas factoriels", fontweight="bold")
        ax.set_ylabel("Valeur du beta")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(["β CMKT", "β SIZE", "β MOM"])
        fig.tight_layout()
        save(fig, "ff_1_betas.png")

    # ── 2. Alpha de Jensen & R²
    fig, ax2 = plt.subplots(figsize=(FIG_W, FIG_H))
    ax2b   = ax2.twinx()
    alphas = ols_df["alpha"].astype(float).values * 100
    colors = ["limegreen" if a > 0 else "tomato" for a in alphas]
    ax2.bar(range(len(ols_df)), alphas, color=colors, alpha=0.85, edgecolor="white")
    ax2b.plot(range(len(ols_df)), ols_df["R2"].astype(float).values,
              "o--", color="steelblue", lw=2, markersize=8, zorder=5)
    ax2.set_xticks(range(len(ols_df)))
    ax2.set_xticklabels(ols_df.index, rotation=30, ha="right")
    ax2.set_title("Alpha de Jensen & R²", fontweight="bold")
    ax2.set_ylabel("Alpha hebdo (%)")
    ax2b.set_ylabel("R²")
    ax2b.set_ylim(0, 1.1)
    ax2.legend(handles=[
        Patch(color="limegreen", label="Alpha > 0"),
        Patch(color="tomato",    label="Alpha < 0"),
        Line2D([0],[0], color="steelblue", marker="o", lw=2, label="R²"),
    ], loc="upper left")
    fig.tight_layout()
    save(fig, "ff_2_alpha_r2.png")

    # ── 3. Frontière efficiente
    fig, ax3 = plt.subplots(figsize=(FIG_W + 2, FIG_H))
    sc = ax3.scatter(mc_df["vol"]*100, mc_df["ret"]*100,
                     c=mc_df["sharpe"], cmap="RdYlGn", alpha=0.5, s=16)
    plt.colorbar(sc, ax=ax3, label="Ratio de Sharpe", pad=0.02)
    rv, vv, sv = opt["min_var"]["stats"]
    rs, vs, ss = opt["max_sharpe"]["stats"]
    ax3.scatter(vv*100, rv*100, marker="*", s=700, color="royalblue",
                zorder=6, label=f"Min Variance  (Sharpe = {sv:.2f})")
    ax3.scatter(vs*100, rs*100, marker="*", s=700, color="goldenrod",
                zorder=6, label=f"Max Sharpe   (Sharpe = {ss:.2f})")
    ax3.set_xlabel("Volatilité annualisée (%)")
    ax3.set_ylabel("Rendement annualisé (%)")
    ax3.set_title(f"Frontière Efficiente — {N_SIMUL} portefeuilles simulés", fontweight="bold")
    ax3.legend(fontsize=12)
    fig.tight_layout()
    save(fig, "ff_3_frontiere.png")

    # ── 4. Poids des portefeuilles optimaux
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    for ax, (label, key) in zip(axes, [("Minimum Variance","min_var"),("Maximum Sharpe","max_sharpe")]):
        wd = {k: v for k, v in opt[key]["w"].items() if v > 0.01}
        if wd:
            wedges, texts, autotexts = ax.pie(
                list(wd.values()), labels=list(wd.keys()),
                autopct="%1.1f%%", startangle=140,
                textprops={"fontsize": 13}, pctdistance=0.80,
            )
            for at in autotexts:
                at.set_fontsize(12)
        r, v, s = opt[key]["stats"]
        ax.set_title(
            f"{label}\nRdt = {r*100:.1f}%   Vol = {v*100:.1f}%   Sharpe = {s:.2f}",
            fontweight="bold", pad=20
        )
    fig.tight_layout(pad=3)
    save(fig, "ff_4_portefeuilles.png")

    # ── 5. Évolution des facteurs
    fig, ax6 = plt.subplots(figsize=(FIG_W + 2, FIG_H))
    factors.astype(float).plot(ax=ax6, lw=1.6)
    ax6.axhline(0, color="k", lw=0.9, ls="--")
    ax6.set_title("Évolution des facteurs dans le temps", fontweight="bold")
    ax6.set_ylabel("Valeur du facteur")
    ax6.legend(["CMKT", "SIZE", "MOM"])
    ax6.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save(fig, "ff_5_facteurs.png")

    # ── 6. Heatmap des corrélations
    corr = returns[selected].astype(float).corr()
    n    = len(corr)
    fig, ax7 = plt.subplots(figsize=(max(8, n * 1.3), max(6, n * 1.1)))
    sns.heatmap(
        corr, ax=ax7,
        annot=True, fmt=".2f", cmap="coolwarm",
        annot_kws={"size": 13}, linewidths=0.6,
        xticklabels=corr.columns, yticklabels=corr.columns,
        vmin=-1, vmax=1
    )
    ax7.set_title("Corrélations — Actifs sélectionnés", fontweight="bold")
    ax7.tick_params(axis="x", rotation=30)
    ax7.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    save(fig, "ff_6_correlations.png")


def plot_endogeneity(endo_df):
    """Heatmap des t-stats d'endogénéité par actif x facteur."""
    set_style()
    factor_names = ["CMKT", "SIZE", "MOM"]
    t_cols  = [f"t_{f}"    for f in factor_names if f"t_{f}"    in endo_df.columns]
    e_cols  = [f"endo_{f}" for f in factor_names if f"endo_{f}" in endo_df.columns]

    t_data = endo_df[t_cols].astype(float)
    t_data.columns = [c.replace("t_", "") for c in t_data.columns]
    endo_mask = endo_df[e_cols].values if e_cols else None

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, max(5, len(endo_df) * 0.8 + 2)))

    ax1 = axes[0]
    sns.heatmap(
        t_data, ax=ax1,
        annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-4, vmax=4,
        annot_kws={"size": 12}, linewidths=0.5,
        xticklabels=t_data.columns, yticklabels=t_data.index
    )
    ax1.set_title("t-statistiques d'endogénéité\n(|t| > 1.96 → endogène)", fontweight="bold")
    ax1.tick_params(axis="x", rotation=0)
    ax1.tick_params(axis="y", rotation=0)

    if endo_mask is not None:
        for i in range(endo_mask.shape[0]):
            for j in range(endo_mask.shape[1]):
                if endo_mask[i, j]:
                    ax1.add_patch(plt.Rectangle(
                        (j, i), 1, 1,
                        fill=False, edgecolor="red", lw=2.5
                    ))

    ax2 = axes[1]
    if e_cols:
        n_endo = endo_df[e_cols].sum(axis=1).astype(int)
        colors = ["tomato" if v > 0 else "limegreen" for v in n_endo]
        ax2.barh(n_endo.index, n_endo.values, color=colors, edgecolor="white")
        ax2.axvline(0, color="k", lw=0.8)
        ax2.set_xlim(0, len(factor_names) + 0.5)
        ax2.set_xticks(range(len(factor_names) + 1))
        ax2.set_xlabel("Nombre de facteurs endogènes")
        ax2.set_title("Endogénéité par actif\n(sur 3 facteurs)", fontweight="bold")
        for i, v in enumerate(n_endo.values):
            if v > 0:
                ax2.text(v + 0.05, i, f" {v}", va="center", fontsize=12,
                         color="tomato", fontweight="bold")

    fig.tight_layout(pad=3)
    save(fig, "ff_7_endogeneite.png")


# ============================================================
#  6. COMPARAISON ÉQUIRÉPARTI vs SHARPE OPTIMAL  ← NOUVEAU
# ============================================================

def _drawdown(cumul: np.ndarray) -> np.ndarray:
    """Calcule la série de drawdown à partir d'une courbe cumulée."""
    roll_max = np.maximum.accumulate(cumul)
    return (cumul - roll_max) / roll_max


def portfolio_comparison(returns,ols_df):
    """
    Compare deux stratégies sur les actifs non-stablecoins :
      - Portefeuille équiréparti (1/N)
      - Portefeuille Sharpe-optimal (Monte Carlo, N_SIMUL_VS simulations)

    Génère ff_8_portfolio_vs.png avec 3 sous-graphiques :
      (A) Performance cumulée — base 1
      (B) Drawdown
      (C) Allocation du portefeuille Sharpe optimal (barres horizontales)
    """
    print("\n[6/6] Comparaison Équiréparti vs Sharpe Optimal...")

    # ── Sélection des actifs (hors stablecoins, suffisamment de données)
    cols = [c for c in returns.columns
            if c not in STABLECOINS and returns[c].notna().sum() > 50]
    ret  = returns[cols].astype(float).fillna(0.0)
    n    = len(cols)
    print(f"  Actifs retenus ({n}) : {cols}")

    # ── Statistiques annualisées
    PER      = {"W": 52, "ME": 12, "D": 252}.get(FREQ, 52)
    mu       = ret.mean().values * PER
    cov_ann  = ret.cov().values  * PER

    def stats(w):
        w = np.array(w)
        r = float(w @ mu)
        v = float(np.sqrt(max(w @ cov_ann @ w, 1e-12)))
        return r, v, (r - RF_ANNUAL) / (v + 1e-9)

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bnds = [(0, 1)] * n
    x0   = np.ones(n) / n
    res  = minimize(lambda w: -stats(w)[2], x0,
                    bounds=bnds, constraints=cons, method="SLSQP")
    opt_weights = res.x

    eq_weights = np.ones(n) / n

    # ── Performances cumulées (sur les log-rendements périodiques)
    dates      = ret.index
    eq_daily   = ret.values @ eq_weights
    opt_daily  = ret.values @ opt_weights
    eq_cumul   = np.cumprod(1 + eq_daily)
    opt_cumul  = np.cumprod(1 + opt_daily)

    # ── Métriques résumées
    def metrics(daily, cumul):
        ann_ret = float(np.mean(daily)) * PER
        ann_vol = float(np.std(daily))  * np.sqrt(PER)
        sharpe  = (ann_ret - RF_ANNUAL) / (ann_vol + 1e-9)
        max_dd  = float(_drawdown(cumul).min())
        total   = float(cumul[-1] - 1)
        return ann_ret, ann_vol, sharpe, max_dd, total

    eq_m  = metrics(eq_daily,  eq_cumul)
    opt_m = metrics(opt_daily, opt_cumul)

    labels_m = ["Rendement ann.", "Volatilité ann.", "Sharpe", "Max Drawdown", "Perf. totale"]
    print(f"\n  {'Métrique':<22} {'Équiréparti':>13} {'Sharpe Opt.':>13}")
    print("  " + "-" * 50)
    for lbl, ev, ov in zip(labels_m, eq_m, opt_m):
        fmt = ".1%" if lbl != "Sharpe" else ".2f"
        print(f"  {lbl:<22} {ev:>{13}{fmt}} {ov:>{13}{fmt}}")

    print(f"\n  Meilleures allocations Sharpe Optimal :")
    top_idx = np.argsort(opt_weights)[::-1]
    for i in top_idx[:8]:
        print(f"    {cols[i]:<8} {opt_weights[i]:.1%}")

    # ── Graphique  ────────────────────────────────────────────────────────────
    set_style()

    fig = plt.figure(figsize=(FIG_W + 2, 14))
    gs  = fig.add_gridspec(3, 2,
                           height_ratios=[3, 1.5, 2],
                           hspace=0.42, wspace=0.35)

    COLOR_EQ  = "#2563eb"   # bleu
    COLOR_OPT = "#d97706"   # ambre

    # ── (A) Performance cumulée ───────────────────────────────────────────────
    ax_perf = fig.add_subplot(gs[0, :])   # pleine largeur
    ax_perf.set_facecolor("#f7f7f7")

    ax_perf.plot(dates, eq_cumul,  color=COLOR_EQ,  lw=2.0,
                 label=f"Équiréparti  | Sharpe {eq_m[2]:.2f} | Perf. {eq_m[4]:+.0%}")
    ax_perf.plot(dates, opt_cumul, color=COLOR_OPT, lw=2.5,
                 label=f"Sharpe Opt.  | Sharpe {opt_m[2]:.2f} | Perf. {opt_m[4]:+.0%}")
    ax_perf.axhline(1, color="#94a3b8", ls="--", lw=0.9)

    ax_perf.fill_between(dates, eq_cumul, opt_cumul,
                         where=(opt_cumul >= eq_cumul),
                         alpha=0.12, color=COLOR_OPT)
    ax_perf.fill_between(dates, eq_cumul, opt_cumul,
                         where=(opt_cumul < eq_cumul),
                         alpha=0.12, color=COLOR_EQ)

    # Annotation de la valeur finale
    for val, col in [(eq_cumul[-1], COLOR_EQ), (opt_cumul[-1], COLOR_OPT)]:
        ax_perf.annotate(f"{val:.2f}x",
                         xy=(dates[-1], val),
                         xytext=(6, 0), textcoords="offset points",
                         color=col, fontsize=11, fontweight="bold",
                         va="center", clip_on=False)

    ax_perf.set_title("Portefeuille Équiréparti vs Sharpe Optimal — Performance cumulée",
                      fontweight="bold", fontsize=15)
    ax_perf.set_ylabel("Valeur du portefeuille (base 1)")
    ax_perf.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}x"))
    ax_perf.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_perf.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax_perf.xaxis.get_majorticklabels(), rotation=25, ha="right")
    ax_perf.grid(True, color="#e2e8f0", lw=0.6)
    ax_perf.legend(fontsize=11, loc="upper left",
                   framealpha=0.9, edgecolor="#cbd5e1")

    # ── (B) Drawdown ──────────────────────────────────────────────────────────
    ax_dd = fig.add_subplot(gs[1, :])    # pleine largeur
    ax_dd.set_facecolor("#f7f7f7")

    dd_eq  = _drawdown(eq_cumul)
    dd_opt = _drawdown(opt_cumul)

    ax_dd.fill_between(dates, dd_eq,  0, alpha=0.55, color=COLOR_EQ,
                       label=f"Équiréparti  (max DD {eq_m[3]:.1%})")
    ax_dd.fill_between(dates, dd_opt, 0, alpha=0.55, color=COLOR_OPT,
                       label=f"Sharpe Opt.  (max DD {opt_m[3]:.1%})")
    ax_dd.axhline(0, color="#94a3b8", lw=0.8, ls="--")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_dd.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax_dd.xaxis.get_majorticklabels(), rotation=25, ha="right")
    ax_dd.grid(True, color="#e2e8f0", lw=0.6)
    ax_dd.set_title("Drawdown", fontweight="bold", fontsize=13)
    ax_dd.legend(fontsize=10, loc="lower left", framealpha=0.9)

    # ── (C) Allocations Sharpe Optimal ───────────────────────────────────────
    ax_wt = fig.add_subplot(gs[2, 0])   # colonne gauche
    ax_wt.set_facecolor("#f7f7f7")

    sorted_idx  = np.argsort(opt_weights)
    sorted_syms = [cols[i] for i in sorted_idx]
    sorted_wts  = opt_weights[sorted_idx]
    eq_wt       = 1.0 / n

    bar_colors = [COLOR_OPT if w >= eq_wt else COLOR_EQ for w in sorted_wts]
    bars = ax_wt.barh(sorted_syms, sorted_wts * 100,
                      color=bar_colors, edgecolor="white", height=0.7)
    ax_wt.axvline(eq_wt * 100, color="#64748b", ls="--", lw=1.2,
                  label=f"Équiréparti ({eq_wt:.1%})")
    ax_wt.set_xlabel("Poids (%)")
    ax_wt.set_title("Allocations — Sharpe Optimal", fontweight="bold", fontsize=13)
    ax_wt.legend(fontsize=10, framealpha=0.9)
    for bar, w in zip(bars, sorted_wts):
        ax_wt.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                   f"{w:.1%}", va="center", fontsize=9,
                   color=COLOR_OPT if w >= eq_wt else COLOR_EQ)
    ax_wt.grid(True, axis="x", color="#e2e8f0", lw=0.6)

    # ── (D) Tableau de métriques ──────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, 1])  # colonne droite
    ax_tbl.set_facecolor("#f7f7f7")
    ax_tbl.axis("off")

    row_labels = labels_m
    col_labels = ["Équiréparti", "Sharpe Opt."]
    cell_text  = []
    for lbl, ev, ov in zip(labels_m, eq_m, opt_m):
        fmt = ".1%" if lbl != "Sharpe" else ".2f"
        cell_text.append([f"{ev:{fmt}}", f"{ov:{fmt}}"])

    tbl = ax_tbl.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.4, 2.0)

    # Colorer les en-têtes
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if r == 0:
            cell.set_facecolor(COLOR_EQ if c == 1 else
                               (COLOR_OPT if c == 2 else "#f1f5f9"))
            cell.set_text_props(color="white" if c in (1, 2) else "black",
                                fontweight="bold")
        elif c == -1:
            cell.set_facecolor("#f1f5f9")
            cell.set_text_props(fontweight="bold")

    ax_tbl.set_title("Métriques récapitulatives", fontweight="bold", fontsize=13, pad=12)

    # ── Sous-titre global
    fig.text(0.5, 0.01,
             f"Période : {str(dates[0])[:10]} → {str(dates[-1])[:10]}  ·  "
             f"Fréquence : {FREQ}  ·  "
             f"Optimisation SLSQP  ·  "
             "Backtest in-sample — hors frais",
             ha="center", fontsize=10, color="#64748b")

    save(fig, "ff_8_portfolio_vs.png")
    print("  8 fichiers PNG générés.")


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  FAMA-FRENCH CRYPTO — Démarrage")
    print("=" * 55 + "\n")

    returns, mcaps       = load_prices()
    factors, size_large  = build_factors(returns, mcaps)
    ols_df               = run_ols(returns, factors)
    endo_df              = test_endogeneity(returns, factors, ols_df)
    mc_df, opt, selected = optimize(returns, ols_df)
    plot_all(ols_df, mc_df, opt, selected, returns, factors, size_large)
    plot_endogeneity(endo_df)
    portfolio_comparison(returns,ols_df)          # ← NOUVEAU

    print("\n" + "=" * 55)
    print("  Pipeline terminé avec succès ✓")
    print("=" * 55 + "\n")