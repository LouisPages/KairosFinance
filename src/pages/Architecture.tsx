import { useState } from "react";
import { useAppMode } from "@/context/AppModeContext";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain,
  ArrowDown,
  Database,
  BarChart3,
  ChevronDown,
  ChevronUp,
  TrendingUp,
  Layers,
  Cpu,
  Bot,
} from "lucide-react";

/* ─────────────────────────────────────────────
   Diagram primitives
───────────────────────────────────────────── */
const Box = ({
  children,
  className = "",
  accent = false,
}: {
  children: React.ReactNode;
  className?: string;
  accent?: boolean;
}) => (
  <div
    className={`border rounded-lg px-4 py-2.5 text-center font-mono text-[11px] leading-relaxed ${
      accent
        ? "border-primary/50 bg-primary/5 text-foreground font-semibold"
        : "border-border bg-background/60 text-foreground/80"
    } ${className}`}
  >
    {children}
  </div>
);

const Arrow = () => (
  <div className="flex justify-center my-1 text-muted-foreground">
    <ArrowDown className="h-4 w-4" />
  </div>
);

const Row = ({ children }: { children: React.ReactNode }) => (
  <div className="flex items-center justify-center gap-3">{children}</div>
);

const Connector = () => (
  <div className="flex justify-center items-center gap-0 my-1">
    <div className="w-px h-4 bg-border" />
  </div>
);


/* ─────────────────────────────────────────────
   Model definitions with detailed content
───────────────────────────────────────────── */
const models = [
  {
    id: "markowitz-classic",
    icon: BarChart3,
    name: "Markowitz Classique",
    badge: "Baseline",
    desc: "Optimisation moyenne-variance fondée sur l'estimation empirique des paramètres (H. Markowitz, 1952).",
    color: "text-blue-400",
    borderColor: "border-blue-400/30",
    detail: {
      summary:
        "Le modèle de référence : μ et Σ sont estimés directement à partir des moyennes et covariances historiques des rendements journaliers. Simple, interprétable, mais sensible au bruit d'estimation.",
      diagram: (
        <div className="space-y-0">
          <Box accent>
            Entrée&nbsp;: prix de clôture ajustés (yfinance) · liste de tickers · période [start, end]
          </Box>
          <Arrow />
          <Box>
            Rendements logarithmiques journaliers<br />
            r&nbsp;=&nbsp;ln(P&nbsp;/&nbsp;P<sub>t-1</sub>)
          </Box>
          <Arrow />
          <Box>
            Estimation empirique — ensemble train (80%)<br />
            μ&nbsp;=&nbsp;moyenne(r)&nbsp;×&nbsp;252<br />
            Σ&nbsp;=&nbsp;cov(r)&nbsp;×&nbsp;252
          </Box>
          <Arrow />
          <Box>
            Monte-Carlo (10&nbsp;000&nbsp;portefeuilles)<br />
            w&nbsp;~&nbsp;Dirichlet&nbsp;—&nbsp;poids&nbsp;≥&nbsp;0,&nbsp;Σw&nbsp;=&nbsp;1<br />
            Sharpe&nbsp;=&nbsp;(wᵀμ&nbsp;−&nbsp;rf)&nbsp;/&nbsp;√(wᵀΣw)<br />
            w*&nbsp;=&nbsp;argmax&nbsp;Sharpe
          </Box>
          <Arrow />
          <Box>
            Backtesting — période de test (20%)<br />
            Comparaison vs SPY (rebasé à 100)<br />
            Drawdown max&nbsp;=&nbsp;max(1&nbsp;−&nbsp;C&nbsp;/&nbsp;max C)
          </Box>
          <Arrow />
          <Box accent>
            Sortie&nbsp;: poids w* · Sharpe · rendement · volatilité · drawdown max · frontière efficiente
          </Box>
        </div>
      ),
      steps: [
        {
          title: "Données",
          text: "Rendements logarithmiques journaliers calculés sur les prix de clôture ajustés. Le split train/test est fixé à 80 % / 20 % des jours de bourse disponibles.",
        },
        {
          title: "Estimation de μ",
          text: "Vecteur des rendements espérés = moyenne empirique des rendements journaliers annualisée × 252. Pas de régularisation, pas de décomposition factorielle.",
        },
        {
          title: "Estimation de Σ",
          text: "Matrice de covariance empirique annualisée × 252. Peut devenir instable sur de petits échantillons ou avec de nombreux actifs.",
        },
        {
          title: "Optimisation",
          text: "10 000 poids aléatoires (Dirichlet), normalisés pour sommer à 1 (pas de vente à découvert). Le portefeuille maximisant le Sharpe est retenu.",
        },
      ],
    },
  },
  {
    id: "capm",
    icon: TrendingUp,
    name: "CAPM (1 facteur)",
    badge: "MEDAF",
    desc: "Estimation de μ par régression OLS sur le facteur de marché (SPY) — fréquence mensuelle.",
    color: "text-emerald-400",
    borderColor: "border-emerald-400/30",
    detail: {
      summary:
        "Le CAPM ancre les rendements espérés sur le bêta de marché plutôt que sur les moyennes historiques, réduisant le bruit d'estimation. Σ reste empirique.",
      diagram: (
        <div className="space-y-0">
          <Box accent>
            Entrée&nbsp;: prix de clôture ajustés (yfinance) · rendement SPY mensuel · T-Bill ^IRX
          </Box>
          <Arrow />
          <Box>
            Rendements arithmétiques mensuels<br />
            R&nbsp;=&nbsp;P&nbsp;/&nbsp;P<sub>t-1</sub>&nbsp;−&nbsp;1
          </Box>
          <Arrow />
          <Box>
            Régression OLS par actif — ensemble train (80%)<br />
            R<sub>i</sub>&nbsp;−&nbsp;Rf&nbsp;=&nbsp;α<sub>i</sub>&nbsp;+&nbsp;β<sub>i</sub>·(Rm&nbsp;−&nbsp;Rf)&nbsp;+&nbsp;ε<sub>i</sub><br />
            → β<sub>i</sub>&nbsp;estimé par OLS (sensibilité au marché)
          </Box>
          <Arrow />
          <Box>
            Rendements espérés annualisés<br />
            μ̂<sub>i</sub>&nbsp;=&nbsp;(R̄f&nbsp;+&nbsp;β̂<sub>i</sub>·mean(Rm−Rf))&nbsp;×&nbsp;12<br />
            α<sub>i</sub>&nbsp;ignoré&nbsp;—&nbsp;MEDAF&nbsp;:&nbsp;α&nbsp;=&nbsp;0&nbsp;à&nbsp;l'équilibre<br />
            Σ empirique × 12
          </Box>
          <Arrow />
          <Box>
            Monte-Carlo (10&nbsp;000 portefeuilles)<br />
            w*&nbsp;=&nbsp;argmax&nbsp;Sharpe&nbsp;→&nbsp;Backtesting mensuel
          </Box>
          <Arrow />
          <Box accent>
            Sortie&nbsp;: poids w* · β par actif · Sharpe · rendement · volatilité · drawdown max · frontière efficiente
          </Box>
        </div>
      ),
      steps: [
        {
          title: "Fréquence mensuelle",
          text: "Les prix sont rééchantillonnés au dernier jour ouvré de chaque mois. Les rendements sont arithmétiques (et non logarithmiques), pour cohérence avec les facteurs Fama-French.",
        },
        {
          title: "Régression CAPM",
          text: "Pour chaque actif, OLS sur l'excès de rendement par rapport au taux sans risque, régressé sur l'excès de rendement du marché (SPY). Le bêta mesure la sensibilité systématique.",
        },
        {
          title: "Estimation de μ via bêta",
          text: "μ̂ᵢ = (R̄f + β̂ᵢ · prime de risque marché) × 12. L'alpha de Jensen est ignoré — conformément à la théorie MEDAF qui postule α = 0 à l'équilibre.",
        },
        {
          title: "Avantage vs Classique",
          text: "Estimations de μ plus stables économiquement, moins sensibles au bruit des moyennes historiques sur des fenêtres courtes.",
        },
      ],
    },
  },
  {
    id: "ff3",
    icon: Layers,
    name: "Fama-French 3 Facteurs",
    badge: "FF3",
    desc: "Extension CAPM avec les primes de taille (SMB) et de valeur (HML) — modèle Fama-French 1993.",
    color: "text-violet-400",
    borderColor: "border-violet-400/30",
    detail: {
      summary:
        "En ajoutant SMB (Small Minus Big) et HML (High Minus Low) au facteur de marché, le modèle capture les effets de taille et de valeur documentés empiriquement par Fama & French (1993).",
      diagram: (
        <div className="space-y-0">
          <Row>
            <Box accent className="flex-1">
              Entrée A<br />
              Prix ajustés (yfinance)<br />
              → R<sub>i,t</sub>&nbsp;mensuels
            </Box>
            <Box accent className="flex-1">
              Entrée B<br />
              Facteurs FF3 (Ken French)<br />
              Mkt-RF · SMB · HML
            </Box>
          </Row>
          <Arrow />
          <Box>
            Jointure interne sur index mensuel<br />
            Découpage train (80%) / test (20%)
          </Box>
          <Arrow />
          <Box>
            Régression OLS par actif — ensemble train<br />
            R<sub>i</sub>&nbsp;−&nbsp;Rf&nbsp;=&nbsp;α<sub>i</sub>&nbsp;+&nbsp;β<sup>Mkt</sup>·(Rm−Rf)&nbsp;+&nbsp;β<sup>SMB</sup>·SMB&nbsp;+&nbsp;β<sup>HML</sup>·HML&nbsp;+&nbsp;ε<sub>i</sub><br />
            β̂&nbsp;=&nbsp;(XᵀX)⁻¹&nbsp;Xᵀy
          </Box>
          <Arrow />
          <Box>
            μ̂<sub>i</sub>&nbsp;=&nbsp;(R̄f&nbsp;+&nbsp;β̂<sup>Mkt</sup>·m̄kt&nbsp;+&nbsp;β̂<sup>SMB</sup>·s̄mb&nbsp;+&nbsp;β̂<sup>HML</sup>·h̄ml)&nbsp;×&nbsp;12<br />
            Σ empirique × 12
          </Box>
          <Arrow />
          <Box>
            Monte-Carlo (10&nbsp;000 portefeuilles)<br />
            w*&nbsp;=&nbsp;argmax&nbsp;Sharpe&nbsp;→&nbsp;Backtesting mensuel
          </Box>
          <Arrow />
          <Box accent>
            Sortie&nbsp;: poids w* · β<sup>Mkt</sup>/β<sup>SMB</sup>/β<sup>HML</sup> par actif · Sharpe · rendement · volatilité · drawdown max · frontière efficiente
          </Box>
        </div>
      ),
      steps: [
        {
          title: "Source des facteurs",
          text: "Les facteurs SMB et HML sont chargés depuis la Ken French Data Library (fichier F-F_Research_Data_3_Factors). L'index mensuel est aligné avec les rendements par jointure interne.",
        },
        {
          title: "Régression à 3 facteurs",
          text: "OLS sur l'excès de rendement de chaque actif, avec une matrice X = [1, Mkt-RF, SMB, HML]. L'estimateur est β̂ = (XᵀX)⁻¹Xᵀy.",
        },
        {
          title: "Reconstruction de μ",
          text: "Les moyennes empiriques des trois facteurs sur la période d'entraînement multiplient les bêtas correspondants — aucun terme alpha n'est inclus dans μ̂.",
        },
        {
          title: "SMB — prime de taille",
          text: "Petites capitalisations surperforment historiquement les grandes. SMB capture cet effet ; un β^SMB positif signale une exposition aux small caps.",
        },
        {
          title: "HML — prime de valeur",
          text: "Actions value (book-to-market élevé) surperforment les growth. Un β^HML positif indique une exposition au style value.",
        },
      ],
    },
  },
  {
    id: "ff5",
    icon: Cpu,
    name: "Fama-French 5 Facteurs",
    badge: "FF5",
    desc: "Extension FF3 avec les primes de profitabilité (RMW) et d'investissement (CMA) — modèle Fama-French 2015.",
    color: "text-amber-400",
    borderColor: "border-amber-400/30",
    detail: {
      summary:
        "Le modèle à 5 facteurs de Fama & French (2015) enrichit FF3 en ajoutant RMW (Robust Minus Weak, prime de profitabilité) et CMA (Conservative Minus Aggressive, prime d'investissement), documentant que les entreprises profitables et peu investisseuses surperforment.",
      diagram: (
        <div className="space-y-0">
          <Row>
            <Box accent className="flex-1">
              Entrée A<br />
              Prix ajustés (yfinance)<br />
              → R<sub>i,t</sub>&nbsp;mensuels
            </Box>
            <Box accent className="flex-1">
              Entrée B<br />
              Facteurs FF5 (Ken French)<br />
              Mkt-RF · SMB · HML · RMW · CMA
            </Box>
          </Row>
          <Arrow />
          <Box>
            Jointure interne sur index mensuel<br />
            Découpage train (80%) / test (20%)
          </Box>
          <Arrow />
          <Box>
            Régression OLS par actif — ensemble train<br />
            R<sub>i</sub>&nbsp;−&nbsp;Rf&nbsp;=&nbsp;α<sub>i</sub>&nbsp;+&nbsp;β<sup>Mkt</sup>·(Rm−Rf)&nbsp;+&nbsp;β<sup>SMB</sup>·SMB&nbsp;+&nbsp;β<sup>HML</sup>·HML&nbsp;+&nbsp;β<sup>RMW</sup>·RMW&nbsp;+&nbsp;β<sup>CMA</sup>·CMA&nbsp;+&nbsp;ε<sub>i</sub><br />
            Fallback ridge (λ=10⁻⁴) si multicolinéarité détectée
          </Box>
          <Arrow />
          <Box>
            μ̂<sub>i</sub>&nbsp;=&nbsp;(R̄f&nbsp;+&nbsp;β̂<sup>Mkt</sup>·m̄kt&nbsp;+&nbsp;β̂<sup>SMB</sup>·s̄mb&nbsp;+&nbsp;β̂<sup>HML</sup>·h̄ml&nbsp;+&nbsp;β̂<sup>RMW</sup>·r̄mw&nbsp;+&nbsp;β̂<sup>CMA</sup>·c̄ma)&nbsp;×&nbsp;12<br />
            Σ empirique × 12
          </Box>
          <Arrow />
          <Box>
            Monte-Carlo (10&nbsp;000 portefeuilles)<br />
            w*&nbsp;=&nbsp;argmax&nbsp;Sharpe&nbsp;→&nbsp;Backtesting mensuel
          </Box>
          <Arrow />
          <Box accent>
            Sortie&nbsp;: poids w* · β<sup>Mkt/SMB/HML/RMW/CMA</sup> par actif · Sharpe · rendement · volatilité · drawdown max · frontière efficiente
          </Box>
        </div>
      ),
      steps: [
        {
          title: "RMW — prime de profitabilité",
          text: "Robust Minus Weak : les entreprises à profitabilité opérationnelle robuste surperforment les entreprises faibles. Un β^RMW positif indique une exposition aux entreprises profitables.",
        },
        {
          title: "CMA — prime d'investissement",
          text: "Conservative Minus Aggressive : les entreprises investissant peu surperforment celles qui investissent agressivement. Un β^CMA positif signale une exposition aux entreprises conservatrices.",
        },
        {
          title: "Multicolinéarité",
          text: "HML, RMW et CMA peuvent être corrélés entre eux. En cas de quasi-singularité de XᵀX (VIF élevé), une régression ridge légère (λ=10⁻⁴) est utilisée comme fallback.",
        },
        {
          title: "Construction de μ",
          text: "Identique à FF3 mais étendu à 5 facteurs : β̂ × moyennes empiriques des facteurs sur le train, annualisé × 12.",
        },
      ],
    },
  },
  {
    id: "llm-dynamic",
    icon: Bot,
    name: "LLM Dynamique",
    badge: "IA",
    desc: "Sélection mensuelle des facteurs Fama-French guidée par LLM (Mistral : news AFP + sélection des facteurs) sur base de l'actualité économique.",
    color: "text-rose-400",
    borderColor: "border-rose-400/30",
    detail: {
      summary:
        "Le pipeline le plus avancé : chaque mois, un premier appel Mistral résume les actualités AFP par ticker, puis un second appel Mistral décide quels facteurs Fama-French sont pertinents. La régression et l'optimisation Markowitz n'utilisent que les facteurs retenus.",
      diagram: (
        <div className="space-y-0">
          <Row>
            <Box accent className="flex-1">
              Entrée A<br />
              Liste de tickers · date cible n<br />
              Prix ajustés (yfinance)
            </Box>
            <Box accent className="flex-1">
              Entrée B<br />
              Facteurs FF5 (Ken French)<br />
              Mkt-RF · SMB · HML · RMW · CMA
            </Box>
          </Row>
          <Arrow />
          <Box>
            <span className="font-semibold text-foreground">Phase 1 — Collecte des news</span><br />
            Mistral Le Chat + accès AFP · fenêtre [n−3 mois, fin mois n]<br />
            → JSON par ticker : &#123; summary, key_events, sentiment &#125;<br />
            Cache indexé par (ticker, année_mois)
          </Box>
          <Arrow />
          <Box>
            <span className="font-semibold text-foreground">Phase 2 — Sélection des facteurs</span><br />
            Mistral (API) · input : résumé + définitions FF<br />
            → masque JSON : &#123; Mkt-RF, SMB, HML, RMW, CMA : bool &#125;<br />
            Masque global = union des facteurs retenus par ticker
          </Box>
          <Arrow />
          <Box>
            <span className="font-semibold text-foreground">Phase 3 — Régression sur fenêtre glissante</span><br />
            R<sub>i</sub> − Rf = α<sub>i</sub> + Σ<sub>k∈Fn</sub> β<sub>i,k</sub>·F<sub>k</sub> + ε<sub>i</sub> (OLS, facteurs du masque uniquement)<br />
            Fenêtre : 80% des données jusqu'au mois n<br />
            μ̂<sub>i</sub> = (R̄f + Σ β̂<sub>i,k</sub>·F̄<sub>k</sub>) × 12
          </Box>
          <Arrow />
          <Box>
            <span className="font-semibold text-foreground">Phase 4 — Optimisation Monte-Carlo</span><br />
            10 000 portefeuilles → w* = argmax Sharpe pour le mois n+1<br />
            Mode prédiction : 1 mois cible | Mode backtest : fenêtre glissante mensuelle
          </Box>
          <Arrow />
          <Box accent>
            Sortie : poids w* · facteurs retenus par ticker · résumés news · Sharpe · rendement · volatilité · drawdown max · frontière efficiente
          </Box>
        </div>
      ),
      steps: [
        {
          title: "Phase 1 — Mistral Le Chat & AFP",
          text: "Mistral dispose d'un accès natif aux dépêches AFP. Pour chaque ticker, une requête avec fenêtre glissante de 3 mois produit un résumé JSON structuré (summary, key_events, sentiment). Les résultats sont mis en cache par (ticker, année_mois).",
        },
        {
          title: "Phase 2 — Agent de sélection (Mistral)",
          text: "Le modèle Mistral configuré (SELECTOR_PROVIDER, défaut mistral) reçoit le résumé économique et les définitions des facteurs FF. Il retourne un JSON booléen pour chaque facteur. Le masque global est l'union des facteurs sélectionnés sur tous les tickers. Fallback vers tous les facteurs si moins de 1 est retenu.",
        },
        {
          title: "Phase 3 — Régression dynamique",
          text: "La régression OLS n'utilise que les facteurs du masque global F_n. Les bêtas sont estimés sur une fenêtre glissante (80% des données jusqu'au mois courant), permettant une adaptation continue aux régimes de marché.",
        },
        {
          title: "Phase 4 — Deux modes de fonctionnement",
          text: "Mode prédiction : calcul des poids pour un seul mois cible n+1. Mode backtest glissant : recalcul mensuel sur toute la période de test, pour mesurer la performance historique du pipeline complet.",
        },
        {
          title: "Gestion des clés API",
          text: "MISTRAL_API_KEY (obligatoire pour ce pipeline), éventuellement OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY si SELECTOR_PROVIDER pointe vers un autre fournisseur. Variables lues depuis l'environnement ou .env (python-dotenv). Température = 0 pour reproductibilité. Requêtes Mistral parallèles via asyncio + httpx.",
        },
      ],
    },
  },
];

const fade = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.08, duration: 0.45 },
  }),
};

const expandVariants = {
  hidden: { opacity: 0, height: 0 },
  visible: { opacity: 1, height: "auto", transition: { duration: 0.35 } },
  exit: { opacity: 0, height: 0, transition: { duration: 0.25 } },
};

const cryptoFfModel = {
  id: "crypto-ff3",
  icon: Layers,
  name: "Fama-French crypto (CMKT, SIZE, MOM)",
  badge: "Crypto",
  desc: "Adaptation du paradigme Fama-French aux cryptomonnaies : facteurs construits sur l’univers CSV (CoinGecko), régressions OLS et optimisation Markowitz.",
  color: "text-orange-400",
  borderColor: "border-orange-400/30",
  detail: {
    summary:
      "Le script crypto_fama_french.py charge les prix et capitalisations depuis des CSV locaux, rééchantillonne (hebdo dans le script batch, mensuel pour le site), construit trois facteurs — CMKT (moyenne des rendements), SIZE (petites vs grandes caps via market cap), MOM (momentum long/short sur quantiles) — puis estime les bêtas par actif, optimise le portefeuille (Sharpe / variance) et produit analyses d’endogénéité et comparaisons 1/N vs Sharpe.",
    diagram: (
      <div className="space-y-0">
        <Box accent>
          Entrée&nbsp;: fichiers CSV dans gestion/crypto/données · colonnes snapped_at, price, market_cap
        </Box>
        <Arrow />
        <Box>
          Rééchantillonnage (mensuel sur le site) · rendements arithmétiques<br />
          Clip extrêmes, alignement des séries
        </Box>
        <Arrow />
        <Box>
          Facteurs<br />
          CMKT = moyenne cross-section des rendements<br />
          SIZE = moyenne(petites caps) − moyenne(grandes caps), grandes = top N par mcap moyen<br />
          MOM = performance winners − losers (fenêtre glissante)
        </Box>
        <Arrow />
        <Box>
          OLS par crypto — ensemble train (80 %)<br />
          r<sub>i</sub> = α + β<sub>CMKT</sub>·CMKT + β<sub>SIZE</sub>·SIZE + β<sub>MOM</sub>·MOM + ε<br />
          μ̂<sub>i</sub> (mensuel) = α̂ + β̂′·F̄ → annualisé × 12 ; Σ empirique × 12
        </Box>
        <Arrow />
        <Box>
          Optimisation Markowitz (Monte-Carlo et/ou gradient à pas fixe / optimal)<br />
          w* = argmax Sharpe sous contraintes w ≥ 0, Σw = 1 · stablecoins exclus du noyau d’optimisation
        </Box>
        <Arrow />
        <Box accent>
          Sortie web&nbsp;: poids w* · Sharpe · rendement / volatilité attendus · backtest 20 % · courbe vs moyenne de marché (CMKT)
        </Box>
      </div>
    ),
    steps: [
      {
        title: "Données",
        text: "Un CSV par actif (format CoinGecko max). Le dossier est colocalisé avec crypto_fama_french.py (données). Les stablecoins listés dans le script sont exclus de l’optimisation pour éviter des poids triviaux.",
      },
      {
        title: "CMKT",
        text: "Facteur « marché crypto » : à chaque date, moyenne des rendements de l’univers chargé. C’est le parallèle du facteur de marché agrégé.",
      },
      {
        title: "SIZE",
        text: "Les N plus grosses capitalisations moyennes forment le pôle « large » ; les autres « small ». SIZE est long small, short large (sur rendements).",
      },
      {
        title: "MOM",
        text: "Sur une fenêtre de rendements passés, les actifs au-dessus du 70e quantile (winners) et en dessous du 30e (losers) alimentent un spread de rendement contemporain.",
      },
      {
        title: "Pipeline batch (script)",
        text: "En ligne de commande, le script génère aussi des graphiques (bêtas, α/R², frontière, poids, facteurs, corrélations, endogénéité, équipondéré vs Sharpe). L’API web réutilise la logique facteurs + OLS + optimisation sans produire les PNG.",
      },
    ],
  },
};

/* ─────────────────────────────────────────────
   Component
───────────────────────────────────────────── */
function ArchitectureActions() {
  const [openId, setOpenId] = useState<string | null>(null);

  const toggle = (id: string) => setOpenId((prev) => (prev === id ? null : id));

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
        <p className="section-label mb-2">Architecture</p>
        <h1 className="section-title mb-1">Modèles d'optimisation</h1>
        <p className="mb-10 text-sm text-muted-foreground">
          Cinq approches basées sur la théorie moderne du portefeuille, du Markowitz classique au pipeline LLM dynamique.
          Cliquez sur un modèle pour afficher sa mise en œuvre technique détaillée.
        </p>
      </motion.div>

      {/* ── Global pipeline ── */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="glass-card mb-12 p-6 md:p-8"
      >
        <h2 className="font-display text-base font-bold text-foreground mb-6">
          Pipeline commune à tous les modèles
        </h2>

        {/* Horizontal pipeline */}
        <div className="flex items-stretch justify-center gap-0">
          {/* Step 1 — Input */}
          <div className="flex flex-col items-center justify-center rounded-lg border border-primary/40 bg-primary/5 px-6 py-6 text-center flex-1">
            <Database className="h-6 w-6 text-primary mb-3" />
            <span className="text-sm font-semibold text-foreground">Données</span>
            <span className="text-xs text-muted-foreground mt-1.5">Prix historiques<br />Yahoo Finance</span>
          </div>

          <div className="flex items-center px-2 text-muted-foreground shrink-0">
            <svg width="32" height="16" viewBox="0 0 32 16" fill="none">
              <path d="M0 8h26M20 2l8 6-8 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>

          {/* Step 2 — Modélisation */}
          <div className="flex flex-col items-center justify-center rounded-lg border border-border bg-background px-6 py-6 text-center flex-1">
            <Brain className="h-6 w-6 text-primary mb-3" />
            <span className="text-sm font-semibold text-foreground">Modélisation</span>
            <span className="text-xs text-muted-foreground mt-1.5">Estimation du rendement<br />et de la volatilité</span>
            <span className="text-xs text-muted-foreground mt-1.5">Optimisation ratio de Sharpe</span>
          </div>

          <div className="flex items-center px-2 text-muted-foreground shrink-0">
            <svg width="32" height="16" viewBox="0 0 32 16" fill="none">
              <path d="M0 8h26M20 2l8 6-8 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>

          {/* Step 3 — Output */}
          <div className="flex flex-col items-center justify-center rounded-lg border border-primary/40 bg-primary/5 px-6 py-6 text-center flex-1">
            <BarChart3 className="h-6 w-6 text-primary mb-3" />
            <span className="text-sm font-semibold text-foreground">Résultats</span>
            <span className="text-xs text-muted-foreground mt-1.5">Backtest sur 20% des données historiques avec comparaison portefeuille/marché</span>
          </div>
        </div>

        {/* Divergence note */}
        <div className="mt-5 rounded-lg border border-border bg-muted/30 px-5 py-3">
          <p className="text-xs text-muted-foreground text-center">
            <span className="font-semibold text-foreground">Point de divergence :</span> l'étape "Modélisation" est celle où chaque modèle se distingue — de l'estimation empirique brute jusqu'à la régression factorielle pilotée par LLM.
          </p>
        </div>
      </motion.div>

      {/* ── Model cards ── */}
      <div className="flex flex-col gap-4">
        {models.map((model, i) => {
          const isOpen = openId === model.id;
          const Icon = model.icon;

          return (
            <motion.div
              key={model.id}
              custom={i}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={fade}
              className={`glass-card overflow-hidden border ${model.borderColor}`}
            >
              {/* Card header — always visible */}
              <button
                onClick={() => toggle(model.id)}
                className="w-full flex items-center justify-between gap-4 p-6 text-left hover:bg-muted/20 transition-colors"
              >
                <div className="flex items-center gap-4">
                  <Icon className={`h-6 w-6 shrink-0 ${model.color}`} />
                  <div>
                    <div className="flex items-center gap-2 mb-0.5">
                      <h3 className="font-display text-base font-bold text-foreground">
                        {model.name}
                      </h3>
                    </div>
                    <p className="text-sm text-muted-foreground">{model.desc}</p>
                  </div>
                </div>
                <div className="shrink-0 text-muted-foreground">
                  {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </div>
              </button>

              {/* Expandable detail */}
              <AnimatePresence initial={false}>
                {isOpen && (
                  <motion.div
                    key="content"
                    variants={expandVariants}
                    initial="hidden"
                    animate="visible"
                    exit="exit"
                    className="overflow-hidden"
                  >
                    <div className="border-t border-border px-6 pb-6 pt-5 space-y-5">
                      {/* Summary */}
                      <p className="text-sm text-muted-foreground">{model.detail.summary}</p>

                      {/* Diagram */}
                      <div className="rounded-lg bg-muted/40 border border-border p-4">
                        {model.detail.diagram}
                      </div>

                      {/* Step-by-step breakdown */}
                      <div className="grid gap-3 sm:grid-cols-2">
                        {model.detail.steps.map((step, j) => (
                          <div
                            key={j}
                            className="rounded-lg border border-border bg-background/50 px-4 py-3"
                          >
                            <p className={`text-xs font-semibold mb-1 ${model.color}`}>
                              {step.title}
                            </p>
                            <p className="text-xs text-muted-foreground leading-relaxed">
                              {step.text}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

function ArchitectureCrypto() {
  const [openId, setOpenId] = useState<string | null>(null);
  const model = cryptoFfModel;
  const isOpen = openId === model.id;
  const Icon = model.icon;

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
        <p className="section-label mb-2">Architecture</p>
        <h1 className="section-title mb-1">Modèle crypto Fama-French</h1>
        <p className="mb-10 text-sm text-muted-foreground">
          Description du pipeline implémenté dans gestion/crypto/crypto_fama_french.py et exposé sur le site via l’API (données CSV, facteurs CMKT / SIZE / MOM, OLS, Markowitz).
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="glass-card mb-12 p-6 md:p-8"
      >
        <h2 className="font-display text-base font-bold text-foreground mb-6">
          Pipeline (mode cryptos)
        </h2>
        <div className="flex items-stretch justify-center gap-0">
          <div className="flex flex-col items-center justify-center rounded-lg border border-primary/40 bg-primary/5 px-6 py-6 text-center flex-1">
            <Database className="h-6 w-6 text-primary mb-3" />
            <span className="text-sm font-semibold text-foreground">Données</span>
            <span className="text-xs text-muted-foreground mt-1.5">CSV CoinGecko<br />gestion/crypto/données</span>
          </div>
          <div className="flex items-center px-2 text-muted-foreground shrink-0">
            <svg width="32" height="16" viewBox="0 0 32 16" fill="none">
              <path d="M0 8h26M20 2l8 6-8 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div className="flex flex-col items-center justify-center rounded-lg border border-border bg-background px-6 py-6 text-center flex-1">
            <Brain className="h-6 w-6 text-primary mb-3" />
            <span className="text-sm font-semibold text-foreground">Modélisation</span>
            <span className="text-xs text-muted-foreground mt-1.5">Facteurs CMKT, SIZE, MOM<br />OLS · μ̂ et Σ · Sharpe max</span>
          </div>
          <div className="flex items-center px-2 text-muted-foreground shrink-0">
            <svg width="32" height="16" viewBox="0 0 32 16" fill="none">
              <path d="M0 8h26M20 2l8 6-8 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div className="flex flex-col items-center justify-center rounded-lg border border-primary/40 bg-primary/5 px-6 py-6 text-center flex-1">
            <BarChart3 className="h-6 w-6 text-primary mb-3" />
            <span className="text-sm font-semibold text-foreground">Résultats</span>
            <span className="text-xs text-muted-foreground mt-1.5">Backtest 20 % · vs moyenne de marché</span>
          </div>
        </div>
        <div className="mt-5 rounded-lg border border-border bg-muted/30 px-5 py-3">
          <p className="text-xs text-muted-foreground text-center">
            <span className="font-semibold text-foreground">Spécificité crypto :</span> pas de facteurs Ken French — tout est endogène à l’univers des cryptos chargées ; un test d’endogénéité (Durbin-Wu-Hausman) est prévu dans le script complet.
          </p>
        </div>
      </motion.div>

      <div className="flex flex-col gap-4">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fade}
          custom={0}
          className={`glass-card overflow-hidden border ${model.borderColor}`}
        >
          <button
            type="button"
            onClick={() => setOpenId((p) => (p === model.id ? null : model.id))}
            className="w-full flex items-center justify-between gap-4 p-6 text-left hover:bg-muted/20 transition-colors"
          >
            <div className="flex items-center gap-4">
              <Icon className={`h-6 w-6 shrink-0 ${model.color}`} />
              <div>
                <h3 className="font-display text-base font-bold text-foreground">{model.name}</h3>
                <p className="text-sm text-muted-foreground">{model.desc}</p>
              </div>
            </div>
            <div className="shrink-0 text-muted-foreground">
              {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </div>
          </button>
          <AnimatePresence initial={false}>
            {isOpen && (
              <motion.div
                key="content"
                variants={expandVariants}
                initial="hidden"
                animate="visible"
                exit="exit"
                className="overflow-hidden"
              >
                <div className="border-t border-border px-6 pb-6 pt-5 space-y-5">
                  <p className="text-sm text-muted-foreground">{model.detail.summary}</p>
                  <div className="rounded-lg bg-muted/40 border border-border p-4">{model.detail.diagram}</div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    {model.detail.steps.map((step, j) => (
                      <div key={j} className="rounded-lg border border-border bg-background/50 px-4 py-3">
                        <p className={`text-xs font-semibold mb-1 ${model.color}`}>{step.title}</p>
                        <p className="text-xs text-muted-foreground leading-relaxed">{step.text}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </div>
  );
}

const Architecture = () => {
  const { mode } = useAppMode();
  return mode === "crypto" ? <ArchitectureCrypto /> : <ArchitectureActions />;
};

export default Architecture;
