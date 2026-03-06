import { useState, useRef } from "react";
import { useLocation, Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Play, BarChart3, TrendingUp, ChevronDown, ChevronUp, Brain,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend, ReferenceDot,
} from "recharts";
import {
  runSimulation, runLlmSimulationStream,
  type SimulateResult, type LlmSimulateResult, type LlmMonthStep, type LlmPromptExample,
  type LlmProgressEvent,
} from "@/lib/api";
import { loadSavedSymbols } from "@/lib/portfolioStorage";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const models = [
  {
    id: "markowitz-classic",
    name: "Markowitz classique",
    desc: "Optimisation moyenne-variance avec matrice de covariance historique.",
  },
  {
    id: "markowitz-1factor",
    name: "Un facteur de risque (CAPM)",
    desc: "Rendements espérés via le modèle à un facteur (prime de marché).",
  },
  {
    id: "markowitz-3factors",
    name: "Trois facteurs (Fama & French)",
    desc: "Rendements espérés avec Mkt-RF, SMB et HML.",
  },
  {
    id: "markowitz-5factors",
    name: "Cinq facteurs (Fama & French)",
    desc: "Rendements espérés avec Mkt-RF, SMB, HML, RMW et CMA.",
    badge: "Bientôt",
  },
  {
    id: "markowitz-llm",
    name: "Choix dynamique des facteurs",
    desc: "Sélection mensuelle des facteurs par LLM selon l'actualité économique.",
    badge: "IA",
  },
];

const FACTOR_COLORS: Record<string, string> = {
  "Mkt-RF":    "#3b82f6",
  SMB:         "#8b5cf6",
  HML:         "#f59e0b",
  RMW:         "#10b981",
  CMA:         "#ef4444",
  UMD:         "#06b6d4",
  HY_SPREAD:   "#f97316",
  TERM_SPREAD: "#a855f7",
  VIX:         "#ec4899",
};

const ALL_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "HY_SPREAD", "TERM_SPREAD", "VIX"];

const SENTIMENT: Record<string, { dot: string; label: string }> = {
  positif: { dot: "bg-emerald-400", label: "Positif" },
  neutre: { dot: "bg-yellow-400", label: "Neutre" },
  "négatif": { dot: "bg-red-400", label: "Négatif" },
};

const INITIAL_VALUE = 10_000;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function yAxisLabel(text: string) {
  return {
    value: text,
    angle: -90 as const,
    position: "insideLeft" as const,
    style: { fontSize: 11, fill: "hsl(var(--muted-foreground))" },
    content: (props: { viewBox?: { x?: number; y?: number; width?: number; height?: number } }) => {
      const { viewBox } = props;
      if (!viewBox || viewBox.height == null) return null;
      const x = (viewBox.x ?? 0) + (viewBox.width ?? 0) / 2 - 14;
      const y = (viewBox.y ?? 0) + viewBox.height / 2;
      return (
        <text x={x} y={y} textAnchor="middle" dominantBaseline="middle"
          transform={`rotate(-90, ${x}, ${y})`}
          style={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}>
          {text}
        </text>
      );
    },
  };
}

function tooltipStyle() {
  return {
    contentStyle: {
      backgroundColor: "hsl(var(--card))",
      border: "1px solid hsl(var(--border))",
      borderRadius: 8,
      fontSize: 12,
    },
  };
}

function domainFromValues(vals: number[]): [number, number] | undefined {
  if (!vals.length) return undefined;
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const pad = Math.max(5, (hi - lo) * 0.05);
  return [Math.max(0, Math.floor((lo - pad) / 5) * 5), Math.ceil((hi + pad) / 5) * 5];
}

// ---------------------------------------------------------------------------
// Accordion
// ---------------------------------------------------------------------------

function Accordion({ title, children, defaultOpen = false }: {
  title: React.ReactNode; children: React.ReactNode; defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-border rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen((p) => !p)}
        className="w-full flex items-center justify-between px-4 py-3 text-xs font-semibold text-foreground bg-muted/30 hover:bg-muted/60 transition-colors"
      >
        <span className="flex-1 min-w-0">{title}</span>
        {open
          ? <ChevronUp className="h-3.5 w-3.5 shrink-0 ml-2 text-muted-foreground" />
          : <ChevronDown className="h-3.5 w-3.5 shrink-0 ml-2 text-muted-foreground" />}
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="overflow-hidden"
          >
            <div className="px-4 py-3">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Factor badges (masque booléen actif / inactif)
// ---------------------------------------------------------------------------

/** Badge pour un facteur : actif (coloré) ou inactif (grisé). */
function FactorBadge({ factor, active }: { factor: string; active: boolean }) {
  const color = FACTOR_COLORS[factor] ?? "#94a3b8";
  return (
    <span
      className="inline-flex items-center rounded px-1.5 py-0.5 text-[9px] font-bold border"
      style={
        active
          ? { color, borderColor: color, background: `${color}18` }
          : { color: "hsl(var(--muted-foreground))", borderColor: "hsl(var(--border))", opacity: 0.45 }
      }
    >
      {factor}
    </span>
  );
}

/** Grille compacte des 5 facteurs pour l'en-tête d'un accordéon. */
function FactorBadgesCompact({ factorMask }: { factorMask: Record<string, boolean> }) {
  return (
    <div className="flex flex-wrap gap-1 min-w-[150px]">
      {ALL_FACTORS.map((f) => (
        <FactorBadge key={f} factor={f} active={factorMask[f] ?? true} />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Classic model result
// ---------------------------------------------------------------------------

function ClassicResult({ result, chartStart, chartEnd, setChartStart, setChartEnd }: {
  result: SimulateResult;
  chartStart: string; chartEnd: string;
  setChartStart: (v: string) => void; setChartEnd: (v: string) => void;
}) {
  const hasPeriods = !!(result.trainPeriodStart && result.trainPeriodEnd && result.testPeriodStart && result.testPeriodEnd);
  const filtered = result.comparisonData.filter(
    (d) => (!chartStart || d.date >= chartStart) && (!chartEnd || d.date <= chartEnd)
  );
  type CP = { date: string; portfolio: number; market: number; portfolioTrain?: number; portfolioBacktest?: number };
  const chartData: CP[] = hasPeriods
    ? filtered.map((d) => ({
        ...d,
        portfolioTrain: d.date >= result.trainPeriodStart! && d.date <= result.trainPeriodEnd! ? d.portfolio : undefined,
        portfolioBacktest: d.date >= result.testPeriodStart! && d.date <= result.testPeriodEnd! ? d.portfolio : undefined,
      }))
    : filtered;

  const vals = chartData.flatMap((d) =>
    [d.portfolio, d.portfolioTrain, d.portfolioBacktest, d.market].filter((v): v is number => v != null)
  );

  return (
    <div className="space-y-8">
      {(result.numPortfolios != null || result.trainPeriodStart != null) && (
        <p className="text-[11px] text-muted-foreground/80">
          {result.numPortfolios != null && <span>{result.numPortfolios.toLocaleString("fr-FR")} portefeuilles générés. </span>}
          {result.trainPeriodStart && result.trainPeriodEnd && <>Entraînement 80 % : {result.trainPeriodStart} → {result.trainPeriodEnd}. </>}
          {result.testPeriodStart && result.testPeriodEnd && <>Test 20 % : {result.testPeriodStart} → {result.testPeriodEnd}.</>}
        </p>
      )}

      {/* Allocation */}
      <div className="glass-card p-6">
        <h3 className="font-display text-sm font-bold text-foreground mb-4">Allocation optimale</h3>
        <div className="space-y-2">
          {Object.entries(result.weights).map(([sym, w]) => (
            <div key={sym} className="flex items-center gap-3">
              <span className="w-14 text-xs font-semibold text-foreground">{sym}</span>
              <div className="flex-1 rounded-full bg-secondary h-3 overflow-hidden">
                <motion.div initial={{ width: 0 }} animate={{ width: `${w * 100}%` }}
                  transition={{ duration: 0.8, delay: 0.2 }} className="h-full rounded-full bg-primary" />
              </div>
              <span className="w-12 text-right text-xs font-medium text-muted-foreground">{(w * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* KPIs */}
      <div className="grid gap-4 md:grid-cols-4">
        {[
          { label: "Ratio de Sharpe", value: result.sharpe.toFixed(2), icon: BarChart3 },
          { label: "Rendement attendu", value: `${result.expectedReturn.toFixed(1)}%`, icon: TrendingUp },
          { label: "Volatilité", value: `${result.volatility.toFixed(1)}%`, icon: BarChart3 },
          { label: "Max Drawdown", value: `-${result.maxDrawdown.toFixed(1)}%`, icon: TrendingUp },
        ].map((kpi) => (
          <div key={kpi.label} className="glass-card p-5 text-center">
            <kpi.icon className="mx-auto h-5 w-5 text-primary" />
            <p className="mt-2 font-display text-xl font-bold text-foreground">{kpi.value}</p>
            <p className="mt-1 text-xs text-muted-foreground">{kpi.label}</p>
          </div>
        ))}
      </div>

      {/* Courbe */}
      <div className="glass-card p-6">
        <h3 className="font-display text-sm font-bold text-foreground mb-4">Performance : Portefeuille vs Marché</h3>
        <div className="flex items-center justify-end gap-3 flex-wrap mb-4">
          <label className="text-xs text-muted-foreground">Du <input type="date" value={chartStart} onChange={(e) => setChartStart(e.target.value)} className="bg-background border border-input rounded px-2 py-1.5 text-foreground text-xs" /></label>
          <label className="text-xs text-muted-foreground">Au <input type="date" value={chartEnd} onChange={(e) => setChartEnd(e.target.value)} className="bg-background border border-input rounded px-2 py-1.5 text-foreground text-xs" /></label>
        </div>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ left: 8, right: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 12 }} stroke="hsl(var(--muted-foreground))" domain={domainFromValues(vals)} tickFormatter={(v) => `${v}`} label={yAxisLabel("Indice (base 100)")} />
              <Tooltip {...tooltipStyle()} formatter={(v: number, n: string) => [v != null ? Number(v).toFixed(1) : "—", n]} labelFormatter={(l) => `Date : ${l}`} />
              <Legend />
              {hasPeriods ? (
                <>
                  <Line type="monotone" dataKey="portfolioTrain" name="Portefeuille (entraînement)" stroke="#3b82f6" strokeWidth={2} dot={false} connectNulls={false} />
                  <Line type="monotone" dataKey="portfolioBacktest" name="Portefeuille (backtest)" stroke="#ef4444" strokeWidth={2} dot={false} connectNulls={false} />
                </>
              ) : (
                <Line type="monotone" dataKey="portfolio" name="Portefeuille" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} />
              )}
              <Line type="monotone" dataKey="market" name="Marché (S&P 500)" stroke="hsl(var(--muted-foreground))" strokeWidth={2} dot={false} strokeDasharray="5 5" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-[10px] text-muted-foreground/70">Courbes normalisées à 100 au premier jour du backtest. Marché : ETF SPY.</p>
      </div>

      {/* Frontière efficiente */}
      {result.efficientFrontier != null && result.efficientFrontier.length > 0 && (
        <div className="glass-card p-6">
          <h3 className="font-display text-sm font-bold text-foreground mb-4">Frontière efficiente</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={[...result.efficientFrontier].sort((a, b) => a.volatility - b.volatility)} margin={{ left: 20, right: 16, top: 8, bottom: 24 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="volatility" type="number" tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" domain={["auto", "auto"]} tickFormatter={(v) => `${v} %`}
                  label={{ value: "Volatilité (%)", position: "insideBottom", offset: -12, style: { fontSize: 11, fill: "hsl(var(--muted-foreground))" } }} />
                <YAxis dataKey="expectedReturn" type="number" tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" domain={["auto", "auto"]} tickFormatter={(v) => `${v} %`} label={yAxisLabel("Rendement attendu (%)")} />
                <Tooltip {...tooltipStyle()} content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const p = payload[0]?.payload as { volatility: number; expectedReturn: number; sharpe?: number; backtestReturn?: number };
                  return (
                    <div className="rounded-lg border border-border bg-card px-3 py-2 text-xs shadow-sm">
                      <p>Volatilité : {p?.volatility?.toFixed(2) ?? "—"} %</p>
                      <p>Rendement attendu : {p?.expectedReturn?.toFixed(2) ?? "—"} %</p>
                      {p?.backtestReturn != null && <p>Rendement réel : {p.backtestReturn.toFixed(2)} %</p>}
                      {p?.sharpe != null && <p>Sharpe : {p.sharpe.toFixed(2)}</p>}
                    </div>
                  );
                }} />
                <Line type="monotone" dataKey="expectedReturn" stroke="hsl(var(--primary))" strokeWidth={2} dot={{ fill: "hsl(var(--primary))", r: 4 }} name="Frontière efficiente" />
                <ReferenceDot x={result.volatility} y={result.expectedReturn} r={6} fill="hsl(var(--primary))" stroke="hsl(var(--foreground))" strokeWidth={2} label={{ value: "Optimal", position: "top", fontSize: 10 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Prompt explorer (tous les prompts, navigables par ticker puis mois)
// ---------------------------------------------------------------------------

function PromptExplorer({ examples }: { examples: LlmPromptExample[] }) {
  const tickers = Array.from(new Set(examples.map((e) => e.ticker))).sort();
  const [activeTicker, setActiveTicker] = useState<string>(tickers[0] ?? "");

  const monthsForTicker = examples
    .filter((e) => e.ticker === activeTicker)
    .map((e) => e.month)
    .sort();
  const [activeMonth, setActiveMonth] = useState<string>(monthsForTicker[0] ?? "");

  // Sync month when ticker changes
  const handleTickerChange = (t: string) => {
    setActiveTicker(t);
    const first = examples.filter((e) => e.ticker === t).map((e) => e.month).sort()[0] ?? "";
    setActiveMonth(first);
  };

  const example = examples.find((e) => e.ticker === activeTicker && e.month === activeMonth);

  return (
    <div className="space-y-3">
      {/* Sélecteur ticker */}
      <div className="flex flex-wrap gap-1.5 items-center">
        <span className="text-[9px] font-semibold uppercase tracking-widest text-muted-foreground w-10 shrink-0">Action</span>
        {tickers.map((t) => (
          <button
            key={t}
            onClick={() => handleTickerChange(t)}
            className={`rounded-full px-2.5 py-1 text-[10px] font-semibold transition-colors ${
              activeTicker === t
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-muted-foreground hover:bg-muted/80"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Sélecteur mois */}
      <div className="flex flex-wrap gap-1.5 items-center">
        <span className="text-[9px] font-semibold uppercase tracking-widest text-muted-foreground w-10 shrink-0">Mois</span>
        {monthsForTicker.map((m) => (
          <button
            key={m}
            onClick={() => setActiveMonth(m)}
            className={`rounded-full px-2.5 py-1 text-[10px] font-mono transition-colors ${
              activeMonth === m
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-muted-foreground hover:bg-muted/80"
            }`}
          >
            {m}
          </button>
        ))}
      </div>

      {/* Contenu du prompt sélectionné */}
      {example && (
        <div className="space-y-2 pt-1">
          <div className="flex items-center gap-2 pb-1">
            <span className="text-[9px] font-mono text-muted-foreground/60">{example.provider}</span>
          </div>
          {[
            { label: "Système", content: example.system, highlight: false },
            { label: "Utilisateur", content: example.user, highlight: false },
            { label: "Réponse", content: example.response, highlight: true },
          ].map(({ label, content, highlight }) => (
            <div key={label}>
              <p className="text-[9px] font-semibold uppercase tracking-widest text-muted-foreground mb-1">{label}</p>
              <pre
                className={`whitespace-pre-wrap font-mono text-[10px] rounded-lg p-3 leading-relaxed overflow-x-auto ${
                  highlight
                    ? "text-foreground bg-primary/5 border border-primary/20"
                    : "text-muted-foreground bg-muted/30"
                }`}
              >
                {content}
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// LLM model result
// ---------------------------------------------------------------------------

function LlmResult({ result, classicResult }: {
  result: LlmSimulateResult;
  classicResult: SimulateResult | null;
}) {
  const [selectedMonth, setSelectedMonth] = useState<string>(
    result.monthlyHistory.length > 0 ? result.monthlyHistory[result.monthlyHistory.length - 1].month : ""
  );

  const step: LlmMonthStep | undefined = result.monthlyHistory.find((s) => s.month === selectedMonth);

  // Courbe $10k : fusion LLM + classique converti en $10k + marché
  // Classic dates: YYYY-MM-DD  |  LLM dates: YYYY-MM  → normalise via YYYY-MM prefix
  const classicMap = new Map<string, number>();
  if (classicResult) {
    const cd = classicResult.comparisonData;
    const testStart = result.testPeriodStart; // "YYYY-MM"
    // Classic points whose YYYY-MM prefix >= testStart
    const testPts = cd.filter((d) => d.date.slice(0, 7) >= testStart);
    if (testPts.length > 0) {
      const base = testPts[0].portfolio;
      testPts.forEach((d) => {
        // Store under YYYY-MM key so LLM mergedCurve lookup works
        classicMap.set(d.date.slice(0, 7), INITIAL_VALUE * (d.portfolio / base));
      });
    }
  }

  const mergedCurve = result.comparisonData.map((pt) => ({
    date: pt.date,
    llm: pt.portfolio,
    market: pt.market != null ? INITIAL_VALUE * (pt.market / (result.comparisonData[0].market ?? 1)) : undefined,
    classic: classicMap.get(pt.date),
  }));

  const allVals = mergedCurve.flatMap((d) => [d.llm, d.market, d.classic].filter((v): v is number => v != null));
  const domY = domainFromValues(allVals);

  // KPIs finaux
  const classicFinal = classicResult
    ? (() => {
        const testStart = result.testPeriodStart;
        const pts = classicResult.comparisonData.filter((d) => d.date >= testStart);
        if (pts.length < 2) return null;
        const base = pts[0].portfolio;
        const last = pts[pts.length - 1].portfolio;
        return { totalReturn: (last / base - 1) * 100, maxDrawdown: classicResult.maxDrawdown };
      })()
    : null;

  const spyFinalValue = mergedCurve.length > 0 ? mergedCurve[mergedCurve.length - 1].market : undefined;
  const spyTotalReturn = spyFinalValue != null ? (spyFinalValue / INITIAL_VALUE - 1) * 100 : null;

  return (
    <div className="space-y-8">
      {/* Méta */}
      <p className="text-[11px] text-muted-foreground/80">
        Backtest glissant sur {result.numMonths} mois.{" "}
        Entraînement : {result.trainPeriodStart} → {result.trainPeriodEnd}.{" "}
        Test : {result.testPeriodStart} → {result.testPeriodEnd}.
      </p>

      {/* KPIs LLM */}
      <div className="grid gap-4 md:grid-cols-3">
        {[
          { label: "Valeur finale (LLM)", value: `$${result.finalValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}`, sub: `+${result.totalReturn.toFixed(1)}% sur la période`, positive: result.totalReturn >= 0 },
          { label: "Max Drawdown", value: `-${result.maxDrawdown.toFixed(1)}%`, sub: "Perte maximale observée", positive: false },
          { label: "Mois backtestés", value: String(result.numMonths), sub: "Recalcul LLM chaque mois", positive: true },
        ].map((kpi) => (
          <div key={kpi.label} className="glass-card p-5 text-center">
            <p className={`font-display text-xl font-bold ${kpi.positive ? "text-foreground" : "text-muted-foreground"}`}>{kpi.value}</p>
            <p className="mt-1 text-xs font-semibold text-foreground">{kpi.label}</p>
            <p className="mt-0.5 text-[10px] text-muted-foreground">{kpi.sub}</p>
          </div>
        ))}
      </div>

      {/* Courbe $10 000 */}
      <div className="glass-card p-6">
        <h3 className="font-display text-sm font-bold text-foreground mb-1">
          Performance — base $10 000
        </h3>
        <p className="text-[11px] text-muted-foreground mb-4">
          Valeur d'un portefeuille de $10 000 investi au début de la période de test, rebalancé chaque mois selon les décisions du LLM.
        </p>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={mergedCurve} margin={{ left: 8, right: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 12 }} stroke="hsl(var(--muted-foreground))" domain={domY} tickFormatter={(v) => `$${v.toLocaleString("en-US", { maximumFractionDigits: 0 })}`} label={yAxisLabel("Valeur ($)")} width={72} />
              <Tooltip
                {...tooltipStyle()}
                formatter={(v: number, n: string) => [`$${v != null ? Number(v).toLocaleString("en-US", { maximumFractionDigits: 0 }) : "—"}`, n]}
                labelFormatter={(l) => `Mois : ${l}`}
              />
              <Legend />
              <Line type="monotone" dataKey="llm" name="LLM dynamique" stroke="hsl(var(--primary))" strokeWidth={2.5} dot={false} />
              {classicResult && (
                <Line type="monotone" dataKey="classic" name="Markowitz classique" stroke="#94a3b8" strokeWidth={2} dot={false} strokeDasharray="5 3" />
              )}
              <Line type="monotone" dataKey="market" name="Marché (S&P 500)" stroke="#64748b" strokeWidth={1.5} dot={false} strokeDasharray="2 4" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-[10px] text-muted-foreground/70">
          Marché : ETF SPY rebased à $10 000. Le LLM rebalance le portefeuille à chaque début de mois.
        </p>
      </div>

      {/* Tableau de comparaison finale */}
      {(classicFinal || spyTotalReturn != null) && (
        <div className="glass-card p-6">
          <h3 className="font-display text-sm font-bold text-foreground mb-4">Comparaison des performances</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="pb-2 text-left text-muted-foreground font-semibold">Modèle</th>
                  <th className="pb-2 text-right text-muted-foreground font-semibold">Valeur finale</th>
                  <th className="pb-2 text-right text-muted-foreground font-semibold">Rendement total</th>
                  <th className="pb-2 text-right text-muted-foreground font-semibold">Max Drawdown</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/40">
                <tr>
                  <td className="py-2 font-semibold text-primary flex items-center gap-1.5"><Brain className="h-3 w-3" /> LLM dynamique</td>
                  <td className="py-2 text-right font-semibold">${result.finalValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}</td>
                  <td className={`py-2 text-right font-semibold ${result.totalReturn >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                    {result.totalReturn >= 0 ? "+" : ""}{result.totalReturn.toFixed(1)}%
                  </td>
                  <td className="py-2 text-right text-muted-foreground">-{result.maxDrawdown.toFixed(1)}%</td>
                </tr>
                {classicFinal && (
                  <tr>
                    <td className="py-2 text-muted-foreground">Markowitz classique</td>
                    <td className="py-2 text-right">
                      ${(INITIAL_VALUE * (1 + classicFinal.totalReturn / 100)).toLocaleString("en-US", { maximumFractionDigits: 0 })}
                    </td>
                    <td className={`py-2 text-right ${classicFinal.totalReturn >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {classicFinal.totalReturn >= 0 ? "+" : ""}{classicFinal.totalReturn.toFixed(1)}%
                    </td>
                    <td className="py-2 text-right text-muted-foreground">-{classicFinal.maxDrawdown.toFixed(1)}%</td>
                  </tr>
                )}
                {spyTotalReturn != null && spyFinalValue != null && (
                  <tr>
                    <td className="py-2 text-muted-foreground">Marché (S&P 500)</td>
                    <td className="py-2 text-right">${spyFinalValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}</td>
                    <td className={`py-2 text-right ${spyTotalReturn >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {spyTotalReturn >= 0 ? "+" : ""}{spyTotalReturn.toFixed(1)}%
                    </td>
                    <td className="py-2 text-right text-muted-foreground">—</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Prompts LLM — tous les appels, navigables */}
      {result.promptExamples && result.promptExamples.length > 0 && (
        <Accordion title={
          <span className="flex items-center gap-2">
            <Brain className="h-3.5 w-3.5 text-primary shrink-0" />
            <span>Prompts envoyés au LLM</span>
            <span className="ml-1 font-mono text-[9px] text-muted-foreground">
              {result.promptExamples.length} appel{result.promptExamples.length > 1 ? "s" : ""}
            </span>
          </span>
        }>
          <PromptExplorer examples={result.promptExamples} />
        </Accordion>
      )}

      {/* Historique mensuel : onglets déroulants */}
      <div className="glass-card p-6">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="h-4 w-4 text-primary" />
          <h3 className="font-display text-sm font-bold text-foreground">Décisions mensuelles du LLM</h3>
        </div>
        <p className="text-[11px] text-muted-foreground mb-4">
          Chaque mois, le LLM sélectionne les facteurs et recalcule l'allocation optimale. Cliquez sur un mois pour voir le détail.
        </p>

        {/* Sélecteur de mois */}
        <div className="flex flex-wrap gap-1.5 mb-4">
          {result.monthlyHistory.map((s) => (
            <button
              key={s.month}
              onClick={() => setSelectedMonth(s.month)}
              className={`rounded-full px-2.5 py-1 text-[10px] font-semibold transition-colors ${
                selectedMonth === s.month
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground hover:bg-muted/80"
              }`}
            >
              {s.month}
            </button>
          ))}
        </div>

        {/* Détail du mois sélectionné */}
        {step && (
          <div className="space-y-3">
            {/* Allocation du mois */}
            <div className="rounded-xl border border-border bg-muted/20 p-4">
              <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-2">Allocation — {step.month}</p>
              <div className="space-y-1.5">
                {Object.entries(step.weights).map(([sym, w]) => (
                  <div key={sym} className="flex items-center gap-2">
                    <span className="w-12 text-xs font-semibold text-foreground">{sym}</span>
                    <div className="flex-1 rounded-full bg-secondary h-2 overflow-hidden">
                      <div className="h-full rounded-full bg-primary" style={{ width: `${w * 100}%` }} />
                    </div>
                    <span className="w-10 text-right text-[10px] text-muted-foreground">{(w * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
              {step.sharpe != null && (
                <div className="flex gap-4 mt-3 pt-3 border-t border-border/50">
                  <span className="text-[10px] text-muted-foreground">Sharpe estimé : <span className="font-semibold text-foreground">{step.sharpe.toFixed(2)}</span></span>
                  {step.expectedReturn != null && <span className="text-[10px] text-muted-foreground">Rendement attendu : <span className="font-semibold text-foreground">{step.expectedReturn.toFixed(1)}%</span></span>}
                  {step.volatility != null && <span className="text-[10px] text-muted-foreground">Volatilité : <span className="font-semibold text-foreground">{step.volatility.toFixed(1)}%</span></span>}
                </div>
              )}
            </div>

            {/* Facteurs (masque booléen) + news par ticker */}
            <div className="space-y-2">
              {Object.keys(step.weights).map((ticker) => {
                const factorMask: Record<string, boolean> =
                  (step.selectedFactors?.[ticker] as Record<string, boolean> | undefined) ??
                  Object.fromEntries(ALL_FACTORS.map((f) => [f, true]));
                const news = step.newsSummaries?.[ticker];
                const sent = SENTIMENT[news?.sentiment ?? "neutre"] ?? SENTIMENT["neutre"];
                return (
                  <Accordion
                    key={ticker}
                    title={
                      <span className="flex items-center gap-3 min-w-0 w-full">
                        <span className="font-bold shrink-0">{ticker}</span>
                        <span className="flex-1 min-w-0">
                          <FactorBadgesCompact factorMask={factorMask} />
                        </span>
                        <span className="flex items-center gap-1 shrink-0">
                          <span className={`h-1.5 w-1.5 rounded-full ${sent.dot}`} />
                          <span className="text-[10px] text-muted-foreground">{sent.label}</span>
                        </span>
                      </span>
                    }
                  >
                    {news?.summary
                      ? <p className="text-[11px] text-muted-foreground leading-relaxed">{news.summary}</p>
                      : <p className="text-[11px] text-muted-foreground italic">Résumé indisponible.</p>}
                  </Accordion>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Timeline facteurs — présence/absence LLM par mois */}
      <div className="glass-card p-6">
        <h3 className="font-display text-sm font-bold text-foreground mb-1">Sélection LLM par facteur — évolution mensuelle</h3>
        <p className="text-[11px] text-muted-foreground mb-3">
          Pour chaque mois et chaque facteur, indique si le LLM l'a jugé pertinent (actif) ou non (inactif) — agrégé sur l'ensemble des actions du portefeuille.
        </p>
        <div className="overflow-x-auto">
          <table className="text-[10px] w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="pb-1.5 text-left text-muted-foreground font-semibold pr-3 w-20">Mois</th>
                {ALL_FACTORS.map((f) => (
                  <th key={f} className="pb-1.5 text-center font-semibold px-3" style={{ color: FACTOR_COLORS[f] }}>{f}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border/30">
              {result.monthlyHistory.map((s) => {
                const tickers = Object.keys(s.selectedFactors ?? {});
                // Pour chaque facteur, compte combien de tickers l'ont activé
                const activeCounts: Record<string, number> = {};
                for (const f of ALL_FACTORS) {
                  if (tickers.length === 0) { activeCounts[f] = 0; continue; }
                  activeCounts[f] = tickers.filter((t) => {
                    const fw = s.selectedFactors?.[t];
                    if (!fw) return true;
                    return (fw as Record<string, boolean>)[f] !== false;
                  }).length;
                }
                return (
                  <tr
                    key={s.month}
                    onClick={() => setSelectedMonth(s.month)}
                    className={`cursor-pointer transition-colors hover:bg-muted/30 ${selectedMonth === s.month ? "bg-muted/40" : ""}`}
                  >
                    <td className="py-1.5 pr-3 font-mono text-muted-foreground">{s.month}</td>
                    {ALL_FACTORS.map((f) => {
                      const count = activeCounts[f] ?? 0;
                      const total = tickers.length || 1;
                      const allActive = count === total;
                      const noneActive = count === 0;
                      const color = FACTOR_COLORS[f] ?? "#94a3b8";
                      return (
                        <td key={f} className="py-1.5 text-center px-3">
                          {allActive ? (
                            <span
                              className="inline-flex items-center justify-center rounded px-1.5 py-0.5 text-[8px] font-bold border"
                              style={{ color, borderColor: color, background: `${color}18` }}
                            >
                              ✓ {count}/{total}
                            </span>
                          ) : noneActive ? (
                            <span className="inline-flex items-center justify-center rounded px-1.5 py-0.5 text-[8px] font-bold border border-border text-muted-foreground opacity-40">
                              ✗ 0/{total}
                            </span>
                          ) : (
                            <span
                              className="inline-flex items-center justify-center rounded px-1.5 py-0.5 text-[8px] font-bold border"
                              style={{ color, borderColor: color, background: `${color}10`, opacity: 0.7 }}
                            >
                              {count}/{total}
                            </span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="flex flex-wrap gap-4 mt-3 pt-3 border-t border-border/50 items-center">
          {ALL_FACTORS.map((f) => (
            <span key={f} className="flex items-center gap-1 text-[10px] text-muted-foreground">
              <span className="h-2 w-3 rounded-sm inline-block" style={{ background: FACTOR_COLORS[f] }} /> {f}
            </span>
          ))}
          <span className="text-[10px] text-muted-foreground ml-auto">
            ✓ = facteur actif pour tous les tickers · X = facteur inactif pour tous
          </span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page principale
// ---------------------------------------------------------------------------

const Simulation = () => {
  const location = useLocation();
  const symbolsFromState = (location.state as { symbols?: string[] } | null)?.symbols;
  const symbols: string[] = symbolsFromState?.length ? symbolsFromState : loadSavedSymbols();

  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SimulateResult | null>(null);
  const [llmResult, setLlmResult] = useState<LlmSimulateResult | null>(null);
  const [classicResult, setClassicResult] = useState<SimulateResult | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [chartStart, setChartStart] = useState("");
  const [chartEnd, setChartEnd] = useState("");
  const [llmProgress, setLlmProgress] = useState<LlmProgressEvent | null>(null);
  const cancelStreamRef = useRef<(() => void) | null>(null);

  const canRun = selectedModel != null && symbols.length >= 2;
  const isLlm = selectedModel === "markowitz-llm";

  const handleRun = () => {
    if (!canRun) return;
    setApiError(null);
    setRunning(true);
    setResult(null);
    setLlmResult(null);
    setClassicResult(null);
    setLlmProgress(null);

    if (isLlm) {
      // Markowitz classique en parallèle (non-streaming)
      runSimulation("markowitz-classic", symbols)
        .then((classic) => setClassicResult(classic))
        .catch(() => { /* classique optionnel */ });

      // LLM via SSE streaming
      const cancel = runLlmSimulationStream(
        symbols,
        (evt) => setLlmProgress(evt),
        (llm) => {
          setLlmResult(llm);
          setRunning(false);
          setLlmProgress(null);
          cancelStreamRef.current = null;
        },
        (msg) => {
          setApiError(msg);
          setRunning(false);
          setLlmProgress(null);
          cancelStreamRef.current = null;
        },
      );
      cancelStreamRef.current = cancel;
    } else {
      runSimulation(selectedModel!, symbols)
        .then((res) => {
          setResult(res);
          if (res.comparisonData.length > 0) {
            setChartStart(res.comparisonData[0].date);
            setChartEnd(res.comparisonData[res.comparisonData.length - 1].date);
          }
        })
        .catch((e) => setApiError(e.message))
        .finally(() => setRunning(false));
    }
  };

  return (
    <div className="px-6 py-10">
      <div>
        <p className="section-label mb-2">Simulation</p>
        <h1 className="section-title mb-1">Optimisez votre portefeuille</h1>
        <p className="mb-8 text-sm text-muted-foreground">
          Choisissez un modèle puis lancez l'optimisation. Les résultats incluent un backtesting sur 20 % des données historiques.
        </p>
      </div>

      {symbols.length < 2 && (
        <div className="mb-6 p-4 rounded-xl bg-muted/50 border border-border">
          <p className="text-xs text-muted-foreground">
            Sélectionnez au moins 2 actions dans l'onglet{" "}
            <Link to="/portfolio" className="text-primary font-medium underline">Mon Portefeuille</Link>{" "}
            pour lancer une simulation.
          </p>
        </div>
      )}

      {/* Sélection du modèle */}
      <div className="mb-8 grid gap-4 md:grid-cols-3 xl:grid-cols-5">
        {models.map((m) => {
          const disabled = m.badge === "Bientôt";
          const isAi = m.badge === "IA";
          return (
            <button
              key={m.id}
              onClick={() => { if (!disabled) { setSelectedModel(m.id); setResult(null); setLlmResult(null); setClassicResult(null); setApiError(null); } }}
              disabled={disabled}
              className={`glass-card relative p-5 text-left transition-shadow focus:outline-none focus:ring-0 active:ring-0 ${
                selectedModel === m.id ? "!ring-2 !ring-primary" : ""
              } ${disabled ? "opacity-60 cursor-not-allowed" : ""}`}
            >
              {m.badge && (
                <span className={`absolute right-3 top-3 rounded-full px-2 py-0.5 text-[10px] font-medium ${
                  isAi ? "bg-primary/20 text-primary" : "bg-muted text-muted-foreground"
                }`}>
                  {m.badge}
                </span>
              )}
              <h3 className="font-display text-sm font-bold text-foreground">{m.name}</h3>
              <p className="mt-1 text-xs text-muted-foreground">{m.desc}</p>
            </button>
          );
        })}
      </div>

      <Button onClick={handleRun} disabled={!canRun || running} className="gap-2 rounded-xl font-semibold">
        {running ? (
          <>
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
            {isLlm ? "Backtest LLM en cours…" : "Optimisation en cours…"}
          </>
        ) : (
          <><Play className="h-4 w-4" /> Lancer la simulation</>
        )}
      </Button>

      {isLlm && !running && !llmResult && (
        <p className="mt-2 text-[11px] text-muted-foreground/80">
          Ce modèle interroge Mistral (news) + Claude Sonnet (sélection des facteurs via Anthropic) pour chaque mois de la période de test. Prévoyez 1–3 min selon le nombre d'actions et de mois.
        </p>
      )}

      {/* Indicateur de progression LLM */}
      <AnimatePresence>
        {isLlm && running && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            className="mt-4 glass-card p-4 space-y-3"
          >
            <div className="flex items-center gap-2">
              <Brain className="h-4 w-4 text-primary animate-pulse" />
              <span className="text-xs font-semibold text-foreground">Backtest LLM dynamique</span>
            </div>

            {llmProgress ? (
              <>
                <p className="text-xs text-muted-foreground">{llmProgress.message}</p>

                {llmProgress.type === "month" && llmProgress.total != null && llmProgress.current != null && (
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                      <span>Mois {llmProgress.current} / {llmProgress.total}</span>
                      <span className="font-mono">{llmProgress.month}</span>
                    </div>
                    <div className="w-full rounded-full bg-secondary h-2 overflow-hidden">
                      <motion.div
                        className="h-full rounded-full bg-primary"
                        initial={{ width: 0 }}
                        animate={{ width: `${(llmProgress.current / llmProgress.total) * 100}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                    <p className="text-[10px] text-muted-foreground/60 text-right">
                      {Math.round((llmProgress.current / llmProgress.total) * 100)}%
                    </p>
                  </div>
                )}

                {llmProgress.type === "status" && llmProgress.step !== "backtest_start" && (
                  <div className="flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-primary animate-pulse" />
                    <span className="text-[10px] text-muted-foreground font-mono">{llmProgress.step}</span>
                  </div>
                )}
              </>
            ) : (
              <p className="text-xs text-muted-foreground/70 animate-pulse">Initialisation…</p>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {apiError && (
        <div className="mt-4 p-4 rounded-xl bg-destructive/10 text-destructive text-xs">{apiError}</div>
      )}

      <AnimatePresence>
        {(result || llmResult) && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-10"
          >
            {result && (
              <ClassicResult
                result={result}
                chartStart={chartStart}
                chartEnd={chartEnd}
                setChartStart={setChartStart}
                setChartEnd={setChartEnd}
              />
            )}
            {llmResult && (
              <LlmResult result={llmResult} classicResult={classicResult} />
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Simulation;
