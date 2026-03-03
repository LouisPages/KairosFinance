import { useState } from "react";
import { useLocation, Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Play, BarChart3, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, ReferenceDot } from "recharts";
import { runSimulation, type SimulateResult } from "@/lib/api";
import { loadSavedSymbols } from "@/lib/portfolioStorage";

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
];

const Simulation = () => {
  const location = useLocation();
  const symbolsFromState = (location.state as { symbols?: string[] } | null)?.symbols;
  const symbols: string[] = symbolsFromState?.length ? symbolsFromState : loadSavedSymbols();
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SimulateResult | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [chartStart, setChartStart] = useState<string>("");
  const [chartEnd, setChartEnd] = useState<string>("");

  const canRun = selectedModel != null && symbols.length >= 2;

  const handleRun = () => {
    if (!canRun) return;
    setApiError(null);
    setRunning(true);
    setResult(null);
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
  };

  return (
    <div className="px-6 py-10">
      <div>
        <p className="section-label mb-2">Simulation</p>
        <h1 className="section-title mb-1">Optimisez votre portefeuille</h1>
        <p className="mb-8 text-sm text-muted-foreground">
          Choisissez un modèle puis lancez l'optimisation. Les résultats incluent un backtesting sur 20% des données historiques.
        </p>
      </div>

      {symbols.length < 2 && (
        <div className="mb-6 p-4 rounded-xl bg-muted/50 border border-border">
          <p className="text-xs text-muted-foreground">
            Sélectionnez au moins 2 actions dans l'onglet{" "}
            <Link to="/portfolio" className="text-primary font-medium underline">
              Mon Portefeuille
            </Link>{" "}
            pour lancer une simulation.
          </p>
        </div>
      )}

      <div className="mb-8 grid gap-4 md:grid-cols-3">
        {models.map((m) => (
          <button
            key={m.id}
            onClick={() => { setSelectedModel(m.id); setResult(null); setApiError(null); }}
            className={`glass-card p-5 text-left transition-shadow focus:outline-none focus:ring-0 active:ring-0 ${
              selectedModel === m.id ? "!ring-2 !ring-primary hover:!ring-2 hover:!ring-primary active:!ring-2 active:!ring-primary" : ""
            }`}
          >
            <h3 className="font-display text-sm font-bold text-foreground">{m.name}</h3>
            <p className="mt-1 text-xs text-muted-foreground">{m.desc}</p>
          </button>
        ))}
      </div>

      <Button
        onClick={handleRun}
        disabled={!canRun || running}
        className="gap-2 rounded-xl font-semibold"
      >
        {running ? (
          <>
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
            Optimisation en cours…
          </>
        ) : (
          <>
            <Play className="h-4 w-4" /> Lancer la simulation
          </>
        )}
      </Button>

      {apiError && (
        <div className="mt-4 p-4 rounded-xl bg-destructive/10 text-destructive text-xs">
          {apiError}
        </div>
      )}

      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-10 space-y-8"
          >


            
            {(result.numPortfolios != null || result.trainPeriodStart != null) && (
              <p className="text-[11px] text-muted-foreground/80">
                {result.numPortfolios != null && (
                  <span>{result.numPortfolios.toLocaleString("fr-FR")} portefeuilles générés aléatoirement.</span>
                )}
                {result.trainPeriodStart != null && result.trainPeriodEnd != null && (
                  <> Entraînement 80 % : {result.trainPeriodStart} → {result.trainPeriodEnd}.</>
                )}
                {result.testPeriodStart != null && result.testPeriodEnd != null && (
                  <> Test 20 % : {result.testPeriodStart} → {result.testPeriodEnd}.</>
                )}
              </p>
            )}
            <div className="glass-card p-6">
              <h3 className="font-display text-sm font-bold text-foreground mb-4">
                Allocation optimale
              </h3>
              <div className="space-y-2">
                {Object.entries(result.weights).map(([sym, w]) => (
                  <div key={sym} className="flex items-center gap-3">
                    <span className="w-14 text-xs font-semibold text-foreground">{sym}</span>
                    <div className="flex-1 rounded-full bg-secondary h-3 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${w * 100}%` }}
                        transition={{ duration: 0.8, delay: 0.2 }}
                        className="h-full rounded-full bg-primary"
                      />
                    </div>
                    <span className="w-12 text-right text-xs font-medium text-muted-foreground">{(w * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
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


            <div className="glass-card p-6">
              <h3 className="font-display text-sm font-bold text-foreground mb-4">
                Performance : Portefeuille optimisé vs Marché
              </h3>
              {(() => {
                const hasPeriods = result.trainPeriodStart != null && result.trainPeriodEnd != null && result.testPeriodStart != null && result.testPeriodEnd != null;
                const filtered = result.comparisonData.filter(
                  (d) => (!chartStart || d.date >= chartStart) && (!chartEnd || d.date <= chartEnd)
                );
                type ChartPoint = { date: string; portfolio: number; market: number; portfolioTrain?: number; portfolioBacktest?: number };
                const chartData: ChartPoint[] = hasPeriods
                  ? filtered.map((d) => {
                      const isTrain = d.date >= result.trainPeriodStart! && d.date <= result.trainPeriodEnd!;
                      const isTest = d.date >= result.testPeriodStart! && d.date <= result.testPeriodEnd!;
                      return {
                        ...d,
                        portfolioTrain: isTrain ? d.portfolio : undefined,
                        portfolioBacktest: isTest ? d.portfolio : undefined,
                      };
                    })
                  : filtered;
                const vals = chartData.flatMap((d) =>
                  [d.portfolio, d.portfolioTrain, d.portfolioBacktest, d.market].filter((v): v is number => v != null)
                );
                const domainY = vals.length
                  ? (() => {
                      const lo = Math.min(...vals);
                      const hi = Math.max(...vals);
                      const pad = Math.max(5, (hi - lo) * 0.05);
                      return [Math.max(0, Math.floor((lo - pad) / 5) * 5), Math.ceil((hi + pad) / 5) * 5] as [number, number];
                    })()
                  : undefined;
                return (
                  <>
                    <div className="flex items-center justify-end gap-3 flex-wrap mb-4">
                      <label className="text-xs text-muted-foreground">
                        Du{" "}
                        <input
                          type="date"
                          value={chartStart}
                          onChange={(e) => setChartStart(e.target.value)}
                          className="bg-background border border-input rounded px-2 py-1.5 text-foreground text-xs"
                        />
                      </label>
                      <label className="text-xs text-muted-foreground">
                        Au{" "}
                        <input
                          type="date"
                          value={chartEnd}
                          onChange={(e) => setChartEnd(e.target.value)}
                          className="bg-background border border-input rounded px-2 py-1.5 text-foreground text-xs"
                        />
                      </label>
                    </div>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData} margin={{ left: 8, right: 8 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                          <XAxis dataKey="date" tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" interval="preserveStartEnd" />
                          <YAxis
                            tick={{ fontSize: 12 }}
                            stroke="hsl(var(--muted-foreground))"
                            domain={domainY}
                            tickFormatter={(v) => `${v}`}
                            label={{
                              value: "Indice (base 100)",
                              angle: -90,
                              position: "insideLeft",
                              style: { fontSize: 11, fill: "hsl(var(--muted-foreground))" },
                              content: (props: { viewBox?: { x?: number; y?: number; width?: number; height?: number } }) => {
                                const { viewBox } = props;
                                if (!viewBox || viewBox.height == null) {
                                  return (
                                    <text x={0} y={0} textAnchor="middle" dominantBaseline="middle" transform="rotate(-90 0 0)" style={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}>
                                      Indice (base 100)
                                    </text>
                                  );
                                }
                                const offsetLeft = 14;
                                const x = (viewBox.x ?? 0) + (viewBox.width ?? 0) / 2 - offsetLeft;
                                const y = (viewBox.y ?? 0) + viewBox.height / 2;
                                return (
                                  <text x={x} y={y} textAnchor="middle" dominantBaseline="middle" transform={`rotate(-90, ${x}, ${y})`} style={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}>
                                    Indice (base 100)
                                  </text>
                                );
                              },
                            }}
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "hsl(var(--card))",
                              border: "1px solid hsl(var(--border))",
                              borderRadius: 8,
                              fontSize: 12,
                            }}
                            formatter={(value: number, name: string) => [value != null ? Number(value).toFixed(1) : "—", name]}
                            labelFormatter={(label) => `Date : ${label}`}
                          />
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
                    <p className="mt-2 text-[10px] text-muted-foreground/70">
                      Courbes normalisées à 100 au premier jour du backtest ; données sur 100 % de la période. Entraînement 80 % / backtest 20 % (rouge). Marché : ETF SPY (S&P 500).
                    </p>
                  </>
                );
              })()}
            </div>

            {result.efficientFrontier != null && result.efficientFrontier.length > 0 && (
              <div className="glass-card p-6">
                <h3 className="font-display text-sm font-bold text-foreground mb-4">
                  Frontière efficiente
                </h3>
                <p className="text-xs text-muted-foreground mb-4">
                  Portefeuilles générés situés sur la frontière efficiente (risque / rendement optimal). Les points hors frontière ne sont pas affichés.
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={[...result.efficientFrontier].sort((a, b) => a.volatility - b.volatility)}
                      margin={{ left: 20, right: 16, top: 8, bottom: 24 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis
                        dataKey="volatility"
                        type="number"
                        tick={{ fontSize: 11 }}
                        stroke="hsl(var(--muted-foreground))"
                        domain={["auto", "auto"]}
                        tickFormatter={(v) => `${v} %`}
                        label={{
                          value: "Volatilité (%)",
                          position: "insideBottom",
                          offset: -12,
                          style: { fontSize: 11, fill: "hsl(var(--muted-foreground))" },
                        }}
                      />
                      <YAxis
                        dataKey="expectedReturn"
                        type="number"
                        tick={{ fontSize: 11 }}
                        stroke="hsl(var(--muted-foreground))"
                        domain={["auto", "auto"]}
                        tickFormatter={(v) => `${v} %`}
                        label={{
                          value: "Rendement attendu (%)",
                          angle: -90,
                          position: "insideLeft",
                          style: { fontSize: 11, fill: "hsl(var(--muted-foreground))" },
                          content: (props: { viewBox?: { x?: number; y?: number; width?: number; height?: number } }) => {
                            const { viewBox } = props;
                            if (!viewBox || viewBox.height == null) {
                              return (
                                <text x={0} y={0} textAnchor="middle" dominantBaseline="middle" transform="rotate(-90 0 0)" style={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}>
                                  Rendement attendu (%)
                                </text>
                              );
                            }
                            const offsetLeft = 14;
                            const x = (viewBox.x ?? 0) + (viewBox.width ?? 0) / 2 - offsetLeft;
                            const y = (viewBox.y ?? 0) + viewBox.height / 2;
                            return (
                              <text x={x} y={y} textAnchor="middle" dominantBaseline="middle" transform={`rotate(-90, ${x}, ${y})`} style={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}>
                                Rendement attendu (%)
                              </text>
                            );
                          },
                        }}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: 8,
                          fontSize: 12,
                        }}
                        content={({ active, payload }) => {
                          if (!active || payload == null || payload.length === 0) return null;
                          const p = payload[0]?.payload as { volatility: number; expectedReturn: number; sharpe?: number; backtestReturn?: number };
                          return (
                            <div className="rounded-lg border border-border bg-card px-3 py-2 text-xs shadow-sm">
                              <p>Volatilité : {p?.volatility?.toFixed(2) ?? "—"} %</p>
                              <p>Rendement attendu : {p?.expectedReturn?.toFixed(2) ?? "—"} %</p>
                              <p>Rendement réel (backtest) : {p?.backtestReturn != null ? `${p.backtestReturn.toFixed(2)} %` : "—"}</p>
                              <p>Ratio de Sharpe : {p?.sharpe != null ? p.sharpe.toFixed(2) : "—"}</p>
                            </div>
                          );
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="expectedReturn"
                        stroke="hsl(var(--primary))"
                        strokeWidth={2}
                        dot={{ fill: "hsl(var(--primary))", r: 4 }}
                        name="Frontière efficiente"
                        isAnimationActive={true}
                      />
                      <ReferenceDot
                        x={result.volatility}
                        y={result.expectedReturn}
                        r={6}
                        fill="hsl(var(--primary))"
                        stroke="hsl(var(--foreground))"
                        strokeWidth={2}
                        label={{ value: "Optimal", position: "top", fontSize: 10 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Simulation;
