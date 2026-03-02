import { useState } from "react";
import { useLocation, Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Play, BarChart3, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from "recharts";
import { runSimulation } from "@/lib/api";
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

interface SimResult {
  sharpe: number;
  expectedReturn: number;
  volatility: number;
  maxDrawdown: number;
  weights: Record<string, number>;
  comparisonData: { date: string; portfolio: number; market: number }[];
}

const Simulation = () => {
  const location = useLocation();
  const symbolsFromState = (location.state as { symbols?: string[] } | null)?.symbols;
  const symbols: string[] = symbolsFromState?.length ? symbolsFromState : loadSavedSymbols();
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SimResult | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const canRun = selectedModel != null && symbols.length >= 2;

  const handleRun = () => {
    if (!canRun) return;
    setApiError(null);
    setRunning(true);
    setResult(null);
    runSimulation(selectedModel!, symbols)
      .then(setResult)
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
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={result.comparisonData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="date" tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" interval="preserveStartEnd" />
                    <YAxis tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="portfolio" name="Portefeuille" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="market" name="Marché (S&P 500)" stroke="hsl(var(--muted-foreground))" strokeWidth={2} dot={false} strokeDasharray="5 5" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

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
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Simulation;
