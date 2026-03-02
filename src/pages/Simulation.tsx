import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, CheckCircle2, BarChart3, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from "recharts";
import { generateComparisonData } from "@/data/mockStocks";

const models = [
  {
    id: "markowitz-classic",
    name: "Markowitz Classique",
    desc: "Optimisation moyenne-variance avec matrice de covariance historique.",
  },
  {
    id: "markowitz-shrinkage",
    name: "Markowitz Shrinkage",
    desc: "Estimation de Ledoit-Wolf pour une matrice de covariance plus robuste.",
  },
  {
    id: "markowitz-blacklitterman",
    name: "Black-Litterman",
    desc: "Intègre des vues subjectives de l'investisseur dans le cadre Markowitz.",
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

function runMockSimulation(): SimResult {
  const symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"];
  const raw = symbols.map(() => Math.random());
  const total = raw.reduce((a, b) => a + b, 0);
  const weights: Record<string, number> = {};
  symbols.forEach((s, i) => (weights[s] = Math.round((raw[i] / total) * 1000) / 10));
  return {
    sharpe: Math.round((1.2 + Math.random() * 0.8) * 100) / 100,
    expectedReturn: Math.round((8 + Math.random() * 7) * 100) / 100,
    volatility: Math.round((10 + Math.random() * 8) * 100) / 100,
    maxDrawdown: Math.round((5 + Math.random() * 10) * 100) / 100,
    weights,
    comparisonData: generateComparisonData(90),
  };
}

const Simulation = () => {
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SimResult | null>(null);

  const handleRun = () => {
    if (!selectedModel) return;
    setRunning(true);
    setResult(null);
    setTimeout(() => {
      setResult(runMockSimulation());
      setRunning(false);
    }, 1500);
  };

  return (
    <div className="px-6 py-10">
      <div>
        <p className="section-label mb-2">Simulation</p>
        <h1 className="section-title mb-1">Optimisez votre portefeuille</h1>
        <p className="mb-8 text-base text-muted-foreground">
          Choisissez un modèle de prédiction puis lancez l'optimisation. Les résultats incluent un backtesting sur 20% des données historiques.
        </p>
      </div>

      {/* Model selection */}
      <div className="mb-8 grid gap-4 md:grid-cols-3">
        {models.map((m) => (
          <button
            key={m.id}
            onClick={() => { setSelectedModel(m.id); setResult(null); }}
            className={`glass-card p-5 text-left transition-shadow focus:outline-none focus:ring-0 active:ring-0 ${
              selectedModel === m.id ? "!ring-2 !ring-primary hover:!ring-2 hover:!ring-primary active:!ring-2 active:!ring-primary" : ""
            }`}
          >
            <h3 className="font-display text-base font-bold text-foreground">{m.name}</h3>
            <p className="mt-1 text-sm text-muted-foreground">{m.desc}</p>
          </button>
        ))}
      </div>

      <Button onClick={handleRun} disabled={!selectedModel || running} className="gap-2 rounded-xl font-semibold">
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

      {/* Results */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-10 space-y-8"
          >
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
                  <p className="mt-2 font-display text-2xl font-bold text-foreground">{kpi.value}</p>
                  <p className="mt-1 text-sm text-muted-foreground">{kpi.label}</p>
                </div>
              ))}
            </div>

            {/* Comparison chart */}
            <div className="glass-card p-6">
              <h3 className="font-display text-base font-bold text-foreground mb-4">
                Performance : Portefeuille optimisé vs Marché
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={result.comparisonData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="date" tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" interval={14} />
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

            {/* Weights */}
            <div className="glass-card p-6">
              <h3 className="font-display text-base font-bold text-foreground mb-4">
                Allocation optimale
              </h3>
              <div className="space-y-2">
                {Object.entries(result.weights).map(([sym, w]) => (
                  <div key={sym} className="flex items-center gap-3">
                    <span className="w-14 text-sm font-semibold text-foreground">{sym}</span>
                    <div className="flex-1 rounded-full bg-secondary h-3 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${w}%` }}
                        transition={{ duration: 0.8, delay: 0.2 }}
                        className="h-full rounded-full bg-primary"
                      />
                    </div>
                    <span className="w-12 text-right text-sm font-medium text-muted-foreground">{w}%</span>
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
