import { useState, useRef } from "react";
import { useLocation, Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Brain } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  runSimulation, runLlmSimulationStream,
  type SimulateResult, type LlmSimulateResult,
  type LlmProgressEvent,
} from "@/lib/api";
import { loadSavedSymbols } from "@/lib/portfolioStorage";
import { saveToHistory } from "@/lib/simulationHistory";
import { ClassicResult, LlmResult, ComparisonResult } from "@/components/SimulationResults";

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
  },
  {
    id: "markowitz-llm",
    name: "Choix dynamique des facteurs",
    desc: "Sélection mensuelle des facteurs par LLM selon l'actualité économique.",
  },
];

type OptimizationMethodId = "monte_carlo" | "gradient_fixe" | "gradient_optimal" | "comparison";
const OPTIMIZATION_METHODS: { id: OptimizationMethodId; label: string }[] = [
  { id: "monte_carlo", label: "Monte-Carlo" },
  { id: "gradient_fixe", label: "Gradient à pas fixe" },
  { id: "gradient_optimal", label: "Gradient à pas optimal" },
  { id: "comparison", label: "Comparaison (Monte-Carlo vs Gradient à pas optimal)" },
];
const COMPARISON_GRADIENT_LABEL = "Gradient à pas optimal";

// ---------------------------------------------------------------------------
// Page principale
// ---------------------------------------------------------------------------

const Simulation = () => {
  const location = useLocation();
  const symbolsFromState = (location.state as { symbols?: string[] } | null)?.symbols;
  const symbols: string[] = symbolsFromState?.length ? symbolsFromState : loadSavedSymbols();

  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [optimizationMethod, setOptimizationMethod] = useState<OptimizationMethodId>("gradient_optimal");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SimulateResult | null>(null);
  const [llmResult, setLlmResult] = useState<LlmSimulateResult | null>(null);
  const [classicResult, setClassicResult] = useState<SimulateResult | null>(null);
  const [comparisonData, setComparisonData] = useState<{
    monteCarlo: SimulateResult;
    bestGradient: SimulateResult;
    bestGradientLabel: string;
  } | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [chartStart, setChartStart] = useState("");
  const [chartEnd, setChartEnd] = useState("");
  const [llmProgress, setLlmProgress] = useState<LlmProgressEvent | null>(null);
  const cancelStreamRef = useRef<(() => void) | null>(null);
  const classicResultRef = useRef<SimulateResult | null>(null);

  const canRun = selectedModel != null && symbols.length >= 2;
  const isLlm = selectedModel === "markowitz-llm";

  const handleRun = () => {
    if (!canRun) return;
    setApiError(null);
    setRunning(true);
    setResult(null);
    setLlmResult(null);
    setClassicResult(null);
    setComparisonData(null);
    setLlmProgress(null);
    classicResultRef.current = null;

    if (isLlm) {
      runSimulation("markowitz-classic", symbols)
        .then((classic) => {
          setClassicResult(classic);
          classicResultRef.current = classic;
        })
        .catch(() => { /* classique optionnel */ });

      const cancel = runLlmSimulationStream(
        symbols,
        (evt) => setLlmProgress(evt),
        (llm) => {
          setLlmResult(llm);
          setRunning(false);
          setLlmProgress(null);
          cancelStreamRef.current = null;
          saveToHistory({
            modelId: "markowitz-llm",
            symbols,
            result: null,
            llmResult: llm,
            classicResult: classicResultRef.current,
          });
        },
        (msg) => {
          setApiError(msg);
          setRunning(false);
          setLlmProgress(null);
          cancelStreamRef.current = null;
        },
      );
      cancelStreamRef.current = cancel;
    } else if (optimizationMethod === "comparison") {
      // Exécution séquentielle pour éviter tout mélange de réponses (même données, ordre garanti)
      runSimulation(selectedModel!, symbols, "monte_carlo")
        .then((monteCarlo) =>
          runSimulation(selectedModel!, symbols, "gradient_optimal").then((gradientOptimal) => ({
            monteCarlo,
            gradientOptimal,
          }))
        )
        .then(({ monteCarlo, gradientOptimal }) => {
          setComparisonData({
            monteCarlo,
            bestGradient: gradientOptimal,
            bestGradientLabel: COMPARISON_GRADIENT_LABEL,
          });
          if (monteCarlo.comparisonData.length > 0) {
            setChartStart(monteCarlo.comparisonData[0].date);
            setChartEnd(monteCarlo.comparisonData[monteCarlo.comparisonData.length - 1].date);
          }
          saveToHistory({
            modelId: selectedModel!,
            symbols,
            result: monteCarlo,
            llmResult: null,
            classicResult: null,
            comparisonData: {
              monteCarlo,
              bestGradient: gradientOptimal,
              bestGradientLabel: COMPARISON_GRADIENT_LABEL,
            },
          });
        })
        .catch((e) => setApiError(e.message))
        .finally(() => setRunning(false));
    } else {
      runSimulation(selectedModel!, symbols, optimizationMethod)
        .then((res) => {
          setResult(res);
          if (res.comparisonData.length > 0) {
            setChartStart(res.comparisonData[0].date);
            setChartEnd(res.comparisonData[res.comparisonData.length - 1].date);
          }
          saveToHistory({
            modelId: selectedModel!,
            symbols,
            result: res,
            llmResult: null,
            classicResult: null,
          });
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
        {models.map((m) => (
          <button
            key={m.id}
            onClick={() => {
              setSelectedModel(m.id);
              setResult(null);
              setLlmResult(null);
              setClassicResult(null);
              setComparisonData(null);
              setApiError(null);
            }}
            className={`glass-card relative p-5 text-left cursor-pointer transition-shadow focus:outline-none focus:ring-0 active:ring-0 ${
              selectedModel === m.id ? "!ring-2 !ring-primary" : ""
            }`}
          >
            <h3 className="font-display text-sm font-bold text-foreground">{m.name}</h3>
            <p className="mt-1 text-xs text-muted-foreground">{m.desc}</p>
          </button>
        ))}
      </div>

      {/* Méthode d'optimisation (toujours visible ; ignorée pour le modèle LLM) */}
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <label className="text-sm font-medium text-foreground">Méthode d'optimisation</label>
        <Select
          value={optimizationMethod}
          onValueChange={(v) => { setOptimizationMethod(v as OptimizationMethodId); setResult(null); setComparisonData(null); setApiError(null); }}
        >
          <SelectTrigger className="w-[320px] cursor-pointer rounded-xl border-border bg-background/80">
            <SelectValue placeholder="Choisir une méthode" />
          </SelectTrigger>
          <SelectContent>
            {OPTIMIZATION_METHODS.map((opt) => (
              <SelectItem key={opt.id} value={opt.id}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {isLlm && (
          <span className="text-xs text-muted-foreground">(non utilisée pour ce modèle)</span>
        )}
      </div>

      <Button onClick={handleRun} disabled={!canRun || running} className="gap-2 rounded-xl font-semibold">
        {running ? (
          <>
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
            {isLlm
              ? "Backtest LLM en cours…"
              : optimizationMethod === "comparison"
                ? "Comparaison (2 simulations) en cours…"
                : "Optimisation en cours…"}
          </>
        ) : (
          <><Play className="h-4 w-4" /> Lancer la simulation</>
        )}
      </Button>

      {isLlm && !running && !llmResult && (
        <p className="mt-2 text-[11px] text-muted-foreground/80">
          Ce modèle interroge Mistral Le Chat (collecte des news AFP) + Gemini 2.0 Flash (sélection des facteurs) pour chaque mois de la période de test. Prévoyez 1–3 min selon le nombre d'actions et de mois.
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
        {(result || llmResult || comparisonData) && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-10"
          >
            {comparisonData && (
              <ComparisonResult
                monteCarlo={comparisonData.monteCarlo}
                bestGradient={comparisonData.bestGradient}
                bestGradientLabel={comparisonData.bestGradientLabel}
                chartStart={chartStart}
                chartEnd={chartEnd}
                setChartStart={setChartStart}
                setChartEnd={setChartEnd}
              />
            )}
            {result && !comparisonData && (
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
