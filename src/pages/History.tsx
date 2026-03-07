import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Trash2, RotateCcw, History as HistoryIcon, Brain } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ClassicResult, LlmResult } from "@/components/SimulationResults";
import {
  loadHistory, deleteFromHistory, clearHistory,
  type SimulationEntry, MODEL_LABELS,
} from "@/lib/simulationHistory";

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString("fr-FR", {
    day: "2-digit", month: "2-digit", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

function KpiPill({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2.5 py-1 text-[10px] text-muted-foreground">
      <span className="font-semibold text-foreground">{value}</span>
      <span>{label}</span>
    </span>
  );
}

function EntryCard({
  entry,
  isSelected,
  onSelect,
  onDelete,
}: {
  entry: SimulationEntry;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
}) {
  const isLlm = entry.modelId === "markowitz-llm";
  const label = MODEL_LABELS[entry.modelId] ?? entry.modelId;

  const kpis: { label: string; value: string }[] = [];
  if (entry.result) {
    kpis.push(
      { label: "Sharpe", value: entry.result.sharpe.toFixed(2) },
      { label: "Rendement", value: `${entry.result.expectedReturn.toFixed(1)}%` },
      { label: "Volatilité", value: `${entry.result.volatility.toFixed(1)}%` },
    );
  } else if (entry.llmResult) {
    kpis.push(
      { label: "Valeur finale", value: `$${entry.llmResult.finalValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}` },
      { label: "Rendement", value: `${entry.llmResult.totalReturn.toFixed(1)}%` },
      { label: "Max DD", value: `-${entry.llmResult.maxDrawdown.toFixed(1)}%` },
    );
  }

  return (
    <button
      onClick={onSelect}
      className={`glass-card relative w-full p-5 text-left transition-shadow focus:outline-none focus:ring-0 active:ring-0 ${
        isSelected ? "!ring-2 !ring-primary" : ""
      }`}
    >
      {isLlm && (
        <span className="absolute right-10 top-3 rounded-full px-2 py-0.5 text-[10px] font-medium bg-primary/20 text-primary">
          IA
        </span>
      )}
      <button
        onClick={(e) => { e.stopPropagation(); onDelete(); }}
        className="absolute right-3 top-3 rounded-md p-1 text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
        title="Supprimer"
      >
        <Trash2 className="h-3.5 w-3.5" />
      </button>

      <div className="pr-6">
        <p className="font-display text-sm font-bold text-foreground leading-tight">{label}</p>
        <p className="mt-0.5 text-[10px] text-muted-foreground font-mono">{formatDate(entry.date)}</p>
        <div className="mt-2 flex flex-wrap gap-1">
          {entry.symbols.map((s) => (
            <span key={s} className="rounded px-1.5 py-0.5 text-[9px] font-semibold bg-secondary text-secondary-foreground">{s}</span>
          ))}
        </div>
        {kpis.length > 0 && (
          <div className="mt-2.5 flex flex-wrap gap-1.5">
            {kpis.map((k) => <KpiPill key={k.label} {...k} />)}
          </div>
        )}
      </div>
    </button>
  );
}

const History = () => {
  const [history, setHistory] = useState<SimulationEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [chartStart, setChartStart] = useState("");
  const [chartEnd, setChartEnd] = useState("");

  useEffect(() => {
    loadHistory().then((entries) => {
      setHistory(entries);
      setSelectedId(entries[0]?.id ?? null);
      setLoading(false);
    });
  }, []);

  const selected = history.find((e) => e.id === selectedId) ?? null;

  const handleSelect = useCallback((entry: SimulationEntry) => {
    setSelectedId(entry.id);
    if (entry.result?.comparisonData.length) {
      setChartStart(entry.result.comparisonData[0].date);
      setChartEnd(entry.result.comparisonData[entry.result.comparisonData.length - 1].date);
    } else {
      setChartStart("");
      setChartEnd("");
    }
  }, []);

  const handleDelete = useCallback(async (id: string) => {
    await deleteFromHistory(id);
    const updated = await loadHistory();
    setHistory(updated);
    if (selectedId === id) {
      setSelectedId(updated[0]?.id ?? null);
    }
  }, [selectedId]);

  const handleClearAll = useCallback(async () => {
    await clearHistory();
    setHistory([]);
    setSelectedId(null);
  }, []);

  return (
    <div className="px-6 py-10">
      <div className="flex items-start justify-between mb-8">
        <div>
          <p className="section-label mb-2">Historique</p>
          <h1 className="section-title mb-1">Simulations sauvegardées</h1>
          <p className="text-sm text-muted-foreground">
            Retrouvez l'ensemble des simulations réalisées. Cliquez sur une simulation pour afficher ses résultats.
          </p>
        </div>
        {!loading && history.length > 0 && (
          <Button
            variant="outline"
            size="sm"
            onClick={handleClearAll}
            className="gap-2 text-destructive border-destructive/30 hover:bg-destructive/10 hover:text-destructive shrink-0 mt-1"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Tout effacer
          </Button>
        )}
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <span className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent mb-4" />
          <p className="text-sm text-muted-foreground">Chargement de l'historique…</p>
        </div>
      ) : history.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <HistoryIcon className="h-12 w-12 text-muted-foreground/30 mb-4" />
          <p className="text-sm font-medium text-muted-foreground">Aucune simulation enregistrée</p>
          <p className="mt-1 text-xs text-muted-foreground/60">
            Lancez une simulation depuis l'onglet <span className="font-semibold">Simulation</span> pour la retrouver ici.
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-10">
          {/* Sélecteur de simulation */}
          <div className="grid gap-4 md:grid-cols-3 xl:grid-cols-4">
            {history.map((entry) => (
              <EntryCard
                key={entry.id}
                entry={entry}
                isSelected={selectedId === entry.id}
                onSelect={() => handleSelect(entry)}
                onDelete={() => handleDelete(entry.id)}
              />
            ))}
          </div>

          {/* Résultats de la simulation sélectionnée */}
          <AnimatePresence mode="wait">
            {selected && (
              <motion.div
                key={selected.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.22 }}
              >
                <div className="mb-6 flex items-center gap-3">
                  <div className="h-px flex-1 bg-border" />
                  <span className="flex items-center gap-2 text-xs font-semibold text-muted-foreground uppercase tracking-widest">
                    {selected.modelId === "markowitz-llm"
                      ? <><Brain className="h-3.5 w-3.5 text-primary" /> Résultats — {MODEL_LABELS[selected.modelId]}</>
                      : <>Résultats — {MODEL_LABELS[selected.modelId] ?? selected.modelId}</>
                    }
                  </span>
                  <div className="h-px flex-1 bg-border" />
                </div>

                {selected.result && (
                  <ClassicResult
                    result={selected.result}
                    chartStart={chartStart}
                    chartEnd={chartEnd}
                    setChartStart={setChartStart}
                    setChartEnd={setChartEnd}
                  />
                )}
                {selected.llmResult && (
                  <LlmResult result={selected.llmResult} classicResult={selected.classicResult} />
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
};

export default History;
