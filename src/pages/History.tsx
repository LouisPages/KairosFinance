import { useState, useCallback, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Trash2, History as HistoryIcon, Brain, PencilLine, Check, X } from "lucide-react";
import { ClassicResult, LlmResult, ComparisonResult } from "@/components/SimulationResults";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  loadHistory, deleteFromHistory, updateDescription, updateObservedInterpretation,
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
  onDescriptionSave,
}: {
  entry: SimulationEntry;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onDescriptionSave: (id: string, description: string) => void;
}) {
  const isLlm = entry.modelId === "markowitz-llm";
  const label = MODEL_LABELS[entry.modelId] ?? entry.modelId;
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(entry.description ?? "");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (editing) {
      textareaRef.current?.focus();
      textareaRef.current?.select();
    }
  }, [editing]);

  const handleEditClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setDraft(entry.description ?? "");
    setEditing(true);
  };

  const handleSave = (e: React.MouseEvent) => {
    e.stopPropagation();
    onDescriptionSave(entry.id, draft.trim());
    setEditing(false);
  };

  const handleCancel = (e: React.MouseEvent) => {
    e.stopPropagation();
    setDraft(entry.description ?? "");
    setEditing(false);
  };

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
        isSelected ? "!ring-2 !ring-inset !ring-primary" : ""
      }`}
    >
      <div className="absolute right-3 top-3 flex flex-col items-center gap-1">
        <button
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          className="rounded-md p-1 text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
          title="Supprimer"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={handleEditClick}
          className="rounded-md p-1 text-muted-foreground hover:text-primary hover:bg-primary/10 transition-colors"
          title="Ajouter une description"
        >
          <PencilLine className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="pr-6">
        <div className="flex items-center gap-2 flex-wrap">
          <p className="font-display text-sm font-bold text-foreground leading-tight">{label}</p>
          {(entry.assetMode === "crypto" || entry.modelId === "markowitz-crypto-ff3") ? (
            <span className="rounded px-1.5 py-0.5 text-[9px] font-semibold bg-amber-500/20 text-amber-700 dark:text-amber-400">
              Crypto
            </span>
          ) : (
            <span className="rounded px-1.5 py-0.5 text-[9px] font-semibold bg-slate-500/15 text-slate-600 dark:text-slate-400">
              Actions
            </span>
          )}
          {entry.comparisonData && (
            <span className="rounded px-1.5 py-0.5 text-[9px] font-semibold bg-primary/20 text-primary">
              Comparaison
            </span>
          )}
          {entry.personTag && (
            <span className="rounded px-1.5 py-0.5 text-[9px] font-semibold bg-emerald-500/15 text-emerald-700 dark:text-emerald-400">
              {entry.personTag}
            </span>
          )}
        </div>
        <p className="mt-0.5 text-[10px] text-muted-foreground font-mono">{formatDate(entry.date)}</p>
        {entry.description && !editing && (
          <p className="mt-1 text-[10px] text-muted-foreground italic leading-snug">{entry.description}</p>
        )}
        {editing && (
          <div className="mt-1.5" onClick={(e) => e.stopPropagation()}>
            <textarea
              ref={textareaRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Ajouter une description…"
              rows={2}
              className="w-full resize-none rounded-md border border-border bg-background/60 px-2 py-1 text-[10px] text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onDescriptionSave(entry.id, draft.trim()); setEditing(false); }
                if (e.key === "Escape") { setDraft(entry.description ?? ""); setEditing(false); }
              }}
            />
            <div className="mt-1 flex gap-1">
              <button
                onClick={handleSave}
                className="flex items-center gap-0.5 rounded px-1.5 py-0.5 text-[9px] font-semibold bg-primary/20 text-primary hover:bg-primary/30 transition-colors"
              >
                <Check className="h-2.5 w-2.5" /> Enregistrer
              </button>
              <button
                onClick={handleCancel}
                className="flex items-center gap-0.5 rounded px-1.5 py-0.5 text-[9px] font-semibold bg-muted text-muted-foreground hover:bg-muted/80 transition-colors"
              >
                <X className="h-2.5 w-2.5" /> Annuler
              </button>
            </div>
          </div>
        )}
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
  const [personFilter, setPersonFilter] = useState<string>("all");
  const [modelFilter, setModelFilter] = useState<string>("all");
  const [analysisDraft, setAnalysisDraft] = useState("");
  const [analysisLocked, setAnalysisLocked] = useState(false);
  const [analysisSaving, setAnalysisSaving] = useState(false);
  const [analysisSavedFlash, setAnalysisSavedFlash] = useState(false);

  useEffect(() => {
    loadHistory().then((entries) => {
      setHistory(entries);
      setSelectedId(entries[0]?.id ?? null);
      setLoading(false);
    });
  }, []);

  const people = Array.from(new Set(history.map((entry) => entry.personTag).filter(Boolean) as string[]));
  const models = Array.from(new Set(history.map((entry) => entry.modelId)));
  const filteredHistory = history.filter((entry) => {
    const matchesPerson = personFilter === "all" || (entry.personTag ?? "") === personFilter;
    const matchesModel = modelFilter === "all" || entry.modelId === modelFilter;
    return matchesPerson && matchesModel;
  });
  const selected = filteredHistory.find((e) => e.id === selectedId) ?? filteredHistory[0] ?? null;

  const handleSelect = useCallback((entry: SimulationEntry) => {
    setSelectedId(entry.id);
    setAnalysisDraft(entry.observedInterpretation ?? "");
    const cd = entry.comparisonData?.monteCarlo?.comparisonData ?? entry.result?.comparisonData;
    if (cd?.length) {
      setChartStart(cd[0].date);
      setChartEnd(cd[cd.length - 1].date);
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

  const handleDescriptionSave = useCallback(async (id: string, description: string) => {
    await updateDescription(id, description);
    setHistory((prev) =>
      prev.map((e) => (e.id === id ? { ...e, description: description || undefined } : e))
    );
  }, []);

  useEffect(() => {
    if (selected) {
      setAnalysisDraft(selected.observedInterpretation ?? "");
      setAnalysisLocked(Boolean(selected.observedInterpretation?.trim()));
      setAnalysisSavedFlash(false);
      return;
    }
    setAnalysisDraft("");
    setAnalysisLocked(false);
    setAnalysisSavedFlash(false);
  }, [selected]);

  useEffect(() => {
    if (!selectedId && filteredHistory.length > 0) {
      handleSelect(filteredHistory[0]);
      return;
    }
    if (selectedId && !filteredHistory.some((entry) => entry.id === selectedId)) {
      setSelectedId(filteredHistory[0]?.id ?? null);
    }
  }, [filteredHistory, selectedId, handleSelect]);

  const handleSaveAnalysis = useCallback(async () => {
    if (!selected || analysisLocked) return;
    const next = analysisDraft.trim();
    setAnalysisSaving(true);
    await updateObservedInterpretation(selected.id, next);
    setHistory((prev) =>
      prev.map((entry) =>
        entry.id === selected.id ? { ...entry, observedInterpretation: next } : entry
      )
    );
    setAnalysisLocked(true);
    setAnalysisSavedFlash(true);
    setTimeout(() => setAnalysisSavedFlash(false), 1600);
    setAnalysisSaving(false);
  }, [selected, analysisDraft, analysisLocked]);

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
          <div className="flex flex-wrap items-center gap-3">
            <Select value={personFilter} onValueChange={setPersonFilter}>
              <SelectTrigger className="w-[220px] rounded-xl border-border bg-background/80 text-xs">
                <SelectValue placeholder="Filtrer par personne" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Toutes les personnes</SelectItem>
                {people.map((person) => (
                  <SelectItem key={person} value={person}>
                    {person}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={modelFilter} onValueChange={setModelFilter}>
              <SelectTrigger className="w-[280px] rounded-xl border-border bg-background/80 text-xs">
                <SelectValue placeholder="Filtrer par modèle" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Tous les modèles</SelectItem>
                {models.map((modelId) => (
                  <SelectItem key={modelId} value={modelId}>
                    {MODEL_LABELS[modelId] ?? modelId}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Sélecteur de simulation */}
          <div className="overflow-x-auto pb-2">
            <div className="flex min-w-max gap-4">
              {filteredHistory.map((entry) => (
                <div key={entry.id} className="w-[300px] shrink-0">
                  <EntryCard
                    entry={entry}
                    isSelected={selectedId === entry.id}
                    onSelect={() => handleSelect(entry)}
                    onDelete={() => handleDelete(entry.id)}
                    onDescriptionSave={handleDescriptionSave}
                  />
                </div>
              ))}
            </div>
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
                <div className="mb-6 glass-card p-4">
                  <p className="text-xs font-semibold text-foreground">Résultat observé / interprétation</p>
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    Complétez votre analyse qualitative de cette simulation.
                  </p>
                  <textarea
                    value={analysisDraft}
                    onChange={(e) => setAnalysisDraft(e.target.value)}
                    rows={3}
                    placeholder="Observations, conclusions, décisions..."
                    disabled={analysisLocked}
                    className={`mt-3 w-full resize-y rounded-md border border-border px-3 py-2 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-primary ${
                      analysisLocked ? "cursor-not-allowed bg-muted/40 text-muted-foreground" : "bg-background/70"
                    }`}
                  />
                  <div className="mt-3 flex items-center justify-end gap-2">
                    <button
                      type="button"
                      onClick={() => setAnalysisLocked(false)}
                      disabled={!analysisLocked}
                      className="rounded-lg border border-border bg-background px-3 py-1.5 text-xs font-semibold text-foreground disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      Modifier l&apos;analyse
                    </button>
                    <button
                      type="button"
                      onClick={handleSaveAnalysis}
                      disabled={analysisLocked || analysisSaving}
                      className="rounded-lg bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {analysisSaving ? "Sauvegarde..." : "Sauvegarder l&apos;analyse"}
                    </button>
                  </div>
                  <AnimatePresence>
                    {analysisSavedFlash && (
                      <motion.div
                        initial={{ opacity: 0, y: 6, scale: 0.98 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -6, scale: 0.98 }}
                        className="mt-2 inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-2.5 py-1 text-[11px] font-semibold text-emerald-700 dark:text-emerald-400"
                      >
                        <Check className="h-3 w-3" />
                        Analyse sauvegardée
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {selected.comparisonData && (
                  <ComparisonResult
                    monteCarlo={selected.comparisonData.monteCarlo}
                    bestGradient={selected.comparisonData.bestGradient}
                    bestGradientLabel={selected.comparisonData.bestGradientLabel}
                    chartStart={chartStart}
                    chartEnd={chartEnd}
                    setChartStart={setChartStart}
                    setChartEnd={setChartEnd}
                  />
                )}
                {selected.result && !selected.comparisonData && (
                  <ClassicResult
                    result={selected.result}
                    chartStart={chartStart}
                    chartEnd={chartEnd}
                    setChartStart={setChartStart}
                    setChartEnd={setChartEnd}
                    benchmarkLineName={
                      selected.assetMode === "crypto" || selected.modelId === "markowitz-crypto-ff3"
                        ? "Marché (moyenne cross-section)"
                        : undefined
                    }
                    benchmarkFootnote={
                      selected.assetMode === "crypto" || selected.modelId === "markowitz-crypto-ff3"
                        ? "Courbes normalisées à 100 au début du backtest. Marché : moyenne des rendements des cryptos à chaque date (proxy proche du facteur CMKT)."
                        : undefined
                    }
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
