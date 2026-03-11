import type { SimulateResult, LlmSimulateResult } from "./api";

/** Données de comparaison Monte-Carlo vs Gradient (sauvegardées en historique). */
export interface ComparisonPayload {
  monteCarlo: SimulateResult;
  bestGradient: SimulateResult;
  bestGradientLabel: string;
}

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export type SimulationModelId =
  | "markowitz-classic"
  | "markowitz-1factor"
  | "markowitz-3factors"
  | "markowitz-llm";

export const MODEL_LABELS: Record<string, string> = {
  "markowitz-classic": "Markowitz classique",
  "markowitz-1factor": "Un facteur de risque (CAPM)",
  "markowitz-3factors": "Trois facteurs (Fama & French)",
  "markowitz-5factors": "Cinq facteurs (Fama & French)",
  "markowitz-llm": "Choix dynamique des facteurs (LLM)",
};

export interface SimulationEntry {
  id: string;
  date: string;
  modelId: string;
  symbols: string[];
  result: SimulateResult | null;
  llmResult: LlmSimulateResult | null;
  classicResult: SimulateResult | null;
  /** Comparaison Monte-Carlo vs Gradient (mode "comparison") — pour affichage dans l'historique. */
  comparisonData?: ComparisonPayload | null;
  description?: string;
}

export async function loadHistory(): Promise<SimulationEntry[]> {
  try {
    const res = await fetch(`${API_BASE}/api/history/list`);
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}

export async function saveToHistory(
  entry: Omit<SimulationEntry, "id" | "date">
): Promise<SimulationEntry> {
  const newEntry: SimulationEntry = {
    ...entry,
    id: `sim-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    date: new Date().toISOString(),
  };
  try {
    await fetch(`${API_BASE}/api/history/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(newEntry),
    });
  } catch {
    // fire-and-forget : on ignore les erreurs réseau
  }
  return newEntry;
}

export async function deleteFromHistory(id: string): Promise<void> {
  try {
    await fetch(`${API_BASE}/api/history/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
  } catch {
    // ignore
  }
}

export async function clearHistory(): Promise<void> {
  try {
    await fetch(`${API_BASE}/api/history`, { method: "DELETE" });
  } catch {
    // ignore
  }
}

export async function updateDescription(id: string, description: string): Promise<void> {
  try {
    await fetch(`${API_BASE}/api/history/${encodeURIComponent(id)}/description`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ description }),
    });
  } catch {
    // ignore
  }
}
