import type { SimulationEntry } from "./simulationHistory";
import rawHistory from "../../server/simulation_history.json";

const MODEL_IDS = [
  "markowitz-classic",
  "markowitz-1factor",
  "markowitz-3factors",
  "markowitz-5factors",
  "markowitz-llm",
  "markowitz-crypto-ff3",
] as const;

function pickOneEntryPerModel(entries: unknown): SimulationEntry[] {
  if (!Array.isArray(entries)) return [];
  const byModel = new Map<string, SimulationEntry>();

  for (const item of entries) {
    if (!item || typeof item !== "object") continue;
    const entry = item as SimulationEntry;
    if (!entry.modelId || byModel.has(entry.modelId)) continue;
    byModel.set(entry.modelId, entry);
    if (byModel.size >= MODEL_IDS.length) break;
  }

  return MODEL_IDS
    .map((modelId) => byModel.get(modelId))
    .filter((entry): entry is SimulationEntry => Boolean(entry));
}

export const SAMPLE_HISTORY_ENTRIES: SimulationEntry[] = pickOneEntryPerModel(rawHistory);
