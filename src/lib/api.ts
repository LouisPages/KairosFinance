const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export interface StockItem {
  symbol: string;
  name: string;
  index: "NASDAQ" | "DOW JONES" | "S&P 500";
}

export async function fetchStocks(): Promise<StockItem[]> {
  const r = await fetch(`${API_BASE}/api/stocks`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchHistory(
  symbols: string[],
  start: string,
  end: string,
  interval: "daily" | "monthly" | "annual"
): Promise<{ dates: string[]; series: Record<string, number[]> }> {
  const params = new URLSearchParams({
    symbols: symbols.join(","),
    start,
    end,
    interval: interval === "annual" ? "1y" : interval === "monthly" ? "1mo" : "1d",
  });
  const r = await fetch(`${API_BASE}/api/history?${params}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export interface EfficientFrontierPoint {
  volatility: number;
  expectedReturn: number;
  sharpe?: number;
  backtestReturn?: number;
}

// Résultat des modèles classiques (Markowitz simple, CAPM, FF3)
export interface SimulateResult {
  weights: Record<string, number>;
  sharpe: number;
  expectedReturn: number;
  volatility: number;
  maxDrawdown: number;
  comparisonData: { date: string; portfolio: number; market: number }[];
  numPortfolios?: number;
  trainPeriodStart?: string;
  trainPeriodEnd?: string;
  testPeriodStart?: string;
  testPeriodEnd?: string;
  efficientFrontier?: EfficientFrontierPoint[];
}

// Résumé news par ticker
export interface NewsSummary {
  summary: string;
  sentiment: "positif" | "neutre" | "négatif";
}

// Un pas mensuel du backtest LLM
// selectedFactors : { ticker: { "Mkt-RF": true, "SMB": false, "HML": false, "RMW": true, "CMA": true } }
export interface LlmMonthStep {
  month: string;
  weights: Record<string, number>;
  selectedFactors: Record<string, Record<string, boolean>>;
  newsSummaries: Record<string, NewsSummary>;
  sharpe: number | null;
  expectedReturn: number | null;
  volatility: number | null;
}

export interface LlmPromptExample {
  ticker: string;
  month: string;
  provider: string;
  system: string;
  user: string;
  response: string;
}

// Résultat du modèle LLM (backtest glissant mensuel)
export interface LlmSimulateResult {
  totalReturn: number;
  maxDrawdown: number;
  initialValue: number;
  finalValue: number;
  comparisonData: { date: string; portfolio: number; market: number | null }[];
  monthlyHistory: LlmMonthStep[];
  numMonths: number;
  trainPeriodStart: string;
  trainPeriodEnd: string;
  testPeriodStart: string;
  testPeriodEnd: string;
  promptExamples?: LlmPromptExample[];
}

export async function runSimulation(
  model: string,
  symbols: string[]
): Promise<SimulateResult> {
  const r = await fetch(`${API_BASE}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, symbols }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error((err as { detail?: string }).detail || "Erreur simulation");
  }
  return r.json();
}

export async function runLlmSimulation(
  symbols: string[]
): Promise<LlmSimulateResult> {
  const r = await fetch(`${API_BASE}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "markowitz-llm", symbols }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error((err as { detail?: string }).detail || "Erreur simulation LLM");
  }
  return r.json();
}

export interface LlmProgressEvent {
  type: "status" | "month" | "error";
  step?: string;
  message?: string;
  current?: number;
  total?: number;
  month?: string;
}

export function runLlmSimulationStream(
  symbols: string[],
  onProgress: (evt: LlmProgressEvent) => void,
  onResult: (result: LlmSimulateResult) => void,
  onError: (msg: string) => void,
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const r = await fetch(`${API_BASE}/api/simulate-llm-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "markowitz-llm", symbols }),
        signal: controller.signal,
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: r.statusText }));
        onError((err as { detail?: string }).detail || "Erreur simulation LLM");
        return;
      }
      const reader = r.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          if (!part.trim() || part.startsWith(":")) continue;
          const lines = part.split("\n");
          let eventType = "";
          let dataStr = "";
          for (const line of lines) {
            if (line.startsWith("event: ")) eventType = line.slice(7).trim();
            else if (line.startsWith("data: ")) dataStr = line.slice(6).trim();
          }
          if (!dataStr) continue;
          try {
            const payload = JSON.parse(dataStr);
            if (eventType === "result") {
              onResult(payload as LlmSimulateResult);
            } else if (eventType === "error") {
              onError(payload.message ?? "Erreur inconnue");
            } else if (eventType === "status" || eventType === "month") {
              onProgress(payload as LlmProgressEvent);
            }
          } catch {
            // ignore parse errors
          }
        }
      }
    } catch (e: unknown) {
      if (e instanceof Error && e.name !== "AbortError") {
        onError(e.message);
      }
    }
  })();

  return () => controller.abort();
}
