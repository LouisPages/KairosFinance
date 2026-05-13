/** Dev : API sur le port 8000. Prod : même origine (Docker / reverse-proxy) si VITE_API_URL absent. */
const API_BASE =
  import.meta.env.VITE_API_URL ||
  (import.meta.env.PROD ? "" : "http://localhost:8000");

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

export interface CryptoListItem {
  symbol: string;
  name: string;
}

export async function fetchCryptoList(): Promise<CryptoListItem[]> {
  const r = await fetch(`${API_BASE}/api/crypto/list`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchCryptoHistory(
  symbol: string,
  start: string,
  end: string
): Promise<{ dates: string[]; series: Record<string, number[]> }> {
  const params = new URLSearchParams({ symbol: symbol.trim().toUpperCase(), start, end });
  const r = await fetch(`${API_BASE}/api/crypto/history?${params}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchCryptoNewsYahooSymbol(code: string): Promise<string> {
  const params = new URLSearchParams({ code: code.trim().toUpperCase() });
  const r = await fetch(`${API_BASE}/api/crypto/news-symbol?${params}`);
  if (!r.ok) throw new Error(await r.text());
  const j = (await r.json()) as { yahooSymbol?: string };
  return j.yahooSymbol ?? `${code}-USD`;
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

/** Article d'actualité pour un titre (source: yfinance / Yahoo Finance). */
export interface NewsArticle {
  title: string;
  url: string;
  publisher: string;
  publishedAt: string;
  thumbnail: string | null;
  summary: string | null;
}

export interface NewsResponse {
  symbol: string;
  articles: NewsArticle[];
}

export async function fetchNews(
  symbol: string,
  limit?: number,
  signal?: AbortSignal
): Promise<NewsResponse> {
  const params = new URLSearchParams({ symbol: symbol.trim().toUpperCase() });
  if (limit != null && limit > 0) params.set("limit", String(limit));
  const r = await fetch(`${API_BASE}/api/news?${params}`, { signal });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export interface EfficientFrontierPoint {
  volatility: number;
  expectedReturn: number;
  sharpe?: number;
  backtestReturn?: number;
}

/** Une ligne de test statistique pour un coefficient (alpha ou facteur). */
export interface FactorStatRow {
  beta: number;
  std_err: number;
  t_stat: number;
  p_value: number;
  ci_lower: number;
  ci_upper: number;
}

/** Statistiques globales du modèle OLS (R², F, etc.). */
export interface FactorModelStats {
  r_squared: number;
  adj_r_squared: number;
  n_obs: number;
  df_residual: number;
  f_stat: number | null;
  f_pvalue: number | null;
}

/** Par actif : tests par facteur + stats du modèle. */
export interface FactorTestsForTicker {
  factor_stats: Record<string, FactorStatRow>;
  model_stats: FactorModelStats | null;
}

/** factor_tests renvoyé par les modèles multi-facteurs (1, 3, 5, LLM). */
export type FactorTestsByTicker = Record<string, FactorTestsForTicker>;

// Résultat des modèles classiques (Markowitz simple, CAPM, FF3)
export interface SimulateResult {
  weights: Record<string, number>;
  sharpe: number;
  expectedReturn: number;
  volatility: number;
  maxDrawdown: number;
  /** Rendement réalisé sur la période de test (backtest), cohérent avec la courbe. */
  backtestReturn?: number;
  /** Ratio de Sharpe calculé sur les rendements de la période de test. */
  backtestSharpe?: number;
  comparisonData: { date: string; portfolio: number; market: number }[];
  numPortfolios?: number;
  trainPeriodStart?: string;
  trainPeriodEnd?: string;
  testPeriodStart?: string;
  testPeriodEnd?: string;
  efficientFrontier?: EfficientFrontierPoint[];
  /** Tests statistiques de pertinence des facteurs par actif (modèles 1/3/5 facteurs). */
  factor_tests?: FactorTestsByTicker;
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
  /** Tests statistiques de pertinence des facteurs par actif pour ce mois. */
  factor_tests?: FactorTestsByTicker;
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

export interface SimulationDataBounds {
  commonStart: string;
  commonEnd: string;
  assetMode: string;
}

/** Plage calendaire complète (ajustement + backtest) envoyée à l’API. */
export interface SimulationPeriod {
  startDate: string;
  endDate: string;
}

export async function fetchSimulationDataBounds(
  symbols: string[],
  assetMode: "actions" | "crypto",
): Promise<SimulationDataBounds> {
  const params = new URLSearchParams({
    symbols: symbols.join(","),
    asset_mode: assetMode,
  });
  const r = await fetch(`${API_BASE}/api/simulation-data-bounds?${params}`);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error((err as { detail?: string }).detail || "Impossible de charger les bornes de données.");
  }
  return r.json();
}

// Défaut aligné sur num_portfolios (10_000) dans gestion/markowitz_*.py
export const DEFAULT_MONTE_CARLO_SIMULATIONS = 10_000;

export async function runSimulation(
  model: string,
  symbols: string[],
  method?: "monte_carlo" | "gradient_fixe" | "gradient_optimal",
  period?: SimulationPeriod,
  options?: { monteCarloSimulations?: number },
): Promise<SimulateResult> {
  const body: {
    model: string;
    symbols: string[];
    method?: string;
    start_date?: string;
    end_date?: string;
    monte_carlo_simulations?: number;
  } = { model, symbols };
  if (method) body.method = method;
  if (period) {
    body.start_date = period.startDate;
    body.end_date = period.endDate;
  }
  if (options?.monteCarloSimulations != null) {
    body.monte_carlo_simulations = options.monteCarloSimulations;
  }
  const r = await fetch(`${API_BASE}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error((err as { detail?: string }).detail || "Erreur simulation");
  }
  return r.json();
}

export async function runLlmSimulation(
  symbols: string[],
  period?: SimulationPeriod,
): Promise<LlmSimulateResult> {
  const body: {
    model: string;
    symbols: string[];
    start_date?: string;
    end_date?: string;
  } = { model: "markowitz-llm", symbols };
  if (period) {
    body.start_date = period.startDate;
    body.end_date = period.endDate;
  }
  const r = await fetch(`${API_BASE}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
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
  period?: SimulationPeriod,
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const body: {
        model: string;
        symbols: string[];
        start_date?: string;
        end_date?: string;
      } = { model: "markowitz-llm", symbols };
      if (period) {
        body.start_date = period.startDate;
        body.end_date = period.endDate;
      }
      const r = await fetch(`${API_BASE}/api/simulate-llm-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
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
