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
