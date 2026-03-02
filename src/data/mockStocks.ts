export interface Stock {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  index: "NASDAQ" | "DOW JONES" | "S&P 500";
  sharpe: number;
  volatility: number;
  annualReturn: number;
  popularity: number; // 1-100
}

export const mockStocks: Stock[] = [
  { symbol: "AAPL", name: "Apple Inc.", price: 178.72, change: 2.34, changePercent: 1.33, index: "NASDAQ", sharpe: 1.42, volatility: 18.3, annualReturn: 12.5, popularity: 95 },
  { symbol: "MSFT", name: "Microsoft Corp.", price: 378.91, change: -1.20, changePercent: -0.32, index: "NASDAQ", sharpe: 1.58, volatility: 16.1, annualReturn: 14.2, popularity: 92 },
  { symbol: "GOOGL", name: "Alphabet Inc.", price: 141.80, change: 0.95, changePercent: 0.67, index: "NASDAQ", sharpe: 1.21, volatility: 20.5, annualReturn: 10.8, popularity: 88 },
  { symbol: "AMZN", name: "Amazon.com Inc.", price: 185.07, change: 3.12, changePercent: 1.71, index: "NASDAQ", sharpe: 1.35, volatility: 22.4, annualReturn: 15.1, popularity: 90 },
  { symbol: "NVDA", name: "NVIDIA Corp.", price: 875.28, change: 12.45, changePercent: 1.44, index: "NASDAQ", sharpe: 1.85, volatility: 35.2, annualReturn: 28.7, popularity: 97 },
  { symbol: "TSLA", name: "Tesla Inc.", price: 248.42, change: -4.58, changePercent: -1.81, index: "NASDAQ", sharpe: 0.78, volatility: 42.1, annualReturn: 8.3, popularity: 94 },
  { symbol: "META", name: "Meta Platforms", price: 505.18, change: 7.20, changePercent: 1.45, index: "NASDAQ", sharpe: 1.52, volatility: 25.8, annualReturn: 18.4, popularity: 86 },
  { symbol: "JPM", name: "JPMorgan Chase", price: 198.47, change: 1.85, changePercent: 0.94, index: "DOW JONES", sharpe: 1.18, volatility: 15.6, annualReturn: 9.8, popularity: 82 },
  { symbol: "V", name: "Visa Inc.", price: 279.35, change: 0.67, changePercent: 0.24, index: "DOW JONES", sharpe: 1.45, volatility: 14.2, annualReturn: 11.6, popularity: 78 },
  { symbol: "JNJ", name: "Johnson & Johnson", price: 156.74, change: -0.42, changePercent: -0.27, index: "DOW JONES", sharpe: 0.95, volatility: 12.8, annualReturn: 6.2, popularity: 65 },
  { symbol: "UNH", name: "UnitedHealth Group", price: 527.63, change: 3.42, changePercent: 0.65, index: "DOW JONES", sharpe: 1.62, volatility: 17.4, annualReturn: 16.5, popularity: 72 },
  { symbol: "WMT", name: "Walmart Inc.", price: 165.23, change: 1.10, changePercent: 0.67, index: "DOW JONES", sharpe: 1.08, volatility: 13.5, annualReturn: 8.1, popularity: 70 },
  { symbol: "SPY", name: "SPDR S&P 500 ETF", price: 502.41, change: 2.15, changePercent: 0.43, index: "S&P 500", sharpe: 1.25, volatility: 15.0, annualReturn: 10.2, popularity: 99 },
  { symbol: "BRK.B", name: "Berkshire Hathaway", price: 408.92, change: 1.30, changePercent: 0.32, index: "S&P 500", sharpe: 1.38, volatility: 14.8, annualReturn: 11.0, popularity: 75 },
  { symbol: "LLY", name: "Eli Lilly", price: 782.15, change: 8.92, changePercent: 1.15, index: "S&P 500", sharpe: 1.72, volatility: 22.1, annualReturn: 21.3, popularity: 80 },
  { symbol: "XOM", name: "Exxon Mobil", price: 104.58, change: -0.85, changePercent: -0.81, index: "S&P 500", sharpe: 0.88, volatility: 19.7, annualReturn: 7.4, popularity: 68 },
];

export function generateHistoricalData(days: number = 90) {
  const data = [];
  let price = 100 + Math.random() * 50;
  const now = Date.now();
  for (let i = days; i >= 0; i--) {
    price += (Math.random() - 0.48) * 3;
    price = Math.max(price, 20);
    data.push({
      date: new Date(now - i * 86400000).toLocaleDateString("fr-FR", { day: "2-digit", month: "short" }),
      price: Math.round(price * 100) / 100,
    });
  }
  return data;
}

export function generateComparisonData(days: number = 90) {
  const data = [];
  let portfolio = 100;
  let market = 100;
  const now = Date.now();
  for (let i = days; i >= 0; i--) {
    portfolio += (Math.random() - 0.46) * 2;
    market += (Math.random() - 0.48) * 1.8;
    data.push({
      date: new Date(now - i * 86400000).toLocaleDateString("fr-FR", { day: "2-digit", month: "short" }),
      portfolio: Math.round(portfolio * 100) / 100,
      market: Math.round(market * 100) / 100,
    });
  }
  return data;
}
