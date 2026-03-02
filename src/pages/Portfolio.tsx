import { useState, useMemo } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, TrendingUp, TrendingDown, Search, X, BarChart3, Activity, Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { mockStocks, generateHistoricalData, Stock } from "@/data/mockStocks";

const indices = ["NASDAQ", "DOW JONES", "S&P 500"] as const;
const DEFAULT_BALANCE = 100000;

const Portfolio = () => {
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
  const [selectedStock, setSelectedStock] = useState<Stock | null>(null);
  const [activeIndex, setActiveIndex] = useState<string>("NASDAQ");
  const [search, setSearch] = useState("");

  const chartData = useMemo(() => (selectedStock ? generateHistoricalData(90) : []), [selectedStock]);

  const toggleStock = (symbol: string) => {
    setSelectedSymbols((prev) =>
      prev.includes(symbol) ? prev.filter((s) => s !== symbol) : [...prev, symbol]
    );
  };

  const removeStock = (symbol: string) => {
    setSelectedSymbols((prev) => prev.filter((s) => s !== symbol));
  };

  const evenAllocation = selectedSymbols.length > 0 ? 100 / selectedSymbols.length : 0;

  return (
    <div className="flex flex-col px-6 py-6" style={{ height: "calc(100vh - 64px)" }}>
      {/* Header row */}
      <div className="flex items-start justify-between mb-6 shrink-0">
        <div>
          <p className="section-label mb-2">Mon Portefeuille</p>
          <h1 className="section-title mb-1">Construisez votre portefeuille</h1>
          
        </div>
        <Button asChild className="gap-2 rounded-xl font-semibold shrink-0" disabled={selectedSymbols.length === 0}>
          <Link to="/simulation">
            Commencer une simulation <ArrowRight className="h-4 w-4" />
          </Link>
        </Button>
      </div>

      <div className="grid gap-6 min-h-0 flex-1" style={{ gridTemplateColumns: "1fr 3fr" }}>
        {/* Left — Stock selection (25%) */}
        <div className="flex flex-col min-h-0">
          <Tabs value={activeIndex} onValueChange={setActiveIndex} className="flex flex-col min-h-0 flex-1">
            <TabsList className="mb-3 w-full shrink-0">
              {indices.map((idx) => (
                <TabsTrigger key={idx} value={idx} className="text-sm font-semibold flex-1">
                  {idx}
                </TabsTrigger>
              ))}
            </TabsList>

            <div className="relative mb-3 shrink-0">
              <Search className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Rechercher…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-9 h-9 text-sm"
              />
            </div>

            {indices.map((idx) => (
              <TabsContent key={idx} value={idx} className="space-y-1.5 overflow-y-auto pr-1 min-h-0 flex-1">
                {mockStocks
                  .filter(
                    (s) =>
                      s.index === idx &&
                      (search === "" ||
                        s.symbol.toLowerCase().includes(search.toLowerCase()) ||
                        s.name.toLowerCase().includes(search.toLowerCase()))
                  )
                  .map((stock) => {
                    const isSelected = selectedSymbols.includes(stock.symbol);
                    return (
                      <div
                        key={stock.symbol}
                        className={`rounded-xl bg-card flex cursor-pointer items-center justify-between p-3 transition-colors border ${
                          isSelected ? "border-primary" : "border-border hover:shadow-sm"
                        } ${selectedStock?.symbol === stock.symbol ? "bg-muted/50" : ""}`}
                        onClick={() => setSelectedStock(stock)}
                      >
                        <div className="min-w-0">
                          <span className="font-display text-sm font-bold text-foreground">{stock.symbol}</span>
                          <p className="truncate text-xs text-muted-foreground">{stock.name}</p>
                        </div>
                        <div className="flex items-center gap-2">
                          <p className={`flex items-center gap-0.5 text-xs font-medium ${stock.change >= 0 ? "text-chart-up" : "text-chart-down"}`}>
                            {stock.change >= 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                            {stock.changePercent > 0 ? "+" : ""}{stock.changePercent.toFixed(2)}%
                          </p>
                          <button
                            onClick={(e) => { e.stopPropagation(); toggleStock(stock.symbol); }}
                            className={`rounded-md px-2 py-1 text-[10px] font-semibold transition-colors ${
                              isSelected
                                ? "bg-primary text-primary-foreground"
                                : "border border-input bg-background text-foreground hover:bg-muted"
                            }`}
                          >
                            {isSelected ? "✓" : "+"}
                          </button>
                        </div>
                      </div>
                    );
                  })}
              </TabsContent>
            ))}
          </Tabs>

          {/* Allocation summary — fixed at bottom */}
          <div className="glass-card mt-3 p-4 shrink-0">
            <h3 className="font-display text-sm font-bold text-foreground mb-2">Répartition</h3>
            {selectedSymbols.length === 0 ? (
              <p className="text-[11px] text-muted-foreground">Aucune action sélectionnée</p>
            ) : (
              <>
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="text-muted-foreground">{selectedSymbols.length} action{selectedSymbols.length > 1 ? "s" : ""}</span>
                  <span className="font-semibold text-foreground">{evenAllocation.toFixed(1)}% chacune</span>
                </div>
                <div className="mt-1 h-2 w-full rounded-full bg-secondary overflow-hidden">
                  <div className="h-full rounded-full bg-primary transition-all" style={{ width: "100%" }} />
                </div>
                <ul className="mt-2 space-y-1 max-h-24 overflow-y-auto">
                  {selectedSymbols.map((sym) => (
                    <li key={sym} className="flex items-center justify-between text-[11px]">
                      <div className="flex items-center gap-1.5">
                        <button onClick={() => removeStock(sym)} className="text-muted-foreground hover:text-destructive">
                          <X className="h-3 w-3" />
                        </button>
                        <span className="font-medium text-foreground">{sym}</span>
                      </div>
                      <span className="text-muted-foreground">{evenAllocation.toFixed(1)}% · ${((evenAllocation / 100) * DEFAULT_BALANCE).toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        </div>

        {/* Right — Charts & Info (75%) */}
        <div className="min-h-0 overflow-y-auto flex flex-col gap-4">
          {selectedStock ? (
            <>
              {/* KPI cards */}
              <div className="grid grid-cols-4 gap-3 shrink-0">
                {[
                  { label: "Rendement annuel", value: `${selectedStock.annualReturn.toFixed(1)}%`, icon: TrendingUp, color: selectedStock.annualReturn >= 0 ? "text-chart-up" : "text-chart-down" },
                  { label: "Volatilité", value: `${selectedStock.volatility.toFixed(1)}%`, icon: Activity, color: "text-foreground" },
                  { label: "Ratio de Sharpe", value: selectedStock.sharpe.toFixed(2), icon: BarChart3, color: "text-foreground" },
                  { label: "Popularité", value: `${selectedStock.popularity}/100`, icon: Star, color: "text-foreground" },
                ].map((kpi) => (
                  <div key={kpi.label} className="glass-card p-4 text-center">
                    <kpi.icon className="mx-auto h-4 w-4 text-primary mb-1" />
                    <p className={`font-display text-lg font-bold ${kpi.color}`}>{kpi.value}</p>
                    <p className="text-[10px] text-muted-foreground mt-0.5">{kpi.label}</p>
                  </div>
                ))}
              </div>

              {/* Chart */}
              <div className="glass-card p-6 flex-1 min-h-0 flex flex-col">
                <div className="flex items-center justify-between mb-4 shrink-0">
                  <div>
                    <h3 className="font-display text-lg font-bold text-foreground">{selectedStock.symbol}</h3>
                    <p className="text-base text-muted-foreground">{selectedStock.name}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-xl font-bold text-foreground">${selectedStock.price.toFixed(2)}</p>
                    <p className={`flex items-center justify-end gap-1 text-base font-medium ${selectedStock.change >= 0 ? "text-chart-up" : "text-chart-down"}`}>
                      {selectedStock.change >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                      {selectedStock.change >= 0 ? "+" : ""}{selectedStock.change.toFixed(2)} ({selectedStock.changePercent > 0 ? "+" : ""}{selectedStock.changePercent.toFixed(2)}%)
                    </p>
                  </div>
                </div>
                <div className="flex-1 min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis dataKey="date" tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" interval={14} />
                      <YAxis tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" domain={["auto", "auto"]} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: 8,
                          fontSize: 12,
                        }}
                      />
                      <Line type="monotone" dataKey="price" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          ) : (
            <div className="glass-card flex h-full items-center justify-center p-6">
              <p className="text-base text-muted-foreground">Sélectionnez une action pour afficher ses données historiques.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Portfolio;
