import { useState, useMemo, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowRight, TrendingUp, Search, X, BarChart3, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { fetchStocks, fetchHistory } from "@/lib/api";
import type { StockItem } from "@/lib/api";

const indices = ["NASDAQ", "DOW JONES", "S&P 500"] as const;

const endDate = new Date();
const startDate = new Date();
startDate.setFullYear(startDate.getFullYear() - 5);

const Portfolio = () => {
  const [stocks, setStocks] = useState<StockItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
  const [selectedStock, setSelectedStock] = useState<StockItem | null>(null);
  const [activeIndex, setActiveIndex] = useState<string>("NASDAQ");
  const [search, setSearch] = useState("");
  const [chartStart, setChartStart] = useState(startDate.toISOString().slice(0, 10));
  const [chartEnd, setChartEnd] = useState(endDate.toISOString().slice(0, 10));
  const [chartInterval, setChartInterval] = useState<"daily" | "monthly" | "annual">("daily");
  const [historyData, setHistoryData] = useState<{ date: string; price: number }[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  useEffect(() => {
    fetchStocks()
      .then(setStocks)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedStock) {
      setHistoryData([]);
      return;
    }
    setHistoryLoading(true);
    fetchHistory([selectedStock.symbol], chartStart, chartEnd, chartInterval)
      .then((res) => {
        const dates = res.dates || [];
        const series = res.series?.[selectedStock.symbol];
        if (dates.length && series?.length) {
          setHistoryData(dates.map((d, i) => ({ date: d, price: series[i] ?? 0 })));
        } else {
          setHistoryData([]);
        }
      })
      .catch(() => setHistoryData([]))
      .finally(() => setHistoryLoading(false));
  }, [selectedStock, chartStart, chartEnd, chartInterval]);

  const chartData = useMemo(() => historyData, [historyData]);

  const toggleStock = (symbol: string) => {
    setSelectedSymbols((prev) =>
      prev.includes(symbol) ? prev.filter((s) => s !== symbol) : [...prev, symbol]
    );
  };

  const removeStock = (symbol: string) => {
    setSelectedSymbols((prev) => prev.filter((s) => s !== symbol));
  };

  const evenAllocation = selectedSymbols.length > 0 ? 100 / selectedSymbols.length : 0;

  const displayStocks = useMemo(() => {
    return stocks.filter(
      (s) =>
        s.index === activeIndex &&
        (search === "" ||
          s.symbol.toLowerCase().includes(search.toLowerCase()) ||
          s.name.toLowerCase().includes(search.toLowerCase()))
    );
  }, [stocks, activeIndex, search]);

  const lastPrice = chartData.length ? chartData[chartData.length - 1]?.price : null;
  const firstPrice = chartData.length ? chartData[0]?.price : null;
  const change = lastPrice != null && firstPrice != null && firstPrice > 0 ? lastPrice - firstPrice : 0;
  const changePercent = firstPrice != null && firstPrice > 0 ? (change / firstPrice) * 100 : 0;

  const kpiCards = useMemo(
    () => [
      {
        label: "Rendement (période)",
        value: lastPrice != null && firstPrice != null ? ((lastPrice / firstPrice - 1) * 100).toFixed(1) + "%" : "—",
        icon: TrendingUp,
        color: change >= 0 ? "text-chart-up" : "text-chart-down",
      },
      {
        label: "Variation",
        value: lastPrice != null ? (change >= 0 ? "+" : "") + changePercent.toFixed(2) + "%" : "—",
        icon: Activity,
        color: change >= 0 ? "text-chart-up" : "text-chart-down",
      },
      { label: "Dernier cours", value: lastPrice != null ? "$" + lastPrice.toFixed(2) : "—", icon: BarChart3, color: "text-foreground" },
    ],
    [lastPrice, firstPrice, change, changePercent]
  );

  return (
    <div className="flex flex-col px-6 py-6" style={{ height: "calc(100vh - 64px)" }}>
      <div className="flex items-start justify-between mb-6 shrink-0">
        <div>
          <p className="section-label mb-2">Mon Portefeuille</p>
          <h1 className="section-title mb-1">Construisez votre portefeuille</h1>
          
        </div>
        <Button
          asChild
          className="gap-2 rounded-xl font-semibold shrink-0"
          disabled={selectedSymbols.length === 0}
        >
          <Link to="/simulation" state={{ symbols: selectedSymbols }}>
            Commencer une simulation <ArrowRight className="h-4 w-4" />
          </Link>
        </Button>
      </div>

      <div className="grid gap-6 min-h-0 flex-1" style={{ gridTemplateColumns: "1fr 3fr" }}>
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
                {loading ? (
                  <p className="text-sm text-muted-foreground">Chargement…</p>
                ) : error ? (
                  <p className="text-sm text-destructive">{error}</p>
                ) : (
                  displayStocks.map((stock) => {
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
                  })
                )}
              </TabsContent>
            ))}
          </Tabs>

          <div className="glass-card mt-3 p-4 shrink-0">
            <h3 className="font-display text-sm font-bold text-foreground mb-2">Composition</h3>
            {selectedSymbols.length === 0 ? (
              <p className="text-[11px] text-muted-foreground">Aucune action sélectionnée</p>
            ) : (
              <>
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="text-muted-foreground">{selectedSymbols.length} action{selectedSymbols.length > 1 ? "s" : ""}</span>
                </div>
                <div className="mt-1 h-2 w-full rounded-full bg-secondary overflow-hidden">
                  <div className="h-full rounded-full bg-primary transition-all" style={{ width: "100%" }} />
                </div>
                <ul className="mt-2 space-y-1 max-h-24 overflow-y-auto">
                  {selectedSymbols.map((sym) => (
                    <li key={sym} className="flex items-center gap-1.5 text-[11px]">
                      <button onClick={() => removeStock(sym)} className="text-muted-foreground hover:text-destructive">
                        <X className="h-3 w-3" />
                      </button>
                      <span className="font-medium text-foreground">{sym}</span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        </div>

        <div className="min-h-0 overflow-y-auto flex flex-col gap-4">
          {selectedStock ? (
            <>
              <div className="grid grid-cols-3 gap-3 shrink-0">
                {kpiCards.map((kpi) => (
                  <div key={kpi.label} className="glass-card p-4 text-center">
                    <kpi.icon className="mx-auto h-4 w-4 text-primary mb-1" />
                    <p className={`font-display text-lg font-bold ${kpi.color}`}>{kpi.value}</p>
                    <p className="text-[10px] text-muted-foreground mt-0.5">{kpi.label}</p>
                  </div>
                ))}
              </div>

              <div className="glass-card p-6 flex-1 min-h-0 flex flex-col">
                <div className="flex items-center justify-between mb-4 shrink-0 flex-wrap gap-2">
                  <div>
                    <h3 className="font-display text-lg font-bold text-foreground">{selectedStock.symbol}</h3>
                    <p className="text-base text-muted-foreground">{selectedStock.name}</p>
                  </div>
                  <div className="flex items-center gap-3 flex-wrap">
                    <label className="text-xs text-muted-foreground">
                      Du{" "}
                      <input
                        type="date"
                        value={chartStart}
                        onChange={(e) => setChartStart(e.target.value)}
                        className="bg-background border rounded px-2 py-1 text-foreground"
                      />
                    </label>
                    <label className="text-xs text-muted-foreground">
                      Au{" "}
                      <input
                        type="date"
                        value={chartEnd}
                        onChange={(e) => setChartEnd(e.target.value)}
                        className="bg-background border rounded px-2 py-1 text-foreground"
                      />
                    </label>
                    <select
                      value={chartInterval}
                      onChange={(e) => setChartInterval(e.target.value as "daily" | "monthly" | "annual")}
                      className="bg-background border rounded px-2 py-1 text-sm text-foreground"
                    >
                      <option value="daily">Journalier</option>
                      <option value="monthly">Mensuel</option>
                      <option value="annual">Annuel</option>
                    </select>
                  </div>
                </div>
                <div className="flex-1 min-h-[300px]">
                  {historyLoading ? (
                    <p className="text-muted-foreground">Chargement des données…</p>
                  ) : chartData.length === 0 ? (
                    <p className="text-muted-foreground">Aucune donnée pour cette plage.</p>
                  ) : (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis dataKey="date" tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" interval="preserveStartEnd" />
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
                  )}
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
