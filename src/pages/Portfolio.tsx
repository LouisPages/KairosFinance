import { useState, useMemo, useEffect } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, TrendingUp, Search, X, BarChart3, Activity, Newspaper, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import {
  fetchStocks,
  fetchHistory,
  fetchNews,
  fetchCryptoList,
  fetchCryptoHistory,
  fetchCryptoNewsYahooSymbol,
} from "@/lib/api";
import { loadSavedSymbols, saveSymbols, loadSavedCryptoSymbols, saveCryptoSymbols } from "@/lib/portfolioStorage";
import type { StockItem, NewsArticle, CryptoListItem } from "@/lib/api";
import { useAppMode } from "@/context/AppModeContext";

const endDate = new Date();
const startDate = new Date("2005-01-01");
const cryptoChartStartDefault = "2015-01-01";

function PortfolioStocks() {
  const [stocks, setStocks] = useState<StockItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(loadSavedSymbols);
  const [selectedStock, setSelectedStock] = useState<StockItem | null>(null);
  const [search, setSearch] = useState("");
  const [chartStart, setChartStart] = useState(startDate.toISOString().slice(0, 10));
  const [chartEnd, setChartEnd] = useState(endDate.toISOString().slice(0, 10));
  const [chartInterval, setChartInterval] = useState<"daily" | "monthly" | "annual">("daily");
  const [historyData, setHistoryData] = useState<{ date: string; price: number }[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [newsArticles, setNewsArticles] = useState<NewsArticle[]>([]);
  const [newsLoading, setNewsLoading] = useState(false);

  useEffect(() => {
    fetchStocks()
      .then(setStocks)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    saveSymbols(selectedSymbols);
  }, [selectedSymbols]);

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

  useEffect(() => {
    if (!selectedStock) {
      setNewsArticles([]);
      return;
    }
    const controller = new AbortController();
    setNewsLoading(true);
    fetchNews(selectedStock.symbol, 12, controller.signal)
      .then((res) => setNewsArticles(res.articles || []))
      .catch(() => setNewsArticles([]))
      .finally(() => setNewsLoading(false));
    return () => controller.abort();
  }, [selectedStock]);

  const chartData = useMemo(() => historyData, [historyData]);

  const formatNewsDate = (iso: string) => {
    if (!iso) return "";
    try {
      const d = new Date(iso);
      const now = new Date();
      const diffMs = now.getTime() - d.getTime();
      const diffH = Math.floor(diffMs / (1000 * 60 * 60));
      const diffD = Math.floor(diffMs / (1000 * 60 * 60 * 24));
      if (diffH < 1) return "À l'instant";
      if (diffH < 24) return `Il y a ${diffH}h`;
      if (diffD < 7) return `Il y a ${diffD}j`;
      return d.toLocaleDateString("fr-FR", { day: "numeric", month: "short", year: "numeric" });
    } catch {
      return iso;
    }
  };

  const toggleStock = (symbol: string) => {
    setSelectedSymbols((prev) =>
      prev.includes(symbol) ? prev.filter((s) => s !== symbol) : [...prev, symbol]
    );
  };

  const removeStock = (symbol: string) => {
    setSelectedSymbols((prev) => prev.filter((s) => s !== symbol));
  };

  const displayStocks = useMemo(() => {
    const q = search.toLowerCase();
    return stocks.filter(
      (s) =>
        s.index === "S&P 500" &&
        (q === "" || s.symbol.toLowerCase().includes(q) || s.name.toLowerCase().includes(q))
    );
  }, [stocks, search]);

  const selectAllDisplayed = () => {
    const symbols = displayStocks.map((s) => s.symbol);
    setSelectedSymbols((prev) => Array.from(new Set([...prev, ...symbols])));
  };

  const allDisplayedSelected =
    displayStocks.length > 0 && displayStocks.every((s) => selectedSymbols.includes(s.symbol));

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
          <div className="flex gap-2 mb-3 shrink-0">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Rechercher…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-9 h-9 text-xs"
              />
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 shrink-0 text-xs font-semibold"
              onClick={selectAllDisplayed}
              disabled={loading || displayStocks.length === 0 || allDisplayedSelected}
            >
              Tout sélectionner
            </Button>
          </div>

          <div className="space-y-1.5 overflow-y-auto pr-1 min-h-0 flex-1">
            {loading ? (
              <p className="text-xs text-muted-foreground">Chargement…</p>
            ) : error ? (
              <p className="text-xs text-destructive">{error}</p>
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
                      <span className="font-display text-xs font-bold text-foreground">{stock.symbol}</span>
                      <p className="truncate text-[11px] text-muted-foreground">{stock.name}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleStock(stock.symbol);
                        }}
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
          </div>

          <div className="glass-card mt-3 p-4 shrink-0">
            <h3 className="font-display text-xs font-bold text-foreground mb-2">Composition</h3>
            {selectedSymbols.length === 0 ? (
              <p className="text-[11px] text-muted-foreground">Aucune action sélectionnée</p>
            ) : (
              <>
                <div className="flex items-center justify-between text-xs mb-2">
                  <span className="text-muted-foreground">
                    {selectedSymbols.length} action{selectedSymbols.length > 1 ? "s" : ""}
                  </span>
                </div>
                <div className="mt-1 h-2 w-full rounded-full bg-secondary overflow-hidden">
                  <div className="h-full rounded-full bg-primary transition-all" style={{ width: "100%" }} />
                </div>
                <ul className="mt-2 space-y-1 max-h-24 overflow-y-auto">
                  {selectedSymbols.map((sym) => (
                    <li key={sym} className="flex items-center gap-1.5 text-[11px]">
                      <button
                        onClick={() => removeStock(sym)}
                        className="text-muted-foreground hover:text-destructive"
                      >
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
                    <p className={`font-display text-base font-bold ${kpi.color}`}>{kpi.value}</p>
                    <p className="text-[10px] text-muted-foreground mt-0.5">{kpi.label}</p>
                  </div>
                ))}
              </div>

              <div className="glass-card p-6 flex flex-col shrink-0 min-h-[55vh]">
                <div className="flex items-center justify-between mb-4 shrink-0 flex-wrap gap-2">
                  <div>
                    <h3 className="font-display text-base font-bold text-foreground">{selectedStock.symbol}</h3>
                    <p className="text-sm text-muted-foreground">{selectedStock.name}</p>
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
                  </div>
                </div>
                <div className="flex-1 min-h-[320px]">
                  {historyLoading ? (
                    <p className="text-muted-foreground">Chargement des données…</p>
                  ) : chartData.length === 0 ? (
                    <p className="text-muted-foreground">Aucune donnée pour cette plage.</p>
                  ) : (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis
                          dataKey="date"
                          tick={{ fontSize: 10 }}
                          stroke="hsl(var(--muted-foreground))"
                          interval="preserveStartEnd"
                        />
                        <YAxis
                          tick={{ fontSize: 11 }}
                          stroke="hsl(var(--muted-foreground))"
                          domain={["auto", "auto"]}
                          tickFormatter={(v) =>
                            `$${Number(v).toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`
                          }
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "hsl(var(--card))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: 8,
                            fontSize: 12,
                          }}
                          formatter={(value: number) => [`$${Number(value).toFixed(2)}`, "Cours"]}
                        />
                        <Line type="monotone" dataKey="price" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </div>

              <div className="glass-card p-4 shrink-0">
                <h3 className="font-display text-sm font-bold text-foreground mb-3 flex items-center gap-2">
                  <Newspaper className="h-4 w-4 text-primary" />
                  Actualités récentes — {selectedStock.symbol}
                </h3>
                {newsLoading ? (
                  <div className="flex items-center gap-2 text-muted-foreground text-sm py-4">
                    <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                    Chargement des actualités…
                  </div>
                ) : newsArticles.length === 0 ? (
                  <p className="text-sm text-muted-foreground py-2">Aucune actualité récente pour le moment.</p>
                ) : (
                  <ul className="space-y-3 max-h-[280px] overflow-y-auto pr-1">
                    {newsArticles.map((article, idx) => (
                      <li key={idx}>
                        <a
                          href={article.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex gap-3 p-3 rounded-lg border border-border bg-card hover:bg-muted/50 transition-colors group"
                        >
                          {article.thumbnail ? (
                            <img
                              src={article.thumbnail}
                              alt=""
                              className="w-16 h-16 rounded-md object-cover shrink-0 bg-muted"
                              loading="lazy"
                            />
                          ) : (
                            <div className="w-16 h-16 rounded-md bg-muted shrink-0 flex items-center justify-center">
                              <Newspaper className="h-6 w-6 text-muted-foreground" />
                            </div>
                          )}
                          <div className="min-w-0 flex-1">
                            <p className="text-xs font-semibold text-foreground line-clamp-2 group-hover:text-primary">
                              {article.title}
                            </p>
                            <div className="flex items-center gap-2 mt-1 text-[11px] text-muted-foreground">
                              {article.publisher && <span>{article.publisher}</span>}
                              {article.publishedAt && (
                                <>
                                  {article.publisher && <span>·</span>}
                                  <span>{formatNewsDate(article.publishedAt)}</span>
                                </>
                              )}
                            </div>
                            {article.summary && (
                              <p className="text-[11px] text-muted-foreground mt-1 line-clamp-2">{article.summary}</p>
                            )}
                          </div>
                          <ExternalLink className="h-3.5 w-3.5 text-muted-foreground shrink-0 self-center opacity-0 group-hover:opacity-100 transition-opacity" />
                        </a>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </>
          ) : (
            <div className="glass-card flex h-full items-center justify-center p-6">
              <p className="text-sm text-muted-foreground">Sélectionnez une action pour afficher ses données historiques.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function PortfolioCrypto() {
  const [cryptos, setCryptos] = useState<CryptoListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(loadSavedCryptoSymbols);
  const [selectedCrypto, setSelectedCrypto] = useState<CryptoListItem | null>(null);
  const [search, setSearch] = useState("");
  const [chartStart, setChartStart] = useState(cryptoChartStartDefault);
  const [chartEnd, setChartEnd] = useState(endDate.toISOString().slice(0, 10));
  const [historyData, setHistoryData] = useState<{ date: string; price: number }[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [newsArticles, setNewsArticles] = useState<NewsArticle[]>([]);
  const [newsLoading, setNewsLoading] = useState(false);

  useEffect(() => {
    saveCryptoSymbols(selectedSymbols);
  }, [selectedSymbols]);

  useEffect(() => {
    fetchCryptoList()
      .then(setCryptos)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedCrypto) {
      setHistoryData([]);
      return;
    }
    setHistoryLoading(true);
    fetchCryptoHistory(selectedCrypto.symbol, chartStart, chartEnd)
      .then((res) => {
        const dates = res.dates || [];
        const series = res.series?.[selectedCrypto.symbol];
        if (dates.length && series?.length) {
          setHistoryData(dates.map((d, i) => ({ date: d, price: series[i] ?? 0 })));
        } else {
          setHistoryData([]);
        }
      })
      .catch(() => setHistoryData([]))
      .finally(() => setHistoryLoading(false));
  }, [selectedCrypto, chartStart, chartEnd]);

  useEffect(() => {
    if (!selectedCrypto) {
      setNewsArticles([]);
      return;
    }
    const controller = new AbortController();
    setNewsLoading(true);
    fetchCryptoNewsYahooSymbol(selectedCrypto.symbol)
      .then((yahoo) => fetchNews(yahoo, 12, controller.signal))
      .then((res) => setNewsArticles(res.articles || []))
      .catch(() => setNewsArticles([]))
      .finally(() => setNewsLoading(false));
    return () => controller.abort();
  }, [selectedCrypto]);

  const formatNewsDate = (iso: string) => {
    if (!iso) return "";
    try {
      const d = new Date(iso);
      const now = new Date();
      const diffMs = now.getTime() - d.getTime();
      const diffH = Math.floor(diffMs / (1000 * 60 * 60));
      const diffD = Math.floor(diffMs / (1000 * 60 * 60 * 24));
      if (diffH < 1) return "À l'instant";
      if (diffH < 24) return `Il y a ${diffH}h`;
      if (diffD < 7) return `Il y a ${diffD}j`;
      return d.toLocaleDateString("fr-FR", { day: "numeric", month: "short", year: "numeric" });
    } catch {
      return iso;
    }
  };

  const toggleCrypto = (symbol: string) => {
    setSelectedSymbols((prev) =>
      prev.includes(symbol) ? prev.filter((s) => s !== symbol) : [...prev, symbol]
    );
  };

  const removeCrypto = (symbol: string) => {
    setSelectedSymbols((prev) => prev.filter((s) => s !== symbol));
  };

  const displayCryptos = useMemo(() => {
    const q = search.toLowerCase();
    return cryptos.filter(
      (c) => q === "" || c.symbol.toLowerCase().includes(q) || c.name.toLowerCase().includes(q)
    );
  }, [cryptos, search]);

  const selectAllDisplayed = () => {
    const symbols = displayCryptos.map((c) => c.symbol);
    setSelectedSymbols((prev) => Array.from(new Set([...prev, ...symbols])));
  };

  const allDisplayedSelected =
    displayCryptos.length > 0 && displayCryptos.every((c) => selectedSymbols.includes(c.symbol));

  const chartData = useMemo(() => historyData, [historyData]);
  const lastPrice = chartData.length ? chartData[chartData.length - 1]?.price : null;
  const firstPrice = chartData.length ? chartData[0]?.price : null;
  const change = lastPrice != null && firstPrice != null && firstPrice > 0 ? lastPrice - firstPrice : 0;
  const changePercent = firstPrice != null && firstPrice > 0 ? (change / firstPrice) * 100 : 0;

  const priceFmt = (p: number) => {
    if (p >= 1) return "$" + p.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (p >= 0.0001) return "$" + p.toFixed(6);
    return "$" + p.toExponential(2);
  };

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
      { label: "Dernier cours (USD)", value: lastPrice != null ? priceFmt(lastPrice) : "—", icon: BarChart3, color: "text-foreground" },
    ],
    [lastPrice, firstPrice, change, changePercent]
  );

  return (
    <div className="flex flex-col px-6 py-6" style={{ height: "calc(100vh - 64px)" }}>
      <div className="flex items-start justify-between mb-6 shrink-0">
        <div>
          <p className="section-label mb-2">Mon Portefeuille</p>
          <h1 className="section-title mb-1">Portefeuille crypto</h1>
          <p className="text-xs text-muted-foreground max-w-xl">
            Données de prix locales (CSV). Actualités via Yahoo Finance pour la paire USD.
          </p>
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
          <div className="flex gap-2 mb-3 shrink-0">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Rechercher une crypto…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-9 h-9 text-xs"
              />
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 shrink-0 text-xs font-semibold"
              onClick={selectAllDisplayed}
              disabled={loading || displayCryptos.length === 0 || allDisplayedSelected}
            >
              Tout sélectionner
            </Button>
          </div>

          <div className="space-y-1.5 overflow-y-auto pr-1 min-h-0 flex-1">
            {loading ? (
              <p className="text-xs text-muted-foreground">Chargement…</p>
            ) : error ? (
              <p className="text-xs text-destructive">{error}</p>
            ) : (
              displayCryptos.map((c) => {
                const isSelected = selectedSymbols.includes(c.symbol);
                return (
                  <div
                    key={c.symbol}
                    className={`rounded-xl bg-card flex cursor-pointer items-center justify-between p-3 transition-colors border ${
                      isSelected ? "border-primary" : "border-border hover:shadow-sm"
                    } ${selectedCrypto?.symbol === c.symbol ? "bg-muted/50" : ""}`}
                    onClick={() => setSelectedCrypto(c)}
                  >
                    <div className="min-w-0">
                      <span className="font-display text-xs font-bold text-foreground">{c.symbol}</span>
                      <p className="truncate text-[11px] text-muted-foreground">{c.name}</p>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleCrypto(c.symbol);
                      }}
                      className={`rounded-md px-2 py-1 text-[10px] font-semibold transition-colors ${
                        isSelected
                          ? "bg-primary text-primary-foreground"
                          : "border border-input bg-background text-foreground hover:bg-muted"
                      }`}
                    >
                      {isSelected ? "✓" : "+"}
                    </button>
                  </div>
                );
              })
            )}
          </div>

          <div className="glass-card mt-3 p-4 shrink-0">
            <h3 className="font-display text-xs font-bold text-foreground mb-2">Composition</h3>
            {selectedSymbols.length === 0 ? (
              <p className="text-[11px] text-muted-foreground">Aucune crypto sélectionnée</p>
            ) : (
              <>
                <div className="flex items-center justify-between text-xs mb-2">
                  <span className="text-muted-foreground">
                    {selectedSymbols.length} crypto{selectedSymbols.length > 1 ? "s" : ""}
                  </span>
                </div>
                <div className="mt-1 h-2 w-full rounded-full bg-secondary overflow-hidden">
                  <div className="h-full rounded-full bg-primary transition-all" style={{ width: "100%" }} />
                </div>
                <ul className="mt-2 space-y-1 max-h-24 overflow-y-auto">
                  {selectedSymbols.map((sym) => (
                    <li key={sym} className="flex items-center gap-1.5 text-[11px]">
                      <button
                        onClick={() => removeCrypto(sym)}
                        className="text-muted-foreground hover:text-destructive"
                      >
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
          {selectedCrypto ? (
            <>
              <div className="grid grid-cols-3 gap-3 shrink-0">
                {kpiCards.map((kpi) => (
                  <div key={kpi.label} className="glass-card p-4 text-center">
                    <kpi.icon className="mx-auto h-4 w-4 text-primary mb-1" />
                    <p className={`font-display text-base font-bold ${kpi.color}`}>{kpi.value}</p>
                    <p className="text-[10px] text-muted-foreground mt-0.5">{kpi.label}</p>
                  </div>
                ))}
              </div>

              <div className="glass-card p-6 flex flex-col shrink-0 min-h-[55vh]">
                <div className="flex items-center justify-between mb-4 shrink-0 flex-wrap gap-2">
                  <div>
                    <h3 className="font-display text-base font-bold text-foreground">{selectedCrypto.symbol}</h3>
                    <p className="text-sm text-muted-foreground">{selectedCrypto.name}</p>
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
                  </div>
                </div>
                <div className="flex-1 min-h-[320px]">
                  {historyLoading ? (
                    <p className="text-muted-foreground">Chargement des données…</p>
                  ) : chartData.length === 0 ? (
                    <p className="text-muted-foreground">Aucune donnée pour cette plage.</p>
                  ) : (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis
                          dataKey="date"
                          tick={{ fontSize: 10 }}
                          stroke="hsl(var(--muted-foreground))"
                          interval="preserveStartEnd"
                        />
                        <YAxis
                          tick={{ fontSize: 11 }}
                          stroke="hsl(var(--muted-foreground))"
                          domain={["auto", "auto"]}
                          tickFormatter={(v) => priceFmt(Number(v))}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "hsl(var(--card))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: 8,
                            fontSize: 12,
                          }}
                          formatter={(value: number) => [priceFmt(Number(value)), "Cours USD"]}
                        />
                        <Line type="monotone" dataKey="price" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </div>

              <div className="glass-card p-4 shrink-0">
                <h3 className="font-display text-sm font-bold text-foreground mb-3 flex items-center gap-2">
                  <Newspaper className="h-4 w-4 text-primary" />
                  Actualités — {selectedCrypto.symbol}
                </h3>
                {newsLoading ? (
                  <div className="flex items-center gap-2 text-muted-foreground text-sm py-4">
                    <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                    Chargement des actualités…
                  </div>
                ) : newsArticles.length === 0 ? (
                  <p className="text-sm text-muted-foreground py-2">Aucune actualité récente pour le moment.</p>
                ) : (
                  <ul className="space-y-3 max-h-[280px] overflow-y-auto pr-1">
                    {newsArticles.map((article, idx) => (
                      <li key={idx}>
                        <a
                          href={article.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex gap-3 p-3 rounded-lg border border-border bg-card hover:bg-muted/50 transition-colors group"
                        >
                          {article.thumbnail ? (
                            <img
                              src={article.thumbnail}
                              alt=""
                              className="w-16 h-16 rounded-md object-cover shrink-0 bg-muted"
                              loading="lazy"
                            />
                          ) : (
                            <div className="w-16 h-16 rounded-md bg-muted shrink-0 flex items-center justify-center">
                              <Newspaper className="h-6 w-6 text-muted-foreground" />
                            </div>
                          )}
                          <div className="min-w-0 flex-1">
                            <p className="text-xs font-semibold text-foreground line-clamp-2 group-hover:text-primary">
                              {article.title}
                            </p>
                            <div className="flex items-center gap-2 mt-1 text-[11px] text-muted-foreground">
                              {article.publisher && <span>{article.publisher}</span>}
                              {article.publishedAt && (
                                <>
                                  {article.publisher && <span>·</span>}
                                  <span>{formatNewsDate(article.publishedAt)}</span>
                                </>
                              )}
                            </div>
                            {article.summary && (
                              <p className="text-[11px] text-muted-foreground mt-1 line-clamp-2">{article.summary}</p>
                            )}
                          </div>
                          <ExternalLink className="h-3.5 w-3.5 text-muted-foreground shrink-0 self-center opacity-0 group-hover:opacity-100 transition-opacity" />
                        </a>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </>
          ) : (
            <div className="glass-card flex h-full items-center justify-center p-6">
              <p className="text-sm text-muted-foreground">
                Sélectionnez une crypto pour afficher le cours et les actualités.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const Portfolio = () => {
  const { mode } = useAppMode();
  return mode === "crypto" ? <PortfolioCrypto /> : <PortfolioStocks />;
};

export default Portfolio;
