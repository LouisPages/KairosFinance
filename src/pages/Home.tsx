import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { BarChart3, Brain, ArrowRight, PieChart, LineChart, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { fetchStocks } from "@/lib/api";

const STATS_DEFAULTS = [
  { label: "Actions supportées", value: "—" },
  { label: "Modèles de prédiction", value: "3" },
];

const features = [
  {
    icon: PieChart,
    title: "Gestion de portefeuille",
    desc: "Composez votre portefeuille à partir des actions du S&P 500. Visualisez les performances historiques.",
  },
  {
    icon: Brain,
    title: "Modèles de prédiction",
    desc: "Trois modèles d'optimisation basés sur la théorie de Markowitz pour trouver le portefeuille optimal.",
  },
  {
    icon: LineChart,
    title: "Backtesting & Comparaison",
    desc: "80% de données pour l'entraînement, 20% pour le test. Comparez les performances de votre portefeuille optimisé au marché.",
  },
  {
    icon: Shield,
    title: "Métriques de risque",
    desc: "Ratio de Sharpe, volatilité, rendement attendu pour évaluer la qualité de votre portefeuille.",
  },
];

const fade = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({ opacity: 1, y: 0, transition: { delay: i * 0.1, duration: 0.5 } }),
};

const Home = () => {
  const [stocks, setStocks] = useState<Awaited<ReturnType<typeof fetchStocks>> | null>(null);

  useEffect(() => {
    fetchStocks()
      .then(setStocks)
      .catch(() => setStocks(null));
  }, []);

  const sp500Count = stocks !== null ? stocks.filter((s) => s.index === "S&P 500").length : 0;
  const stats = [
    { label: "Actions supportées", value: stocks !== null ? String(sp500Count) : "—" },
    ...STATS_DEFAULTS.slice(1),
  ];

  return (
    <div className="mx-auto max-w-5xl px-6">
      {/* Hero */}
      <section className="flex flex-col items-center py-20 text-center">
        <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="section-label mb-3">
          PE 25 — École Centrale Lyon
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="font-display font-bold leading-tight text-foreground"
          style={{ fontSize: "clamp(1.5rem, 2.2vw, 2.5rem)" }}
        >
          Portfolio Manager
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15, duration: 0.5 }}
          className="mt-4 max-w-xl text-base text-muted-foreground"
        >
          Optimisation de portefeuille boursier par des modèles quantitatifs. Simulez, comparez et analysez vos stratégies d'investissement.
        </motion.p>
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="mt-8">
          <Button asChild size="lg" className="gap-2 rounded-xl px-8 text-sm font-semibold">
            <Link to="/portfolio">
              Commencer <ArrowRight className="h-4 w-4" />
            </Link>
          </Button>
        </motion.div>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-6 text-xs text-muted-foreground"
        >
          {stats.map((s, i) => (
            <span key={s.label} className={i > 0 ? "ml-8" : undefined}>
              <span className="text-muted-foreground">{s.value} {s.label.toLowerCase()}</span>
            </span>
          ))}
        </motion.div>
      </section>

      {/* Problem */}
      <section className="glass-card mb-16 p-8 md:p-10">
        <p className="section-label mb-2">Le défi</p>
        <h2 className="section-title mb-6">Optimiser un portefeuille est complexe</h2>
        <div className="grid gap-4 md:grid-cols-3">
          {[
            { icon: "🤯", title: "Données massives", desc: "Des milliers de points de données historiques à analyser pour chaque action." },
            { icon: "🔍", title: "Corrélations", desc: "Les interdépendances entre actifs rendent l'optimisation manuelle quasi-impossible." },
            { icon: "⚠️", title: "Risque", desc: "Trouver le bon équilibre rendement/risque demande des outils quantitatifs avancés." },
          ].map((item) => (
            <div key={item.title} className="rounded-lg border border-border bg-background p-5 text-center">
              <span className="text-2xl">{item.icon}</span>
              <h3 className="mt-2 font-display text-base font-bold text-foreground">{item.title}</h3>
              <p className="mt-1 text-sm text-muted-foreground">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="mb-20">
        <p className="section-label mb-2">Fonctionnalités</p>
        <h2 className="section-title mb-8">Ce que propose l'application</h2>
        <div className="grid gap-5 md:grid-cols-2">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              custom={i}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={fade}
              className="glass-card p-6"
            >
              <f.icon className="h-6 w-6 text-primary" />
              <h3 className="mt-3 font-display text-base font-bold text-foreground">{f.title}</h3>
              <p className="mt-1 text-sm text-muted-foreground">{f.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="mb-20 flex flex-col items-center rounded-xl bg-primary px-8 py-12 text-center text-primary-foreground">
        <BarChart3 className="mb-4 h-8 w-8" />
        <h2 className="font-display text-xl font-bold">Prêt à optimiser votre portefeuille ?</h2>
        <p className="mt-2 max-w-md text-xs opacity-80">
          Construisez votre portefeuille, lancez une simulation et analysez les résultats.
        </p>
        <Button asChild variant="secondary" size="lg" className="mt-6 gap-2 rounded-xl font-semibold">
          <Link to="/portfolio">
            Accéder au portefeuille <ArrowRight className="h-4 w-4" />
          </Link>
        </Button>
      </section>
    </div>
  );
};

export default Home;
