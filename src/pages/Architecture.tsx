import { motion } from "framer-motion";
import { Brain, ArrowDown, Server, Database, BarChart3 } from "lucide-react";

const models = [
  {
    id: "markowitz-classic",
    name: "Markowitz Classique",
    desc: "L'approche fondamentale d'optimisation moyenne-variance proposée par Harry Markowitz en 1952.",
    details: [
      "Matrice de covariance historique des rendements",
      "Frontière efficiente par optimisation quadratique",
      "Contrainte de poids positifs (pas de vente à découvert)",
      "Maximisation du ratio de Sharpe",
    ],
  },
  {
    id: "markowitz-shrinkage",
    name: "Markowitz Shrinkage",
    desc: "Amélioration de l'estimation de la matrice de covariance par la méthode de Ledoit-Wolf.",
    details: [
      "Shrinkage de Ledoit-Wolf pour la covariance",
      "Réduction du bruit d'estimation",
      "Portefeuilles plus stables et diversifiés",
      "Même optimisation quadratique que le classique",
    ],
  },
  {
    id: "black-litterman",
    name: "Black-Litterman",
    desc: "Combine l'équilibre du marché avec des vues subjectives de l'investisseur.",
    details: [
      "Rendements d'équilibre via le CAPM",
      "Intégration de vues (views) subjectives",
      "Paramètre tau pour la confiance dans le prior",
      "Allocation plus intuitive et robuste",
    ],
  },
];

const fade = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({ opacity: 1, y: 0, transition: { delay: i * 0.12, duration: 0.5 } }),
};

const Architecture = () => {
  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
        <p className="section-label mb-2">Architecture</p>
        <h1 className="section-title mb-1">Modèles d'optimisation</h1>
        <p className="mb-10 text-sm text-muted-foreground">
          Trois approches basées sur la théorie moderne du portefeuille, connectées via FastAPI.
        </p>
      </motion.div>

      {/* Pipeline */}
      <div className="glass-card mb-12 p-6 md:p-8">
        <h2 className="font-display text-base font-bold text-foreground mb-6">Pipeline de bout en bout</h2>
        <div className="flex flex-col items-center gap-3">
          {[
            { icon: Database, label: "Données historiques (Yahoo Finance)" },
            { icon: ArrowDown, label: "" },
            { icon: Server, label: "API FastAPI — Prétraitement & Feature Engineering" },
            { icon: ArrowDown, label: "" },
            { icon: Brain, label: "Modèle d'optimisation (Markowitz / Shrinkage / BL)" },
            { icon: ArrowDown, label: "" },
            { icon: BarChart3, label: "Backtesting & Rapport (Sharpe, Vol, Drawdown)" },
          ].map((step, i) =>
            step.label === "" ? (
              <ArrowDown key={i} className="h-5 w-5 text-muted-foreground" />
            ) : (
              <div key={i} className="flex items-center gap-3 rounded-lg border border-border bg-background px-5 py-3 w-full max-w-md">
                <step.icon className="h-5 w-5 shrink-0 text-primary" />
                <span className="text-xs font-medium text-foreground">{step.label}</span>
              </div>
            )
          )}
        </div>
      </div>

      {/* Models */}
      <div className="grid gap-6 md:grid-cols-3">
        {models.map((model, i) => (
          <motion.div
            key={model.id}
            custom={i}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fade}
            className="glass-card p-6"
          >
            <Brain className="h-6 w-6 text-primary" />
            <h3 className="mt-3 font-display text-base font-bold text-foreground">{model.name}</h3>
            <p className="mt-1 text-sm text-muted-foreground">{model.desc}</p>
            <ul className="mt-4 space-y-1.5">
              {model.details.map((d) => (
                <li key={d} className="flex items-start gap-2 text-xs text-muted-foreground">
                  <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                  {d}
                </li>
              ))}
            </ul>
          </motion.div>
        ))}
      </div>

      {/* Tech stack note */}
      <div className="mt-12 glass-card p-6 text-center">
        <p className="text-xs text-muted-foreground">
          <span className="font-semibold text-foreground">Stack technique :</span> React + TypeScript (frontend) · FastAPI + Python (backend) · NumPy / SciPy / cvxpy (optimisation)
        </p>
      </div>
    </div>
  );
};

export default Architecture;
