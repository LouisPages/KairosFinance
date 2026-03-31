import { Link, useLocation } from "react-router-dom";
import { TrendingUp, LineChart, Coins } from "lucide-react";
import { useAppMode } from "@/context/AppModeContext";

const navItems = [
  { label: "Accueil", path: "/" },
  { label: "Mon Portefeuille", path: "/portfolio" },
  { label: "Simulation", path: "/simulation" },
  { label: "Historique", path: "/history" },
  { label: "Architecture", path: "/architecture" },
];

const Navbar = () => {
  const location = useLocation();
  const { mode, setMode } = useAppMode();

  return (
    <nav className="sticky top-0 z-50 border-b border-border bg-card/80 backdrop-blur-md px-6">
      <div className="flex h-14 items-center justify-between gap-4">
        <Link to="/" className="flex shrink-0 items-center gap-2 font-display text-base font-bold text-foreground">
          <TrendingUp className="h-5 w-5 text-primary" />
          Portfolio Manager
        </Link>
        <div className="flex min-w-0 flex-1 items-center justify-end gap-1">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                location.pathname === item.path
                  ? "text-primary"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {item.label}
            </Link>
          ))}
          <div
            className="ml-2 flex shrink-0 items-center rounded-full border border-border bg-muted/50 p-0.5"
            title="Type d'actifs"
            role="group"
            aria-label="Choisir le mode actions ou cryptomonnaies"
          >
            <button
              type="button"
              onClick={() => setMode("actions")}
              className={`flex items-center gap-1.5 rounded-full px-2.5 py-1.5 text-xs font-semibold transition-colors ${
                mode === "actions"
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <LineChart className="h-3.5 w-3.5" aria-hidden />
              Actions
            </button>
            <button
              type="button"
              onClick={() => setMode("crypto")}
              className={`flex items-center gap-1.5 rounded-full px-2.5 py-1.5 text-xs font-semibold transition-colors ${
                mode === "crypto"
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Coins className="h-3.5 w-3.5" aria-hidden />
              Cryptos
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
