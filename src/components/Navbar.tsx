import { Link, useLocation } from "react-router-dom";
import { TrendingUp } from "lucide-react";

const navItems = [
  { label: "Accueil", path: "/" },
  { label: "Mon Portefeuille", path: "/portfolio" },
  { label: "Simulation", path: "/simulation" },
  { label: "Architecture", path: "/architecture" },
];

const Navbar = () => {
  const location = useLocation();

  return (
    <nav className="sticky top-0 z-50 border-b border-border bg-card/80 backdrop-blur-md px-6">
      <div className="flex h-14 items-center justify-between">
        <Link to="/" className="flex items-center gap-2 font-display text-base font-bold text-foreground">
          <TrendingUp className="h-5 w-5 text-primary" />
          Portfolio Manager
        </Link>
        <div className="flex h-full items-center gap-1">
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
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
