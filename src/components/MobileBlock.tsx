import { TrendingUp } from "lucide-react";

const MobileBlock = () => (
  <div className="flex min-h-screen flex-col items-center justify-center bg-background px-6">
    <div className="glass-card flex max-w-md flex-col items-center gap-6 p-8 text-center md:p-10">
      <div className="flex items-center gap-2 font-display text-base font-bold text-foreground">
        <TrendingUp className="h-6 w-6 shrink-0 text-primary" />
        <span>Portfolio Optimizer</span>
      </div>
      <p className="text-sm text-muted-foreground leading-relaxed">
        Application réservée à des écrans plus larges, passez sur un ordinateur ou une tablette.
      </p>
    </div>
  </div>
);

export default MobileBlock;
