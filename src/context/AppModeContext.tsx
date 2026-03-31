import { createContext, useCallback, useContext, useMemo, useState, useEffect } from "react";

export type AppAssetMode = "actions" | "crypto";

const STORAGE_KEY = "pe25-app-asset-mode";

type AppModeContextValue = {
  mode: AppAssetMode;
  setMode: (m: AppAssetMode) => void;
  toggleMode: () => void;
};

const AppModeContext = createContext<AppModeContextValue | null>(null);

export function AppModeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setModeState] = useState<AppAssetMode>(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw === "crypto" || raw === "actions") return raw;
    } catch {
      /* ignore */
    }
    return "actions";
  });

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, mode);
    } catch {
      /* ignore */
    }
  }, [mode]);

  const setMode = useCallback((m: AppAssetMode) => setModeState(m), []);
  const toggleMode = useCallback(
    () => setModeState((prev) => (prev === "actions" ? "crypto" : "actions")),
    []
  );

  const value = useMemo(
    () => ({ mode, setMode, toggleMode }),
    [mode, setMode, toggleMode]
  );

  return <AppModeContext.Provider value={value}>{children}</AppModeContext.Provider>;
}

export function useAppMode(): AppModeContextValue {
  const ctx = useContext(AppModeContext);
  if (!ctx) throw new Error("useAppMode must be used within AppModeProvider");
  return ctx;
}
