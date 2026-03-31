const PORTFOLIO_STORAGE_KEY = "pe25-portfolio-symbols";
const CRYPTO_PORTFOLIO_STORAGE_KEY = "pe25-portfolio-crypto-symbols";

export function loadSavedSymbols(): string[] {
  try {
    const raw = localStorage.getItem(PORTFOLIO_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((s): s is string => typeof s === "string") : [];
  } catch {
    return [];
  }
}

export function saveSymbols(symbols: string[]) {
  localStorage.setItem(PORTFOLIO_STORAGE_KEY, JSON.stringify(symbols));
}

export function loadSavedCryptoSymbols(): string[] {
  try {
    const raw = localStorage.getItem(CRYPTO_PORTFOLIO_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((s): s is string => typeof s === "string") : [];
  } catch {
    return [];
  }
}

export function saveCryptoSymbols(symbols: string[]) {
  localStorage.setItem(CRYPTO_PORTFOLIO_STORAGE_KEY, JSON.stringify(symbols));
}
