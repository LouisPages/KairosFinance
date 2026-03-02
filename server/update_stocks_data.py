"""
Génère server/stocks_data.json à partir de sources sans Wikipedia :
- S&P 500 : CSV GitHub (datasets/s-and-p-500-companies)
- NASDAQ-100 et Dow Jones : listes statiques (indices connus)

À lancer manuellement pour mettre à jour les données :
  cd server && python update_stocks_data.py

Les cours sont récupérés via l'API Yahoo Finance (yfinance) dans l'app.
"""
import csv
import json
import urllib.request
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "stocks_data.json"
SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

# NASDAQ-100 (symbol, name) — liste courante
NASDAQ_100 = [
    ("ADBE", "Adobe Inc."),
    ("AMD", "Advanced Micro Devices"),
    ("ABNB", "Airbnb"),
    ("AMZN", "Amazon"),
    ("AMGN", "Amgen"),
    ("ADI", "Analog Devices"),
    ("AAPL", "Apple Inc."),
    ("AMAT", "Applied Materials"),
    ("ASML", "ASML Holding"),
    ("TEAM", "Atlassian"),
    ("ADSK", "Autodesk"),
    ("ADP", "Automatic Data Processing"),
    ("AVGO", "Broadcom"),
    ("BKR", "Baker Hughes"),
    ("BKNG", "Booking Holdings"),
    ("CDNS", "Cadence Design Systems"),
    ("CHTR", "Charter Communications"),
    ("CTAS", "Cintas"),
    ("CSCO", "Cisco"),
    ("CMCSA", "Comcast"),
    ("COST", "Costco"),
    ("CRWD", "CrowdStrike"),
    ("CSGP", "CoStar Group"),
    ("DDOG", "Datadog"),
    ("DXCM", "DexCom"),
    ("FANG", "Diamondback Energy"),
    ("EA", "Electronic Arts"),
    ("EXC", "Exelon"),
    ("FAST", "Fastenal"),
    ("FTNT", "Fortinet"),
    ("GILD", "Gilead Sciences"),
    ("HON", "Honeywell"),
    ("IDXX", "Idexx Laboratories"),
    ("INTC", "Intel"),
    ("INTU", "Intuit"),
    ("ISRG", "Intuitive Surgical"),
    ("KDP", "Keurig Dr Pepper"),
    ("KLAC", "KLA Corporation"),
    ("KHC", "Kraft Heinz"),
    ("LRCX", "Lam Research"),
    ("LIN", "Linde plc"),
    ("MAR", "Marriott International"),
    ("MRVL", "Marvell Technology"),
    ("MELI", "Mercado Libre"),
    ("META", "Meta Platforms"),
    ("MCHP", "Microchip Technology"),
    ("MU", "Micron Technology"),
    ("MSFT", "Microsoft"),
    ("MDB", "MongoDB"),
    ("MDLZ", "Mondelez International"),
    ("MNST", "Monster Beverage"),
    ("NFLX", "Netflix"),
    ("NVDA", "Nvidia"),
    ("NXPI", "NXP Semiconductors"),
    ("ORLY", "O'Reilly Automotive"),
    ("ODFL", "Old Dominion Freight Line"),
    ("PCAR", "Paccar"),
    ("PANW", "Palo Alto Networks"),
    ("PAYX", "Paychex"),
    ("PYPL", "PayPal"),
    ("PEP", "PepsiCo"),
    ("QCOM", "Qualcomm"),
    ("REGN", "Regeneron Pharmaceuticals"),
    ("ROST", "Ross Stores"),
    ("STX", "Seagate Technology"),
    ("SHOP", "Shopify"),
    ("SBUX", "Starbucks"),
    ("SNPS", "Synopsys"),
    ("TMUS", "T-Mobile US"),
    ("TSLA", "Tesla"),
    ("TXN", "Texas Instruments"),
    ("VRSK", "Verisk Analytics"),
    ("VRTX", "Vertex Pharmaceuticals"),
    ("WBD", "Warner Bros. Discovery"),
    ("WDAY", "Workday"),
    ("XEL", "Xcel Energy"),
    ("ZS", "Zscaler"),
    ("GOOGL", "Alphabet Inc. Class A"),
    ("GOOG", "Alphabet Inc. Class C"),
    ("CEG", "Constellation Energy"),
    ("CPRT", "Copart"),
    ("CSX", "CSX Corporation"),
    ("DASH", "DoorDash"),
    ("FER", "Ferrovial"),
    ("GEHC", "GE HealthCare"),
    ("INSM", "Insmed"),
    ("APP", "AppLovin"),
    ("ARM", "Arm Holdings"),
    ("AXON", "Axon Enterprise"),
    ("CCEP", "Coca-Cola Europacific Partners"),
    ("CTSH", "Cognizant"),
    ("PDD", "PDD Holdings"),
    ("PLTR", "Palantir Technologies"),
    ("MSTR", "MicroStrategy"),
    ("ROP", "Roper Technologies"),
    ("TRI", "Thomson Reuters"),
    ("WMT", "Walmart"),
    ("WDC", "Western Digital"),
]

# Dow Jones Industrial Average (30 composants)
DOW_30 = [
    ("MMM", "3M"),
    ("AXP", "American Express"),
    ("AMGN", "Amgen"),
    ("AMZN", "Amazon"),
    ("AAPL", "Apple Inc."),
    ("BA", "Boeing"),
    ("CAT", "Caterpillar"),
    ("CVX", "Chevron"),
    ("CSCO", "Cisco"),
    ("KO", "Coca-Cola"),
    ("DIS", "Walt Disney"),
    ("GS", "Goldman Sachs"),
    ("HD", "Home Depot"),
    ("HON", "Honeywell"),
    ("IBM", "IBM"),
    ("JNJ", "Johnson & Johnson"),
    ("JPM", "JPMorgan Chase"),
    ("MCD", "McDonald's"),
    ("MRK", "Merck"),
    ("MSFT", "Microsoft"),
    ("NKE", "Nike"),
    ("NVDA", "Nvidia"),
    ("PG", "Procter & Gamble"),
    ("CRM", "Salesforce"),
    ("SHW", "Sherwin-Williams"),
    ("TRV", "Travelers"),
    ("UNH", "UnitedHealth"),
    ("VZ", "Verizon"),
    ("V", "Visa"),
    ("WMT", "Walmart"),
]


def fetch_sp500() -> list[dict]:
    out = []
    try:
        with urllib.request.urlopen(urllib.request.Request(
            SP500_CSV_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PE25/1.0)"},
        ), timeout=15) as r:
            reader = csv.DictReader(r.read().decode("utf-8").splitlines())
            for row in reader:
                symbol = (row.get("Symbol") or "").strip()
                name = (row.get("Security") or "").strip()
                if symbol:
                    out.append({"symbol": symbol, "name": name or symbol, "index": "S&P 500"})
    except Exception as e:
        print(f"Erreur fetch S&P 500: {e}")
    return out


def main() -> None:
    seen: set[tuple[str, str]] = set()
    result: list[dict] = []

    # S&P 500 depuis GitHub
    for s in fetch_sp500():
        key = (s["symbol"], s["index"])
        if key not in seen:
            seen.add(key)
            result.append(s)

    # NASDAQ-100
    for symbol, name in NASDAQ_100:
        key = (symbol, "NASDAQ")
        if key not in seen:
            seen.add(key)
            result.append({"symbol": symbol, "name": name, "index": "NASDAQ"})

    # Dow 30
    for symbol, name in DOW_30:
        key = (symbol, "DOW JONES")
        if key not in seen:
            seen.add(key)
            result.append({"symbol": symbol, "name": name, "index": "DOW JONES"})

    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Écrit {len(result)} actions dans {OUTPUT}")


if __name__ == "__main__":
    main()
