"""
Charge les facteurs Fama-French depuis le site de Ken French.
Supporte les datasets 3 facteurs (Mkt-RF, SMB, HML, RF)
et 5 facteurs (Mkt-RF, SMB, HML, RMW, CMA, RF).
"""
import io
import zipfile
import pandas as pd
import httpx

_3FACTORS_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
_5FACTORS_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
_MOMENTUM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"


def _fetch_and_parse(url: str, col_names: list[str], start: str | None, end: str | None) -> pd.DataFrame:
    with httpx.Client(timeout=30) as client:
        resp = client.get(url)
        resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content), "r") as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            data = f.read().decode("utf-8", errors="ignore")

    lines = [ln.strip() for ln in data.replace("\r\n", "\n").split("\n") if ln.strip()]
    start_row = 0
    for i, line in enumerate(lines):
        parts = line.split()
        if parts and len(parts[0]) >= 6 and parts[0][:6].isdigit():
            start_row = i
            break

    # Stopper à la première ligne non numérique (section annuelle)
    end_row = len(lines)
    for i in range(start_row + 1, len(lines)):
        parts = lines[i].split(",")
        if parts and len(parts[0].strip()) == 4:
            try:
                int(parts[0].strip())
                end_row = i
                break
            except ValueError:
                pass

    csv_content = "\n".join(lines[start_row:end_row])
    df = pd.read_csv(
        io.StringIO(csv_content),
        index_col=0,
        header=None,
        names=col_names,
    )
    df.index = df.index.astype(str).str.strip()
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    df.index = pd.to_datetime(df.index.astype(str) + "01", format="%Y%m%d").to_period("M")

    try:
        start_p = pd.Period(pd.Timestamp(start).strftime("%Y-%m"), freq="M") if start else None
        end_p = pd.Period(pd.Timestamp(end).strftime("%Y-%m"), freq="M") if end else None
        if start_p is not None:
            df = df.loc[df.index >= start_p]
        if end_p is not None:
            df = df.loc[df.index <= end_p]
    except Exception:
        pass

    return df


def load_famafrench_factors(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Charge les 3 facteurs Fama-French mensuels (Mkt-RF, SMB, HML, RF)."""
    return _fetch_and_parse(
        _3FACTORS_URL,
        ["Date", "Mkt-RF", "SMB", "HML", "RF"],
        start,
        end,
    )


def load_famafrench_5factors(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Charge les 5 facteurs Fama-French mensuels (Mkt-RF, SMB, HML, RMW, CMA, RF)."""
    return _fetch_and_parse(
        _5FACTORS_URL,
        ["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
        start,
        end,
    )


def load_momentum_factor(start: str | None = None, end: str | None = None) -> pd.Series:
    """Charge le facteur Momentum mensuel (UMD) de Ken French."""
    df = _fetch_and_parse(
        _MOMENTUM_URL,
        ["Date", "UMD"],
        start,
        end,
    )
    return df["UMD"]
