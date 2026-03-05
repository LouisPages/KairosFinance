"""
Charge les facteurs Fama-French depuis le site de Ken French.
Contourne pandas_datareader qui est incompatible avec pandas 2.2+ (deprecate_kwarg).
"""
import io
import zipfile
import pandas as pd
import httpx


def load_famafrench_factors(start: str, end: str) -> pd.DataFrame:
    """Charge les facteurs Fama-French (Mkt-RF, SMB, HML, RF) mensuels."""
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
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
    csv_content = "\n".join(lines[start_row:])
    df = pd.read_csv(
        io.StringIO(csv_content),
        index_col=0,
        header=None,
        names=["Date", "Mkt-RF", "SMB", "HML", "RF"],
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
    df.columns = ["Mkt-RF", "SMB", "HML", "RF"]
    return df
