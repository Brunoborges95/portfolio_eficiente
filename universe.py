"""Investable universe: B3 stocks, BDRs, FIIs, Fiagros, FI-Infras, ETFs, Tesouro Direto bonds
and the CDI.

A snapshot is saved as datasets/<YYYY-MM-DD>/universe.csv with that day's liquidity and
analyst data, so that a backtest can use only what was known on each rebalance date. Older
snapshots in the legacy df_stocks_info.csv format are read too.
"""
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import analysts
import market_data
from market_data import CDI

DATASETS = Path(__file__).resolve().parent / "datasets"
BDR_MAP_PATH = DATASETS / "reference" / "bdr_underlying.csv"
BRAPI_URL = "https://brapi.dev/api/quote/list"

ASSET_CLASSES = ["Stocks", "BDRs", "FIIs", "Fixed income", "Brazil ETFs", "Global ETFs", "Crypto", "Gold & FX"]
# How many of the most traded tickers of each kind enter a snapshot
DEFAULT_TOP = {"Stock": 250, "BDR": 200, "FII": 150, "Fiagro": 30, "FI-Infra": 30, "ETF": 150}
US_EXCHANGES = {"NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS", "NAS", "NYS"}
TICKER_RE = r"^[A-Z0-9]{4}\d{1,2}$"
SHARE_CLASSES = {"3": "ON", "4": "PN", "5": "PNA", "6": "PNB", "7": "PNC", "8": "PND", "11": "Unit"}
# Fund name (upper case, no accents) -> asset class; the first match wins
FUND_RULES = [
    ("Crypto", r"BITCOIN|ETHER|CRYPTO|CRIPTO|SOLANA|\bXRP\b|BLOCKCHAIN"),
    ("Gold & FX", r"\bOURO\b|\bGOLD\b|\bDOLAR\b|\bPRATA\b|\bSILVER\b"),
    ("Fixed income", r"INFRA|INCENTIVAD|TESOURO|SELIC|IMA-?B|IRF-?M|RENDA FIXA|DEBENTURE|LETRAS? FINANCEIRA"
                     r"|TREASURY|IPCA|\bPRE\b|IBOXX|LOAN|\bDI\b|\bCDI\b|\bIDKA\b|LIQUIDEZ|CREDITO PRIVADO"),
    ("Brazil ETFs", r"\bBRAZIL\b|\bBRASIL\b|IBOVESPA|\bIBOV\b|IBRX"),
    ("Global ETFs", r"S&P|NASDAQ|NYSE|FANG|MSCI|FTSE|WORLD|GLOBAL|ALL COUNTRIES|CHINA|EUROP|\bUS\b|U\.S|USA|CRSP"
                    r"|EXTERIOR|ACWI|JAPAN|INDIA|EMERG"),
    ("FIIs", r"\bIFIX\b"),
]
LEGACY_CLASSES = {"ON": ("Stocks", "ON"), "PN": ("Stocks", "PN"), "PNA": ("Stocks", "PNA"),
                  "PNB": ("Stocks", "PNB"), "PNC": ("Stocks", "PNC"), "FII/Unit": ("FIIs", "FII/Unit"),
                  "BDR": ("BDRs", "BDR")}
CDI_ROW = {"symbol": CDI, "name": "CDI (Tesouro Selic, CDBs at 100% of the CDI)",
           "asset_class": "Fixed income", "sub_class": "CDI", "instrument": "CDI", "currency": "BRL"}


def _plain(text):
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    return text.upper()


def _same_name(a, b):
    def norm(s):
        return re.sub(r"[^a-z0-9]", "", _plain(s).lower())
    return bool(a) and norm(a) == norm(b)


def fetch_b3_listing(token=None):
    """Every B3 ticker with its type, sector and last session's volume, from brapi.dev
    (a BRAPI_TOKEN environment variable is sent if set)."""
    token = token or os.getenv("BRAPI_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    frames = []
    for kind in ("stock", "fund", "bdr"):
        page = 1
        while True:
            r = requests.get(BRAPI_URL, headers=headers, timeout=60, params={
                "type": kind, "limit": 5000, "page": page, "sortBy": "volume", "sortOrder": "desc"})
            r.raise_for_status()
            payload = r.json()
            frames.append(pd.DataFrame(payload.get("stocks", [])))
            if not payload.get("hasNextPage"):
                break
            page += 1
    listing = pd.concat(frames, ignore_index=True).rename(columns={"stock": "symbol"})
    return listing.drop_duplicates("symbol")


def _instrument(kind, sub_type):
    if kind == "stock":
        return "Stock"
    if kind == "bdr":
        return "BDR"
    return {"fii": "FII", "fi-agro": "Fiagro", "fi-infra": "FI-Infra", "etf": "ETF"}.get(sub_type)


def classify(symbol, instrument, name):
    """(asset_class, sub_class) of a B3 ticker."""
    if instrument == "Stock":
        return "Stocks", SHARE_CLASSES.get(symbol[4:], "Other")
    if instrument == "BDR":
        return "BDRs", "BDR ETF" if symbol.endswith("39") else "BDR"
    if instrument in ("FII", "Fiagro"):
        return "FIIs", instrument
    if instrument == "FI-Infra":
        return "Fixed income", "FI-Infra"
    text = _plain(name)
    for asset_class, pattern in FUND_RULES:
        if re.search(pattern, text):
            if asset_class == "Fixed income":
                return asset_class, "FI-Infra" if re.search("INFRA|INCENTIVAD", text) else "Fixed income ETF"
            return asset_class, "FII ETF" if asset_class == "FIIs" else "ETF"
    return "Brazil ETFs", "ETF"


def _quote_info(ticker, retries=3):
    for attempt in range(retries):
        try:
            return yf.Ticker(ticker).info or {}
        except Exception as e:
            if "RateLimit" in type(e).__name__ or "Too Many Requests" in str(e):
                time.sleep(5 * 2 ** attempt)
                continue
            return {}
    return {}


def fetch_info(tickers, workers=8):
    """Yahoo Finance quote of each ticker ({} when unavailable)."""
    with ThreadPoolExecutor(workers) as ex:
        return dict(zip(tickers, ex.map(_quote_info, tickers)))


def _search_underlying(name):
    try:
        quotes = yf.Search(name, max_results=8, news_count=0).quotes
    except Exception:
        return None, None
    for q in quotes:
        listed = q.get("exchange") in US_EXCHANGES and q.get("quoteType") in ("EQUITY", "ETF")
        if listed and _same_name(q.get("longname") or q.get("shortname"), name):
            return q["symbol"], q.get("quoteType")
    return None, None


def map_bdrs(names, workers=6, retry_misses_after=90):
    """Underlying US ticker of each BDR (AAPL34 -> AAPL, ROXO34 -> NU), found by searching
    the company name on Yahoo Finance and accepting only exact name matches on US
    exchanges. names: {bdr: company name}. Results are cached in BDR_MAP_PATH."""
    columns = ["bdr", "name", "underlying", "quote_type", "checked"]
    cache = pd.read_csv(BDR_MAP_PATH) if BDR_MAP_PATH.exists() else pd.DataFrame(columns=columns)
    cache = cache.drop_duplicates("bdr", keep="last").set_index("bdr")
    age = (pd.Timestamp.today() - pd.to_datetime(cache["checked"])).dt.days
    known = set(cache.index[cache["underlying"].notna() | (age < retry_misses_after)])
    todo = [b for b, n in names.items() if b not in known and n]
    with ThreadPoolExecutor(workers) as ex:
        found = list(ex.map(lambda b: (b, *_search_underlying(names[b])), todo))
    if found:
        today = date.today().isoformat()
        new = pd.DataFrame([{"bdr": b, "name": names[b], "underlying": u, "quote_type": q, "checked": today}
                            for b, u, q in found]).set_index("bdr")
        cache = pd.concat([cache.drop(index=new.index, errors="ignore"), new])
        BDR_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache.reset_index().sort_values("bdr")[columns].to_csv(BDR_MAP_PATH, index=False)
    return cache.reindex(list(names))[["underlying", "quote_type"]]


def tesouro_rows(as_of=None):
    """Universe rows of the zero-coupon Tesouro Direto bonds quoted on `as_of`."""
    bonds = market_data.tesouro_bonds(as_of)
    return pd.DataFrame({
        "symbol": bonds.index,
        "name": [f"{t} {m:%Y}" for t, m in zip(bonds["type"], bonds["maturity"])],
        "asset_class": "Fixed income", "sub_class": bonds["type"].to_numpy(), "instrument": "Tesouro",
        "currency": "BRL", "price": bonds["price"].to_numpy(), "bond_yield": bonds["yield"].to_numpy(),
        "maturity": bonds["maturity"].dt.strftime("%Y-%m-%d").to_numpy(),
    })


def build_universe(day=None, top=None, token=None, workers=8, log=print):
    """Builds and saves the snapshot of `day` (default: today): the most traded tickers of
    each kind with liquidity, fundamentals and the analyst consensus (for BDRs, the
    consensus on the underlying company), plus the CDI."""
    day = pd.Timestamp(day or date.today()).strftime("%Y-%m-%d")
    top = {**DEFAULT_TOP, **(top or {})}
    log("Listing B3 tickers (brapi.dev)...")
    listing = fetch_b3_listing(token)
    sub_types = listing["subType"] if "subType" in listing else [None] * len(listing)
    listing["instrument"] = [_instrument(k, s) for k, s in zip(listing["type"], sub_types)]
    listing = listing[listing["symbol"].str.match(TICKER_RE) & (listing["close"] >= 1) & listing["instrument"].notna()]
    listing = listing.assign(fin_volume=listing["volume"] * listing["close"]).sort_values("fin_volume", ascending=False)
    listing = pd.concat([g.head(top.get(k, 0)) for k, g in listing.groupby("instrument")])

    log(f"Downloading quotes and analyst data for {len(listing)} tickers (Yahoo Finance)...")
    info = fetch_info([s + ".SA" for s in listing["symbol"]], workers)
    rows = []
    for r in listing.itertuples():
        i = info.get(r.symbol + ".SA", {})
        name = i.get("longName") or i.get("shortName") or r.name
        price = i.get("regularMarketPrice") or i.get("currentPrice") or r.close
        avg_volume = i.get("averageVolume")
        asset_class, sub_class = classify(r.symbol, r.instrument, name)
        dy = i.get("dividendYield")
        row = {
            "symbol": r.symbol, "name": name, "asset_class": asset_class, "sub_class": sub_class,
            "instrument": r.instrument, "sector": i.get("sector") or getattr(r, "sector", None),
            "industry": i.get("industry"), "currency": i.get("currency"), "price": price,
            "adtv": avg_volume * price if avg_volume and price else np.nan,
            "market_cap": i.get("marketCap"), "dividend_yield": dy / 100 if dy is not None else np.nan,
            "price_to_book": i.get("priceToBook"),
        }
        if asset_class == "Stocks":
            row.update(analysts.extract_info(i), analyst_source=r.symbol + ".SA")
        rows.append(row)
    df = pd.DataFrame(rows)

    bdr = df["asset_class"] == "BDRs"
    log(f"Mapping {bdr.sum()} BDRs to their underlying companies...")
    mapping = map_bdrs(dict(zip(df.loc[bdr, "symbol"], df.loc[bdr, "name"])))
    df.loc[bdr, "underlying"] = df.loc[bdr, "symbol"].map(mapping["underlying"])
    is_etf = df["symbol"].map(mapping["quote_type"]).eq("ETF")
    df.loc[bdr & is_etf, "sub_class"] = "BDR ETF"
    stocks_abroad = bdr & df["underlying"].notna() & ~is_etf
    sources = df.loc[stocks_abroad, "underlying"].tolist()
    log(f"Downloading the analyst consensus of {len(sources)} underlying companies...")
    under = fetch_info(sources, workers)
    for idx in df.index[stocks_abroad]:
        u = df.at[idx, "underlying"]
        for col, value in analysts.extract_info(under.get(u, {})).items():
            df.at[idx, col] = value
        df.at[idx, "analyst_source"] = u
        if pd.isna(df.at[idx, "sector"]):
            df.at[idx, "sector"] = under.get(u, {}).get("sector")

    covered = df.loc[pd.to_numeric(df["n_analysts"], errors="coerce").fillna(0) >= 1, "analyst_source"]
    log(f"Downloading rating trends of {len(covered)} covered companies...")
    trends = analysts.fetch_trends(covered.tolist(), workers)
    for col in ("rec_mean_now", "rec_mean_3m", "n_trend"):
        df[col] = df["analyst_source"].map(lambda s, c=col: trends.get(s, {}).get(c, np.nan))

    log("Downloading Tesouro Direto bonds (National Treasury)...")
    try:
        bonds = tesouro_rows()
    except (requests.RequestException, ValueError) as e:
        log(f"Tesouro Direto data unavailable, snapshot saved without bonds: {e}")
        bonds = pd.DataFrame()
    df = pd.concat([df, bonds, pd.DataFrame([CDI_ROW])], ignore_index=True)
    log("Computing the stocks' fundamentals (CVM filings)...")
    try:
        import fundamentals
        fund = fundamentals.snapshot_columns(df, day, log=log)
        df = df.merge(fund[fundamentals.SNAPSHOT_COLUMNS], left_on="symbol", right_index=True, how="left")
    except (requests.RequestException, FileNotFoundError, ValueError) as e:
        log(f"Fundamentals unavailable, snapshot saved without them: {e}")
    df["snapshot"] = day
    folder = DATASETS / day
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / "universe.csv", index=False)
    log(f"Saved {len(df)} assets to {folder / 'universe.csv'}")
    return df


def list_snapshots():
    """Dates (YYYY-MM-DD) of the saved snapshots, oldest first."""
    if not DATASETS.exists():
        return []
    return sorted(
        d.name for d in DATASETS.iterdir()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name)
        and ((d / "universe.csv").exists() or (d / "df_stocks_info.csv").exists())
    )


def _from_legacy(df):
    """Converts a df_stocks_info.csv snapshot (rating only, volume in shares)."""
    df = df.drop_duplicates("symbol")
    classes = df["Type"].map(LEGACY_CLASSES)
    df, classes = df[classes.notna()], classes[classes.notna()]
    out = pd.DataFrame({
        "symbol": df["symbol"], "name": df["full_name"],
        "asset_class": classes.str[0], "sub_class": classes.str[1],
        "avg_volume": df["Avg_Volume"],
        "rec_key": df["Recommendation"].where(df["Recommendation"].isin(analysts.RATINGS)),
    })
    return pd.concat([out, pd.DataFrame([CDI_ROW])], ignore_index=True)


def read_universe(day=None):
    """The latest snapshot on or before `day` (point in time) or, when none is that old, the
    earliest one, with analyst features and scores. Returns (universe, snapshot_day)."""
    snapshots = list_snapshots()
    if not snapshots:
        raise FileNotFoundError("No universe snapshot found; run `python build_dataset.py`.")
    day = pd.Timestamp(day or date.today()).strftime("%Y-%m-%d")
    eligible = [s for s in snapshots if s <= day]
    snapshot = eligible[-1] if eligible else snapshots[0]
    folder = DATASETS / snapshot
    if (folder / "universe.csv").exists():
        df = pd.read_csv(folder / "universe.csv")
        # Classes are derived from the stored names, so rule fixes reach old snapshots too
        funds = df["instrument"].isin(["Stock", "FII", "Fiagro", "FI-Infra", "ETF"])
        if funds.any():
            df.loc[funds, ["asset_class", "sub_class"]] = [
                classify(s, i, n) for s, i, n in df.loc[funds, ["symbol", "instrument", "name"]].itertuples(index=False)
            ]
    else:
        df = _from_legacy(pd.read_csv(folder / "df_stocks_info.csv"))
    if "adtv" not in df or df["adtv"].isna().all():
        df["adtv"] = np.nan
    df = analysts.composite_score(analysts.add_features(df))
    return df.reset_index(drop=True), snapshot
