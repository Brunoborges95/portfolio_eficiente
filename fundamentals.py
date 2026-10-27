"""Company fundamentals from the CVM's open data (dados.cvm.gov.br), point in time.

The annual (DFP) and quarterly (ITR) statements of every listed company since 2011 come with
the date each filing was first received by the CVM, so an indicator computed for a past date
only uses what had been published by then. The values are those of the latest version of
each filing: restatements (about 5% of filings, usually within days) are a small look-ahead.
Tickers come from the securities table of the FCA (Formulario Cadastral).
"""
import time
import unicodedata
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CVM_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/{doc}/DADOS/{name}"
DATASETS = Path(__file__).resolve().parent / "datasets"
RAW_DIR = DATASETS / "cvm"  # downloaded zips, not versioned
TABLE_PATH = DATASETS / "reference" / "cvm_fundamentals.csv.gz"
TICKERS_PATH = DATASETS / "reference" / "cvm_tickers.csv"
FIRST_YEAR = 2011
STATEMENTS = ("DRE", "BPA", "BPP")
# Columns that a universe snapshot keeps for each stock
SNAPSHOT_COLUMNS = ["cnpj", "filing", "pe", "pb", "roe", "net_margin", "ebit_margin", "net_debt_to_equity",
                    "revenue_growth", "earnings_growth", "profitable_years", "fiscal_years", "roe_5y",
                    "momentum", "return_12m", "volatility", "value_score", "quality_score", "momentum_score",
                    "low_risk_score", "fundamental_score"]
# Indicator groups of the factor score; the sign makes "higher is better"
FACTORS = {
    "value": {"earnings_yield": 1, "book_to_market": 1, "dividend_yield": 1},
    "quality": {"roe": 1, "net_margin": 1, "net_debt_to_equity": -1, "profitable_years": 1},
    "momentum": {"momentum": 1},
    "low_risk": {"volatility": -1},
}


def _plain(text):
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower().strip()


# ---- download ----
def _zip_path(doc, year):
    return RAW_DIR / f"{doc.lower()}_cia_aberta_{year}.zip"


def download(doc, year, max_age_days=7):
    """Zip of one year of DFP, ITR or FCA filings, cached in RAW_DIR. Recent years are
    downloaded again when the copy is older than max_age_days (the CVM updates them weekly)."""
    path = _zip_path(doc, year)
    recent = year >= date.today().year - 1
    if path.exists() and (not recent or time.time() - path.stat().st_mtime < max_age_days * 86400):
        return path
    url = CVM_URL.format(doc=doc, name=path.name)
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=300)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
            return path
        except requests.RequestException:
            if attempt == 2:
                if path.exists():
                    return path
                raise
            time.sleep(5 * 2 ** attempt)


# ---- extraction ----
def _read(zf, name, **kw):
    if name not in zf.namelist():
        return pd.DataFrame()
    return pd.read_csv(zf.open(name), sep=";", encoding="latin-1", dtype={"CD_CONTA": str, "CNPJ_CIA": str}, **kw)


def classify_accounts(code, label):
    """Field name of each statement line (None for lines that are not used). Accounts are
    found by code and description, because banks and insurers use other charts of accounts."""
    label = label.map(_plain)
    level2 = code.str.fullmatch(r"\d\.\d\d")
    field = pd.Series(None, index=code.index, dtype=object)
    field[code == "3.01"] = "revenue"
    field[(code == "3.05") & label.str.contains("antes do resultado financeiro")] = "ebit"
    net_income = r"lucro.{0,3}(ou )?prejuizo (liquido )?(consolidado )?do periodo"
    field[level2 & code.str.startswith("3.") & label.str.fullmatch(net_income)] = "net_income"
    field[code.str.fullmatch(r"3\.\d\d\.01") & label.str.contains("controladora")
          & ~label.str.contains("nao controlador")] = "net_income_ctrl"
    field[code == "1"] = "assets"
    field[(code == "1.01.01") & label.str.contains("caixa")] = "cash"
    field[(code == "1.01.02") & label.str.contains("aplicac")] = "investments"
    field[level2 & code.str.startswith("2.") & label.str.startswith("patrimonio liquido")] = "equity"
    field[code.str.fullmatch(r"2\.\d\d\.\d\d") & label.str.contains("nao controlador")] = "minority"
    field[code.isin(["2.01.04", "2.02.01"]) & label.str.contains("emprestimos")] = "debt"
    return field


def extract_year(doc, year):
    """Long table (cnpj, period_end, field, value) of one year of DFP or ITR filings, with
    the date each filing was first received. Prefers consolidated statements and falls back
    to the individual ones for companies without subsidiaries."""
    path = download(doc, year)
    if path is None:
        return pd.DataFrame()
    name = f"{doc.lower()}_cia_aberta"
    with zipfile.ZipFile(path) as zf:
        index = _read(zf, f"{name}_{year}.csv")
        received = (index.groupby(["CNPJ_CIA", "DT_REFER"])
                    .agg(received=("DT_RECEB", "min"), cd_cvm=("CD_CVM", "first"), company=("DENOM_CIA", "first"))
                    .reset_index())
        parts = []
        for statement in STATEMENTS:
            for basis in ("con", "ind"):
                df = _read(zf, f"{name}_{statement}_{basis}_{year}.csv")
                if df.empty:
                    continue
                df["field"] = classify_accounts(df["CD_CONTA"], df["DS_CONTA"])
                df = df[df["field"].notna()].copy()
                if statement == "DRE":
                    # Year-to-date rows only (quarterly filings also carry the quarter alone);
                    # the previous year's comparative becomes <field>_prev
                    df = df[df["DT_INI_EXERC"] == df.groupby(["CNPJ_CIA", "DT_REFER", "ORDEM_EXERC", "CD_CONTA"])
                            ["DT_INI_EXERC"].transform("min")]
                    df.loc[df["ORDEM_EXERC"] != "ÚLTIMO", "field"] += "_prev"
                    # Banks and insurers do not report "revenue from sales of goods and services"
                    revenue = df[(df["CD_CONTA"] == "3.01") & (df["ORDEM_EXERC"] == "ÚLTIMO")]
                    flag = ~revenue["DS_CONTA"].map(_plain).str.startswith("receita de venda")
                    parts.append(pd.DataFrame({"CNPJ_CIA": revenue["CNPJ_CIA"], "DT_REFER": revenue["DT_REFER"],
                                               "field": "financial", "value": flag.astype(float), "basis": basis}))
                else:
                    df = df[df["ORDEM_EXERC"] == "ÚLTIMO"]
                scale = np.where(df["ESCALA_MOEDA"].str.upper() == "MIL", 1000.0, 1.0)
                df["value"] = df["VL_CONTA"] * scale
                df = df.groupby(["CNPJ_CIA", "DT_REFER", "field"], as_index=False)["value"].sum()  # debt: short + long term
                df["basis"] = basis
                parts.append(df)
        capital = _read(zf, f"{name}_composicao_capital_{year}.csv")
    if capital is not None and not capital.empty:
        shares = pd.DataFrame({
            "CNPJ_CIA": capital["CNPJ_CIA"], "DT_REFER": capital["DT_REFER"],
            "shares": capital["QT_ACAO_TOTAL_CAP_INTEGR"] - capital["QT_ACAO_TOTAL_TESOURO"].fillna(0),
            "shares_on": capital["QT_ACAO_ORDIN_CAP_INTEGR"] - capital["QT_ACAO_ORDIN_TESOURO"].fillna(0),
        }).melt(["CNPJ_CIA", "DT_REFER"], var_name="field")
        shares["basis"] = "shares"
        parts.append(shares)
    long = pd.concat(parts, ignore_index=True)
    # Consolidated figures win over individual ones, company by company
    has_con = long[long["basis"] == "con"].groupby(["CNPJ_CIA", "DT_REFER"]).size().rename("has_con")
    long = long.join(has_con, on=["CNPJ_CIA", "DT_REFER"])
    keep = long["basis"].isin(["con", "shares"]) | long["has_con"].isna()
    long = long[keep].drop(columns=["basis", "has_con"])
    long = long.merge(received, on=["CNPJ_CIA", "DT_REFER"], how="inner")
    long["doc"] = doc
    return long.rename(columns={"CNPJ_CIA": "cnpj", "DT_REFER": "period_end"})


def build_table(first_year=FIRST_YEAR, last_year=None, workers=4, log=print):
    """Extracts every year of DFP and ITR filings into TABLE_PATH (a few MB)."""
    last_year = last_year or date.today().year
    jobs = [(doc, y) for y in range(first_year, last_year + 1) for doc in ("DFP", "ITR")]
    log(f"Downloading and reading {len(jobs)} CVM files (cached in {RAW_DIR})...")
    with ThreadPoolExecutor(workers) as ex:
        frames = list(ex.map(lambda job: extract_year(*job), jobs))
    table = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)
    log(f"Saved {len(table):,} values of {table['cnpj'].nunique()} companies to {TABLE_PATH}")
    return table


def update_table(log=print):
    """Re-reads the current and previous years (the CVM updates them weekly)."""
    if not TABLE_PATH.exists():
        return build_table(log=log)
    table = load_table()
    recent = [date.today().year - 1, date.today().year]
    table = table[~pd.to_datetime(table["period_end"]).dt.year.isin(recent)]
    fresh = [extract_year(doc, y) for y in recent for doc in ("DFP", "ITR")]
    table = pd.concat([table] + [f for f in fresh if not f.empty], ignore_index=True)
    table.to_csv(TABLE_PATH, index=False)
    log(f"Updated {TABLE_PATH} ({table['cnpj'].nunique()} companies)")
    return table


def load_table():
    return pd.read_csv(TABLE_PATH, dtype={"cnpj": str})


def ticker_table(year=None):
    """B3 tickers of each company (CNPJ), from the FCA securities table, cached in TICKERS_PATH."""
    year = year or date.today().year
    path = download("FCA", year) or download("FCA", year - 1)
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if "valor_mobiliario" in n)
        vm = pd.read_csv(zf.open(name), sep=";", encoding="latin-1", dtype=str)
    vm = vm[vm["Codigo_Negociacao"].notna() & vm["Mercado"].eq("Bolsa")]
    out = vm.rename(columns={"CNPJ_Companhia": "cnpj", "Codigo_Negociacao": "symbol", "Valor_Mobiliario": "security"})
    out = out[["cnpj", "symbol", "security"]].drop_duplicates("symbol")
    out.to_csv(TICKERS_PATH, index=False)
    return out


def load_tickers():
    if TICKERS_PATH.exists():
        return pd.read_csv(TICKERS_PATH, dtype=str)
    return ticker_table()


# ---- point-in-time panel ----
def filings(table=None):
    """One row per filing (company, period end) with its fields, the trailing-12-month (TTM)
    income figures and year-over-year growth.

    For a quarterly filing, TTM = last fiscal year + year to date - same period a year
    earlier (both year-to-date figures come in the same filing); an annual filing is TTM."""
    table = load_table() if table is None else table
    wide = table.pivot_table(index=["cnpj", "doc", "period_end", "received", "cd_cvm", "company"],
                             columns="field", values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    for col in ["revenue", "revenue_prev", "ebit", "ebit_prev", "net_income", "net_income_prev",
                "net_income_ctrl", "net_income_ctrl_prev", "equity", "minority", "assets", "cash",
                "investments", "debt", "shares", "shares_on", "financial"]:
        if col not in wide:
            wide[col] = np.nan
    wide["period_end"] = pd.to_datetime(wide["period_end"])
    wide["received"] = pd.to_datetime(wide["received"])
    # Net income attributable to the controlling shareholders when reported
    for suffix in ("", "_prev"):
        wide["ni" + suffix] = wide["net_income_ctrl" + suffix].fillna(wide["net_income" + suffix])
    wide["book"] = wide["equity"] - wide["minority"].fillna(0.0)
    # Some companies report the share count in thousands (Itau, Vale) and the file has no
    # unit column: a book value above R$1,000 per share means thousands
    thousands = (wide["book"].abs() / wide["shares"]) > 1000
    wide.loc[thousands, ["shares", "shares_on"]] *= 1000
    annual = wide[wide["doc"] == "DFP"]
    fy = annual[["cnpj", "period_end", "ni", "revenue", "ebit"]].rename(
        columns={"period_end": "fy_end", "ni": "fy_ni", "revenue": "fy_revenue", "ebit": "fy_ebit"})
    wide = wide.sort_values("period_end")
    fy = fy.sort_values("fy_end")
    wide = pd.merge_asof(wide, fy, left_on="period_end", right_on="fy_end", by="cnpj",
                         allow_exact_matches=False, direction="backward")
    quarterly = wide["doc"] == "ITR"
    for name, fy_col in (("ni", "fy_ni"), ("revenue", "fy_revenue"), ("ebit", "fy_ebit")):
        ttm = wide[name].where(~quarterly, wide[fy_col] + wide[name] - wide[name + "_prev"])
        # A fiscal year older than ~15 months means a missing annual filing
        stale = quarterly & ((wide["period_end"] - wide["fy_end"]).dt.days > 370)
        wide[name + "_ttm"] = ttm.mask(stale)
    wide["revenue_growth"] = wide["revenue"] / wide["revenue_prev"] - 1
    wide["earnings_growth"] = (wide["ni"] - wide["ni_prev"]) / wide["ni_prev"].abs()
    wide["net_debt"] = wide["debt"].fillna(0.0) - wide["cash"].fillna(0.0) - wide["investments"].fillna(0.0)
    return wide.sort_values(["cnpj", "received", "period_end"]).reset_index(drop=True)


def annual_history(panel):
    """Fiscal years of each company: net income, revenue, book equity and ROE."""
    annual = panel[panel["doc"] == "DFP"].copy()
    annual["roe"] = annual["ni"] / annual["book"].where(annual["book"] > 0)
    return annual[["cnpj", "period_end", "received", "ni", "revenue", "book", "roe"]]


def state_at(panel, when):
    """Latest filing of each company published on or before `when`."""
    known = panel[panel["received"] <= pd.Timestamp(when)]
    known = known[known["period_end"] == known.groupby("cnpj")["period_end"].transform("max")]
    return known.sort_values("received").groupby("cnpj").tail(1).set_index("cnpj")


def split_factor_after(splits, symbol, day):
    """Product of the split ratios of `symbol` after `day` (Yahoo's prices are adjusted for
    them retroactively, while share counts are as filed)."""
    if splits is None or symbol not in splits:
        return 1.0
    s = splits[symbol].dropna()
    return float(s[s.index > pd.Timestamp(day)].prod()) if len(s) else 1.0


def market_caps(panel, companies, quotes, when, state, last_filings=3):
    """Market value of each company on `when`: ON shares x ON price + PN shares x PN price
    (or all shares x the one traded class), with the share counts of the filings brought to
    `when` by the split factors. Robust to a bad filing: the median over the last
    `last_filings` filings with share counts. Share counts reported in thousands (the file
    has no unit column) are detected when they give a market value below 2% of the book
    value (or of the revenue). Share counts start in 2020 in the CVM's files: earlier dates
    use the first filings that have one (exact for splits, not for issuances or buybacks)."""
    when = pd.Timestamp(when)
    recent = quotes["close"].loc[:when].iloc[-10:]

    def last_price(s):
        return recent[s].dropna().iloc[-1] if isinstance(s, str) and s in recent and recent[s].notna().any() else np.nan

    with_shares = panel[panel["cnpj"].isin(companies.index)].dropna(subset=["shares"]).sort_values("received")
    known = with_shares[with_shares["received"] <= when]
    sources = known.groupby("cnpj").tail(last_filings)
    missing = ~companies.index.isin(sources["cnpj"])
    sources = pd.concat([sources, with_shares[with_shares["cnpj"].isin(companies.index[missing])].groupby("cnpj").head(last_filings)])
    ref = state["book"].abs().where(state["book"].abs() > 0).fillna(state["revenue_ttm"].abs())
    out = {}
    for cnpj, rows in sources.groupby("cnpj"):
        c = companies.loc[cnpj]
        on, pn = c.get("on_symbol"), c.get("pn_symbol")
        p_on, p_pn, p_main = last_price(on), last_price(pn), last_price(c["price_symbol"])
        values = []
        for r in rows.itertuples():
            on_shares = r.shares_on if pd.notna(r.shares_on) else np.nan
            if pd.notna(p_on) and pd.notna(p_pn) and pd.notna(on_shares):
                f_on = split_factor_after(quotes.get("splits"), on, r.period_end)
                f_pn = split_factor_after(quotes.get("splits"), pn, r.period_end)
                value = on_shares * f_on * p_on + (r.shares - on_shares) * f_pn * p_pn
            else:
                value = r.shares * split_factor_after(quotes.get("splits"), c["price_symbol"], r.period_end) * p_main
            if pd.notna(ref.get(cnpj)) and value < 0.02 * ref[cnpj]:
                value *= 1000
            values.append(value)
        out[cnpj] = np.nanmedian(values) if np.isfinite(values).any() else np.nan
    return pd.Series(out, dtype=float).reindex(companies.index)


def indicators(panel, companies, quotes, when, history=None):
    """Valuation, profitability, leverage, growth and market indicators of each company on
    `when`, using only filings published by then and prices up to that day.

    companies: DataFrame indexed by cnpj with `price_symbol` (for the market indicators) and
    `on_symbol` / `pn_symbol` (for the market value); quotes: market_data.download_quotes
    output covering at least a year before `when`."""
    when = pd.Timestamp(when)
    state = state_at(panel, when).reindex(companies.index)
    close = quotes["close"].loc[:when]
    if close.empty:
        return pd.DataFrame(index=companies.index)
    day = close.index[-1]
    from market_data import total_return_index  # local import: market_data is heavier
    tr = total_return_index(close, quotes["dividends"].loc[:when])
    out = pd.DataFrame(index=companies.index)
    out["filing"] = state["period_end"]
    out["received"] = state["received"]
    out["financial"] = state["financial"].fillna(0.0) > 0
    sym = companies["price_symbol"]
    out["market_cap"] = market_caps(panel, companies, quotes, when, state)
    out["pe"] = (out["market_cap"] / state["ni_ttm"]).where(state["ni_ttm"] > 0)
    out["earnings_yield"] = state["ni_ttm"] / out["market_cap"]
    out["pb"] = (out["market_cap"] / state["book"]).where(state["book"] > 0)
    out["book_to_market"] = state["book"] / out["market_cap"]
    out["roe"] = (state["ni_ttm"] / state["book"]).where(state["book"] > 0)
    out["net_margin"] = (state["ni_ttm"] / state["revenue_ttm"]).where(~out["financial"] & (state["revenue_ttm"] > 0))
    out["ebit_margin"] = (state["ebit_ttm"] / state["revenue_ttm"]).where(~out["financial"] & (state["revenue_ttm"] > 0))
    out["net_debt_to_equity"] = (state["net_debt"] / state["book"]).where(~out["financial"] & (state["book"] > 0))
    out["revenue_growth"] = state["revenue_growth"]
    out["earnings_growth"] = state["earnings_growth"]
    hist = annual_history(panel) if history is None else history
    hist = hist[hist["received"] <= when].sort_values("period_end").groupby("cnpj").tail(5)
    stats = hist.groupby("cnpj").agg(profitable_years=("ni", lambda x: int((x > 0).sum())),
                                     fiscal_years=("ni", "size"), roe_5y=("roe", "mean"))
    out = out.join(stats)
    # Market indicators from the total return index of the priced ticker
    past = tr.loc[:day]

    def window_return(s, start, end):
        if s not in past or len(past) <= start:
            return np.nan
        col = past[s]
        return col.iloc[-1 - end] / col.iloc[-1 - start] - 1 if pd.notna(col.iloc[-1 - start]) else np.nan

    out["momentum"] = [window_return(s, 252, 21) for s in sym]
    out["return_12m"] = [window_return(s, 252, 0) for s in sym]
    returns = past.iloc[-253:].pct_change(fill_method=None)
    out["volatility"] = [returns[s].std() * np.sqrt(252) if s in returns else np.nan for s in sym]
    paid = quotes["dividends"].loc[day - pd.Timedelta(days=365):day]
    last = close.iloc[-10:].ffill().iloc[-1]
    out["dividend_yield"] = [paid[s].sum() / last[s] if s in paid and s in last and last[s] > 0 else np.nan for s in sym]
    return out


def factor_scores(ind, sectors, min_peers=5):
    """Sector-neutral z-scores of each indicator (winsorized at the 5th/95th percentiles),
    averaged into the groups of FACTORS and then into a composite score."""
    sectors = sectors.reindex(ind.index).fillna("Other")
    out = pd.DataFrame(index=ind.index)
    for group, members in FACTORS.items():
        zs = []
        for col, sign in members.items():
            if col not in ind:
                continue
            x = pd.to_numeric(ind[col], errors="coerce") * sign
            lo, hi = x.quantile(0.05), x.quantile(0.95)
            x = x.clip(lo, hi)
            z = (x - x.mean()) / x.std()
            # Within the sector when it has enough companies
            by_sector = x.groupby(sectors)
            sector_z = (x - by_sector.transform("mean")) / by_sector.transform("std")
            big = sectors.map(sectors.value_counts()) >= min_peers
            zs.append(sector_z.where(big, z).clip(-3, 3))
        if zs:
            out[f"{group}_score"] = pd.concat(zs, axis=1).mean(axis=1)
    out["fundamental_score"] = out.mean(axis=1)
    return out


def companies_for(universe_rows, tickers=None):
    """Companies (cnpj) behind the stocks of the universe, with the tickers whose prices give
    the market value: the most traded ON share, the most traded PN share, and the most traded
    of the two for the market indicators (units bundle several shares and are not used)."""
    tickers = load_tickers() if tickers is None else tickers
    rows = universe_rows.merge(tickers, on="symbol", how="inner")
    rows = rows[~rows["symbol"].str.endswith("11")].sort_values("adtv", ascending=False)
    main = rows.groupby("cnpj").agg(price_symbol=("symbol", "first"), sector=("sector", "first"))
    on = rows[rows["security"].str.contains("Ordin", na=False)].groupby("cnpj")["symbol"].first()
    pn = rows[rows["security"].str.contains("Prefer", na=False)].groupby("cnpj")["symbol"].first()
    main = main.join(on.rename("on_symbol")).join(pn.rename("pn_symbol"))
    return main, tickers[tickers["cnpj"].isin(main.index)]


def quote_symbols(companies):
    """Every ticker needed by indicators() for these companies."""
    cols = [c for c in ("price_symbol", "on_symbol", "pn_symbol") if c in companies]
    return sorted(set(companies[cols].stack().dropna()))


def snapshot_columns(universe_df, when=None, quotes=None, log=print):
    """Fundamental indicators and scores of the universe's stocks on `when` (default: today),
    one row per ticker, to be stored in a universe snapshot."""
    from market_data import download_quotes
    when = pd.Timestamp(when or date.today())
    stocks = universe_df[universe_df["asset_class"] == "Stocks"]
    companies, tickers = companies_for(stocks[["symbol", "sector", "adtv"]])
    if quotes is None:
        log(f"Downloading two years of prices of {len(companies)} companies...")
        quotes = download_quotes(quote_symbols(companies), when - pd.Timedelta(days=760), when + pd.Timedelta(days=1))
    panel = filings()
    ind = indicators(panel, companies, quotes, when)
    scores = factor_scores(ind, companies["sector"])
    per_company = ind.join(scores).join(companies[["price_symbol"]])
    per_company.index.name = "cnpj"
    out = tickers[tickers["symbol"].isin(stocks["symbol"])].merge(per_company.reset_index(), on="cnpj")
    return out.drop(columns=["security"]).set_index("symbol")
