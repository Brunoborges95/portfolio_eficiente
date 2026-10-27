"""Market data: B3 prices (Yahoo Finance), Tesouro Direto bonds (National Treasury), the CDI
and the Focus survey (Central Bank of Brazil)."""
import functools
import io
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import yfinance as yf

CDI = "CDI"  # synthetic ticker: post-fixed fixed income paying 100% of the CDI (Tesouro Selic, CDBs)
TESOURO_PREFIX = "TD-"  # synthetic tickers of Tesouro Direto bonds, e.g. TD-IPCA2035
SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series}/dados"
FOCUS_URL = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/"
TESOURO_URL = ("https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/"
               "resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv")
# Zero-coupon bonds only: coupon bonds' prices drop on each payment
TESOURO_TYPES = {"Tesouro IPCA+": "IPCA", "Tesouro Prefixado": "PRE"}
TESOURO_CACHE = Path(__file__).resolve().parent / "datasets" / "reference" / "tesouro_direto.csv.gz"
SELIC_CDI_GAP = 0.0010  # the CDI trades about 0.10 p.p. below the Selic target
TRADING_DAYS = 252


def _get(url, params=None, retries=4, timeout=60):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 429 and attempt < retries - 1:  # rate limited: wait and retry
                time.sleep(5 * 2 ** attempt)
                continue
            return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _get_json(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = _get(url, params)
            if r.status_code == 404:
                return []
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def parse_tesouro_csv(content):
    """The Treasury's PrecoTaxaTesouroDireto.csv, reduced to the zero-coupon bonds."""
    df = pd.read_csv(io.BytesIO(content), sep=";", decimal=",", encoding="latin-1")
    df = df[df["Tipo Titulo"].isin(TESOURO_TYPES)].copy()
    df["maturity"] = pd.to_datetime(df["Data Vencimento"], dayfirst=True)
    df["date"] = pd.to_datetime(df["Data Base"], dayfirst=True)
    df["symbol"] = TESOURO_PREFIX + df["Tipo Titulo"].map(TESOURO_TYPES) + df["maturity"].dt.year.astype(str)
    df = df.rename(columns={"Tipo Titulo": "type", "PU Base Manha": "price", "Taxa Compra Manha": "yield"})
    df["yield"] = df["yield"] / 100
    return df[["symbol", "type", "maturity", "date", "price", "yield"]].sort_values(["symbol", "date"])


@functools.lru_cache(maxsize=1)
def tesouro_table():
    """Daily prices and yields of the zero-coupon Tesouro Direto bonds since 2004 (National
    Treasury's open data). Symbols look like TD-IPCA2035 and TD-PRE2029. The table is cached
    in TESOURO_CACHE, refreshed once a day, and the cache is used as is when the Treasury's
    site is unavailable (it often answers 429, too many requests)."""
    fresh = TESOURO_CACHE.exists() and time.time() - TESOURO_CACHE.stat().st_mtime < 86400
    if not fresh:
        try:
            r = _get(TESOURO_URL, timeout=180)
            r.raise_for_status()
            TESOURO_CACHE.parent.mkdir(parents=True, exist_ok=True)
            parse_tesouro_csv(r.content).to_csv(TESOURO_CACHE, index=False)
        except requests.RequestException:
            if not TESOURO_CACHE.exists():
                raise
    return pd.read_csv(TESOURO_CACHE, parse_dates=["maturity", "date"])


def tesouro_bonds(as_of=None, min_years_left=1.0):
    """Bonds quoted on the last date up to `as_of` that mature in more than `min_years_left`."""
    table = tesouro_table()
    if as_of is not None:
        table = table[table["date"] <= pd.Timestamp(as_of)]
    last = table[table["date"] == table["date"].max()]
    last = last[last["maturity"] > last["date"] + pd.DateOffset(days=int(365 * min_years_left))]
    return last.set_index("symbol")[["type", "maturity", "yield", "price"]]


def tesouro_prices(symbols, start, end):
    table = tesouro_table()
    rows = table[table["symbol"].isin(symbols) & (table["date"] >= pd.Timestamp(start)) & (table["date"] <= pd.Timestamp(end))]
    return rows.pivot(index="date", columns="symbol", values="price")


@functools.lru_cache(maxsize=256)
def _expected_ipca(as_of):
    filters = "Indicador eq 'IPCA' and Suavizada eq 'S' and baseCalculo eq 0" + (f" and Data le '{as_of}'" if as_of else "")
    df = _odata("ExpectativasMercadoInflacao12Meses", filters, top=5)
    return float(df["Mediana"].iloc[0]) / 100 if len(df) else None


def expected_ipca(as_of=None):
    """IPCA inflation the market expects over the next 12 months (Focus survey), or None."""
    try:
        return _expected_ipca(pd.Timestamp(as_of).strftime("%Y-%m-%d") if as_of is not None else None)
    except (requests.RequestException, KeyError, ValueError):
        return None


def bond_carry(symbols, as_of=None):
    """Expected 12-month return of each Tesouro bond if yields do not change (carry): the
    nominal yield of a Prefixado, and the real yield compounded with the Focus expected IPCA
    for an IPCA+. Uses the yields and the survey of `as_of` (point in time)."""
    try:
        bonds = tesouro_bonds(as_of, min_years_left=0).reindex(symbols)
    except requests.RequestException:
        return pd.Series(dtype=float)
    ipca = expected_ipca(as_of)
    carry = bonds["yield"].copy()
    linked = bonds["type"] == "Tesouro IPCA+"
    carry[linked] = (1 + bonds.loc[linked, "yield"]) * (1 + (ipca if ipca is not None else np.nan)) - 1
    return carry.dropna()


@functools.lru_cache(maxsize=32)
def _cdi_rates(start, end):
    frames = []
    chunk_start, end = pd.Timestamp(start), pd.Timestamp(end)
    # SGS serves daily series in windows of at most 10 years
    while chunk_start <= end:
        chunk_end = min(chunk_start + pd.DateOffset(years=10) - pd.Timedelta(days=1), end)
        rows = _get_json(SGS_URL.format(series=12), {
            "formato": "json",
            "dataInicial": chunk_start.strftime("%d/%m/%Y"),
            "dataFinal": chunk_end.strftime("%d/%m/%Y"),
        })
        frames.append(pd.DataFrame(rows, columns=["data", "valor"]))
        chunk_start = chunk_end + pd.Timedelta(days=1)
    df = pd.concat(frames)
    rates = pd.Series(df["valor"].astype(float).to_numpy() / 100,
                      index=pd.to_datetime(df["data"], dayfirst=True), name=CDI)
    return rates[~rates.index.duplicated()].sort_index()


def fetch_cdi(start, end):
    """Daily CDI rate (fraction per business day) from the Central Bank's SGS series 12."""
    return _cdi_rates(pd.Timestamp(start).strftime("%Y-%m-%d"), pd.Timestamp(end).strftime("%Y-%m-%d")).copy()


def cdi_index(start, end):
    """Value of R$1 invested in the CDI. The rate of day d accrues overnight, so it shows up
    in the index on the next business day, like a B3 close-to-close return."""
    rates = fetch_cdi(start, end)
    return (1 + rates).cumprod().shift(1, fill_value=1.0)


def _odata(endpoint, filters, top=200):
    query = f"$filter={quote(filters, safe=chr(39))}&$orderby=Data%20desc&$top={top}&$format=json"
    return pd.DataFrame(_get_json(f"{FOCUS_URL}{endpoint}?{query}")["value"])


@functools.lru_cache(maxsize=256)
def focus_selic_path(as_of=None):
    """Median Selic the market expects (Focus survey) at each upcoming Copom meeting, from the
    latest survey published on or before `as_of` (YYYY-MM-DD; default: the latest one)."""
    filters = "baseCalculo eq 0" + (f" and Data le '{as_of}'" if as_of else "")
    df = _odata("ExpectativasMercadoSelic", filters)
    if df.empty:
        return df
    df = df[df["Data"] == df["Data"].max()]
    key = df["Reuniao"].str.extract(r"R(\d)/(\d{4})").astype(int)
    df = df.assign(meeting=key[0], year=key[1]).sort_values(["year", "meeting"])
    return df[["Data", "Reuniao", "Mediana"]].reset_index(drop=True)


def expected_cdi(as_of=None, months=12):
    """Average annual CDI the market expects over the next `months` months, compounding the
    Focus Selic path (8 Copom meetings a year, each rate holding for about 1.5 months).
    Returns None when the survey is not available."""
    as_of = pd.Timestamp(as_of).strftime("%Y-%m-%d") if as_of is not None else None
    try:
        path = focus_selic_path(as_of)
    except (requests.RequestException, KeyError, ValueError):
        return None
    if path.empty:
        return None
    n = max(1, round(months / 12 * 8))
    selic = path["Mediana"].to_numpy()[:n] / 100
    selic = np.pad(selic, (0, n - len(selic)), mode="edge")
    return float(np.prod((1 + selic - SELIC_CDI_GAP) ** (1 / 8)) ** (8 / n) - 1)


def annual_to_daily(rate):
    return (1 + rate) ** (1 / TRADING_DAYS) - 1


def remove_stale_prices(prices, max_run=10):
    """Blanks prices repeated for more than `max_run` days in a row, after the first day of
    the run: Yahoo serves frozen quotes for some funds (IRFM11, IMAB11 and B5P211 sat at
    the same value from 2022 to January 2026), which would look like a riskless asset."""
    same = prices.eq(prices.shift(1)) & prices.notna()
    # Length of the run of repeated prices each day belongs to
    run_id = (~same).cumsum()
    run_len = same.groupby(run_id).transform("sum") if prices.ndim == 1 else same.apply(
        lambda col: col.groupby(run_id[col.name]).transform("sum"))
    return prices.mask(same & (run_len >= max_run))


def remove_bad_ticks(prices, jump=0.4, window=5, tolerance=0.15):
    """Blanks isolated bad prints: a move of more than `jump` that comes back to within
    `tolerance` of the previous price in at most `window` days (Yahoo has some, e.g. XPML11
    quoted at R$0.13 instead of R$110 for three days in January 2026)."""
    clean = prices.copy()
    big, near = np.log(1 + jump), np.log(1 + tolerance)
    for col in prices.columns:
        s = np.log(prices[col].dropna())
        v, idx = s.to_numpy(), s.index
        i = 1
        while i < len(v):
            if abs(v[i] - v[i - 1]) > big:
                back = [k for k in range(i + 1, min(i + 1 + window, len(v))) if abs(v[k] - v[i - 1]) < near]
                if back:
                    clean.loc[idx[i:back[0]], col] = np.nan
                    i = back[0]
            i += 1
    return clean


def download_quotes(symbols, start, end):
    """Closing prices (split-adjusted, not dividend-adjusted), volumes, dividends and split
    ratios of B3 symbols from Yahoo Finance, on the Ibovespa's calendar, with frozen quotes
    and bad prints removed. Returns a dict of DataFrames plus `dropped` {symbol: reason}."""
    data = yf.download([s + ".SA" for s in symbols] + ["^BVSP"], start=start, end=end, auto_adjust=False,
                       actions=True, progress=False, threads=True)

    def field(name):
        if name not in data.columns.get_level_values(0):
            return pd.DataFrame(index=data.index)
        return data[name].rename(columns=lambda c: str(c).removesuffix(".SA"))

    close = field("Close")
    calendar = close["^BVSP"].dropna().index
    close = close.drop(columns="^BVSP")
    if len(calendar) == 0:  # no index data: keep the days on which most assets traded
        calendar = close.index[close.notna().mean(axis=1) >= 0.5]
    close = close.reindex(calendar)
    dropped = {s: "no price data" for s in symbols if s not in close.columns or close[s].notna().sum() < 2}
    close = remove_bad_ticks(remove_stale_prices(close.drop(columns=[s for s in dropped if s in close.columns])))

    def aligned(name):
        return field(name).reindex(columns=close.columns).reindex(close.index.union(field(name).index))

    # Dividends and splits on non-trading days move to the next trading day
    events = {name: aligned(name).fillna(0.0) for name in ("Dividends", "Stock Splits")}
    dividends = events["Dividends"].groupby(close.index.searchsorted(events["Dividends"].index)).sum()
    dividends.index = close.index[np.minimum(dividends.index, len(close.index) - 1)]
    splits = events["Stock Splits"].replace(0.0, np.nan).dropna(how="all")
    return {
        "close": close,
        "volume": field("Volume").reindex(index=calendar, columns=close.columns),
        "dividends": dividends.reindex(close.index).fillna(0.0),
        "splits": splits,
        "dropped": dropped,
    }


def total_return_index(close, dividends):
    """Value of one share with dividends reinvested, from closing prices and the dividends
    paid on each ex-date. Built here instead of using Yahoo's adjusted close, whose
    adjustment factors are sometimes wrong (XPML11's whole history before January 2026 is
    divided by ~8 there)."""
    previous = close.ffill().shift(1)
    r = (close + dividends.reindex_like(close).fillna(0.0)) / previous - 1
    return (1 + r.fillna(0.0)).cumprod().where(close.notna())


def collect_prices(symbols, start, end, max_gap=5, require_full_history=True, with_traded_value=False):
    """Total return prices (dividends reinvested, which matters for FIIs) on the B3 calendar
    for B3 symbols such as "PETR4" or "AAPL34", Tesouro Direto bonds ("TD-IPCA2035") and the
    CDI index.

    The calendar is the Ibovespa's trading days. Frozen quotes and bad prints are removed
    (remove_stale_prices, remove_bad_ticks), gaps of up to `max_gap` days are filled with
    the last price, and assets that still jump more than 100% in a day are dropped. With
    require_full_history (portfolio optimization), assets that start after `start` or have
    longer gaps are dropped too; without it (backtests), they are kept so that each
    rebalance can use whatever was investable then.
    Returns (prices, dropped) where `dropped` maps each discarded symbol to the reason, plus
    the daily traded value in R$ (close x volume) when with_traded_value.
    """
    symbols = list(dict.fromkeys(symbols))
    bonds = [s for s in symbols if s.startswith(TESOURO_PREFIX)]
    listed = [s for s in symbols if s != CDI and s not in bonds]
    dropped = {}
    prices = traded = pd.DataFrame()
    if listed:
        quotes = download_quotes(listed, start, end)
        dropped.update(quotes["dropped"])
        traded = quotes["close"] * quotes["volume"]
        prices = total_return_index(quotes["close"], quotes["dividends"]).ffill(limit=max_gap)
    if bonds:
        try:
            quotes = tesouro_prices(bonds, pd.Timestamp(start) - pd.Timedelta(days=10), end)
        except requests.RequestException:
            dropped.update({s: "Tesouro Direto data unavailable (the Treasury's site did not answer)" for s in bonds})
            quotes = pd.DataFrame()
        for s in bonds:
            if s not in dropped and (s not in quotes or quotes[s].notna().sum() < 2):
                dropped[s] = "no price data"
        if not quotes.empty:
            quotes = remove_bad_ticks(remove_stale_prices(quotes))
            calendar = prices.index if not prices.empty else quotes.loc[pd.Timestamp(start):].index
            # The Treasury publishes with a lag of a few days: the last quotes are carried forward
            quotes = quotes.reindex(quotes.index.union(calendar)).ffill(limit=max_gap).reindex(calendar)
            prices = pd.concat([prices, quotes.drop(columns=[s for s in dropped if s in quotes])], axis=1)
    if not prices.empty:
        jumps = prices.pct_change(fill_method=None).abs().max()
        for s in jumps.index[jumps > 1.0]:
            dropped[s] = f"suspicious price jump of {jumps[s]:.0%} in a day"
        prices = prices.drop(columns=[s for s in dropped if s in prices.columns])
        if require_full_history:
            for s in prices.columns:
                col = prices[s]
                first = col.first_valid_index()
                if col.iloc[: max_gap + 1].isna().all():
                    dropped[s] = f"history starts on {first:%Y-%m-%d}" if first is not None else "no price data"
                elif col.loc[first:].isna().any():
                    dropped[s] = f"no valid price for more than {max_gap} days in a row (missing or frozen quotes)"
            prices = prices.drop(columns=[s for s in dropped if s in prices.columns])
    if CDI in symbols:
        index = cdi_index(pd.Timestamp(start) - pd.Timedelta(days=10), end)
        if prices.empty:
            prices = index.loc[pd.Timestamp(start):].to_frame()
        else:
            prices[CDI] = index.reindex(prices.index.union(index.index)).ffill().reindex(prices.index)
    if require_full_history:
        prices = prices.dropna()
    prices.index.name = "Date"
    if with_traded_value:
        return prices, dropped, traded.reindex(index=prices.index, columns=prices.columns)
    return prices, dropped
