import numpy as np
import pandas as pd
import pytest

import market_data
import universe
from market_data import CDI


def test_parse_tesouro_csv_keeps_zero_coupon_bonds():
    csv = (
        "Tipo Titulo;Data Vencimento;Data Base;Taxa Compra Manha;Taxa Venda Manha;PU Compra Manha;PU Venda Manha;PU Base Manha\n"
        "Tesouro IPCA+;15/05/2035;15/10/2026;6,84;6,96;2700,10;2686,72;2686,72\n"
        "Tesouro Prefixado;01/01/2029;15/10/2026;12,64;12,76;793,10;791,20;791,20\n"
        "Tesouro Selic;01/03/2029;15/10/2026;0,05;0,06;20073,14;20058,12;20058,12\n"
        "Tesouro IPCA+ com Juros Semestrais;15/05/2035;15/10/2026;6,80;6,92;4500,00;4480,00;4480,00\n"
    ).encode("latin-1")
    table = market_data.parse_tesouro_csv(csv)
    assert list(table["symbol"]) == ["TD-IPCA2035", "TD-PRE2029"]
    assert np.isclose(table.set_index("symbol").loc["TD-IPCA2035", "yield"], 0.0684)
    assert table.set_index("symbol").loc["TD-PRE2029", "price"] == 791.20


def test_remove_stale_prices():
    days = pd.bdate_range("2026-01-05", periods=30)
    prices = pd.DataFrame({
        "FROZEN": [100.0] * 5 + [63.63] * 20 + [101.0, 102.0, 103.0, 104.0, 105.0],
        "QUIET": [10.0, 10.0, 10.0] + list(np.linspace(10.1, 12.0, 27)),  # a few unchanged days are fine
    }, index=days)
    clean = market_data.remove_stale_prices(prices, max_run=10)
    assert clean["FROZEN"].iloc[6:25].isna().all() and clean["FROZEN"].iloc[5] == 63.63
    assert clean["FROZEN"].iloc[:5].notna().all() and clean["FROZEN"].iloc[25:].notna().all()
    assert clean["QUIET"].notna().all()


def test_remove_bad_ticks_keeps_real_moves():
    days = pd.bdate_range("2026-01-05", periods=10)
    prices = pd.DataFrame({
        "BAD": [110, 111, 13.8, 0.13, 0.13, 101, 102, 103, 104, 105],  # misprint that comes back
        "CRASH": [100, 100, 50, 52, 51, 53, 52, 54, 53, 55],  # real fall: stays down
    }, index=days, dtype=float)
    clean = market_data.remove_bad_ticks(prices)
    assert clean["BAD"].iloc[2:5].isna().all() and clean["BAD"].drop(days[2:5]).notna().all()
    assert np.allclose(clean["CRASH"], prices["CRASH"])


@pytest.mark.parametrize("symbol, instrument, name, expected", [
    ("PETR4", "Stock", "Petrobras", ("Stocks", "PN")),
    ("TAEE11", "Stock", "Taesa", ("Stocks", "Unit")),
    ("HGLG11", "FII", "CSHG Logistica", ("FIIs", "FII")),
    ("KNCA11", "Fiagro", "Kinea Credito Agro", ("FIIs", "Fiagro")),
    ("AAPL34", "BDR", "Apple Inc.", ("BDRs", "BDR")),
    ("BIAU39", "BDR", "iShares Gold Trust", ("BDRs", "BDR ETF")),
    ("LFTS11", "ETF", "INVESTO TEVA TESOURO SELIC ETF", ("Fixed income", "Fixed income ETF")),
    ("IMAB11", "ETF", "IT NOW ID ETF IMA-B FUNDO DE ÍNDICE", ("Fixed income", "Fixed income ETF")),
    ("CPTI11", "ETF", "Capitania Infra Fic Fi Infra Rf Cp", ("Fixed income", "FI-Infra")),
    ("GOLD11", "ETF", "TREND ETF LBMA OURO CLASSE DE ÍNDICE INVESTIMENTO NO EXTERIOR", ("Gold & FX", "ETF")),
    ("HASH11", "ETF", "HASHDEX NASDAQ CME CRYPTO INDEX", ("Crypto", "ETF")),
    ("IVVB11", "ETF", "ISHARES S&P 500 CLASSE DE ÍNDICE", ("Global ETFs", "ETF")),
    ("BZEQ11", "ETF", "Btg Pactual Reference Msci Brazil 25/50", ("Brazil ETFs", "ETF")),
    ("BOVA11", "ETF", "ISHARES IBOVESPA CLASSE DE ÍNDICE", ("Brazil ETFs", "ETF")),
    ("XFIX11", "ETF", "TREND ETF IFIX-L CLASSE DE ÍNDICE", ("FIIs", "FII ETF")),
])
def test_classify(symbol, instrument, name, expected):
    assert universe.classify(symbol, instrument, name) == expected


def test_same_name_ignores_case_and_punctuation():
    assert universe._same_name("Alphabet Inc.", "ALPHABET INC")
    assert not universe._same_name("iShares MSCI World ETF", "iShares Semiconductor ETF")


def test_read_universe_is_point_in_time(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "DATASETS", tmp_path)
    (tmp_path / "2025-01-10").mkdir()
    pd.DataFrame({
        "country": "brazil", "name": "X", "full_name": ["Petrobras", "Some FII"], "isin": "-",
        "currency": "BRL", "symbol": ["PETR4", "ABCD11"], "Type": ["PN", "FII/Unit"],
        "Avg_Volume": [1e7, 1e4], "Recommendation": ["buy", "none"],
    }).to_csv(tmp_path / "2025-01-10" / "df_stocks_info.csv", index=False)
    (tmp_path / "2025-06-10").mkdir()
    pd.DataFrame({"symbol": ["VALE3", CDI], "name": ["Vale", "CDI"], "asset_class": ["Stocks", "Fixed income"],
                  "sub_class": ["ON", "CDI"], "instrument": ["Stock", "CDI"], "adtv": [1e8, None]}
                 ).to_csv(tmp_path / "2025-06-10" / "universe.csv", index=False)
    assert universe.list_snapshots() == ["2025-01-10", "2025-06-10"]
    legacy, day = universe.read_universe("2025-03-01")
    assert day == "2025-01-10"
    assert set(legacy.symbol) == {"PETR4", "ABCD11", CDI}
    assert legacy.set_index("symbol").loc["PETR4", "rec_key"] == "buy"
    assert pd.isna(legacy.set_index("symbol").loc["ABCD11", "rec_key"])
    assert universe.read_universe("2025-07-01")[1] == "2025-06-10"
    # Before the first snapshot: the earliest one (the caller flags it as look-ahead)
    assert universe.read_universe("2024-01-01")[1] == "2025-01-10"
