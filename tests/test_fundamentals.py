import numpy as np
import pandas as pd

import fundamentals as f


def test_classify_accounts():
    rows = pd.DataFrame([
        ("3.01", "Receita de Venda de Bens e/ou Serviços", "revenue"),
        ("3.05", "Resultado Antes do Resultado Financeiro e dos Tributos", "ebit"),
        ("3.05", "Resultado antes dos Tributos sobre o Lucro", None),  # banks: not EBIT
        ("3.11", "Lucro/Prejuízo Consolidado do Período", "net_income"),
        ("3.11", "Lucro ou Prejuízo Líquido Consolidado do Período", "net_income"),
        ("3.11", "Lucro/Prejuízo do Período", "net_income"),
        ("3.09", "Lucro ou Prejuízo antes das Participações e Contribuições Estatutárias", None),
        ("3.99", "Lucro por Ação - (Reais / Ação)", None),
        ("3.11.01", "Atribuído a Sócios da Empresa Controladora", "net_income_ctrl"),
        ("3.11.02", "Atribuído a Sócios Não Controladores", None),
        ("2.03", "Patrimônio Líquido Consolidado", "equity"),
        ("2.08", "Patrimônio Líquido Consolidado", "equity"),
        ("2.03.09", "Participação dos Acionistas Não Controladores", "minority"),
        ("2.01.04", "Empréstimos e Financiamentos", "debt"),
        ("2.02.01", "Depósitos", None),
        ("1", "Ativo Total", "assets"),
    ], columns=["code", "label", "expected"])
    got = f.classify_accounts(rows["code"], rows["label"])
    assert got.fillna("-").tolist() == rows["expected"].fillna("-").tolist()


def long_table():
    """A company that files an annual statement for 2024, then Q1 2025 (in thousands of shares)."""
    rows = []
    def add(doc, period, received, **fields):
        for k, v in fields.items():
            rows.append({"cnpj": "1", "doc": doc, "period_end": period, "received": received, "cd_cvm": 1,
                         "company": "ACME", "field": k, "value": v})
    add("DFP", "2024-12-31", "2025-03-10", ni=0, net_income=100.0, net_income_prev=80.0, revenue=1000.0,
        revenue_prev=900.0, equity=500.0, shares=50.0, shares_on=50.0, financial=0.0)
    add("ITR", "2025-03-31", "2025-05-10", net_income=30.0, net_income_prev=20.0, revenue=260.0,
        revenue_prev=200.0, equity=530.0, shares=0.05, shares_on=0.05, financial=0.0)
    df = pd.DataFrame(rows)
    return df[df["field"] != "ni"]


def test_ttm_growth_and_thousands():
    panel = f.filings(long_table()).set_index("doc")
    assert panel.loc["ITR", "ni_ttm"] == 100 + 30 - 20
    assert panel.loc["ITR", "revenue_ttm"] == 1000 + 260 - 200
    assert np.isclose(panel.loc["ITR", "revenue_growth"], 0.3)
    assert np.isclose(panel.loc["ITR", "earnings_growth"], 0.5)
    # A book value of R$10,600 per share means the count was in thousands
    assert panel.loc["ITR", "shares"] == 50.0 and panel.loc["DFP", "shares"] == 50.0


def test_state_is_point_in_time():
    panel = f.filings(long_table())
    assert f.state_at(panel, "2025-04-01").loc["1", "doc"] == "DFP"  # the ITR was not out yet
    assert f.state_at(panel, "2025-05-10").loc["1", "doc"] == "ITR"
    assert "1" not in f.state_at(panel, "2025-01-01").index


def test_split_factor_after():
    splits = pd.DataFrame({"AAA3": [2.0, 0.5]}, index=pd.to_datetime(["2024-06-03", "2025-06-02"]))
    assert f.split_factor_after(splits, "AAA3", "2024-01-01") == 1.0
    assert f.split_factor_after(splits, "AAA3", "2024-12-31") == 0.5
    assert f.split_factor_after(splits, "BBB3", "2024-12-31") == 1.0


def quotes(on_price=10.0, pn_price=5.0):
    days = pd.bdate_range("2025-06-02", periods=5)
    close = pd.DataFrame({"AAA3": on_price, "AAA4": pn_price}, index=days)
    return {"close": close, "dividends": close * 0, "splits": pd.DataFrame(columns=close.columns)}


def panel_with(shares_list, shares_on_list, book=1000.0):
    rows = []
    for i, (s, s_on) in enumerate(zip(shares_list, shares_on_list)):
        rows.append({"cnpj": "1", "doc": "ITR", "period_end": pd.Timestamp("2024-03-31") + pd.DateOffset(months=3 * i),
                     "received": pd.Timestamp("2024-05-01") + pd.DateOffset(months=3 * i), "shares": s,
                     "shares_on": s_on, "book": book, "revenue_ttm": 2000.0})
    return pd.DataFrame(rows)


def test_market_caps_dual_class_and_bad_filings():
    companies = pd.DataFrame({"price_symbol": ["AAA3"], "on_symbol": ["AAA3"], "pn_symbol": ["AAA4"]}, index=["1"])
    when = pd.Timestamp("2025-06-06")
    panel = panel_with([100.0, 100.0, 100.0], [40.0, 40.0, 40.0])
    other = panel.assign(cnpj="2")  # a company outside the universe is ignored
    state = f.state_at(panel, when)
    assert f.market_caps(pd.concat([panel, other]), companies, quotes(), when, state)["1"] == 40 * 10 + 60 * 5
    # One filing with 10x the shares (a reporting error) does not move the median
    panel = panel_with([100.0, 100.0, 1000.0], [40.0, 40.0, 400.0])
    assert np.isclose(f.market_caps(panel, companies, quotes(), when, f.state_at(panel, when))["1"], 700)
    # Shares in thousands: a market value below 2% of the book value is scaled up
    panel = panel_with([0.1, 0.1, 0.1], [0.04, 0.04, 0.04], book=100_000.0)
    assert np.isclose(f.market_caps(panel, companies, quotes(), when, f.state_at(panel, when))["1"], 700)


def test_factor_scores_are_sector_neutral():
    ind = pd.DataFrame({
        "earnings_yield": [0.20, 0.10, 0.05, 0.02, 0.15, 0.30, 0.25, 0.20, 0.10, 0.05],
        "roe": np.linspace(0.05, 0.30, 10), "momentum": np.linspace(-0.2, 0.4, 10),
        "volatility": np.linspace(0.2, 0.5, 10),
    }, index=[f"c{i}" for i in range(10)])
    sectors = pd.Series(["Banks"] * 5 + ["Energy"] * 5, index=ind.index)
    scores = f.factor_scores(ind, sectors)
    assert scores.loc["c0", "value_score"] > scores.loc["c3", "value_score"]
    # Within each sector the value scores average zero
    assert np.allclose(scores["value_score"].groupby(sectors).mean(), 0, atol=1e-9)
    assert scores.loc["c9", "low_risk_score"] < scores.loc["c0", "low_risk_score"]
