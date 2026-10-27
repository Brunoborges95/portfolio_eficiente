from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import explainer
import optimizer as opt
import portfolio
from market_data import CDI


def test_ptbr_formatting():
    assert explainer.pct(0.1234) == "12,3%"
    assert explainer.pct(-0.032) == "-3,2%"
    assert explainer.times(6.25) == "6,2x" or explainer.times(6.25) == "6,3x"
    assert explainer.money(110.129e9) == "R$ 110,1 bi"
    assert explainer.money(3.5e6) == "R$ 3,5 mi"
    assert explainer.pct(np.nan) is None and explainer.times(None) is None


def test_unverified_numbers():
    facts = '{"P/L": "6,2x (mediana do setor: 9,1x)", "ROE": "18,4%", "receita": "R$ 1.234,5 mi", "ano": "2025", "variação": "-3,2%"}'
    text = "PETR4 tem P/L de 6,2x contra 9,1x do setor, ROE de 18,4% em 2025, receita de R$ 1.234,5 mi e queda de 3,2%."
    assert explainer.unverified_numbers(text, facts) == []
    assert explainer.unverified_numbers("O ROE de 21,7% é alto.", facts) == ["21,7"]


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def fake_client(response):
    messages = FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def sample_explanation(text="O modelo inclui AAA3 porque o ROE de 18,4% supera a mediana."):
    return explainer.PortfolioExplanation(
        overview="Carteira com 2 ativos.",
        holdings=[explainer.HoldingExplanation(symbol="AAA3", headline="ROE alto", justification=text,
                                               counterpoints="P/L de 6,2x.")])


def test_explain_request_and_checks(monkeypatch):
    monkeypatch.setattr(explainer, "load_env", lambda *a, **k: None)
    client, messages = fake_client(SimpleNamespace(stop_reason="end_turn", parsed_output=sample_explanation(
        "O modelo inclui AAA3 porque o ROE de 18,4% supera a mediana; o lucro cresceu 99,9%.")))
    port = {"número de ativos": "2"}
    holdings = [{"ticker": "AAA3", "fundamentos": {"ROE": "18,4%", "P/L": "6,2x"}}]
    result, checks, facts = explainer.explain(port, holdings, client=client)
    assert result.holdings[0].symbol == "AAA3"
    assert checks["AAA3"] == ["99,9"]  # invented number flagged
    assert checks["overview"] == []
    kw = messages.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_format"] is explainer.PortfolioExplanation
    assert kw["output_config"] == {"effort": "medium"}
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert "18,4%" in kw["messages"][0]["content"]


@pytest.mark.parametrize("response, message", [
    (SimpleNamespace(stop_reason="refusal", parsed_output=None), "declined"),
    (SimpleNamespace(stop_reason="max_tokens", parsed_output=None), "cut short"),
    (TypeError("Could not resolve authentication method. Expected one of api_key..."), "ANTHROPIC_API_KEY"),
])
def test_explain_errors(monkeypatch, response, message):
    monkeypatch.setattr(explainer, "load_env", lambda *a, **k: None)
    client, _ = fake_client(response)
    with pytest.raises(explainer.ExplanationError, match=message):
        explainer.explain({}, [], client=client)


def test_build_facts():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2024-01-01", periods=300)
    X = pd.DataFrame(rng.normal(0.0005, 0.015, (300, 3)), index=days, columns=["AAA3", "BBB34", "TD-IPCA2035"])
    X[CDI] = 0.0005
    prices = (1 + X).cumprod()
    assets = pd.DataFrame({
        "asset_class": ["Stocks", "BDRs", "Fixed income", "Fixed income"],
        "name": ["Empresa A", "Company B", "Tesouro IPCA+ 2035", "CDI"],
        "sector": ["Energy", "Technology", None, None],
        "pe": [6.2, np.nan, np.nan, np.nan], "roe": [0.184, np.nan, np.nan, np.nan],
        "rec_key": ["buy", "strong_buy", None, None], "n_analysts": [12, 40, np.nan, np.nan],
        "upside": [0.18, 0.25, np.nan, np.nan], "underlying": [None, "BBB", None, None],
        "bond_yield": [np.nan, np.nan, 0.0684, np.nan], "maturity": [None, None, "2035-05-15", None],
    }, index=["AAA3", "BBB34", "TD-IPCA2035", CDI])
    R = X.iloc[1:]
    mu = opt.expected_returns(R, rf=R[CDI])
    settings = portfolio.ModelSettings(max_weight=0.5, risk_budget=0.05)
    problem = opt.CVaRProblem(R, mu, max_weight=0.5, groups=assets["asset_class"], uncapped=(CDI,))
    w = pd.Series({"AAA3": 0.4, "BBB34": 0.3, "TD-IPCA2035": 0.2, CDI: 0.1})
    port, holdings = explainer.build_facts(problem, w, assets, mu, settings, {"rf": 0.0005, "focus_cdi": 0.1266}, prices)
    assert port["número de ativos"] == "4"
    by_symbol = {h["ticker"]: h for h in holdings}
    assert by_symbol["AAA3"]["peso na carteira"] == "40,0%"
    assert by_symbol["AAA3"]["fundamentos (CVM)"]["P/L"] == "6,2x"
    assert by_symbol["BBB34"]["analistas (da empresa no exterior, BBB)"]["número de analistas"] == "40"
    assert by_symbol["TD-IPCA2035"]["taxa de compra"] == "IPCA + 6,84% ao ano"
    assert by_symbol[CDI]["CDI esperado pelo Focus para os próximos 12 meses"] == "12,66%"
    # The CDI's rate appears once, compounded as it is quoted
    assert "retorno esperado pelo modelo (ao ano)" not in by_symbol[CDI]
    assert by_symbol[CDI]["taxa considerada pelo modelo (ao ano)"] == explainer.pct(1.0005 ** 252 - 1, 2)
    assert port["como o retorno esperado é estimado"].startswith("média dos retornos")
    # Every value is a formatted string (or a nested structure of them)
    def strings(x):
        if isinstance(x, dict):
            return all(strings(v) for v in x.values())
        if isinstance(x, list):
            return all(strings(v) for v in x)
        return isinstance(x, str)
    assert strings(port) and all(strings(h) for h in holdings)
