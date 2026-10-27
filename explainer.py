"""Grounded explanations of a portfolio, written by Claude.

The numbers come from the project's own data (CVM fundamentals, analyst consensus, the
optimizer) and are sent already formatted; Claude only writes the text, citing them as
given, and every number in the text is checked against the facts that were sent. The text
explains why the model chose each asset. It is not investment advice: in Brazil,
recommendations to third parties require a certified analyst (CVM Resolution 20/2021).
"""
import json
import os
import re
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

import optimizer as opt
import portfolio
from market_data import CDI, TESOURO_PREFIX

MODEL = "claude-opus-5-5"
ENV_PATH = Path(__file__).resolve().parent / ".env"
RATING_PT = {"strong_buy": "compra forte", "buy": "compra", "hold": "neutro",
             "underperform": "desempenho abaixo da média", "sell": "venda"}
CLASS_PT = {"Stocks": "Ações", "BDRs": "BDRs", "FIIs": "Fundos imobiliários", "Fixed income": "Renda fixa",
            "Brazil ETFs": "ETFs de ações brasileiras", "Global ETFs": "ETFs internacionais",
            "Crypto": "Cripto", "Gold & FX": "Ouro e câmbio"}
RETURN_METHODS_PT = {"historical": "média dos retornos no período analisado (reflete o momentum recente)",
                     "shrinkage": "médias históricas aproximadas de um mesmo retorno por unidade de risco (James-Stein)"}

SYSTEM = """Você explica, em {language}, por que um otimizador de carteiras incluiu cada ativo numa carteira.

Regras:
- Use somente os fatos do JSON enviado. Não use conhecimento próprio sobre empresas, setores, notícias ou preços.
- Cite números exatamente como estão escritos nos fatos, com a mesma formatação. Não calcule, arredonde nem estime números novos.
- Para cada ativo, explique o que nos dados sustenta a escolha: fundamentos comparados à mediana do setor, histórico anual, consenso dos analistas e o papel do ativo na carteira (peso, parcela do risco de cauda, correlação com o restante, taxa ou carrego). Depois, os principais pontos contra, também com dados.
- Seja equilibrado e factual: sem adjetivos promocionais, sem prever preços, sem prometer retorno.
- Quando um dado não estiver nos fatos, não o mencione.
- Descreva a escolha do modelo ("o modelo inclui... porque"); não faça recomendação pessoal de investimento.
"""


class HoldingExplanation(BaseModel):
    symbol: str
    headline: str = Field(description="Uma frase com o principal motivo da escolha")
    justification: str = Field(description="De 60 a 110 palavras: o que nos dados sustenta a escolha")
    counterpoints: str = Field(description="Os principais pontos contra ou riscos, com dados")


class PortfolioExplanation(BaseModel):
    overview: str = Field(description="De 2 a 4 frases sobre a carteira como um todo")
    holdings: list[HoldingExplanation]


class ExplanationError(RuntimeError):
    """Claude could not write the explanation (credentials, API error, refusal...)."""


# ---- number formatting (pt-BR) ----
def _num(x, digits=1, sign=False):
    text = f"{x:+,.{digits}f}" if sign else f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def _ok(x):
    return x is not None and not (isinstance(x, float) and np.isnan(x)) and np.isfinite(float(x))


def pct(x, digits=1):
    return _num(100 * float(x), digits) + "%" if _ok(x) else None


def times(x):
    return _num(float(x), 1) + "x" if _ok(x) else None


def money(x):
    if not _ok(x):
        return None
    x = float(x)
    if abs(x) >= 1e9:
        return f"R$ {_num(x / 1e9, 1)} bi"
    if abs(x) >= 1e6:
        return f"R$ {_num(x / 1e6, 1)} mi"
    return f"R$ {_num(x, 0)}"


def _clean(d):
    """Drops empty facts so the model never sees placeholders."""
    if isinstance(d, dict):
        out = {k: _clean(v) for k, v in d.items()}
        return {k: v for k, v in out.items() if v not in (None, {}, [], "")}
    if isinstance(d, list):
        return [x for x in (_clean(v) for v in d) if x not in (None, {}, [], "")]
    return d


# ---- grounding check ----
_NUMBER = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*")


def _value(token):
    if "," in token:
        token = token.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+", token):
        token = token.replace(".", "")
    return round(float(token), 6)


def unverified_numbers(text, facts_text):
    """Numbers written in `text` that do not appear in the facts (digits inside tickers such
    as PETR4 are ignored; signs too, so "queda de 3,2%" matches a fact of "-3,2%")."""
    allowed = {_value(t) for t in _NUMBER.findall(facts_text)}
    return sorted({t for t in _NUMBER.findall(text) if _value(t) not in allowed})


# ---- facts ----
def _median_note(value, median, fmt):
    v, m = fmt(value), fmt(median)
    if v is None:
        return None
    return f"{v} (mediana do setor: {m})" if m is not None else v


def _sector_medians(universe_df):
    stocks = universe_df[universe_df["asset_class"] == "Stocks"].drop_duplicates("cnpj") if "cnpj" in universe_df else pd.DataFrame()
    cols = ["pe", "pb", "roe", "net_margin", "net_debt_to_equity", "dividend_yield"]
    if stocks.empty:
        return pd.DataFrame(columns=cols)
    return stocks.groupby("sector")[[c for c in cols if c in stocks]].median()


def holding_facts(symbol, weight, row, problem, mu, prices, risk_share, corr, medians, history, details):
    """Facts about one holding, every number formatted as it may be cited."""
    cls = row.get("asset_class")
    window = prices[symbol].dropna().iloc[-253:] if symbol in prices else pd.Series(dtype=float)
    ret_12m = window.iloc[-1] / window.iloc[0] - 1 if len(window) > 200 else None
    vol = window.pct_change().std() * np.sqrt(opt.TRADING_DAYS) if len(window) > 60 else None
    facts = {
        "ticker": symbol, "nome": row.get("name"), "classe": CLASS_PT.get(cls, cls),
        "peso na carteira": pct(weight),
        "retorno esperado pelo modelo (ao ano)": pct(mu.get(symbol, np.nan) * opt.TRADING_DAYS),
        "parcela do risco de cauda da carteira": pct(risk_share),
        "correlação com o restante da carteira": _num(corr, 2) if _ok(corr) else None,
    }
    if symbol != CDI and not symbol.startswith(TESOURO_PREFIX):
        facts["retorno nos últimos 12 meses"] = pct(ret_12m)
        facts["volatilidade anual"] = pct(vol)
    if cls == "Stocks":
        sector = row.get("sector")
        med = medians.loc[sector] if isinstance(medians, pd.DataFrame) and sector in medians.index else pd.Series(dtype=float)
        facts["setor"] = sector
        filing = pd.to_datetime(row.get("filing"), errors="coerce")
        facts[f"fundamentos (CVM, balanço de {filing:%Y-%m-%d})" if pd.notna(filing) else "fundamentos (CVM)"] = {
            "P/L": _median_note(row.get("pe"), med.get("pe"), times),
            "P/VP": _median_note(row.get("pb"), med.get("pb"), times),
            "ROE": _median_note(row.get("roe"), med.get("roe"), pct),
            "margem líquida": _median_note(row.get("net_margin"), med.get("net_margin"), pct),
            "dívida líquida / patrimônio": _median_note(row.get("net_debt_to_equity"), med.get("net_debt_to_equity"), times),
            "crescimento da receita no ano (vs. mesmo período do ano anterior)": pct(row.get("revenue_growth")),
            "crescimento do lucro no ano (vs. mesmo período do ano anterior)": pct(row.get("earnings_growth")),
            "anos com lucro entre os últimos 5": f"{int(row['profitable_years'])} de {int(row['fiscal_years'])}"
            if _ok(row.get("profitable_years")) and _ok(row.get("fiscal_years")) else None,
            "ROE médio dos últimos 5 anos": pct(row.get("roe_5y")),
            "dividend yield (12 meses)": _median_note(row.get("dividend_yield"), med.get("dividend_yield"), pct),
        }
        if _ok(row.get("fundamental_score")):
            facts["score fundamentalista (desvios-padrão em relação ao setor)"] = {
                "total": _num(row["fundamental_score"], 2, sign=True),
                "valor": _num(row["value_score"], 2, sign=True) if _ok(row.get("value_score")) else None,
                "qualidade": _num(row["quality_score"], 2, sign=True) if _ok(row.get("quality_score")) else None,
                "momento": _num(row["momentum_score"], 2, sign=True) if _ok(row.get("momentum_score")) else None,
                "baixo risco": _num(row["low_risk_score"], 2, sign=True) if _ok(row.get("low_risk_score")) else None,
            }
        if history is not None and _ok_text(row.get("cnpj")):
            years = history[history["cnpj"] == row["cnpj"]].sort_values("period_end").tail(5)
            facts["histórico anual"] = [
                {"ano": str(pd.Timestamp(y.period_end).year), "lucro líquido": money(y.ni), "receita": money(y.revenue),
                 "ROE": pct(y.roe)} for y in years.itertuples()]
    if cls in ("Stocks", "BDRs") and _ok(row.get("n_analysts")) and row.get("n_analysts") > 0:
        revision = row.get("revision")
        facts["analistas" + (f" (da empresa no exterior, {row.get('underlying')})" if cls == "BDRs" else "")] = {
            "consenso": RATING_PT.get(row.get("rec_key")),
            "número de analistas": str(int(row["n_analysts"])),
            "potencial até o preço-alvo médio": pct(row.get("upside")),
            "revisão do consenso em 3 meses": None if not _ok(revision) else (
                "estável" if abs(revision) < 0.05 else
                f"{'melhorou' if revision > 0 else 'piorou'} {_num(abs(revision), 2)} ponto na escala de 1 a 5"),
        }
    if cls == "FIIs":
        facts["dividend yield (12 meses)"] = pct(row.get("dividend_yield"))
        facts["P/VP"] = times(row.get("price_to_book"))
    if symbol.startswith(TESOURO_PREFIX):
        rate = row.get("bond_yield")
        linked = "IPCA" in symbol
        facts["taxa de compra"] = (f"IPCA + {pct(rate, 2)} ao ano" if linked else f"{pct(rate, 2)} ao ano") if _ok(rate) else None
        facts["vencimento"] = str(row.get("maturity"))[:4] if _ok_text(row.get("maturity")) else None
    if symbol == CDI:
        # Its expected return is the rate itself (today's, or the Focus one with projections), quoted compounded
        facts.pop("retorno esperado pelo modelo (ao ano)")
        facts["taxa considerada pelo modelo (ao ano)"] = pct((1 + details["rf"]) ** opt.TRADING_DAYS - 1, 2)
        if details.get("focus_cdi") is not None:
            facts["CDI esperado pelo Focus para os próximos 12 meses"] = pct(details["focus_cdi"], 2)
    return _clean(facts)


def _ok_text(x):
    return isinstance(x, str) and x.strip() != "" and x.lower() != "nan"


def build_facts(problem, w, assets, mu, settings, details, prices, universe_df=None, history=None, top=12):
    """Facts about the portfolio and its `top` largest holdings, ready to send."""
    w = w.reindex(problem.assets).fillna(0.0)
    held = w[w > 1e-6].sort_values(ascending=False)
    stats = problem.stats(w)
    shares = problem.risk_contributions(w)
    X = problem.X_dev
    classes = assets.reindex(problem.assets)["asset_class"]
    at_cap = [s for s in held.index if s != CDI and held[s] >= settings.max_weight - 1e-4]
    medians = _sector_medians(universe_df) if universe_df is not None else None
    port = {
        "objetivo": portfolio.OBJECTIVES.get(settings.objective, settings.objective),
        "orçamento de CVaR mensal": pct(settings.risk_budget) if settings.objective == "risk_budget" else None,
        f"CVaR {pct(problem.beta, 0)} em {problem.h} dias úteis (perda média nos piores meses)": pct(stats["cvar"]),
        "retorno esperado (ao ano)": pct(stats["annual_return"]),
        "como o retorno esperado é estimado": RETURN_METHODS_PT.get(settings.return_method),
        "volatilidade anual": pct(stats["annual_vol"]),
        "número de ativos": str(int(len(held))),
        "alocação por classe": {CLASS_PT.get(c, c): pct(v) for c, v in held.groupby(classes).sum().sort_values(ascending=False).items()},
        "peso máximo por ativo": pct(settings.max_weight, 0),
        "ativos no peso máximo": at_cap,
        "projeções prospectivas": "ligadas" if settings.projections else "desligadas",
    }
    holdings = []
    for s in held.index[:top]:
        i = problem.assets.index(s)
        rest = np.delete(X, i, axis=1) @ np.delete(w.to_numpy(), i)
        corr = np.corrcoef(X[:, i], rest)[0, 1] if np.std(rest) > 0 and np.std(X[:, i]) > 0 else np.nan
        row = assets.loc[s].to_dict() if s in assets.index else {"asset_class": None}
        holdings.append(holding_facts(s, held[s], row, problem, mu, prices, shares[s], corr, medians, history, details))
    return _clean(port), holdings


# ---- Claude ----
def load_env(path=ENV_PATH):
    """Reads KEY=VALUE lines of a .env file into the environment, without overriding it."""
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def explain(portfolio_facts, holdings_facts, client=None, model=MODEL, effort="medium", language="português do Brasil"):
    """Asks Claude to explain each holding. Returns (explanation, unverified numbers by
    symbol, facts sent). Raises ExplanationError with a message for the user."""
    load_env()
    client = client or anthropic.Anthropic()
    facts = json.dumps({"carteira": portfolio_facts, "ativos": holdings_facts}, ensure_ascii=False, indent=1)
    try:
        response = client.beta.messages.parse(
            model=model,
            max_tokens=16000,
            system=SYSTEM.format(language=language),
            messages=[{"role": "user", "content": f"Fatos da carteira (JSON):\n{facts}\n\nExplique a carteira e cada ativo."}],
            output_format=PortfolioExplanation,
            output_config={"effort": effort},
            # On a policy decline the API retries on Anthropic's recommended fallback model
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except TypeError as e:
        if "authentication" in str(e).lower():
            raise ExplanationError("No Claude credentials: set ANTHROPIC_API_KEY (in the environment or in .env).") from e
        raise
    except anthropic.AuthenticationError as e:
        raise ExplanationError("The Claude API key was rejected; check ANTHROPIC_API_KEY.") from e
    except anthropic.RateLimitError as e:
        raise ExplanationError("Claude API rate limit reached; try again in a moment.") from e
    except anthropic.APIConnectionError as e:
        raise ExplanationError("Could not reach the Claude API (network).") from e
    except anthropic.APIStatusError as e:
        raise ExplanationError(f"Claude API error {e.status_code}: {e.message}") from e
    if response.stop_reason == "refusal":
        raise ExplanationError("The model declined to write this explanation.")
    if response.stop_reason == "max_tokens" or response.parsed_output is None:
        raise ExplanationError("The explanation was cut short; explain fewer holdings.")
    result = response.parsed_output
    checks = {h.symbol: unverified_numbers(" ".join([h.headline, h.justification, h.counterpoints]), facts)
              for h in result.holdings}
    checks["overview"] = unverified_numbers(result.overview, facts)
    return result, checks, facts
