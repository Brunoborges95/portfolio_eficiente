import pandas as pd
import numpy as np
import utils
import pytest
from datetime import datetime, timedelta
import yfinance as yfin

yfin.pdr_override()

@pytest.fixture()
def historic_stocks():
    start = datetime.now() - timedelta(days=400)
    end = datetime.now()

    df_stocks_info = utils.read_stocks_info(end)

    stock_type = ['BDR', 'ON', 'PN', 'PNA', 'PNB', 'Subscrição', 'PNC', 'PND']
    stock_sector = list(df_stocks_info["Setor"].unique())
    stock_technical = ["Compra Forte", "Compra", "Venda", "Venda Forte", "Neutro"]

    df_filter = df_stocks_info.query(
        "Tipo in @stock_type and Setor in @stock_sector and Mensal in @stock_technical"
    )
    stocks_codes = [i + ".SA" for i in df_filter.Códigos.unique()]
    return utils.collect_historico_stocks(stocks_codes, start, end)

def test_df_Stocks_info_exists_and_all_codes_ar_not_null():
    d = datetime.now()
    df = utils.read_stocks_info(d)
    assert len(df)>1000
    assert (df.Códigos.unique()!=np.nan).any() == True

@pytest.mark.timeout(180)
def test_if_historic_stocks_is_collected_without_missing_values_or_null_values(historic_stocks):
    assert historic_stocks.isna().any().any() == False
    assert (historic_stocks.values!=0).any() == True

@pytest.mark.timeout(180)
def test_if_optimization_is_working(historic_stocks): 
    po = utils.Portfolio_optimization(historic_stocks)
    r_dict = po.returns()
    Returns = r_dict["Returns"]
    ExpR = r_dict["Expected Returns"]
    opt_dict = po.optimize(Returns, ExpR)
    meanR = opt_dict["meanR"]
    risk_measure = opt_dict["risk_measure"]
    assert len(risk_measure) == 50
    assert len(meanR) == 50


def test_if_advanced_transformations_is_working(historic_stocks):
    try:
        utils.df_moving_avg(historic_stocks, sma=25)
        utils.df_optimal_pso_points(historic_stocks, 20)
        utils.df_weighted(historic_stocks, recent_weight=.5)
    except Exception as e:
        pytest.fail(f"The function raises an exception: {e}")






