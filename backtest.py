#backtest.py 
#preprocessing_pipe_x.fit_transfroms(crisis_x)-> predict() на загруж предобученной модели-> вывод фин и ml метрик
#to do- сохранить и импоритровать и применить preprocessing_pipe_x; три простых стратегии для тестов; прогнать итог и логику api оформить аналогично осн
from finmetrics import sharpe_ratio, var, cvar
from myparser import fetch_moex_ohlcv
import numpy as np
import pandas as pd

import os
os.chdir("D:\project_root")

#1- raw data crisis_data 2008
import yfinance as yf
data = yf.download('AAPL', start='2007-12-01', end='2008-12-01')  
data.columns.name = None  # убрать name='Price'
data.index.name = "Date"  # явно
data = data.droplevel(0, axis=1)  # если это мультииндекс по колонкам, убери "Price"
data = data.astype(float)
data = data.sort_index()
data
data["target"]= data["close"].diff() 

def run_backtest(model_id: str, data: pd.DataFrame) -> dict:
    """
    Загружает pipeline по model_id, применяет на crisis_data,
    считает метрики.

    Parameters:
        model_id (str): Идентификатор модели в SQLite/.pkl
        crisis_data (pd.DataFrame): Временной ряд

    Returns:
        dict: метрики модели
    """
    pipe = load_pipeline(model_id)  # из models/
    X = data.drop(columns=["target"])
    y_true = crisis_data["target"]
    y_pred = pipe.predict(X)

    return compute_all_metrics(y_true, y_pred)
