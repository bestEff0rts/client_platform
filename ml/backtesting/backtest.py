#backtest.py 
#preprocessing_pipe_x.fit_transfroms(crisis_x)-> predict() на загруж предобученной модели-> вывод фин и ml метрик
#to do- сохранить и импоритровать и применить preprocessing_pipe_x; три простых стратегии для тестов; прогнать итог и логику api оформить аналогично осн
from finmetrics import sharpe_ratio, var, cvar
from myparser import fetch_moex_ohlcv
import numpy as np
import pandas as pd
def логика :
  данные проходят этапы:
    1-расчет таргет\лейбеллинг 
    2-предобработка preprocessing_pipe_x(feature engineering+UniStationarityTransfromer+StandatScaler())
  итог- pandas dataframe x_transformed


import os
os.chdir("D:\project_root")

#1- raw data crisis_data 2008
import yfinance as yf
data = yf.download('AAPL', start='2007-12-01', end='2008-12-01') 
data
data.columns = data.columns.get_level_values(0)
data.columns.name = None

data.index.name = "Date"  
# data = data.droplevel(0, axis=1) 
data = data.astype(float)
data = data.sort_index()
data
data["target"]= data["Close"].diff() 


def run_backtest(model_id: str, data: pd.DataFrame) -> dict:
    """
    Загружает prepocessing pipeline по model_id, применяет на crisis_data,
    считает метрики.

    Parameters:
        model_id (str): Идентификатор модели в SQLite/.pkl
        crisis_data (pd.DataFrame): Временной ряд

    Returns:
        dict: метрики модели
    """
    x = df.drop(columns=["target", "Date"])  #SEPARATE X и y из общего PD DF
    y = df["target"]
    x_transformed = preprocessing_pipe_x.fit_transform(x) #EXECUTE PIPELINES (fit_transform,  тк обучение трансформации)
    x_transformed_df = pd.DataFrame(x_transformed, index=x.index[-len(x_transformed):]) #из np array в pd df сохр временную структуру для сплита
    x_transformed_df.columns = ["open","high","low", "close","volume","log_returns", "volatility_20", "vwap", "sma_5", "sma_20", "rsi"]
    pipe = load_pipeline(model_id)  # из models/
    
    y_true = data ["target"]
    y_pred = pipe.predict(X) #CV??? fit

    return compute_all_metrics(y_true, y_pred)
