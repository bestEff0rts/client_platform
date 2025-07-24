#ОСНОВНОЙ ml_pipeline.py ПРОТЕСТИРОВАНО
def ЛОГИКА:
  Парсинг и расчёт target;DONE
  preprocessing_pipe_x: FeatureEngineering + UniStationarityTransformer + Scaler;DONE
  preprocessing_pipe_x.fit_transform(X) → X_transformed; DONE
  train_test_split(X_transformed, y) (shuffle=False) DOING
  fit(model = build_model("rf"))-ФАБРИКА МОДЕЛЕЙ
  model.fit(X_train, y_train)
  )
  сохр предобученной МОДЕЛИ в .pkl (joblib.dump(rf_fit, "models/rf_fit.pkl")
  *prepocessing_pipe_x сохраняется тоже НО ОДИН РАЗ т к предобработка Х +- идентична*
  joblib.dump(model, "models/model_rf.pkl")
  )

#ИМПОРТЫ
import os
os.chdir("D:\project_root")
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#кастомные импорты 
from myfunctions import compute_rsi, FeatureEngineeringTransformer
from StatTransf import StationarityTransformer

#RAW DATA_public api endpoint_myparser.py
from myparser import fetch_binance_ohlcv, fetch_moex_ohlcv
import asyncio

#BINANCE
bdata = asyncio.run(fetch_binance_ohlcv("BTC/USDT")) #___/___; не тикер а trading pair
len(bdata)
bdata
#в PANDAS DATAFRAME+ РАСЧЕТ TARGET
df = pd.DataFrame(bdata, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms") #timestamp → datetime
print(df.describe())
df["target"]= df["close"].diff() ##РАСЧЕТ TARGET = Rt(returns or log returns) ВНЕ пайплайна 
#df["log_returns"]= np.log(df["close"] / df["close"].shift(1))

#MOEX#RAW DATA_public api endpoint_myparser.py MOEX
import asyncio
import platform
import sys

if sys.platform.startswith("win") and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
#or this for jupyter r-reticulate
#import nest_asyncio
#nest_asyncio.apply()
#data = await fetch_moex_ohlcv("SBER")

from myparser import fetch_binance_ohlcv, fetch_moex_ohlcv
mdata = asyncio.run(fetch_moex_ohlcv("SBER")) 
len(mdata)
mdata
#в PANDAS DATAFRAME+ РАСЧЕТ TARGETimport pandas as pd
moex_df = pd.DataFrame(mdata).reset_index()
moex_df['begin'] = pd.to_datetime(moex_df['begin'])  # convert to datetime
moex_df.set_index('begin', inplace=True)
print(moex_df.head())
print(moex_df.describe())

from UniStatTransf import UniStationarityTransformer
#ПАЙПЛАЙНЫ
preprocessing_pipe_x = Pipeline([
    ("features", FeatureEngineeringTransformer()),          # лог-доходности, SMA, RSI и т.д.
    # fracdiff/diff, если нужно column="close"
    ("stationarity", UniStationarityTransformer(columns=[
    "open", "high", "low", "close", "volume", "log_returns",
    "volatility_20", "vwap", "sma_5", "sma_20", "rsi"])),
    ("scaler", StandardScaler())                            # Масштабирование (or RobustScaler)
])
df
#ТЕСТ ВНЕ ПАЙПЛАНА Feature Engineering
x_test=df.drop(columns=["target","timestamp"])
features= FeatureEngineeringTransformer()
x_f = features.fit_transform(x_test)
x_f.describe()
x_f.columns

#ТЕСТ ВНЕ ПАЙПЛАНА Stationarity
#stat = StationarityTransformer()
stat= UniStationarityTransformer()
x_stat = stat.fit_transform(x_f)
x_stat.describe()
#РАБОТАЕТ

#ТЕСТ ВНЕ ПАЙПЛАНА Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_stat)
#x_scaled_df = pd.DataFrame(x_scaled, columns=x_stat.columns, index=x_stat.index)
#не дескрайбит это же np array пока что
#РАБОТАЕТ

#ТЕСТ ВНЕ ПАЙПЛАНА итого 
x_scaled_df = pd.DataFrame(x_scaled, index=x_test.index[-len(x_scaled):], columns=x_stat.columns)
x_scaled_df.describe()
#РАБОТАЕТ

#pipeline for Y(target)
# pipeline_y = Pipeline([
    #("stationarity", StationarityTransformer(column="target"))
])

#SEPARATE X и y из общего PD DF
x = df.drop(columns=["target", "timestamp"])
y = df["target"]
x.describe()
#EXECUTE PIPELINES (fit_transform,  тк обучение трансформации)
x_transformed = preprocessing_pipe_x.fit_transform(x)
#y_transformed = pipeline_y.fit_transform(pd.DataFrame({"target": y}))

x_transformed_df = pd.DataFrame(x_transformed, index=x.index[-len(x_transformed):]) #из np array в pd df сохр временную структуру для сплита
x_transformed_df.columns = ["open","high","low", "close","volume","log_returns", "volatility_20", "vwap", "sma_5", "sma_20", "rsi"]
x_transformed_df.describe()

#TILL THIS ВСЕ ПРОТЕСТИРОВАНО И РАБОТАЕТ

#TRAIN TEST СПЛИТ shuffle= False cause time series
from sklearn.model_selection import train_test_split
target = df["target"]
target_transformed = target.loc[x_transformed_df.index]
x_train, x_test, y_train, y_test = train_test_split( x_transformed_df, target_transformed, shuffle=False)
x_train.describe()
x_test.describe()
#ПРОТЕСТИРОВАНО РАБОТАЕТ

#for FASTAPI predict()
last_samples = x_test.iloc[-5:].values.tolist()
print(last_samples)

#FIT 
from myModelFactory import build_model
model = build_model("svm") #ФАБРИКА МОДЕЛЕЙ myModelFactory.py
model.fit(x_train, y_train)
model.get_params()
#сохранение ОБУЧЕННОЙ МОДЕЛИ -бинарный файл .pkl
import joblib
joblib.dump(model, "models/MODELTEST_fit.pkl") 
#NOTTHIS#joblib.dump({model}, "models/SVR_fit.pkl") -если get_params 'set' object has no attribute 'get_params'
print("Предобученная модель сохранена")

#проверка что в .pkl именно {model}_fit -предобученная модель загружена 
import joblib
model = joblib.load("models/MODELTEST_fit.pkl")
print(type(model))
print(model)
print(model.get_params())
#ПРОТЕСТИРОВАНО И РАБОТАЕТ


#save in SQL
import sqlite3
#подключение к БД
conn = sqlite3.connect("models.db")
cursor = conn.cursor()
#создать таблицу (если нет)
cursor.execute("""
CREATE TABLE IF NOT EXISTS pipelines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    path TEXT
)
""")
#записывать путь 
cursor.execute("""
INSERT INTO pipelines (name, path)
VALUES (?, ?)
""", ("xgb_fit", "models/pipeline_x.pkl"))

conn.commit()
conn.close()

#ДОРАБОТАТЬ это автоматизация загрузки пока можно и вручную через http://127.0.0.1:8000/docs#/
#save to fastapi swagger ui SQLAlchemy и FastAPI ORM
# from datetime import datetime
# from sqlalchemy.orm import Session
# from server import engine, ModelMeta  # или вынеси в models.py
# 
# model_path = f"models/{model_name}.pkl"
# 
# with Session(bind=engine) as session:
#     record = ModelMeta(
#         name=model_name,
#         uploaded_at=datetime.utcnow(),
#         path=model_path
#     )
#     session.add(record)
#     session.commit()
# 
