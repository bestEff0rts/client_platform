#test_ml_pipeline.py
import os
os.chdir("D:\project_root")
from sklearn.preprocessing import StandardScaler
#кастомные импорты 
from myfunctions import compute_rsi, FeatureEngineeringTransformer
from StatTransf import StationarityTransformer

#RAW DATA_public api endpoint_myparser.py
from myparser import fetch_binance_ohlcv, fetch_moex_ohlcv
import asyncio
data = asyncio.run(fetch_binance_ohlcv("ETH/BTC")) #___-___; не тикер а trading pair
len(data)
#в PANDAS DATAFRAME+ РАСЧЕТ TARGET
df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms") #timestamp → datetime
print(df.describe())
df["target"]= df["close"].diff() ##РАСЧЕТ TARGET = Rt(returns or log returns) ВНЕ пайплайна 
#df["log_returns"]= np.log(df["close"] / df["close"].shift(1))


#ТЕСТ ВНЕ ПАЙПЛАНА Feature Engineering
features= FeatureEngineeringTransformer()
x_f = features.fit_transform(x)
x_f.describe()
#ПРОТЕСТИРОВАНО РАБОАТЕТ 

#ТЕСТ ВНЕ ПАЙПЛАНА Stationarity
stat = StationarityTransformer()
x_stat = stat.fit_transform(x_f)
x_stat.describe()

#ТЕСТ ВНЕ ПАЙПЛАНА Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_stat)
x_scaled.describe()

#ТЕСТ ВНЕ ПАЙПЛАНА итого 
x_scaled_df = pd.DataFrame(x_scaled, index=x.index[-len(x_scaled):], columns=x_stat.columns)
