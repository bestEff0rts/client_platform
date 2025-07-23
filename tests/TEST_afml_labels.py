#afml_labels_test.py
import os
os.chdir("D:\project_root")
#импорты
from afml_labels import get_bins, get_daily_vol, apply_pt_sl_on_t1, get_events, add_vertical_barrier

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#кастомные импорты 
from myfunctions import compute_rsi, FeatureEngineeringTransformer
from StatTransf import StationarityTransformer

import asyncio
import platform
import sys

if sys.platform.startswith("win") and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

#RAW DATA_public api endpoint_myparser.py MOEX
from myparser import fetch_binance_ohlcv, fetch_moex_ohlcv
data = asyncio.run(fetch_moex_ohlcv("SBER")) 
# df = asyncio.run(fetch_moex_ohlcv("SBER", interval=1, days=10))
print(df.head())
len(data)
data
#в PANDAS DATAFRAME+ РАСЧЕТ TARGETimport pandas as pd
df = pd.DataFrame(data)
df['begin'] = pd.to_datetime(df['begin'])  # convert to datetime
df.set_index('begin', inplace=True)
print(df.head())
print(df.describe())
df.columns
close = df["close"]
close
# РАСЧЕТ daily volatility
daily_vol = get_daily_vol(close, span0=100)

# 2. Сформировать события (t_events)
t_events = daily_vol.index

# 3. Добавить вертикальный барьер
t1 = add_vertical_barrier(t_events, close, num_days=5)

# 4. Получить события (с учетом PT/SL/t1)
events = get_events(
    close=close,
    t_events=t_events,
    pt_sl=1,
    trgt=daily_vol,
    min_ret=0.005,
    num_threads=1,
    t1=t1
)

# 5. Получить метки (±1)
labels = get_bins(events, close)
