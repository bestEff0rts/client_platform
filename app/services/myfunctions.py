#myfunctions.py
#тут все кастомные функции для импорта
print("Loaded myfunctions.py")

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility_20"] = df["log_returns"].rolling(window=20).std()
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        df["sma_5"] = df["close"].rolling(window=5).mean()
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["rsi"] = compute_rsi(df["close"])
        return df

#feature engineering calculations
def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from UniStatTransf import UniStationarityTransformer  # кастомные трансформеры

def build_preprocessing_pipeline():
    return Pipeline([
        ("features", FeatureEngineeringTransformer()),
        ("stationarity", UniStationarityTransformer(columns=[
            "open", "high", "low", "close", "volume", "log_returns",
            "volatility_20", "vwap", "sma_5", "sma_20", "rsi"
        ])),
        ("scaler", StandardScaler())
    ])


#метрики ml
def calculate_accuracy(predicted_returns, real_returns):
    predicted_returns = np.reshape(predicted_returns, (-1, 1))
    real_returns = np.reshape(real_returns, (-1, 1))
    hits = sum((np.sign(predicted_returns)) == np.sign(real_returns))
    total_samples = len(predicted_returns)
    accuracy = hits / total_samples
    return accuracy[0] * 100

def model_bias(predicted_returns):
    bullish_forecasts = np.sum(predicted_returns > 0)
    bearish_forecasts = np.sum(predicted_returns < 0)
    return bullish_forecasts / bearish_forecasts
  
def calculate_directional_accuracy(predicted_returns, real_returns):
    # Calculate differences between consecutive elements
    diff_predicted = np.diff(predicted_returns, axis = 0)
    diff_real = np.diff(real_returns, axis = 0)
    # Check if signs of differences are the same
    store = []  
    for i in range(len(predicted_returns)):
        try:            
            if np.sign(diff_predicted[i]) == np.sign(diff_real[i]):                
                store = np.append(store, 1)        
            elif np.sign(diff_predicted[i]) != np.sign(diff_real[i]):                
                store = np.append(store, 0)                  
        except IndexError:           
            pass       
    directional_accuracy = np.sum(store) / len(store)
    return directional_accuracy * 100
