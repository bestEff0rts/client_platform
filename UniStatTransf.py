#NEW StatTransf.py
#UniStatTransf.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.stattools import adfuller, kpss
from fracdiff.sklearn import Fracdiff
import warnings
class UniStationarityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, fracdiff_d=0.4, fracdiff_thres=1e-5):
        self.columns = columns if columns is not None else []
        self.fracdiff_d = fracdiff_d
        self.fracdiff_thres = fracdiff_thres
        self.methods_ = {}  # сохраняем, какой метод сработал по каждой колонке

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.columns:
            series = df[col].dropna()

            if self._is_stationary(series):
                self.methods_[col] = "original"
                df[col] = series
                continue
            try:
                f = Fracdiff(self.fracdiff_d, mode='valid', window=50)
                frac_array = f.fit_transform(series.to_frame()).squeeze()# Берём только последние индексы с нужной длиной
                valid_index = series.index[-len(frac_array):]
                frac_series = pd.Series(frac_array, index=valid_index)
            except Exception as e:
                print(f"Fracdiff error on {col}: {e}")
                frac_series = pd.Series(index=series.index, data=np.nan)

            # try:
            #     f = Fracdiff(self.fracdiff_d, mode='valid', window=50)
            #     frac_series = pd.Series(f.fit_transform(series.to_frame()).squeeze(), index=series.index[-len(series):])
            # except Exception as e:
            #     print(f"Fracdiff error on {col}: {e}")
            #     frac_series = pd.Series(index=series.index, data=np.nan)

            if self._is_stationary(frac_series):
                self.methods_[col] = "fracdiff"
                df[col] = frac_series
            else:
                self.methods_[col] = "diff"
                df[col] = series.diff()

        return df.dropna()

    def _is_stationary(self, series: pd.Series, alpha=0.05) -> bool:
        series = series.dropna()
        if len(series) < 20:
            return False
        try:
            adf_p = adfuller(series)[1]
            kpss_p = kpss(series, regression='c')[1]
            return adf_p < alpha and kpss_p > alpha
        except:
            return False
