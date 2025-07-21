#StatTransf.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.stattools import adfuller, kpss
from fracdiff.sklearn import Fracdiff
import warnings


class StationarityTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer: проверяет стационарность по ADF/KPSS и применяет:
    1) если уже стационарен — возвращает как есть
    2) если fracdiff → стационарен — возвращает его
    3) иначе → обычное .diff()
    """
    def __init__(self, column="target", fracdiff_d=0.4, fracdiff_thres=1e-5):
        self.column = column
        self.fracdiff_d = fracdiff_d
        self.fracdiff_thres = fracdiff_thres
        self.method_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Series, без NaN
        series = df[self.column].dropna()
        if len(series) < 20:
            raise ValueError("Series too short to test for stationarity")

        # Step 1: проверка оригинала
        if self._is_stationary(series):
            self.method_ = "original"
            df[self.column] = series
            return df.dropna()

        # Step 2: fracdiff
        try:
            window = 50
            f = Fracdiff(self.fracdiff_d, mode='valid', window=window)
            frac_array = f.fit_transform(series.to_frame())
            valid_index = series.index[-len(frac_array):]  # выравниваем индекс
            frac_series = pd.Series(frac_array.squeeze(), index=valid_index)

            if self._is_stationary(frac_series):
                self.method_ = "fracdiff"
                df = df.loc[valid_index]
                df[self.column] = frac_series
                return df.dropna()

        except Exception as e:
            print("Fracdiff error:", e)

        # Step 3: fallback на .diff()
        diff_series = series.diff().dropna()
        self.method_ = "diff"
        df = df.loc[diff_series.index]
        df[self.column] = diff_series
        return df.dropna()

    def _is_stationary(self, series: pd.Series, alpha=0.05) -> bool:
        series = series.dropna()
        if len(series) < 20:
            return False

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # подавим InterpolationWarning
                adf_p = adfuller(series)[1]
                kpss_p = kpss(series, regression='c')[1]
            return adf_p < alpha and kpss_p > alpha
        except:
            return False
