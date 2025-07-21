#ModelFactory.py 
#КАСТОМНЫЙ ИПМОРТ: from ModelFactory import build_model
import os
os.chdir("D:\project_root")
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from sklearn.svm import SVR
# from model import build_keras_model  #кастом для Sequential() и тд

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor


def build_model(model_type: str):
    """
    Возвращает ML-модель -ПОКА CONTINUOUS OUTPUT=РЕГРЕССИЯ- по строковому идентификатору.
    Параметры:
    ----------
    model_type : str
        Тип модели, например: "rf", "xgb", "svm", "ridge", "lasso", "elastic.net"
    Возвращает:
    -----------
    Объект модели scikit-learn или совместимый
    """
    model_type = model_type.lower()

    if model_type == "rf":
        return RandomForestRegressor(max_depth=5, random_state=69)
    elif model_type == "xgb":
        return XGBRegressor(n_estimators=100, max_depth=3, random_state=69)
    elif model_type == "svm":
        return SVR(kernel='rbf')  
    elif model_type == "ridge":
        return Ridge(alpha=0.5)
    elif model_type == "lasso":
        return Lasso(alpha=0.9)
    elif model_type == "elastic.net":
        return ElasticNet(alpha=0.1, l1_ratio=0.5)
    else:
        raise ValueError(f"OH NOOO Unknown model type: '{model_type}' — check spelling or extend build_model().")

#пример использования 
#from ModelFactory import build_model
#model = build_model("rf")
#model.fit(X_train_transformed, y_train)

