#CV for tuning parameters- КроссВалидация для подбори гиперпараметров 
#islp: is should come as *no surprise* that we use **CROSS VALIDATION** to estimate the tuning parameters

#GridSearchCV vs RandomizedSearchCV(random_state=69 тк инициализация случайная) = = SGD vs GD(gradient descent)
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
# from afml_labels import get_daily_vol, apply_pt_sl_on_t1, get_events, add_vertical_barrier

#словарь параметров- разные модели- разные параметры
param_grids = {
    'random_forest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    },
    'xgboost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.1]
    },
    'logistic_regression': {
        'C': [0.1, 1.0, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'svm': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'lasso': {
        'alpha': [0.01, 0.1, 1.0, 10],
        'max_iter': [1000, 5000]
    },
    'ridge': {
        'alpha': [0.01, 0.1, 1.0, 10],
        'solver': ['auto', 'svd', 'cholesky']
    },
    'pls': {
        'n_components': [2, 4, 6, 8]
    }

}
param_grid = param_grids["svm"]

#данные-myparser+preprocessing_pipe_x которая сохр и импорт как ФУНКЦИЯ 
from myfunctions import build_preprocessing_pipeline
from UniStatTransf import UniStationarityTransformer
#RAW DATA_public api endpoint_myparser.py
from myparser import fetch_binance_ohlcv, fetch_moex_ohlcv
import asyncio
data = asyncio.run(fetch_binance_ohlcv("AAVE/USDT")) #___/___; не тикер а trading pair
len(data)
data
#в PANDAS DATAFRAME+ РАСЧЕТ TARGET
df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms") #timestamp → datetime
print(df.describe())
##РАСЧЕТ TARGET = Rt(returns or log returns) ВНЕ пайплайна 
df["target"]= df["close"].diff() # CONT OUTCOME НЕПРЕРЫВНЫЙ Y = регрессия
df["target_class"] = (df["close"].shift(-1) > df["close"]).astype(int) #DISCRETE OUTCOME дискретный у т е КЛАССИФИКАТОР ну labelling простой(ост- см afml_labelling.py)

#SEPARATE X и y из общего PD DF
x = df.drop(columns=["target", "timestamp"])
y = df["target"]
x.describe()
assert not x.isnull().values.any(), "NaNs detected in x"

preprocessing_pipe_x = build_preprocessing_pipeline()
#EXECUTE PIPELINES (fit_transform,  тк обучение трансформации)
x_transformed = preprocessing_pipe_x.fit_transform(x)
x_transformed_df = pd.DataFrame(x_transformed, index=x.index[-len(x_transformed):]) #из np array в pd df сохр временную структуру для сплита
x_transformed_df.columns = ["open","high","low", "close","volume","log_returns", "volatility_20", "vwap", "sma_5", "sma_20", "rsi"]
x_transformed_df.describe()

y = y[-len(x_transformed):]  # обрезать y, чтобы соответствовал X


#FIT 
from myModelFactory import build_model
model = build_model("svm") #ФАБРИКА МОДЕЛЕЙ myModelFactory.py
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit 
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=12,
    #scoring='f1_macro',
    cv=TimeSeriesSplit(n_splits=5), #TimeSeriesSplit(n_splits=5)-or StratifiedKFold
    random_state=69
)

random_search.fit(x_transformed_df, y)
best_model = random_search.best_estimator_
print(best_model.get_params())

#метрики
train_size = int(len(x_transformed_df) * 0.8)
X_train, X_test = x_transformed_df[:train_size], x_transformed_df[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, roc_curve

y_pred = best_model.predict(X_test)

# Для классификации (если y — метки 0/1)
if hasattr(best_model, "predict_proba"):
    y_proba = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC: {auc:.4f}")
else:
    # Для регрессии
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.4f}")

