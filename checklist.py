#checklist.py 
| №  | Компонент                            | Файл / Модуль                     | Статус / Цель                            |
| -- | ------------------------------------ | --------------------------------- | ---------------------------------------- |
| _1  | **ML pipeline (X)**                  | `ml_pipeline.py`                  | `Pipeline` из `transformers` без модели  |- DONE
| _2  | **Model factory**                    | `models_factory.py`               | Выбор моделей (RF, XGB, MLP...)          |- DONE 
| _3  | **Preprocessing transformers**       | `transformers.py`                 | `FeatureEng`, `Stationarity`, `Scaler`   |- DONE
| _4  | **Stationarity logic + логирование** | `StationarityTransformer`         | Логика выбора: orig / fracdiff / diff    |без логирования DONE
| _5  | **Save/load pipeline+model**         | `joblib` + `models/`              | Всё сохраняется в `.pkl`                 |без сохр pipe DONE
; 0 прогнать predict() и metrics
+ raw data myparser.py - start = end= limit= 
**************************************************
ПРОГОН RandomizedSarchCV for estimating tuning parameters- так ну ALMOST DONE
@afmllabelling.py Tripple Barrier Method and Fixed Horizon Method- def for import 

| _6  | **Backtest**                         | `backtest.py`+ @декоратор в server| Подгрузка кризисных данных + метрики     | >>>>>>_________ 1
| _7  | **Метрики ML + фин.**                | `metrics.py`                      | MSE, R2 + Sharpe, Sortino, MaxDrawdown   |
| _8  | **Bars / Sampling logic**            | `sampling.py`                     | Dollar, Tick, Imbalance (AFML logic)     |
| _9  | **FastAPI endpoints**                | `server.py`                       | `parse/`, `predict/`, `backtest/`, ...   |+- predict; metrics
| _10 | **Retraining**                       | retrain.py` +@ server             | Повторная прогонка по новым данным       |>>>>>>____ 2 
| _11 | **Model metadata storage**           | `models.db`, `ModelMeta`          | SQLite, `upload`, `info`, `list`         |+-
| _12 | **Unit tests**                       | `tests/test_functions.py`         | Для всех `custom` функций (`pytest`)     |>>>>>__3
| _13 | **Functional tests**                 | `tests/test_pipeline.py`          | Для пайплайна: `fit_transform`, `.score` |>>>
| _14 | **Swagger UI auto-docs**             | `FastAPI` built-in                | Уже работает, скрин у тебя есть          | DONE
| _15 | **Web3/on-chain stub (будет)**       | `web3.py`, `contract.py`          | Подключение к токенам / голоса           |
| _16 | **UI/React stub (будет)**            | `React.js`                        | Дашборд / форма ввода                    |
| _17 | **Архитектура документация**         | `architecture.md`                 | (будет) — описание компонентов + схема   |-ПРОСТО СОБРАТЬ
| _18 | *save to fastapi swagger ui SQLAlchemy и FastAPI ORM* ДОРАБОТАТЬ конец ml_pipeline.py - model_name= ; record = ;
это автоматизация загрузки пока можно и вручную через http://127.0.0.1:8000/docs#/ 
| _19!!! *инпут RESHAPE для разных моделей* сейчас x_train x_test это time series pandas dataframe
|_20!!| *api /model id info- Response body чтобы get_params выдавал как у 1 и 2- КАК ИХ СОХР так надо и ост





***notes for README.MD and ARCHITECTURE.MD
Почему расчет target (например, log_returns) происходит вне Pipeline?
💡 Ключевая причина — data leakage.
Pipeline обучается на X, target — это y.
Если ты в Pipeline добавишь шаг генерации y = f(X) (например, log_returns = log(close/close.shift(1))), ты нарушаешь принцип разделения train/test:
➡️ на этапе .fit() будет использована информация из будущего, т.к. .shift() или .diff() вовлекают данные с t+1, t-1


@model.factory.py
Elastic Net
Диапазон значений: 0 < α < 1. 
aicorespot.io
α = 0 — весь вес — штраф L2.
α = 1 — весь вес — штраф L1.
0 < α < 1 — штраф — комбинация L1 и L2


к 20
Download
{
  "model_id": 2,
  "name": "lob_test_model.pkl",
  "path": "models/lob_test_model.pkl",
  "params": {
    "bootstrap": false,
    "contamination": "auto",
    "max_features": 1,
    "max_samples": "auto",
    "n_estimators": 300,
    "n_jobs": null,
    "random_state": 69,
    "verbose": 0,
    "warm_start": false
  }
}

ЛОГИКА архитектуры с PREDICT DICRETE OUTCOME(=CLASSIFIACTION)
raw_data 
  → feature_engineering (pipeline) 
  → label_generation  ← (target variable: y_class)
  → train_test_split
  → model.fit(X, y_class)
  → predict → backtest (на правилах)
Best practices:
Классическая структура:
y = labeling_function(df) → train_test_split(X, y)
Потому что y зависит и от будущих значений (например, future return > threshold), и не должна быть частью X.

FASTAPI
Формат данных для POST /model/predict
FastAPI ожидает на этом эндпоинте JSON с ключом "data", где значение — двумерный список (List[List[float]]), представляющий матрицу признаков (как X_test).
{
  "data": [
    [0.1, -0.5, 0.3, 1.2, -1.0, 0.04, ...],  // sample 1
    [0.2, -0.3, 0.4, 1.0, -0.8, 0.03, ...]   // sample 2
  ]
}
Где взять такие данные?
После прогонки данных через твой pipeline_x (например, x_transformed_df.iloc[-5:])

Просто преобразуешь к .tolist():
last_samples = x_transformed_df.iloc[-5:].values.tolist()

@backtesting.py ЛОГИКА- y = labeling_function(df) → train_test_split(X, y)
| ML компонент      | Trading стратегия                       |
| ----------------- | --------------------------------------- |
| `X` (features)    | Признаки: цена, RSI, SMA и др.          |
| `y` (target)      | Доходность или сигнал (buy/sell)        |
| `model.predict()` | Сигнал от модели (`ŷ`)                  |
| `thresholding`    | Превращение `ŷ` в сигнал (`+1`, `0`)    |
| `backtest`        | Прогон стратегии по историческим данным |
| `metrics`         | Sharpe, max drawdown, CAGR, etc.        |

Потому что y зависит и от будущих значений (например, future return > threshold), и не должна быть частью X
какой таргет нужен??
| Название                | Цель                           | Пример              | Task           |
| ----------------------- | ------------------------------ | ------------------- | -------------- |
| `next_return`           | Доходность за t+1              | `log(C[t+1]/C[t])`  | Regression     |
| `return_3d`             | Доходность за 3 дня            | `log(C[t+3]/C[t])`  | Regression     |
| `binary_label`          | Классификация "up/down"        | `1 if r > 0 else 0` | Classification |
| `triple_barrier_label`  | ML Labeling по AFML            | tp/sl/horizon       | Classification |
| `moving_average_signal` | Кросс мувингов (как стратегия) | `if SMA5 > SMA20`   | Signal         |
4. Feature engineering: для backtest и модели
Простейшие признаки:
data['ret_1d'] = np.log(data['Close'] / data['Close'].shift(1))
data['ret_5d'] = np.log(data['Close'] / data['Close'].shift(5))
data['volatility'] = data['ret_1d'].rolling(10).std()
data['sma_10'] = data['Close'].rolling(10).mean()
data['sma_50'] = data['Close'].rolling(50).mean()
data['volume_z'] = (data['Volume'] - data['Volume'].rolling(20).mean()) / data['Volume'].rolling(20).std()

признаки не должны смотреть в будущее (только .shift(1) или .rolling(...))
таргет должен соответствовать горизонту стратегии

Backtest — это прогон стратегии на исторических данных с целью оценки её реалистичной доходности и риска. Стратегия — это правило:

"Если model.predict(X_t) > threshold, то buy, иначе hold / sell."
| Предсказание   | Стратегия действия | Алгоритм      |
| -------------- | ------------------ | ------------- |
| `ŷ_t > 0`      | Покупка            | long-only     |
| `ŷ_t > α`      | Покупка, иначе нет | thresholding  |
| `ŷ_class == 1` | Купить, иначе нет  | классификация |
| `ŷ_class == 2` | Short              | multi-class   |

@cv
| Параметр                 | `GridSearchCV`                         | `RandomizedSearchCV`                              |
| ------------------------ | -------------------------------------- | ------------------------------------------------- |
| Алгоритм                 | **Полный перебор** всех сочетаний      | **Сэмплирует случайно** из заданных распределений |
| Быстродействие           | Медленно при росте размерности         | Быстрее, линейно масштабируется                   |
| Риск переобучения        | Ниже (но дороже)                       | Есть стохастический разброс                       |
| Нужно ли `random_state`? | ❌ Нет необходимости                    | ✅ Обязательно, для **детерминированности**        |
| Применимость             | Когда сетка маленькая и важен контроль | Когда параметров много и важна скорость           |


scoring= — какие варианты, и от чего зависят
🔹 Категории моделей:
Классификация — accuracy, f1, precision, recall, roc_auc, log_loss, …

Регрессия — r2, neg_mean_squared_error, neg_mean_absolute_error, explained_variance, …
| Модель                   | Задача        | scoring                                 | Подходит? |
| ------------------------ | ------------- | --------------------------------------- | --------- |
| `RandomForestClassifier` | классификация | `'accuracy'`, `'f1_macro'`, `'roc_auc'` | ✅         |
| `XGBClassifier`          | классификация | `'f1_macro'`, `'log_loss'`, `'roc_auc'` | ✅         |
| `LogisticRegression`     | классификация | `'f1'`, `'precision'`, `'recall'`       | ✅         |
| `RandomForestRegressor`  | регрессия     | `'r2'`, `'neg_mean_squared_error'`      | ✅         |
| `Lasso`, `Ridge`, `PLS`  | регрессия     | `'neg_mean_absolute_error'`, `'r2'`     | ✅         |

Полный список:
https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
🔸 GridSearchCV требует указания scoring, иначе по умолчанию:

классификация: accuracy,

регрессия: r2.

Документация:
https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

| Тип задачи        | Значения `scoring`                                  | Применимо к моделям                               |
| ----------------- | --------------------------------------------------- | ------------------------------------------------- |
| **Классификация** | `"accuracy"`                                        | все классификаторы (`RF`, `XGB`, `SVM`, `LogReg`) |
|                   | `"precision"`, `"recall"`, `"f1"`, `"roc_auc"`      | —                                                 |
|                   | `"f1_macro"` (взвешивает по классам)                | —                                                 |
|                   | `"balanced_accuracy"`                               | для несбалансированных классов                    |
| **Регрессия**     | `"r2"`, `"neg_mean_squared_error"`                  | `Ridge`, `Lasso`, `SVR`, `PLSRegression`          |
|                   | `"neg_mean_absolute_error"`, `"explained_variance"` | —                                                 |

@CV 
Интерпретация: UserWarning: The total space of parameters 12 is smaller than n_iter=20
📌 Что произошло:
Ты передаёшь в RandomizedSearchCV:

n_iter=20 → просишь случайно попробовать 20 комбинаций гиперпараметров

но param_distributions содержит всего 12 уникальных комбинаций

📐 Подсчёт:
python
Копировать
Редактировать
C:         [0.1, 1, 10]              → 3 варианта  
kernel:    ['linear', 'rbf']         → 2 варианта  
gamma:     ['scale', 'auto']         → 2 варианта  
→ 3 × 2 × 2 = 12 комбинаций

@UNIT_TESTS/.
ДЛЯ ПРОДАКШН
ЮНИТ-ТЕСТЫ: логика и пример
Зачем нужны:
Обеспечить корректность функциЙ Гарантировать повторяемость

Упростить отладку в будущем
| Функция                        | Что проверить                                                    |
| ------------------------------ | ---------------------------------------------------------------- |
| `build_model`                  | Возвращает модель нужного класса по имени                        |
| `build_preprocessing_pipeline` | pipeline корректен, `.fit_transform()` работает                  |
| `train_model_with_cv()`        | Работает без ошибок, возвращает обученную модель, метрики в dict |
| `fetch_binance_ohlcv()`        | Возвращает DataFrame нужной формы                                |
| `UniStationarityTransformer`   | Проверка `fit()` и `transform()` дают матрицу нужной размерности |

@afml_labelling.py
3ple barrier labellimg - SL TP TO- stop loss; take profit; time out 
собрать полную цепочку Triple Barrier Labeling:

get_daily_vol() → оценка волатильности

add_vertical_barrier() → устанавливаем горизонты удержания

get_events() → строим события с pt/sl/t1

apply_pt_sl_on_t1() → применяем стопы

get_bins() → финальная метка (лейбл: +1 / -1 / 0)


| Шаг | Логика                                                            |
| --- | ----------------------------------------------------------------- |
| 1   | Выбирается подмножество событий (`molecule`)                      |
| 2   | Вычисляется уровень PT и SL в процентах от `trgt`                 |
| 3   | Проход по каждой точке: строится путь цены `close[loc:t1]`        |
| 4   | Вычисляются кум. доходности с направлением (через `side`)         |
| 5   | Проверка достижения уровней PT/SL                                 |
| 6   | Возвращаются даты, когда сработал PT или SL (или `NaT`, если нет) |

get_events()
Какой формат events?
Фрейм events — это таблица событий. После get_events() он выглядит так:

index (datetime)	t1 (datetime)	trgt (float)
2020-01-01	2020-01-05	0.015
2020-01-07	2020-01-10	0.012

index — дата начала события (например, когда сформировался сигнал).

t1 — дата завершения события (раньше из PT/SL или вертикального барьера).

trgt — значение цели (volatility-adjusted target return).

проверить класс print(type(data))        # <class 'list'>
print(type(data[0]))     # <class 'dict'>

@MOEX
 это для def fetch_moex_ohlcv 
import asyncio
import platform
import sys

if sys.platform.startswith("win") and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
#or this for jupyter r-reticulate
import nest_asyncio
nest_asyncio.apply()
data = await fetch_moex_ohlcv("SBER")

#RAW DATA_public api endpoint_myparser.py MOEX
from myparser import fetch_binance_ohlcv, fetch_moex_ohlcv
data = asyncio.run(fetch_moex_ohlcv("SBER")) 
len(data)
data
#в PANDAS DATAFRAME+ РАСЧЕТ TARGETimport pandas as pd
df = pd.DataFrame(data)
df['begin'] = pd.to_datetime(df['begin'])  # convert to datetime
df.set_index('begin', inplace=True)
print(df.head())
print(df.describe())

это проверка ИМПОРТОВ КАСТОМНЫХ - просто все перезапустить)))
import afml_labels
print(dir(afml_labels))

@3ple barrier labelling
| Функция                | Входы                              | Выходы              |
| ---------------------- | ---------------------------------- | ------------------- |
| `get_daily_vol`        | close                              | daily\_vol (Series) |
| `add_vertical_barrier` | t\_events, close, num\_days        | t1 (Series)         |
| `get_events`           | close, t\_events, pt\_sl, trgt, t1 | events (DataFrame)  |
| `apply_pt_sl_on_t1`    | close, events, pt\_sl, molecule    | df0 (pt/sl: Series) |
| `get_bins`             | events, close                      | labels (ret, bin)   |

