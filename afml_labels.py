#afml_labels.py
#2варианта- Fixed Horizon and Tripple Barrier Method-из книги Advances in Financial ML by Marcos Lopes de Prado

## daily vol, reindexed to close-calculate DAILY VOLATILITY ESTIMATES
import pandas as pd
import numpy as np

def get_daily_vol(close: pd.Series, span0: int = 100) -> pd.Series:
    """
    Calculate daily volatility (standard deviation of daily returns, exponentially weighted).

    Parameters:
    -----------
    close : pd.Series
        Series of closing prices indexed by datetime.
    span0 : int
        Span for the exponentially weighted moving std.

    Returns:
    --------
    pd.Series
        Daily volatility estimate.
    """
    # Найти индексы предыдущего дня
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]  # исключить нулевые смещения
    # Создать серию: индекс — текущая дата, значения — дата T-1
    prev_day_idx = pd.Series(close.index[df0 - 1], index=close.index[-df0.shape[0]:])
    # Вычислить доходности: close[t] / close[t-1] - 1
    daily_ret = close.loc[prev_day_idx.index] / close.loc[prev_day_idx.values].values - 1
    # EWMA стандартное отклонение доходностей
    daily_vol = daily_ret.ewm(span=span0).std()
    return daily_vol

def apply_pt_sl_on_t1(close: pd.Series,
                      events: pd.DataFrame,
                      pt_sl: tuple,
                      molecule: list) -> pd.DataFrame:
    """
    Применение profit-taking и stop-loss барьеров до момента t1 (конца события).

    Parameters:
    -----------
    close : pd.Series
        Временной ряд цен.
    events : pd.DataFrame
        DataFrame с колонками ['t1', 'trgt', 'side'].
    pt_sl : tuple
        Кортеж вида (pt, sl), где pt/sl — множители таргета.
    molecule : list
        Подмножество индексов событий (молекула) для асинхронной обработки.

    Returns:
    --------
    pd.DataFrame
        DataFrame с колонками 'pt' и 'sl' — временем срабатывания барьеров.
    """
    # Скопировать подмножество событий
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)

    # Задать уровни PT/SL (в абсолютных значениях, умноженные на таргет)
    pt = pt_sl[0] * events_['trgt'] if pt_sl[0] > 0 else pd.Series(index=events_.index)
    sl = -pt_sl[1] * events_['trgt'] if pt_sl[1] > 0 else pd.Series(index=events_.index)

    # Обход событий: найти первое достижение PT/SL или конец события t1
    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        # Путь цены от начала до конца события
        price_path = close[loc:t1]
        # Возвраты от точки входа, с учётом направления (side)
        price_returns = (price_path / close[loc] - 1) * events_.at[loc, 'side']
        # Находим первую дату, когда был достигнут SL / PT
        out.loc[loc, 'sl'] = price_returns[price_returns < sl[loc]].index.min()
        out.loc[loc, 'pt'] = price_returns[price_returns > pt[loc]].index.min()

    return out

#get_events
def get_events(close, t_events, pt_sl, trgt, min_ret, num_threads, t1=False):
    """
    Создание событий (events) для triple-barrier labeling.

    Parameters
    ----------
    close : pd.Series
        Цены закрытия (datetime index).
    t_events : pd.Index
        Метки времени, в которых допускается генерация событий.
    pt_sl : float
        Симметричные множители для PT и SL.
    trgt : pd.Series
        Целевое значение (например, волатильность) для каждого события.
    min_ret : float
        Минимальное значение trgt, ниже которого события не рассматриваются.
    num_threads : int
        Количество потоков для multiprocessing.
    t1 : pd.Series or bool, optional
        Время окончания событий (если False — будет заполнено NaT).

    Returns
    -------
    events : pd.DataFrame
        События с колонками: 't1', 'trgt'.
    """
    # 1) Фильтрация по целевому значению
    trgt = trgt.loc[t_events]
    trgt = trgt[trgt > min_ret]

    # 2) Установка вертикального барьера
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=t_events)

    # 3) Формирование объекта events
    side_ = pd.Series(1.0, index=trgt.index)  # assume long-only
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])

    # 4) Применение PT/SL с многопоточностью
    df0 = mpPandasObj(
        func=applyPtSlOnT1,
        pdObj=('molecule', events.index),
        numThreads=num_threads,
        close=close,
        events=events,
        ptSl=[pt_sl, pt_sl]
    )

    # 5) t1 — минимальная дата между PT и SL
    events['t1'] = df0.dropna(how='all').min(axis=1)

    # 6) Удаляем 'side' (если не используешь meta-labeling)
    events = events.drop('side', axis=1)

    return events



def add_vertical_barrier(t_events: pd.Series,
                         close: pd.Series,
                         num_days: int) -> pd.Series:
    """
    Устанавливает вертикальный барьер (максимальный холдинг-период) в днях для каждого события.

    Parameters:
    -----------
    t_events : pd.Series
        Индексы событий (обычно — бары после фильтрации).
    close : pd.Series
        Цены закрытия (datetime index).
    num_days : int
        Кол-во дней до вертикального барьера.

    Returns:
    --------
    pd.Series
        Индексы `t1` (временные метки вертикального барьера).
    """
    t1_idx = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1_idx = t1_idx[t1_idx < close.shape[0]]
    t1 = pd.Series(close.index[t1_idx], index=t_events[:t1_idx.shape[0]])

    return t1

import numpy as np
def get_bins(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """
    Присваивает бинарные метки (labels) на основе результата triple barrier метода.
    ----------
    events : pd.DataFrame с колонками ['t1', 'trgt'], созданный через get_events().
    close : pd.Series Цены закрытия с datetime индексом.
    -------
    out : pd.DataFrame с колонками:
        - 'ret' — доходность между событием и его t1;
        - 'bin' — знак доходности (+1, -1, 0).
    """
    # 1) Синхронизация цен с событиями
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # 2) Создание DataFrame с результатами
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    out['bin'] = np.sign(out['ret'])
    return out

