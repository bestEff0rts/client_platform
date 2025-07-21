#finmetrics.py ДЛЯ  BACKTESTING
#импорт from mymetrics import sharpe_ratio, var, cvar
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
import numpy as np

#фин метрики для backtesting strategies Sharpe Ratio, Sortino, Drawdown, Alpha/Beta 
def sharpe_ratio(returns, risk_free_rate=0):
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)

def var(returns, alpha=0.05):
    return np.percentile(returns, 100 * alpha)

def cvar(returns, alpha=0.05):
    var = value_at_risk(returns, alpha)
    return returns[returns <= var].mean()


#ml metrics
def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    returns = pd.Series(y_pred)

    sharpe_ratio = returns.mean() / returns.std()
    max_drawdown = (returns.cumsum().cummax() - returns.cumsum()).max()

    return {
        "MSE": mse,
        "R²": r2,
        "Sharpe": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }
