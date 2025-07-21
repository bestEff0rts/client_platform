#metrics.py- КАСТОМНЫЕ ФУНКЦИИ РАСЧЕТ МЕТРИК после predict для вывода 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_metrics_regression(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }

#еще будет def compute_metrics_classification
def calculate_accuracy(predicted_returns, real_returns):
    predicted_returns = np.reshape(predicted_returns, (-1, 1))
    real_returns = np.reshape(real_returns, (-1, 1))
    hits = sum((np.sign(predicted_returns)) == np.sign(real_returns))
    total_samples = len(predicted_returns)
    accuracy = hits / total_samples
    
    return accuracy[0] * 100
  
