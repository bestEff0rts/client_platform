#continuous_retraining.py
#update the model every once in a while due to changing market dynamics
import os
os.chdir("D:\project_root")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from myfunctions import compute_rsi,calculate_accuracy, model_bias #кастом добавть в myfunctions.py
from sklearn.metrics import mean_squared_error

#continuous retraining loop
# Store the new forecasts
y_predicted = []
# Reshape x_test to forecast one period
#3 d - latest_values = np.transpose(np.reshape(x_test[0], (-1, 1)))
latest_values = x_test[0].reshape(1, -1)
# Isolate the real values for comparison
y_test_store = y_test
y_train_store = y_train
for i in range(len(y_test)):
    try:
        predicted_value = model.predict(latest_values)
        y_predicted = np.append(y_predicted, predicted_value)
        x_train = np.concatenate((x_train, latest_values), axis = 0)
        y_train = np.append(y_train, y_test[0])
        y_test = y_test[1:]
        x_test = x_test[1:, ]
        model.fit(x_train, y_train)
        latest_values = np.transpose(np.reshape(x_test[0], (-1, 1)))
    except IndexError:
        pass
      
#предсказания In Sample(train) 
y_predicted_train = np.reshape(model.predict(x_train), (-1, 1))
