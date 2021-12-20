import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

crypto_currency = "BTC"
#базовая валюта
against_currency = 'USD'

start = dt.datetime('2016,1,1')
end = dt.datetime.now()

data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo',start,end)

"""
подготовка данных для нейронной сети,приводим к виду 0 и 1,сжимаем все значения между нулем и единицей
Алгоритмы машинного обучения, как правило, работают лучше или сходятся быстрее, когда различные функции 
(переменные) имеют меньший масштаб. Поэтому перед обучением на них моделей машинного обучения 
данные обычно нормализуются.
"""
scaler = MinMaxScaler(feature_range=(0,1))
#преобразовываем один столбец
scaler_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
#количество дней на основе,которых мы прогнозируем один день
prediction_days = 40
x_train, y_train = [], []
#заполним фактическими значениями

for x in range
