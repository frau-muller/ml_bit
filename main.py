
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

start = dt.datetime(2016,1,1)
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
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
#количество дней на основе,которых мы прогнозируем один день
prediction_days = 60
future_day = 20
x_train, y_train = [], []
#заполним фактическими значениями

for x in range(prediction_days, len(scaled_data)-future_day):
    x_train.append(scaled_data[x-prediction_days:x,0])
    y_train.append(scaled_data[x+future_day,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#строим нейронную сеть,которая будет моделью для прогноза pip install numpy == 1.19.5

model = Sequential()
#первый слой 50 единиц
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
#модель при  выпадении 0,2 чтобы предотвратить переобучение
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
#последний слой
model.add(Dense(units=1))

#скомпилируем
model.compile(optimizer='adam', loss='mean_squared_error')
#обучение модели на тестовых данных
model.fit(x_train, y_train, epochs=25, batch_size=32)

#тестируем модель с указанием временных рамок
test_start = dt.datetime(2015,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo',test_start, test_end)
#фактические цены будут тестовыми данными
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'],test_data['Close']),axis=0)

#входные данные модели будут равны общему набору данных
model_inputs = total_dataset[len(total_dataset)-len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

plt.plot(actual_prices,color = 'black',label = 'Actual Prices')
plt.plot(prediction_prices,color = 'green',label = 'Predicted Prices')
plt.title(f'{crypto_currency} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()
#прогноз на следующий день

