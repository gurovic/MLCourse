# **Анализ временных рядов**
## **Введение во временные ряды**  
**Временной ряд** — это последовательность данных, измеренных через равные промежутки времени. Примеры:  
- 📈 Цены акций  
- 🌡️ Температура воздуха  
- 🛒 Продажи товаров  

**Особенности временных рядов:**  
- **Тренд** — долгосрочное увеличение или уменьшение значений  
- **Сезонность** — повторяющиеся колебания (например, продажи мороженого летом)  
- **Шум** — случайные колебания  

**Проблемы в анализе:**  
- Автокорреляция (зависимость текущего значения от предыдущих)  
- Пропуски данных  
- Изменчивость во времени  

---

## **🟢 Базовый уровень (Разведочный анализ и простые методы)**  

### **1.1 Визуализация временного ряда**  
```python
import matplotlib.pyplot as plt

# Пример данных: температура за 30 дней  
dates = pd.date_range("2024-01-01", periods=30)  
temperature = [10, 12, 11, 15, 18, 20, 25, 23, 19, 17, 16, 14, 12, 10, 8, 9, 11, 13, 15, 18, 20, 22, 21, 19, 17, 16, 15, 14, 13, 12]  

plt.figure(figsize=(10, 4))  
plt.plot(dates, temperature, marker='o')  
plt.title("Температура воздуха в январе")  
plt.xlabel("Дата")  
plt.ylabel("Температура (°C)")  
plt.grid(True)  
plt.show()  
```

### **1.2 Скользящее среднее (Moving Average, MA)**  
**Идея:** Сглаживание ряда для выделения тренда.  
```python
window_size = 3  
data['MA'] = data['temperature'].rolling(window=window_size).mean()  

plt.plot(data['date'], data['temperature'], label='Исходные данные')  
plt.plot(data['date'], data['MA'], label=f'Скользящее среднее ({window_size} дня)', color='red')  
plt.legend()  
plt.show()  
```

### **1.3 Лаги (Lags) — зависимость от прошлых значений**  
```python
# Создание лагов (значений за предыдущие дни)  
data['lag_1'] = data['temperature'].shift(1)  # температура вчера  
data['lag_2'] = data['temperature'].shift(2)  # температура позавчера  

# Корреляция текущего значения с лагами  
print(data[['temperature', 'lag_1', 'lag_2']].corr())  
```

### **1.4 Разложение временного ряда (Trend + Seasonality + Noise)**  
```python
from statsmodels.tsa.seasonal import seasonal_decompose  

result = seasonal_decompose(data['temperature'], model='additive', period=7)  
result.plot()  
plt.show()  
```

---

## **🟡 Продвинутый уровень (Прогнозирование и ML)**  

### **2.1 Авторегрессия (AR) и модель ARIMA**  
**ARIMA (p, d, q):**  
- **p** — порядок авторегрессии (зависимость от прошлых значений)  
- **d** — порядок дифференцирования (устранение тренда)  
- **q** — порядок скользящего среднего  

```python
from statsmodels.tsa.arima.model import ARIMA  

model = ARIMA(data['temperature'], order=(1, 1, 1))  
model_fit = model.fit()  
forecast = model_fit.forecast(steps=5)  # прогноз на 5 дней  
print(forecast)  
```

### **2.2 Метод экспоненциального сглаживания (ETS)**  
```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing  

model = ExponentialSmoothing(data['temperature'], trend='add', seasonal='add', seasonal_periods=7)  
model_fit = model.fit()  
forecast = model_fit.forecast(7)  # прогноз на неделю  
```

### **2.3 LSTM для прогнозирования временных рядов**  
```python
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import LSTM, Dense  

# Подготовка данных для LSTM  
X = []  
y = []  
for i in range(len(data) - 3):  
    X.append(data['temperature'][i:i+3])  
    y.append(data['temperature'][i+3])  

# Построение модели  
model = Sequential([  
    LSTM(50, activation='relu', input_shape=(3, 1)),  
    Dense(1)  
])  
model.compile(optimizer='adam', loss='mse')  
model.fit(X, y, epochs=100)  
```

---

## **🔴 Экспертный уровень (Сложные методы и ансамбли)**  

### **3.1 Prophet от Facebook (автоматическое прогнозирование)**  
```python
from prophet import Prophet  

df_prophet = pd.DataFrame({'ds': dates, 'y': temperature})  
model = Prophet()  
model.fit(df_prophet)  

future = model.make_future_dataframe(periods=7)  
forecast = model.predict(future)  
model.plot(forecast)  
plt.show()  
```

### **3.2 Детекция аномалий во временных рядах**  
```python
from sklearn.ensemble import IsolationForest  

model = IsolationForest(contamination=0.05)  
data['anomaly'] = model.fit_predict(data[['temperature']])  

plt.scatter(data.index, data['temperature'], c=data['anomaly'], cmap='coolwarm')  
plt.show()  
```

### **3.3 Мультимодельные ансамбли**  
```python
# Гибрид ARIMA + LSTM  
arima_forecast = arima_model.forecast(30)  
lstm_forecast = lstm_model.predict(X_test)  

ensemble_forecast = (arima_forecast + lstm_forecast) / 2  
```

---

## **📌 Тренировочные задания**  

### **🟢 Базовый уровень**  
1. Для ряда `[10, 12, 15, 13, 16, 20, 18]` рассчитайте:  
   - Скользящее среднее с окном 2  
   - Лаг 1  

### **🟡 Продвинутый уровень**  
1. Обучите модель ARIMA на данных о температуре и сделайте прогноз на 3 дня.  

### **🔴 Экспертный уровень**  
1. Постройте LSTM-модель для прогнозирования цен акций (используйте лаги 5, 10, 15 дней).  

---

## **💡 Заключение**  
**Основные принципы работы с временными рядами:**  
1. **Всегда визуализируйте** данные перед анализом.  
2. **Проверяйте стационарность** (ряд не должен зависеть от времени).  
3. **Экспериментируйте с моделями** (ARIMA, Prophet, LSTM).  
4. **Учитывайте сезонность** (например, распродажи в декабре).  

> **"Прогнозирование временных рядов — это как предсказание погоды: даже небольшие изменения могут привести к неожиданным результатам!"**
