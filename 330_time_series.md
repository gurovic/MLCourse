# **Feature Engineering для временных рядов**  

## **Введение в Feature Engineering для временных рядов**  
Преобразование временных рядов в информативные признаки — ключевой этап для:  
- 📈 **Прогнозирования** (продажи, курс акций)  
- 🕵️ **Детекции аномалий** (мошенничество, поломки оборудования)  
- 🏷️ **Классификации** (определение режимов работы устройства)  

**Особенности временных рядов:**  
- Зависимость между наблюдениями (автокорреляция)  
- Наличие трендов и сезонности  
- Неравномерная частота измерений  

---

## **🟢 Базовый уровень (Статистические признаки)**  

### **1.1 Лаги и скользящие статистики**  
```python
import pandas as pd

# Создание лагов
data['lag_1'] = data['value'].shift(1)  # предыдущее значение
data['lag_7'] = data['value'].shift(7)  # неделю назад

# Скользящее среднее и стандартное отклонение
window = 3
data['rolling_mean'] = data['value'].rolling(window).mean()
data['rolling_std'] = data['value'].rolling(window).std()
```

### **1.2 Разностные признаки**  
```python
# Абсолютные разницы
data['diff_1'] = data['value'].diff(1)  # разница с предыдущим значением
data['diff_7'] = data['value'].diff(7)  # разница с тем же днем недели назад

# Логарифмические возвраты (для финансовых данных)
data['log_return'] = np.log(data['value'] / data['value'].shift(1))
```

### **1.3 Признаки времени**  
```python
# Извлечение компонентов даты
data['hour'] = data.index.hour
data['day_of_week'] = data.index.dayofweek  # 0-понедельник
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

# Праздничные дни (пример для России)
holidays = ['2024-01-01', '2024-03-08'] 
data['is_holiday'] = data.index.date.astype(str).isin(holidays).astype(int)
```

---

## **🟡 Продвинутый уровень (Сложные преобразования)**  

### **2.1 Частотные признаки (FFT)**  
```python
from scipy.fft import fft

def get_fft_features(series, n_components=5):
    fft_result = fft(series.values)
    magnitudes = np.abs(fft_result)[:n_components]  # главные частотные компоненты
    return dict(zip([f'fft_{i}' for i in range(n_components)], magnitudes))

# Применение к скользящему окну
data[['fft_0', 'fft_1']] = data['value'].rolling(24).apply(lambda x: pd.Series(get_fft_features(x)))
```

### **2.2 Признаки сезонности**  
```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(data['value'], period=24)  # для суточной сезонности
data['seasonal'] = decomposition.seasonal
data['trend'] = decomposition.trend
```

### **2.3 Энтропийные признаки**  
```python
from entropylib import sample_entropy

def calculate_entropy(series):
    return sample_entropy(series, m=2, r=0.2*np.std(series))

data['entropy_24h'] = data['value'].rolling(24).apply(calculate_entropy)
```

---

## **🔴 Экспертный уровень (Продвинутые техники)**  

### **3.1 Признаки на основе автоэнкодеров**  
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Простой автоэнкодер
input_layer = Input(shape=(24,))
encoder = Dense(8, activation='relu')(input_layer)
decoder = Dense(24, activation='linear')(encoder)
autoencoder = Model(input_layer, decoder)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=10)

# Извлечение признаков
encoder_model = Model(input_layer, encoder)
data['ae_features'] = encoder_model.predict(data['value'].rolling(24).to_numpy())
```

### **3.2 Признаки изменений (Change Point Detection)**  
```python
from ruptures import Binseg

def detect_changepoints(series):
    algo = Binseg(model="l2").fit(series.values)
    return algo.predict(pen=10)  # точки изменения

data['changepoints'] = data['value'].rolling(100).apply(detect_changepoints)
```

### **3.3 Графовые признаки для временных рядов**  
```python
import networkx as nx

def create_visibility_graph(series):
    G = nx.Graph()
    for i in range(len(series)):
        for j in range(i+1, len(series)):
            if all(series[k] < series[i] + (series[j]-series[i])*(k-i)/(j-i) for k in range(i+1,j)):
                G.add_edge(i,j)
    return G

data['graph_clustering'] = data['value'].rolling(10).apply(lambda x: nx.average_clustering(create_visibility_graph(x)))
```

---

## **📌 Тренировочные задания**  

### **🟢 Базовый уровень**  
1. Для ряда `[10, 12, 15, 13, 16]` создайте:  
   - Лаг 1 и лаг 2  
   - Скользящее среднее с окном 2  

### **🟡 Продвинутый уровень**  
1. Для набора данных об энергопотреблении:  
   - Выделите сезонную компоненту (суточную и недельную)  
   - Рассчитайте энтропию для скользящего окна 24 часа  

### **🔴 Экспертный уровень**  
1. Постройте автоэнкодер для временного ряда и используйте скрытый слой как признаки  
2. Реализуйте детекцию точек изменения для финансового ряда  

---

## **💡 Заключение**  
**Ключевые принципы Feature Engineering для временных рядов:**  
1. **Сохраняйте временную структуру** (не перемешивайте данные!)  
2. **Комбинируйте подходы** (статистика + ML + доменные знания)  
3. **Учитывайте частоту данных** (секунды/дни/годы требуют разных подходов)  
4. **Контролируйте утечки** (признаки не должны использовать "будущее")  

**Практические рекомендации:**  
- Для прогнозирования: лаги + скользящие статистики + сезонность  
- Для аномалий: энтропия + частотные признаки + автоэнкодеры  
- Для классификации: статистики окон + FFT + графовые методы  

> **"Временные ряды — это не просто данные, а рассказ о процессе. Ваша задача — выделить ключевые сюжетные линии."**  

**Инструменты:**  
- `tsfresh` — автоматическая генерация признаков  
- `ruptures` — детекция точек изменения  
- `statsmodels` — анализ сезонности и трендов
