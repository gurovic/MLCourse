# Обнаружение и обработка выбросов  

## Введение в проблему выбросов
**Выбросы (outliers)** - точки данных, которые значительно отличаются от остальных наблюдений. Они могут быть:
- **Ошибками измерения**: 99.9 вместо 9.9
- **Естественными аномалиями**: Гений с IQ 200
- **Событиями "черного лебедя"**: Финансовые кризисы

**Влияние на модели:**
- Искажают статистики (среднее, стандартное отклонение)
- Ломают линейные модели
- Снижают эффективность градиентного спуска
- Искажают метрики качества

---

## 🟢 Базовый уровень (Статистические методы)

### 1.1 Визуализация выбросов
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot
sns.boxplot(x=data['price'])
plt.title('Распределение цен с выбросами')
plt.show()

# Scatter plot
plt.scatter(data['age'], data['income'])
plt.xlabel('Возраст')
plt.ylabel('Доход')
plt.title('Выбросы в пространстве признаков')
plt.show()
```

### 1.2 Правило трех сигм
**Принцип:** 99.7% данных в пределах μ ± 3σ  
```python
def detect_outliers_sigma(data, threshold=3):
    mean = data.mean()
    std = data.std()
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    return (data < lower_bound) | (data > upper_bound)

outliers_mask = detect_outliers_sigma(data['income'])
```

### 1.3 Межквартильный размах (IQR)
**Формула:** IQR = Q3 - Q1  
**Границы:** [Q1 - 1.5·IQR, Q3 + 1.5·IQR]  
```python
def detect_outliers_iqr(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (data < lower_bound) | (data > upper_bound)

outliers_mask = detect_outliers_iqr(data['age'])
```

### 1.4 Простые стратегии обработки
```python
# Удаление
clean_data = data[~outliers_mask]

# Замена на границы
capped_data = data.copy()
capped_data['income'] = np.where(
    outliers_mask, 
    data['income'].median(),  # Альтернатива: границы IQR
    data['income']
)
```

---

## 🟡 Продвинутый уровень (ML-методы)

### 2.1 Isolation Forest
**Принцип:** Изоляция аномалий через случайные разделения  
```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(data[['age', 'income']])

# Визуализация
plt.scatter(data['age'], data['income'], c=outliers, cmap='coolwarm')
plt.colorbar(label='-1=выброс')
plt.show()
```

### 2.2 Local Outlier Factor (LOF)
**Принцип:** Оценка локальной плотности точек  
```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
outliers = lof.fit_predict(data[['x', 'y']])

# Оценка "аномальности"
negative_outlier_factor = -lof.negative_outlier_factor_
```

### 2.3 DBSCAN для обнаружения выбросов
**Принцип:** Плотностная кластеризация с пометкой шума  
```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=10)
clusters = dbscan.fit_predict(data[['feature1', 'feature2']])

# Выбросы помечены как -1
outliers_mask = clusters == -1
```

### 2.4 One-Class SVM
**Принцип:** Построение границы вокруг нормальных данных  
```python
from sklearn.svm import OneClassSVM

svm = OneClassSVM(nu=0.05)  # nu ~ доля выбросов
svm.fit(data[['feature1', 'feature2']])
outliers = svm.predict(data[['feature1', 'feature2']])  # -1 для выбросов
```

---

## 🔴 Экспертный уровень (Продвинутые техники)

### 3.1 Автоматический выбор метода
```python
from sklearn.ensemble import VotingClassifier

# Ансамбль детекторов
iso = IsolationForest(contamination=0.05)
lof = LocalOutlierFactor(novelty=True, contamination=0.05)  # novelty=True для predict
svm = OneClassSVM(nu=0.05)

def ensemble_detect(X):
    votes = (iso.predict(X) == -1).astype(int)
    votes += (lof.predict(X) == -1).astype(int)
    votes += (svm.predict(X) == -1).astype(int)
    return votes >= 2  # Консенсус минимум 2 из 3

# Обучение
iso.fit(X_train)
lof.fit(X_train)
svm.fit(X_train)
```

### 3.2 Выбросы в высоких размерностях
**Метод:** Использование PCA + Isolation Forest  
```python
from sklearn.decomposition import PCA

# Снижение размерности
pca = PCA(n_components=0.95)  # Сохраняем 95% дисперсии
X_pca = pca.fit_transform(data)

# Обнаружение выбросов
iso = IsolationForest(contamination=0.05)
outliers = iso.fit_predict(X_pca)
```

### 3.3 Обработка временных рядов
```python
from statsmodels.tsa.seasonal import STL

# Декомпозиция временного ряда
decomposition = STL(data['value'], period=12).fit()

# Выбросы в остатках
residuals = decomposition.resid
outliers = detect_outliers_iqr(residuals)

# Визуализация
plt.plot(data['date'], data['value'], label='Исходный ряд')
plt.scatter(data['date'][outliers], data['value'][outliers], c='red', label='Выбросы')
plt.legend()
plt.show()
```

### 3.4 Глубокое обучение для выбросов
**Autoencoders:** Реконструкция нормальных данных  
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Архитектура автоэнкодера
input_layer = Input(shape=(n_features,))
encoder = Dense(32, activation='relu')(input_layer)
decoder = Dense(n_features, activation='linear')(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32)

# Оценка ошибки реконструкции
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
outliers = mse > np.quantile(mse, 0.95)  # Верхние 5% ошибок
```

---

## 🛠️ Стратегии обработки выбросов

### Удаление
```python
clean_data = data[~outliers_mask]
```

### Трансформация
```python
# Логарифмирование
data['log_income'] = np.log1p(data['income'])

# Winsorization (ограничение)
def winsorize(data, limits=[0.05, 0.05]):
    return np.clip(data, 
                 np.quantile(data, limits[0]), 
                 np.quantile(data, 1-limits[1]))
```

### Импьютация
```python
# Замена на медиану
data['income'] = np.where(outliers_mask, data['income'].median(), data['income'])
```

### Сегментация
```python
# Раздельное моделирование
normal_model = train_model(data[~outliers_mask])
outlier_model = train_model(data[outliers_mask])
```

---

## 📊 Сравнение методов обнаружения

| **Метод**          | **Тип данных**      | **Преимущества**                  | **Недостатки**               |
|--------------------|---------------------|-----------------------------------|------------------------------|
| **IQR/3-σ**        | Одномерный          | Простота, интерпретируемость      | Не работает в многомерных пространствах |
| **Isolation Forest**| Многомерный         | Быстрый, работает с высокими размерностями | Чувствителен к параметрам    |
| **LOF**            | Многомерный         | Обнаружение локальных выбросов    | Медленный на больших данных  |
| **DBSCAN**         | Пространственный    | Обнаружение кластеров выбросов    | Чувствителен к параметрам    |
| **One-Class SVM**  | Многомерный         | Мощный нелинейный подход          | Вычислительно дорогой       |
| **Autoencoders**   | Сложные данные      | Обнаружение сложных паттернов     | Требует много данных         |

---

## ⚠️ Когда НЕ нужно удалять выбросы?
1. **Целевые выбросы**: В задачах обнаружения аномалий
2. **Естественные вариации**: Физические пределы (температура сверхновой)
3. **Важные события**: Финансовые кризисы, пандемии
4. **Репрезентативные данные**: При работе с реальными распределениями

---

## 🚀 Лучшие практики
1. **Всегда визуализируйте** данные перед обработкой
2. **Изучайте природу** выбросов (ошибка vs реальное явление)
3. **Сравнивайте модели** с обработкой и без
4. **Документируйте** все удаленные или измененные точки
5. **Используйте робастные метрики** (медиана вместо среднего)

---

## 📌 Тренировочные задания

### 🟢 Базовый уровень
1. Для датасета [1, 2, 2, 3, 3, 3, 4, 4, 100]:
   - Рассчитайте границы по IQR
   - Определите выбросы

### 🟡 Продвинутый уровень
1. Примените Isolation Forest к датасету Wine из sklearn
   - Визуализируйте выбросы в 2D PCA проекции
   - Сравните с результатами LOF

### 🔴 Экспертный уровень
1. Реализуйте потоковую систему обнаружения выбросов для временных рядов
   - Используйте скользящее окно + Isolation Forest
   - Визуализируйте результаты в реальном времени

---

```python
# Пример решения 🟢 Задания 1
data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 100])

q1 = np.percentile(data, 25)  # 2
q3 = np.percentile(data, 75)  # 4
iqr = q3 - q1  # 2
lower_bound = q1 - 1.5 * iqr  # 2 - 3 = -1
upper_bound = q3 + 1.5 * iqr  # 4 + 3 = 7

outliers = data[(data < lower_bound) | (data > upper_bound)]  # [100]
print(f"Выбросы: {outliers}")
```

---

## 💎 Заключение
**Ключевые принципы работы с выбросами:**
1. **Не бывает универсального решения** - метод зависит от данных и задачи
2. **Контекст важнее алгоритма** - понимайте природу аномалий
3. **Выбросы ≠ ошибки** - иногда это самая ценная информация
4. **Документирование критично** - сохраняйте историю изменений

**Профессиональный подход:**
- 🟢 Начинайте с простых статистических методов
- 🟡 Переходите к ML-методам для многомерных случаев
- 🔴 Для сложных задач используйте ансамбли и глубокое обучение

> "Выбросы - это не помехи, а сигналы о том, что наши модели неполны.  
> Изучайте их, а не просто удаляйте!" - John Tukey, основоположник EDA
