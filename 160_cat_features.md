
# Категориальные признаки – One-Hot, Target Encoding

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder, CatBoostEncoder
from sklearn.model_selection import train_test_split

# !pip install category_encoders scikit-learn
```

---

## 🟢 Базовый уровень (Основные методы)

### 1.1 One-Hot Encoding (OHE)
**Принцип:** Создание бинарных колонок для каждой категории  
**Плюсы:**  
- Простота интерпретации  
- Сохраняет информацию о категориях  
**Минусы:**  
- Проклятие размерности для признаков с высокой кардинальностью  

```python
# Пример
data = pd.DataFrame({'color': ['red', 'blue', 'green', 'blue']})

# Стандартный подход
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
ohe_data = ohe.fit_transform(data[['color']])
print(ohe_data)
# [[0. 1. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]
```

### 1.2 Frequency Encoding
**Принцип:** Замена категорий на частоту их встречаемости  
**Лучше всего:** Для деревьев и ансамблей  

```python
freq = data['color'].value_counts(normalize=True)
data['color_freq'] = data['color'].map(freq)
print(data)
```

### 1.3 Label Encoding
**Принцип:** Присвоение числовых меток (0, 1, 2,...)  
**Осторожно:** Не подходит для линейных моделей (искусственный порядок)  

```python
data['color_label'] = data['color'].astype('category').cat.codes
print(data)
```

---

## 🟡 Продвинутый уровень (Target-Based Encoding)

### 2.1 Базовый Target Encoding
**Принцип:** Замена категории на среднее значение целевой переменной  
**Риск:** Утечка данных при неправильной реализации  

```python
# Опасный способ (утечка данных)
data = pd.DataFrame({
    'city': ['A','A','B','B','C','C'],
    'price': [100, 120, 200, 180, 300, 320]
})

# Неправильно!
data['city_target'] = data.groupby('city')['price'].transform('mean')

# Правильно (через split)
train, test = train_test_split(data, test_size=0.3)
means = train.groupby('city')['price'].mean()
test['city_target'] = test['city'].map(means)
```

### 2.2 Сглаженный Target Encoding
**Формула:**  
$encoded = \frac{mean \times n_{samples} + global\_mean \times \alpha}{n_{samples} + \alpha}$  

```python
encoder = TargetEncoder(smoothing=5.0)
train_encoded = encoder.fit_transform(train[['city']], train['price'])
test_encoded = encoder.transform(test[['city']])
```

### 2.3 CatBoost Encoding
**Особенность:** Использует принцип ordered boosting для предотвращения утечки  

```python
encoder = CatBoostEncoder()
train_encoded = encoder.fit_transform(train[['city']], train['price'])
test_encoded = encoder.transform(test[['city']])
```

---

## 🔴 Экспертный уровень (Продвинутые техники)

### 3.1 Кросс-валидационное кодирование
```python
from sklearn.model_selection import KFold

def cross_val_encode(df, col, target, n_splits=5):
    df[f'{col}_encoded'] = np.nan
    kf = KFold(n_splits=n_splits)
    
    for train_idx, val_idx in kf.split(df):
        train = df.iloc[train_idx]
        val = df.iloc[val_idx]
        means = train.groupby(col)[target].mean()
        df.loc[val.index, f'{col}_encoded'] = val[col].map(means)
    
    return df

data = cross_val_encode(data, 'city', 'price')
```

### 3.2 Генерация признаков через NLP
```python
# Для текстовых категорий
from sklearn.feature_extraction.text import TfidfVectorizer

categories = ['electronics', 'books', 'home appliances']
vectorizer = TfidfVectorizer(max_features=10)
embeddings = vectorizer.fit_transform(categories)
print(embeddings.toarray())
```

### 3.3 Оптимальное кодирование под задачу
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# Тестирование разных стратегий
encoders = {
    'onehot': OneHotEncoder(),
    'target': TargetEncoder(),
    'catboost': CatBoostEncoder()
}

for name, encoder in encoders.items():
    model = Pipeline([
        ('encode', encoder),
        ('model', RandomForestRegressor())
    ])
    score = cross_val_score(model, X, y, cv=5).mean()
    print(f"{name}: {score:.4f}")
```

### 3.4 GPU-ускорение
```python
# Использование RAPIDS для OHE
import cudf
gdf = cudf.from_pandas(data)
gdf_ohe = gdf.one_hot_encoding(column='city', prefix='city')
```

---

## 📊 Сравнение методов

| **Метод**          | **Подходит для**       | **Риск утечки** | **Размерность** |
|--------------------|------------------------|-----------------|-----------------|
| One-Hot            | Линейные модели        | Нет             | Высокая         |
| Target Encoding    | Деревья, бустинги      | Высокий         | Низкая          |
| CatBoost Encoding  | Градиентный бустинг    | Низкий          | Низкая          |
| Frequency Encoding | Ансамбли               | Нет             | Низкая          |

---

## ⚠️ Антипаттерны
1. **Применение OHE к признакам с >100 категориями** → используйте Target Encoding
2. **Target Encoding без сглаживания** → переобучение
3. **Использование Label Encoding для линейных моделей** → ложные зависимости
4. **Кодирование теста отдельно от трейна** → утечка данных

---

## 🚀 Продвинутые советы
1. **Комбинирование подходов:**
```python
data['city_ohe'] = OneHotEncoder().fit_transform(data[['city']])  # Для линейной части
data['city_target'] = TargetEncoder().fit_transform(data[['city']], data['price'])  # Для деревьев
```

2. **Динамическое сглаживание:**
```python
class AdaptiveEncoder:
    def __init__(self, min_samples=50):
        self.min_samples = min_samples
        
    def encode(self, group, global_mean):
        n = len(group)
        alpha = max(self.min_samples - n, 0)
        return (group.mean() * n + global_mean * alpha) / (n + alpha)
```

3. **Встраивание в пайплайн:**
```python
preprocessor = ColumnTransformer([
    ('ohe', OneHotEncoder(), ['color']),
    ('target', TargetEncoder(), ['city'])
])
pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', xgb.XGBRegressor())
])
```

---

## 📌 Тренировочные задания

### 🟢 Базовый уровень
1. Примените One-Hot Encoding к колонке "product_type" с 5 категориями.
2. Реализуйте Frequency Encoding для колонки "country" (50 уникальных значений).

### 🟡 Продвинутый уровень
1. Сравните Target Encoding и CatBoost Encoding на датасете House Prices.
2. Реализуйте сглаживание в ручном Target Encoding с alpha=10.

### 🔴 Экспертный уровень
1. Реализуйте кросс-валидационное кодирование для временных рядов (с соблюдением порядка).
2. Создайте ансамбль из признаков, закодированных разными методами.

---

```python
# Пример решения 🟢 Задания 1
data = pd.DataFrame({'product_type': ['A', 'B', 'C', 'A', 'D']})
ohe = OneHotEncoder(sparse_output=False)
encoded = ohe.fit_transform(data[['product_type']])
print(encoded)
```

---

## 📌 Заключение
Ключевые принципы:
1. **Выбор метода зависит от модели:**
   - Линейные модели: One-Hot (до 20 категорий)
   - Деревья/бустинги: Target Encoding
2. **Всегда предотвращайте утечки:**
   - Разделение до кодирования
   - Кросс-валидационные схемы
   - Ordered encoding (CatBoost)
3. **Экспериментируйте с комбинациями:**
   - One-Hot + Target Encoding
   - Частоты + Embeddings
4. **Контролируйте размерность:**
   - Уменьшение категорий (объединение редких)
   - Использование хеширования

Помните: нет "серебряной пули" - тестируйте разные подходы для ваших данных!
