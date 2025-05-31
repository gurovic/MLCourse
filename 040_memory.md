```python
# Оптимизация памяти данных

import pandas as pd
import numpy as np
from sys import getsizeof

# !pip install dask cudf pyarrow
```

---

## 🟢 Базовый уровень (Основные методы)

### 1.1 Сжатие числовых типов
```python
# Исходные данные
df = pd.DataFrame({'value': [0, 1, 0, 1, 2, 3, 0]})

# Оптимизация
df['value'] = df['value'].astype('int8')
print(f"Память: {df.memory_usage(deep=True).sum()} байт")
```

### 1.2 Категоризация строк
```python
# До оптимизации
df = pd.DataFrame({'color': ['red', 'blue', 'green'] * 1000})
print(f"Исходно: {getsizeof(df)} байт")

# После
df['color'] = df['color'].astype('category')
print(f"После оптимизации: {getsizeof(df)} байт")
```

### 1.3 Автоматический downcast
```python
df = pd.DataFrame({'price': [100.5, 200.3, 150.0]})
df['price'] = pd.to_numeric(df['price'], downcast='float')
print(f"Тип: {df['price'].dtype}")  # float32 вместо float64
```

---

## 🟡 Продвинутый уровень (Автоматизация)

### 2.1 Функция оптимизации типов
```python
def optimize_dtypes(df):
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'object':
            if df[col].nunique() < 0.5 * len(df):
                df[col] = df[col].astype('category')
                
        elif 'int' in str(col_type):
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        elif 'float' in str(col_type):
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

df = optimize_dtypes(pd.read_csv('data.csv'))
```

### 2.2 Параллельная обработка с Dask
```python
import dask.dataframe as dd

ddf = dd.read_csv('big_data.csv', blocksize=100e6)
ddf['id'] = ddf['id'].astype('int32')
ddf = ddf.categorize(columns=['city'])
```

### 2.3 Анализ памяти
```python
def memory_report(df):
    mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Всего: {mem:.2f} MB")
    
    for col in df.columns:
        col_mem = df[col].memory_usage(deep=True) / 1024**2
        print(f"{col}: {col_mem:.2f} MB ({df[col].dtype})")
```

---

## 🔴 Экспертный уровень (Экстремальная оптимизация)

### 3.1 Распределенные типы
```python
from pandas.api.types import CategoricalDtype

# Создаем свой категориальный тип
dtype = CategoricalDtype(categories=['low', 'medium', 'high'], ordered=True)
df['priority'] = df['priority'].astype(dtype)
```

### 3.2 GPU-ускорение с RAPIDS
```python
import cudf

gdf = cudf.read_csv('data.csv')
gdf['timestamp'] = gdf['timestamp'].astype('datetime64[ms]')  # В 5x быстрее pandas
```

### 3.3 Форматы хранения
```python
# Сохранение в Parquet
df = optimize_dtypes(df)
df.to_parquet('data.parquet', engine='pyarrow')

# Сравнение размеров
print(f"CSV: {os.path.getsize('data.csv')/1e6:.1f} MB")
print(f"Parquet: {os.path.getsize('data.parquet')/1e6:.1f} MB")
```

---

## 📊 Чеклист по уровням

| Уровень | Навыки |
|---------|--------|
| 🟢 | astype(), категоризация, downcast |
| 🟡 | Автоматизация, Dask, анализ памяти |
| 🔴 | Кастомные категории, RAPIDS, Parquet |

---

## ⚠️ Антипаттерны
1. **Категоризация всех строк** (>50% уникальных значений)
2. **Использование int8** для значений >127
3. **Смешанные типы** в колонках (object)
4. **Игнорирование меток времени** (datetime64[ns] → datetime64[s])

---

## 🚀 Продвинутые советы
1. **Словарное сжатие для строк:**
```python
# Замена строк на индексы
categories, uniques = pd.factorize(df['city'])
df['city_id'] = categories.astype('int16')
mapping = pd.Series(uniques, name='city').to_frame()
```

2. **Бинарное хранение булевых значений:**
```python
df['is_active'] = df['is_active'].astype('bool').astype('uint8')  # 1 байт → 1 бит
```

3. **Экономия на временных данных:**
```python
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')  # UNIX time → datetime64
```

---

## 📈 Практический пример: Оптимизация датасета 10M строк

**Исходные данные:**
- 5 числовых колонк (int64)
- 3 строковых (object)
- 2 даты (datetime64[ns])

**После оптимизации:**
- Числовые → int8/int32
- Строки → category
- Даты → datetime64[s]

**Результат:**
- Память: 2.1 GB → 340 MB (-84%)
- Время загрузки: 18s → 4s

---

## 📌 Тренировочные задания

### 🟢 Базовый уровень
1. Преобразуйте колонку `user_id` (значения 0-1000) в оптимальный тип.
2. Категоризируйте колонку `country` (50 уникальных значений).
3. Сравните размер DataFrame до/после.

```python
# Тест:
assert df['user_id'].dtype == 'int16'
```

### 🟡 Продвинутый уровень
1. Напишите функцию для автоматического downcast всех числовых колонок.
2. Обработайте CSV 1GB через Dask с оптимизацией типов.
3. Сравните производительность apply до/после.

### 🔴 Экспертный уровень
1. Реализуйте бинарное хранение для 20 булевых колонок (1 бит на значение).
2. Конвертируйте датасет в Parquet с кастомной схемой типов.
3. Добейтесь 10x сокращения памяти для временных меток.

---

```python
# Пример решения 🟢 Задания 1
df = pd.DataFrame({'user_id': np.arange(0, 1001)})
df['user_id'] = df['user_id'].astype('int16')
print(f"Максимальное значение: {df.user_id.max()}")  # Должно быть 1000
```

---

## 📌 Заключение
Ключевые принципы:
1. **Анализируйте перед оптимизацией** (memory_usage, dtypes)
2. **Используйте минимальные типы** без потери информации
3. **Категоризация** эффективна при <50% уникальных значений
4. **Бинарные форматы** (Parquet) лучше CSV в 2-10x
5. **GPU-ускорение** даёт выигрыш на больших данных

Оптимизация памяти — баланс между экономией ресурсов и сохранением функциональности. Всегда проверяйте, что преобразования не нарушают логику данных!
