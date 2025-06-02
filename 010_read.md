# Чтение данных в Pandas для ML  

```python
!pip install pandas numpy matplotlib seaborn pyarrow fastparquet scipy cudf
```
---
> ⚠️ How to install cudf for google colab
```python
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!python rapidsai-csp-utils/colab/pip-install.py
```
---

## 🟢 Базовый уровень (Must Know для начинающих)

### 1.1 Чтение данных из CSV

```python
import pandas as pd

# Простейший случай
df = pd.read_csv('data.csv')

# Сложнее: работа с "грязными" файлами
df = pd.read_csv(
    'data.csv',
    sep=';',                 # нестандартный разделитель
    encoding='utf-8',        # поддержка кириллицы или других кодировок
    na_values=['NA', '?'],   # обозначения пропусков
    parse_dates=['date_col'] # автоматический парсинг дат
)
```

> ⚠️ Если данные приходят без заголовков, используйте `header=None` и задайте названия колонок через `names`.

---

### 1.2 Просмотр данных

```python
print("Первые 5 строк:")
display(df.head())

print("\nОсновная статистика:")
display(df.describe(include='all'))  # include='all' покажет все типы

print("\nИнформация о типах:")
display(df.info())
```

> 💡 Для быстрого анализа можно использовать `pandas_profiling` или `sweetviz`.

---

### 1.3 Сохранение данных

```python
df.to_csv('processed_data.csv', index=False)  # без индексов
df.to_excel('report.xlsx', sheet_name='Data')  # сохранение в Excel
```

> ✅ Всегда используйте `index=False`, если не нужно сохранять индекс.

---

### 1.4 Решение базовых проблем

```python
# Удаление пропусков
df.dropna(subset=['important_col'], inplace=True)

# Заполнение пропусков медианой
df['age'] = pd.to_numeric(df['age'], errors='coerce') # некорректные значения превращаются в NaN
df.fillna({'age': df['age'].median()}, inplace=True)

# Удаление дубликатов
df = df.drop_duplicates()
```

> 💡 Можно заполнять пропуски и средним, и модой (`mode()`), в зависимости от типа данных.

---

## 🟡 Продвинутый уровень (Для опытных ML-инженеров)

### 2.1 Оптимизация памяти

```python
def optimize_dtypes(df):
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

df = optimize_dtypes(df)
print(f"Память: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

> 📈 Такая оптимизация особенно важна на этапе предобработки больших датасетов.

---

### 2.2 Работа с категориальными признаками

```python
# Преобразование в категориальный тип
df['category_col'] = df['category_col'].astype('category')

# One-Hot Encoding
df = pd.get_dummies(df, columns=['category_col'], prefix='cat')
```

> ⚠️ Для бустинговых моделей (CatBoost, LightGBM) лучше оставлять как `category`.

---

### 2.3 Эффективное чтение больших данных

```python
# Чтение только нужных столбцов
cols = ['feature1', 'feature2', 'target']
df = pd.read_csv('big_data.csv', usecols=cols)

# Сохранение в Parquet (быстрее CSV в 5–10 раз)
df.to_parquet('data.parquet', engine='pyarrow')
```

> ✅ Используйте Parquet или Feather для хранения промежуточных результатов.

---

### 2.4 Интеграция с ML-библиотеками

```python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1).values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

> 💡 Для удобства работы можно оставить данные в виде `DataFrame` и использовать `.to_numpy()` при необходимости.

---

## 🔴 Экспертный уровень (Для хакатонов и Kaggle Grandmasters)

### 3.1 Работа с гигабайтными данными

```python
chunk_size = 10**5  # 100k строк за раз
filtered_rows = []

for chunk in pd.read_csv('terabyte_data.csv', chunksize=chunk_size):
    chunk = chunk[chunk['value'] > 0]  # фильтрация на лету
    filtered_rows.append(chunk)

df = pd.concat(filtered_rows)
```

> 📦 Это позволяет работать с файлами, которые больше доступной оперативной памяти.

---

### 3.2 Разреженные матрицы для текстов/NLP

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

vectorizer = TfidfVectorizer()
sparse_matrix = vectorizer.fit_transform(df['text'])

# Сохранение в формате, совместимом с LibSVM
save_npz('sparse_data.npz', sparse_matrix)
```

> ✅ Разреженные матрицы экономят память и подходят для моделей вроде SVM, LogisticRegression.

---

### 3.3 GPU-ускорение с RAPIDS

```python
import cudf

# Чтение данных на GPU
gdf = cudf.read_csv('big_data.csv')
gdf = gdf.query('value > 0')  # фильтрация в 10–50x быстрее, чем pandas

# Конвертация в numpy для моделей
X = gdf.to_pandas().values
```

> 🚀 Требуется CUDA-совместимая видеокарта и установка библиотек RAPIDS.

---

### 3.4 Экстремальная оптимизация

```python
# Уменьшение точности float64 -> float16
df = df.astype({col: 'float16' for col in df.select_dtypes(include=['float']).columns})

# Словарное сжатие для строк
df['text'] = df['text'].astype('category').cat.codes

# Партицирование данных по ключу
df.groupby('month').apply(lambda x: x.to_parquet(f"data_{x.name}.parquet"))
```

> 🧪 Полезно для очень больших датасетов, где каждый байт важен.

---

### 3.5 Инкрементальное обучение (Online Learning)

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier()
for chunk in pd.read_csv('stream.csv', chunksize=1000):
    X = chunk.drop('target', axis=1)
    y = chunk['target']
    model.partial_fit(X, y, classes=np.unique(y))
```

> 🔄 Подходит для потоковой обработки данных, например, в реальном времени.

---

## 📊 Чеклист по уровням

| Уровень | Навыки |
|--------|--------|
| 🟢     | Чтение CSV/Excel, анализ данных, fillna/drop_duplicates |
| 🟡     | Оптимизация типов, категориальные признаки, parquet, train_test_split |
| 🔴     | Итеративная обработка, GPU-ускорение, разреженные матрицы, online learning |

---

## ⚠️ Антипаттерны

| Уровень | Что делать не стоит |
|---------|---------------------|
| 🟢      | Сохранять данные с индексом без необходимости |
| 🟢      | Читать весь большой файл целиком ради просмотра |
| 🟡      | Не использовать `usecols` при работе с большими файлами |
| 🔴      | Игнорировать `dtype` при чтении больших данных |
| 🔴      | Использовать медленные циклы вместо векторизованных операций |

---

## 🚀 Полезные советы

1. **Тестирование перед обработкой:**  
   ```python
   pd.read_csv('test_data.csv', nrows=10)  # проверка структуры
   ```

2. **Быстрая конвертация в NumPy:**  
   ```python
   X = df.values  # работает быстро
   ```

3. **Проверка памяти:**  
   ```python
   df.memory_usage(deep=True)
   ```

4. **Избежание копий:**  
   Используйте `inplace=True` или переприсваивайте переменные явно.

---

## 📌 Итог

Pandas — основной инструмент для подготовки данных в ML. Он прост в освоении, но мощный в использовании. От базового анализа до экспертных техник вроде GPU-ускорения и онлайн-обучения — он покрывает полный цикл работы с данными.

> 🎯 Главное правило: всегда помни, что ты готовишь данные **для модели**, поэтому делай это эффективно и аккуратно.
