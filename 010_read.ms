```python
# Чтение и вывод данных в pandas для ML (Jupyter Notebook)

# !pip install pandas numpy matplotlib seaborn pyarrow fastparquet scipy cudf
```

---

## 🟢 Базовый уровень (Must Know для начинающих)

### 1.1 Чтение данных из CSV
```python
import pandas as pd

# Базовый пример
df = pd.read_csv('data.csv')

# Параметры для "грязных" данных
df = pd.read_csv(
    'data.csv',
    sep=';',                 # нестандартный разделитель
    encoding='latin1',       # для кириллицы: 'utf-8' или 'cp1251'
    na_values=['NA', '?'],   # обозначения пропусков
    parse_dates=['date_col'] # автоматический парсинг дат
)
```

### 1.2 Просмотр данных
```python
print("Первые 5 строк:")
display(df.head())

print("\nОсновная статистика:")
display(df.describe(include='all'))

print("\nИнформация о типах:")
display(df.info())
```

### 1.3 Сохранение данных
```python
df.to_csv('processed_data.csv', index=False)  # без сохранения индексов
df.to_excel('report.xlsx', sheet_name='Data')
```

### 1.4 Решение базовых проблем
```python
# Пропуски: удаление или заполнение
df.dropna(subset=['important_col'], inplace=True)
df.fillna({'age': df['age'].median()}, inplace=True)

# Дубликаты
df = df.drop_duplicates()
```

---

## 🟡 Продвинутый уровень (Для опытных ML-инженеров)

### 2.1 Оптимизация памяти
```python
# Автоматическое приведение типов
def optimize_dtypes(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df

df = optimize_dtypes(df)
print(f"Память: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

### 2.2 Работа с категориальными признаками
```python
# Конвертация + One-Hot Encoding для моделей
df['category_col'] = df['category_col'].astype('category')
df = pd.get_dummies(df, columns=['category_col'], prefix='cat')
```

### 2.3 Эффективное чтение больших данных
```python
# Чтение только нужных колонок
cols = ['feature1', 'feature2', 'target']
df = pd.read_csv('big_data.csv', usecols=cols)

# Сохранение в бинарных форматах
df.to_parquet('data.parquet', engine='pyarrow')  # быстрее CSV в 5-10x
```

### 2.4 Интеграция с ML-библиотеками
```python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1).values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

---

## 🔴 Экспертный уровень (Для хакатонов и Kaggle Grandmasters)

### 3.1 Работа с гигабайтными данными
```python
# Итеративное чтение с фильтрацией
chunk_size = 10**5  # 100k строк за раз
filtered_rows = []

for chunk in pd.read_csv('terabyte_data.csv', chunksize=chunk_size):
    chunk = chunk[chunk['value'] > 0]  # фильтрация на лету
    filtered_rows.append(chunk)

df = pd.concat(filtered_rows)
```

### 3.2 Разреженные матрицы для текстов/NLP
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

vectorizer = TfidfVectorizer()
sparse_matrix = vectorizer.fit_transform(df['text'])

# Сохранение в формате, совместимом с LibSVM
save_npz('sparse_data.npz', sparse_matrix)
```

### 3.3 GPU-ускорение с RAPIDS
```python
import cudf

# Чтение данных на GPU
gdf = cudf.read_csv('big_data.csv')
gdf = gdf.query('value > 0')  # фильтрация в 10-50x быстрее pandas

# Конвертация в numpy для моделей
X = gdf.to_pandas().values
```

### 3.4 Экстремальная оптимизация
```python
# Уменьшение точности float64 -> float16
df = df.astype({col: 'float16' for col in float_cols})

# Словарное сжатие для строк
df['text'] = df['text'].astype('category').cat.codes

# Партицирование данных по ключу
df.groupby('month').apply(lambda x: x.to_parquet(f"data_{x.name}.parquet"))
```

### 3.5 Инкрементальное обучение (Online Learning)
```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier()
for chunk in pd.read_csv('stream.csv', chunksize=1000):
    X = chunk.drop('target', axis=1)
    y = chunk['target']
    model.partial_fit(X, y, classes=np.unique(y))
```

---

## 📊 Чеклист по уровням

| Уровень | Обязательные навыки |
|---------|---------------------|
| 🟢      | Чтение CSV/Excel, базовый анализ, fillna/drop_duplicates |
| 🟡      | Оптимизация типов, категориальные фичи, parquet, train_test_split |
| 🔴      | Итеративная обработка, GPU-ускорение, разреженные матрицы, online learning |

---

## ⚠️ Антипаттерны
- **Для всех уровней:** 
  - Сохранение с индексом (`index=True`)
  - Чтение всего файла для предпросмотра (используйте `nrows=1000`)
- **🔴 Эксперты:** 
  - Неоптимизированные циклы по DataFrame
  - Игнорирование `dtype` при чтении гигабайтных данных

```python
# Тестирование всех примеров (замените пути на реальные данные)
try:
    pd.read_csv('test_data.csv', nrows=10)
except FileNotFoundError:
    print("⚠️ Создайте тестовый файл или используйте sample данные!")
``` 

Этот ноутбук можно использовать как шаблон для проектов: копируйте ячейки нужного уровня и адаптируйте под конкретную задачу.
