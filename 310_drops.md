```python
# !pip install pandas numpy scikit-learn xgboost
```

---

## 🟢 Базовый уровень (Основные методы)

### 1.1 Обнаружение пропусков
```python
import pandas as pd

# Подсчет пропусков по колонкам
print("Пропуски:\n", df.isnull().sum())

# Визуализация
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
df.isnull().mean().plot(kind='bar')  # Доля пропусков
plt.title('Процент пропусков по признакам')
plt.show()
```

### 1.2 Простые стратегии
```python
# Удаление строк/колонок
df_dropped = df.dropna(axis=0, subset=['target'])  # Удалить строки с пропуском в target
df_dropped_cols = df.drop(columns=['col_with_90%_nulls'])  # Удалить колонку

# Заполнение константой
df_filled = df.fillna({
    'numeric_col': -999,
    'categorical_col': 'Unknown'
})

# Заполнение статистиками
df['age'] = df['age'].fillna(df['age'].median())
df['income'] = df['income'].fillna(df.groupby('education')['income'].transform('mean'))
```

---

## 🟡 Продвинутый уровень (ML-ориентированные методы)

### 2.1 Интерполяция
```python
# Временные ряды
df['price'] = df['price'].interpolate(method='time') 

# Многомерная интерполяция
df['temperature'] = df['temperature'].interpolate(method='linear', limit_direction='both')
```

### 2.2 Предсказание пропусков
```python
from sklearn.ensemble import RandomForestRegressor

def impute_with_model(df, target_col):
    # Разделяем данные
    known = df[df[target_col].notnull()]
    unknown = df[df[target_col].isnull()]
    
    # Обучаем модель
    X = known.drop(columns=[target_col])
    y = known[target_col]
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    
    # Предсказываем и заполняем
    df.loc[df[target_col].isnull(), target_col] = model.predict(unknown.drop(columns=[target_col]))
    return df

df = impute_with_model(df, 'salary')
```

### 2.3 Категориальные данные
```python
# Добавление отдельной категории
df['city'] = df['city'].fillna('Unknown')

# Частотное заполнение
most_common = df['product_type'].mode()[0]
df['product_type'] = df['product_type'].fillna(most_common)
```

---

## 🔴 Экспертный уровень (Для соревнований и Big Data)

### 3.1 Автоматический выбор стратегии
```python
from autoimpute.imputations import MultipleImputer

# Множественное импьютирование
imputer = MultipleImputer(n=5, strategy={'numeric': 'pmm', 'categorical': 'logreg'})
df_imputed = imputer.fit_transform(df)

# Сохранение всех вариантов для ансамблей
for i in range(5):
    df[f'salary_imputed_{i}'] = imputer.imputed_[i]['salary']
```

### 3.2 Градиентный бустинг для больших данных
```python
import xgboost as xgb

# Итеративное заполнение
for col in ['age', 'income']:
    mask = df[col].isnull()
    model = xgb.XGBRegressor()
    model.fit(df[~mask].drop(columns=[col]), df.loc[~mask, col])
    df.loc[mask, col] = model.predict(df[mask].drop(columns=[col]))
```

### 3.3 Оптимизация для GPU
```python
import cudf
from cuml.impute import SimpleImputer

gdf = cudf.from_pandas(df)
imputer = SimpleImputer(strategy='median', missing_values=np.nan)
gdf_imputed = imputer.fit_transform(gdf)
```

### 3.4 Пайплайн для продакшена
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), ['age', 'income']),
    ('cat', SimpleImputer(strategy='most_frequent'), ['city'])
])

pipeline = Pipeline([
    ('imputer', preprocessor),
    ('model', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)
```

---

## 📊 Чеклист по уровням

| Уровень | Навыки |
|---------|--------|
| 🟢 | `fillna()`, `dropna()`, групповое заполнение, визуализация пропусков |
| 🟡 | Интерполяция, ML-модели для заполнения, работа с категориями |
| 🔴 | Ансамбли импьютеров, GPU-ускорение, интеграция в пайплайны |

---

## ⚠️ Антипаттерны
### Для всех уровней:
- **Удаление >30% строк** без анализа природы пропусков
- **Заполнение медианой** для мультимодальных распределений
- **Игнорирование временного контекста** в интерполяции

### 🔴 Эксперты:
- Использование **одинаковой стратегии** для всех колонок
- [**Утечка данных** при предсказании пропусков (не разделяйте train/test!)](310_10_drops_leak.md)

---

## 🚀 Советы
1. **Проверяйте распределения** до/после заполнения:
```python
df['age'].plot(kind='kde', label='After')
df['age'].dropna().plot(kde=True, label='Before')
plt.legend()
```

2. **Создавайте бинарные флаги** пропусков:
```python
df['age_missing'] = df['age'].isnull().astype(int)
```

3. **Экспериментируйте с NaN** как отдельной категорией в деревьях.

---

## 📈 Практический пример
**Задача:** Заполнить пропуски в данных Titanic (возраст, порт посадки).
```python
# 🟡 Продвинутый подход
df['age'] = df.groupby(['pclass', 'sex'])['age'].transform(lambda x: x.fillna(x.median()))
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# 🔴 Экспертный подход (XGBoost)
impute_xgboost(df, 'age')
df['embarked'] = df['embarked'].cat.add_categories('Unknown').fillna('Unknown')
```

```python
# Тестирование
assert df.isnull().sum().sum() == 0, "Есть пропуски!"
```

---

## 📌 Заключение
Правильная обработка пропусков может улучшить качество модели на 10-30%. Выбирайте методы в зависимости от:
1) Причины пропусков (MCAR, MAR, MNAR)  
2) Доли пропусков  
3) Типа данных (числовые/категориальные)  
4) Вычислительных ресурсов
