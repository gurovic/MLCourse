# Кросс-валидация (CV)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, 
    cross_val_score, GridSearchCV, train_test_split
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
```

---

## 🟢 Базовый уровень (Основы)

### 1.1 Зачем нужна кросс-валидация?
**Проблемы train-test split:**
- **Случайность разбиения:** Результат зависит от конкретного разбиения.
- **Неэффективное использование данных:** 20-30% данных не участвуют в обучении.

**Решение:** Кросс-валидация использует все данные для обучения и оценки.

### 1.2 K-Fold (K-блочная)
```python
# Пример с 3 фолдами
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([0, 0, 1, 1, 0, 1, 1, 0])

kf = KFold(n_splits=3, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X):
    print("Train:", train_index, "Test:", test_index)
```

### 1.3 Использование `cross_val_score`
```python
model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
```

---

## 🟡 Продвинутый уровень (Stratified K-Fold)

### 2.1 Стратифицированная выборка
**Проблема:** В KFold может нарушаться пропорция классов.  
**Решение:** `StratifiedKFold` сохраняет распределение классов.

```python
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for train_index, test_index in skf.split(X, y):
    print("Stratified Train:", train_index, "Test:", test_index)
```

### 2.2 Сравнение KFold и StratifiedKFold
```python
from collections import Counter

y_imbalanced = np.array([0]*80 + [1]*20)  # 80% класса 0

kf_counts = [Counter(y_imbalanced[test]) for _, test in KFold(n_splits=5).split(X, y_imbalanced)]
skf_counts = [Counter(y_imbalanced[test]) for _, test in StratifiedKFold(n_splits=5).split(X, y_imbalanced)]

print("KFold распределение:", kf_counts)
print("StratifiedKFold распределение:", skf_counts)
```

### 2.3 Временные характеристики
```python
import time

start = time.time()
cross_val_score(model, X, y, cv=10)
print(f"10-Fold время: {time.time()-start:.2f} сек")

cross_val_score(model, X, y, cv=3)  # Быстрее для больших данных
```

---

## 🔴 Экспертный уровень (Продвинутые техники)

### 3.1 Временные ряды (Time Series Split)
```python
tscv = TimeSeriesSplit(n_splits=3)
for train_index, test_index in tscv.split(X):
    print("Train:", train_index, "Test:", test_index)  # Тест всегда после трейна
```

### 3.2 Nested Cross-Validation
```python
outer_cv = StratifiedKFold(n_splits=5)
inner_cv = StratifiedKFold(n_splits=3)

param_grid = {'C': [0.1, 1, 10]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=inner_cv)

nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv)
print(f"Оценка модели: {nested_scores.mean():.2f}")
```

### 3.3 Кастомные стратегии
```python
from sklearn.model_selection import BaseCrossValidator

class GroupKFoldCustom(BaseCrossValidator):
    def __init__(self, n_splits=3):
        self.n_splits = n_splits
        
    def split(self, X, y=None, groups=None):
        # Пример: Разбиение по уникальным группам (например, клиентам)
        group_indices = np.unique(groups, return_inverse=True)[1]
        for fold in range(self.n_splits):
            train_idx = np.where(group_indices % self.n_splits != fold)[0]
            test_idx = np.where(group_indices % self.n_splits == fold)[0]
            yield train_idx, test_idx
```

### 3.4 Bootstrapping
```python
from sklearn.utils import resample

def bootstrap_metrics(model, X, y, n_iterations=100):
    scores = []
    for _ in range(n_iterations):
        X_sample, y_sample = resample(X, y)
        model.fit(X_sample, y_sample)
        scores.append(model.score(X, y))  # Оценка на полных данных
    return np.percentile(scores, [2.5, 97.5])  # 95% доверительный интервал
```

---

## 📊 Чеклист по уровням

| Уровень | Навыки |
|---------|--------|
| 🟢 | K-Fold, `cross_val_score`, базовое применение |
| 🟡 | `StratifiedKFold`, работа с дисбалансом, время выполнения |
| 🔴 | `TimeSeriesSplit`, Nested CV, кастомные стратегии |

---

## ⚠️ Антипаттерны
1. **Стратификация для регрессии** → Используйте бининг (`pd.cut`) для непрерывных целевых переменных.
2. **KFold для временных рядов** → Используйте `TimeSeriesSplit`.
3. **Утечка данных при предобработке** → Выполняйте нормализацию/кодирование внутри пайплайна.
4. **Слишком много фолдов** → Для больших данных используйте `cv=3-5`.

---

## 🚀 Продвинутые советы
1. **Автоматизация с `Pipeline`:**
```python
pipeline = make_pipeline(StandardScaler(), LogisticRegression())
cross_val_score(pipeline, X, y, cv=5)  # Предобработка внутри CV!
```

2. **Стратификация по непрерывной переменной:**
```python
y_bins = pd.cut(y, bins=5, labels=False)  # Бининг для регрессии
skf = StratifiedKFold(n_splits=3)
for train_index, test_index in skf.split(X, y_bins):
    ...
```

---

## 📌 Тренировочные задания

### 🟢 Базовый уровень
1. Реализуйте 5-Fold CV для линейной регрессии на `fetch_california_housing`.
2. Сравните MSE с single train-test split.

### 🟡 Продвинутый уровень
1. Для дисбалансного датасета (1:10) сравните KFold и StratifiedKFold по F1-score.
2. Проанализируйте распределение классов во фолдах.

### 🔴 Экспертный уровень
1. Реализуйте nested CV для подбора гиперпараметров SVM.
2. Создайте кастомный кросс-валидатор по группам (например, по клиентам).

---

## 🟢 Пример решения задания 1
```python
california = fetch_california_housing()
X, y = california.data, california.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression().fit(X_train, y_train)
mse_single = mean_squared_error(y_test, model.predict(X_test))

scores = -cross_val_score(LinearRegression(), X, y, cv=5, scoring='neg_mean_squared_error')
print(f"Single MSE: {mse_single:.2f}, CV MSE: {scores.mean():.2f}")
```

---

## 📌 Заключение
**Ключевые принципы:**
1. **Используйте кросс-валидацию** для надежной оценки.
2. **Stratify при дисбалансе** классов.
3. **Для временных данных** соблюдайте порядок с `TimeSeriesSplit`.
4. **Nested CV** для подбора гиперпараметров без утечки.
5. **Учитывайте вычислительные затраты** → оптимальное число фолдов.

**Помните:**
- Результаты CV — это интервал (`mean ± std`).
- Предобработка должна быть **внутри** цикла CV.
- Для малых датасетов: `cv=5-10`, для больших: `cv=3-5`.

Кросс-валидация — золотой стандарт оценки ML-моделей!
