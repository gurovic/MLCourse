
# Случайный лес – бутстрэп, агрегация

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# !pip install scikit-learn
```

---

## 🟢 Базовый уровень (Основные концепции)

### 1.1 Что такое случайный лес?
**Ансамблевый метод:** Комбинация множества деревьев решений  
**Два ключевых принципа:**
1. **Бутстрэп (Bootstrap):** Случайная выборка данных с возвращением
2. **Агрегация (Bagging):** Объединение предсказаний деревьев

```python
# Простой пример
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.3f}")
```

### 1.2 Преимущества перед одиночным деревом
- Уменьшение переобучения
- Повышение точности
- Стабильность к выбросам
- Оценка важности признаков

### 1.3 Визуализация деревьев леса
```python
# Первые 3 дерева
plt.figure(figsize=(15, 10))
for i in range(3):
    plt.subplot(1, 3, i+1)
    tree = rf.estimators_[i]
    plot_tree(tree, feature_names=load_breast_cancer().feature_names, 
              filled=True, max_depth=2)
plt.tight_layout()
plt.show()
```

---

## 🟡 Продвинутый уровень (Реализация)

### 2.1 Реализация бэггинга
```python
class SimpleBagging:
    def __init__(self, base_estimator, n_estimators=10):
        self.estimators = [clone(base_estimator) for _ in range(n_estimators)]
        
    def fit(self, X, y):
        for estimator in self.estimators:
            # Бутстрэп выборка
            indices = np.random.choice(len(X), size=len(X), replace=True)
            estimator.fit(X[indices], y[indices])
            
    def predict(self, X):
        predictions = np.array([estimator.predict(X) for estimator in self.estimators])
        return np.mean(predictions, axis=0) > 0.5  # Голосование для классификации
```

### 2.2 Важность признаков
```python
# На основе среднего снижения нечистоты
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

# Визуализация
forest_importances = pd.Series(importances, index=load_breast_cancer().feature_names)
forest_importances.sort_values().plot(kind='barh', xerr=std, figsize=(10, 8))
plt.title("Важность признаков")
plt.xlabel("Среднее снижение нечистоты")
plt.show()
```

### 2.3 Анализ Out-of-Bag ошибки
```python
rf = RandomForestClassifier(oob_score=True, n_estimators=200)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_:.3f}")

# Сравнение с тестовой выборкой
y_pred = rf.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

---

## 🔴 Экспертный уровень (Продвинутые техники)

### 3.1 Extremely Randomized Trees
```python
from sklearn.ensemble import ExtraTreesClassifier

# Случайный выбор порогов разделения
et = ExtraTreesClassifier(n_estimators=100, random_state=42)
et.fit(X_train, y_train)
print(f"ExtraTrees Accuracy: {accuracy_score(y_test, et.predict(X_test)):.3f}")
```

### 3.2 Подбор гиперпараметров
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [5, 10, None]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Лучшие параметры: {grid_search.best_params_}")
```

### 3.3 Регрессия и доверительные интервалы
```python
# Случайный лес для регрессии
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1)
rf_reg = RandomForestRegressor(n_estimators=100)

# Предсказание с интервалом
rf_reg.fit(X_reg, y_reg)
predictions = np.array([tree.predict(X_reg) for tree in rf_reg.estimators_])
mean_pred = np.mean(predictions, axis=0)
std_pred = np.std(predictions, axis=0)

# Визуализация
plt.figure(figsize=(10, 6))
plt.errorbar(y_reg[:50], mean_pred[:50], yerr=1.96*std_pred[:50], fmt='o', alpha=0.7)
plt.plot([min(y_reg), max(y_reg)], [min(y_reg), max(y_reg)], 'r--')
plt.title("Предсказания с доверительным интервалом")
plt.xlabel("Истинные значения")
plt.ylabel("Предсказания")
plt.show()
```

### 3.4 GPU-ускорение
```python
# Использование cuML для GPU
# !pip install cuml

from cuml.ensemble import RandomForestClassifier as cuRFC

gpu_rf = cuRFC(n_estimators=100)
gpu_rf.fit(X_train, y_train)
print(f"GPU Accuracy: {accuracy_score(y_test, gpu_rf.predict(X_test)):.3f}")
```

---

## 📊 Чеклист по уровням

| Уровень | Навыки |
|---------|--------|
| 🟢 | Понимание бутстрэпа и бэггинга, базовое применение |
| 🟡 | Реализация бэггинга, анализ важности, OOB-ошибка |
| 🔴 | ExtraTrees, доверительные интервалы, GPU-ускорение |

---

## ⚠️ Антипаттерны
1. **Использование без настройки гиперпараметров** (особенно max_depth)
2. **Игнорирование OOB-оценки** для быстрой валидации
3. **Слепая интерпретация важности признаков** без проверки
4. **Применение к высокоразмерным данным** без отбора признаков

---

## 🚀 Продвинутые советы
1. **Пост-обработка важности:**
```python
# Удаление коррелированных признаков
corr_matrix = pd.DataFrame(X_train).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
filtered_importances = importances[~np.isin(feature_names, to_drop)]
```

2. **Ансамблирование разных моделей:**
```python
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

ensemble = VotingClassifier([
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('svm', SVC(probability=True))
], voting='soft')
```

3. **Инкрементальное обучение:**
```python
# Через warm_start
rf = RandomForestClassifier(warm_start=True, n_estimators=50)
rf.fit(X_train[:100], y_train[:100])
rf.n_estimators += 50
rf.fit(X_train, y_train)  # Продолжает обучение
```

---

## 📌 Тренировочные задания

### 🟢 Базовый уровень
1. Обучите случайный лес на датасете Iris, сравните accuracy с одиночным деревом.
2. Визуализируйте важность признаков.

### 🟡 Продвинутый уровень
1. Реализуйте простой вариант бэггинга для 10 деревьев.
2. Сравните OOB-оценку с тестовой точностью.

### 🔴 Экспертный уровень
1. Постройте доверительные интервалы для регрессионной задачи.
2. Сравните время обучения на CPU и GPU для датасета 500k строк.

---

```python
# Пример решения 🟢 Задания 1
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Одиночное дерево
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X_train, y_train)
tree_acc = accuracy_score(y_test, tree.predict(X_test))

# Случайный лес
rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

print(f"Tree Accuracy: {tree_acc:.3f}, RF Accuracy: {rf_acc:.3f}")
```

---

## 📌 Заключение
Ключевые принципы:
1. **Разнообразие деревьев** → сила леса (разные данные + разные признаки)
2. **OOB-оценка** как быстрый аналог кросс-валидации
3. **Важность признаков** - побочный продукт обучения
4. **Масштабируемость** - параллельное обучение деревьев

Случайный лес:
- Работает "из коробки" без сложной настройки
- Хорош для табличных данных
- Дает интерпретируемые результаты
- Легко распараллеливается

Используйте его как baseline перед переходом к бустингам!
