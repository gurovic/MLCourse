
# Деревья решений - критерии Gini/энтропия

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# !pip install scikit-learn
```

---

## 🟢 Базовый уровень (Основные концепции)

### 1.1 Что такое дерево решений?
**Алгоритм:** Иерархическая структура условий "если-то", разделяющая данные на классы  
**Преимущества:**  
- Интерпретируемость  
- Работает с категориальными и числовыми признаками  
- Не требует масштабирования данных  

### 1.2 Критерии разделения
**Энтропия:**  
$Entropy = -\sum(p_i \log_2 p_i)$  
Мера неопределенности (0 для чистых узлов)

**Индекс Джини:**  
$Gini = 1 - \sum(p_i^2)$  
Мера нечистоты (0 для чистых узлов)

```python
# Расчет вручную
def gini(p):
    return 1 - sum(p**2)

def entropy(p):
    return -sum(p * np.log2(p))

# Пример для узла с распределением [0.9, 0.1]
print(f"Gini: {gini(np.array([0.9, 0.1])):.3f}")  # 0.18
print(f"Entropy: {entropy(np.array([0.9, 0.1])):.3f}")  # 0.469
```

### 1.3 Построение дерева в sklearn
```python
# Загрузка данных
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Обучение с критерием Джини
clf = DecisionTreeClassifier(criterion='gini', max_depth=2)
clf.fit(X_train, y_train)

# Визуализация
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
```

---

## 🟡 Продвинутый уровень (Реализация)

### 2.1 Расчет прироста информации
**Информационный выигрыш:**  
$IG = I_{parent} - \sum\frac{N_{child}}{N_{parent}}I_{child}$  
где $I$ - энтропия или Gini

```python
def information_gain(parent, children, criterion='gini'):
    if criterion == 'gini':
        parent_impurity = gini(parent)
        child_impurity = sum(gini(child) * len(child) for child in children) / sum(len(c) for c in children)
    else:
        parent_impurity = entropy(parent)
        child_impurity = sum(entropy(child) * len(child) for child in children) / sum(len(c) for c in children)
    
    return parent_impurity - child_impurity

# Пример использования
parent = np.array([0.5, 0.5])
children = [np.array([0.9, 0.1]), np.array([0.2, 0.8])]
print(f"Information Gain: {information_gain(parent, children, 'entropy'):.3f}")
```

### 2.2 Поиск лучшего разделения
```python
def find_best_split(X, y):
    best_ig = -1
    best_feature = None
    best_threshold = None
    
    for feature_idx in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            
            left_probs = np.bincount(y[left_mask]) / len(y[left_mask])
            right_probs = np.bincount(y[right_mask]) / len(y[right_mask])
            
            ig = information_gain(
                np.bincount(y) / len(y),
                [left_probs, right_probs],
                'gini'
            )
            
            if ig > best_ig:
                best_ig = ig
                best_feature = feature_idx
                best_threshold = threshold
                
    return best_feature, best_threshold, best_ig
```

### 2.3 Визуализация границ решений
```python
# Для 2D случая
X = iris.data[:, [0, 2]]  # sepal length и petal length
y = iris.target

clf_2d = DecisionTreeClassifier(max_depth=3)
clf_2d.fit(X, y)

# Построение границ
x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
plt.title("Границы решений дерева")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[2])
plt.show()
```

---

## 🔴 Экспертный уровень (Продвинутые техники)

### 3.1 Кастомные критерии
```python
from sklearn.tree import BaseDecisionTree

class CustomCriterion:
    """Кастомный критерий на основе дисперсии"""
    def __call__(self, y, sample_weight):
        # Расчет "нечистоты" как дисперсии
        mean = np.average(y, weights=sample_weight)
        return np.average((y - mean)**2, weights=sample_weight)
    
    def proxy_impurity_improvement(self, impurity, impurity_children):
        return impurity - np.sum(impurity_children)
    
# Использование в дереве
tree = BaseDecisionTree(criterion=CustomCriterion())
```

### 3.2 Анализ важности признаков
```python
# Важность на основе снижения нечистоты
importances = clf.feature_importances_

# Перестановочная важность
from sklearn.inspection import permutation_importance

result = permutation_importance(clf, X_test, y_test, n_repeats=10)
perm_importances = result.importances_mean

# Визуализация
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].barh(iris.feature_names, importances)
ax[0].set_title('Важность по Gini')
ax[1].barh(iris.feature_names, perm_importances)
ax[1].set_title('Перестановочная важность')
plt.show()
```

### 3.3 Оптимизация гиперпараметров
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Лучшие параметры: {grid_search.best_params_}")
print(f"Точность: {grid_search.best_score_:.3f}")
```

### 3.4 Построение деревьев с GPU
```python
# Использование cuML для GPU-ускорения
# !pip install cuml

from cuml import DecisionTreeClassifier as cuDecisionTreeClassifier

gpu_tree = cuDecisionTreeClassifier(max_depth=5)
gpu_tree.fit(X_train, y_train)
print(f"GPU Accuracy: {gpu_tree.score(X_test, y_test):.3f}")
```

---

## 📊 Чеклист по уровням

| Уровень | Навыки |
|---------|--------|
| 🟢 | Понимание Gini/энтропии, построение деревьев в sklearn |
| 🟡 | Ручная реализация IG, визуализация границ |
| 🔴 | Кастомные критерии, анализ важности, GPU-ускорение |

---

## ⚠️ Антипаттерны
1. **Переобучение** без ограничения глубины
2. **Игнорирование несбалансированных классов**
3. **Использование деревьев без пост-обрейки**
4. **Интерпретация важности как причинности**

---

## 🚀 Продвинутые советы
1. **Контроль переобучения:**
```python
clf = DecisionTreeClassifier(
    ccp_alpha=0.02,  # Параметр обрезки
    min_samples_leaf=5,  # Минимум объектов в листе
    max_leaf_nodes=20  # Максимум листьев
)
```

2. **Визуализация через Graphviz:**
```python
from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(clf, out_file=None, 
                          feature_names=iris.feature_names,
                          class_names=iris.target_names,
                          filled=True)
graphviz.Source(dot_data).render("iris_tree")
```

3. **Ансамблирование деревьев:**
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
```

---

## 📌 Тренировочные задания

### 🟢 Базовый уровень
1. Постройте дерево решений для датасета Wine (sklearn.datasets.load_wine) с max_depth=2.
2. Рассчитайте вручную индекс Джини для узла с распределением классов [20, 10, 5].

### 🟡 Продвинутый уровень
1. Реализуйте функцию расчета энтропии для многоклассовой задачи.
2. Визуализируйте границы решений для датасета Moons (make_moons).

### 🔴 Экспертный уровень
1. Создайте кастомный критерий на основе коэффициента вариации.
2. Сравните скорость обучения на CPU и GPU для датасета 1M строк.

---

```python
# Пример решения 🟢 Задания 2
def manual_gini(class_counts):
    total = sum(class_counts)
    proportions = np.array(class_counts) / total
    return 1 - sum(proportions**2)

print(f"Gini: {manual_gini([20, 10, 5]):.3f}")  
# Расчет: proportions = [20/35, 10/35, 5/35] ≈ [0.57, 0.29, 0.14]
# Gini = 1 - (0.57² + 0.29² + 0.14²) = 1 - (0.325 + 0.084 + 0.020) = 0.571
```

---

## 📌 Заключение
Ключевые принципы:
1. **Gini быстрее энтропии** (без логарифмов)
2. **Деревья склонны к переобучению** → всегда используйте ограничения
3. **Важность признаков** помогает в feature selection
4. **Интерпретируемость** - главное преимущество перед сложными моделями

Деревья решений - фундамент для:
- Случайных лесов
- Градиентного бустинга
- Ансамблевых методов

Помните: простота ≠ примитивность! Грамотно настроенные деревья могут бить нейросети на табличных данных.
