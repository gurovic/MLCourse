
# Градиентный бустинг – XGBoost, LightGBM, CatBoost

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import time

# !pip install xgboost lightgbm catboost scikit-learn
```

---

## 🟢 Базовый уровень (Основы бустинга)

### 1.1 Что такое градиентный бустинг?
**Ансамблевый метод:** Последовательное построение слабых моделей (обычно деревьев), где каждая новая модель исправляет ошибки предыдущих.  
**Ключевые особенности:**
- Оптимизация дифференцируемой функции потерь
- Адаптивное взвешивание ошибочных наблюдений
- Высокая точность на табличных данных

### 1.2 Три фреймворка бустинга
```python
# Данные для примера
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
xgb_model.fit(X_train, y_train)

# LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=-1)
lgb_model.fit(X_train, y_train)

# CatBoost
cb_model = cb.CatBoostClassifier(iterations=100, learning_rate=0.05, depth=6, verbose=0)
cb_model.fit(X_train, y_train)

# Сравнение точности
print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_model.predict(X_test)):.4f}")
print(f"LightGBM Accuracy: {accuracy_score(y_test, lgb_model.predict(X_test)):.4f}")
print(f"CatBoost Accuracy: {accuracy_score(y_test, cb_model.predict(X_test)):.4f}")
```

---

## 🟡 Продвинутый уровень (Особенности фреймворков)

### 2.1 Ключевые различия
| **Характеристика**       | **XGBoost**         | **LightGBM**       | **CatBoost**         |
|--------------------------|---------------------|--------------------|----------------------|
| Стратегия роста          | Level-wise          | Leaf-wise          | Обычный / Ordered   |
| Скорость обучения        | Средняя             | Очень высокая      | Средняя             |
| Память                   | Высокая             | Низкая             | Средняя             |
| Категориальные признаки  | Требуют кодирования | Встроенная поддержка | Нативная обработка |
| GPU-поддержка            | Полная              | Полная             | Полная              |
| Защита от утечек         | -                   | -                  | Ordered boosting    |

### 2.2 Обработка категориальных признаков
```python
# Пример данных с категориями
data = pd.DataFrame({
    'cat_feature': np.random.choice(['A', 'B', 'C'], size=1000),
    'num_feature': np.random.rand(1000),
    'target': np.random.randint(0, 2, 1000)
})

# LightGBM
lgb_dataset = lgb.Dataset(data[['cat_feature', 'num_feature']], data['target'], 
                         categorical_feature=['cat_feature'])

# CatBoost (автоматическая обработка)
cb_model = cb.CatBoostClassifier(iterations=100, cat_features=['cat_feature'], verbose=0)
cb_model.fit(data[['cat_feature', 'num_feature']], data['target'])

# XGBoost (требует One-Hot)
ohe = OneHotEncoder()
X_ohe = ohe.fit_transform(data[['cat_feature']])
xgb_model.fit(pd.concat([pd.DataFrame(X_ohe), data['num_feature']], axis=1), data['target'])
```

### 2.3 Ранняя остановка и валидация
```python
# Общий подход для всех
eval_set = [(X_test, y_test)]

# XGBoost
xgb_model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=10, verbose=False)

# LightGBM
lgb_model.fit(X_train, y_train, eval_set=eval_set, callbacks=[lgb.early_stopping(10)])

# CatBoost
cb_model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=10, verbose=False)
```

### 2.4 Важность признаков
```python
# Сравнение важности
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

xgb.plot_importance(xgb_model, ax=ax[0])
ax[0].set_title('XGBoost')

lgb.plot_importance(lgb_model, ax=ax[1])
ax[1].set_title('LightGBM')

cb.plot_importance(cb_model, ax=ax[2])
ax[2].set_title('CatBoost')

plt.tight_layout()
plt.show()
```

---

## 🔴 Экспертный уровень (Продвинутые техники)

### 3.1 Кастомные функции потерь
```python
# XGBoost
def xgb_custom_loss(y_pred, dtrain):
    y_true = dtrain.get_label()
    grad = np.where(y_true > y_pred, 2*(y_true - y_pred), -1)
    hess = np.where(y_true > y_pred, 2, 1)
    return grad, hess

# LightGBM
def lgb_custom_loss(y_true, y_pred):
    grad = np.where(y_true > y_pred, 2*(y_true - y_pred), -1)
    hess = np.where(y_true > y_pred, 2, 1)
    return grad, hess

# CatBoost
class CatBoostCustomLoss(object):
    def calc_ders_range(self, approxes, targets, weights):
        ders = []
        for i in range(len(targets)):
            der1 = 2*(targets[i] - approxes[i]) if targets[i] > approxes[i] else -1
            der2 = 2 if targets[i] > approxes[i] else 1
            ders.append((der1, der2))
        return ders
```

### 3.2 Оптимизация гиперпараметров
```python
import optuna

def optimize_hyperparams(trial, framework='catboost'):
    if framework == 'catboost':
        params = {
            'learning_rate': trial.suggest_float('lr', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2', 1, 10),
            'bagging_temperature': trial.suggest_float('bagging', 0, 1)
        }
        model = cb.CatBoostClassifier(**params, verbose=0)
    elif framework == 'lightgbm':
        params = {...}
        model = lgb.LGBMClassifier(**params)
    else:
        params = {...}
        model = xgb.XGBClassifier(**params)
    
    return cross_val_score(model, X_train, y_train, cv=3).mean()

# Оптимизация для CatBoost
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: optimize_hyperparams(trial, 'catboost'), n_trials=50)
```

### 3.3 GPU-ускорение
```python
# XGBoost
xgb_gpu = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0)

# LightGBM
lgb_gpu = lgb.LGBMClassifier(device='gpu')

# CatBoost
cb_gpu = cb.CatBoostClassifier(task_type='GPU', devices='0:1')

# Бенчмарк скорости
for name, model in [('XGBoost', xgb_gpu), ('LightGBM', lgb_gpu), ('CatBoost', cb_gpu)]:
    start = time.time()
    model.fit(X_train, y_train)
    print(f"{name} GPU Time: {time.time()-start:.2f}s")
```

### 3.4 Интерпретация моделей (SHAP)
```python
import shap

# Общий подход для всех
explainer_map = {
    'XGBoost': shap.TreeExplainer(xgb_model),
    'LightGBM': shap.TreeExplainer(lgb_model),
    'CatBoost': shap.TreeExplainer(cb_model)
}

for name, explainer in explainer_map.items():
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, title=f"{name} SHAP Summary")
```

---

## 📊 Чеклист по уровням

| Уровень | Навыки |
|---------|--------|
| 🟢 | Базовое применение всех трех фреймворков |
| 🟡 | Обработка категорий, ранняя остановка, анализ важности |
| 🔴 | Кастомные функции потерь, оптимизация, GPU, SHAP |

---

## ⚠️ Антипаттерны
1. **Использование без ранней остановки** → переобучение
2. **Игнорирование обработки категорий** в XGBoost
3. **Неучет особенностей фреймворков** при выборе
4. **Бездумное увеличение n_estimators** без контроля сложности

---

## 🚀 Продвинутые советы
1. **Стартовые параметры:**
```python
# XGBoost
xgb_params = {'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8}

# LightGBM
lgb_params = {'learning_rate': 0.1, 'num_leaves': 31, 'feature_fraction': 0.8}

# CatBoost
cb_params = {'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 3}
```

2. **Выбор фреймворка:**
- **LightGBM**: Большие данные (>1M строк), скорость критична
- **CatBoost**: Данные с категориальными признаками, нужна защита от утечек
- **XGBoost**: Точность важнее скорости, малые/средние данные

3. **Ансамблирование:**
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier([
    ('xgb', xgb.XGBClassifier()),
    ('lgb', lgb.LGBMClassifier()),
    ('cb', cb.CatBoostClassifier(verbose=0))
], voting='soft')
```

---

## 📌 Тренировочные задания

### 🟢 Базовый уровень
1. Обучите все три модели на Iris, сравните accuracy.
2. Примените CatBoost к данным с категориальными признаками без кодирования.

### 🟡 Продвинутый уровень
1. Сравните скорость обучения всех трех фреймворков на датасете 100k строк.
2. Настройте обработку категорий для каждого фреймворка.

### 🔴 Экспертный уровень
1. Реализуйте кастомную функцию потерь для всех трех фреймворков.
2. Проведите оптимизацию гиперпараметров с Optuna для CatBoost.

---

```python
# Пример решения 🟢 Задания 1
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# XGBoost
xgb_model = xgb.XGBClassifier().fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))

# LightGBM
lgb_model = lgb.LGBMClassifier().fit(X_train, y_train)
lgb_acc = accuracy_score(y_test, lgb_model.predict(X_test))

# CatBoost
cb_model = cb.CatBoostClassifier(verbose=0).fit(X_train, y_train)
cb_acc = accuracy_score(y_test, cb_model.predict(X_test))

print(f"XGB: {xgb_acc:.4f}, LGBM: {lgb_acc:.4f}, CatBoost: {cb_acc:.4f}")
```

---

## 📌 Заключение
Ключевые принципы:
1. **LightGBM - самый быстрый**: Лучший выбор для больших данных
2. **CatBoost - лучший для категорий**: Автоматическая обработка, защита от утечек
3. **XGBoost - эталон точности**: Часто дает лучшие результаты на малых данных

Когда выбирать:
- **XGBoost**: Точность важнее скорости, малые данные
- **LightGBM**: Большие данные (>1M строк), ограниченные ресурсы
- **CatBoost**: Категориальные признаки, нужна защита от утечек, автоматизация

Рекомендации:
- Всегда используйте **раннюю остановку**
- Экспериментируйте с **GPU-ускорением**
- Применяйте **SHAP** для интерпретации
- Для максимальной точности - **ансамблируйте** все три подхода

Градиентный бустинг - мощный инструмент, и владение всеми тремя фреймворками делает вас универсальным специалистом по табличным данным!
