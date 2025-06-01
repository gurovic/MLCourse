
# Логистическая регрессия

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# !pip install scikit-learn
```

---

## 🟢 Базовый уровень (Основы)

### 1.1 Сигмоида и вероятность
**Формула сигмоиды:**  
$\sigma(z) = \frac{1}{1 + e^{-z}}$  
где $z = w_0 + w_1x_1 + ... + w_nx_n$

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Пример
z = np.linspace(-7, 7, 100)
plt.plot(z, sigmoid(z))
plt.title('Сигмоида')
plt.xlabel('z')
plt.ylabel('Вероятность')
plt.grid(True)
plt.show()
```

### 1.2 Бинарная классификация
```python
# Пример данных
X = np.array([[1.5], [2.0], [3.0], [4.0], [5.0]])
y = np.array([0, 0, 1, 1, 1])

# Обучение модели
model = LogisticRegression()
model.fit(X, y)

# Предсказание вероятностей
probabilities = model.predict_proba(X)[:, 1]
print(f"Вероятности: {probabilities}")
```

### 1.3 Порог классификации
```python
# По умолчанию порог = 0.5
predictions = model.predict(X)
print(f"Предсказания: {predictions}")

# Изменение порога
custom_predictions = (probabilities > 0.3).astype(int)
print(f"Кастомные предсказания: {custom_predictions}")
```

---

## 🟡 Продвинутый уровень (Реализация)

### 2.1 Функция потерь (Log Loss)
**Формула:**  
$LogLoss = -\frac{1}{N}\sum_{i=1}^{N}[y_i\log(p_i) + (1-y_i)\log(1-p_i)]$

```python
def log_loss(y_true, y_pred_proba):
    epsilon = 1e-15  # Для численной стабильности
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1-epsilon)
    return -np.mean(y_true * np.log(y_pred_proba) + (1-y_true) * np.log(1-y_pred_proba))
```

### 2.2 Градиентный спуск
```python
def logistic_gd(X, y, lr=0.01, epochs=1000):
    w = np.zeros(X.shape[1])
    b = 0
    n = len(X)
    losses = []
    
    for epoch in range(epochs):
        z = np.dot(X, w) + b
        p = sigmoid(z)
        
        # Градиенты
        dw = (1/n) * np.dot(X.T, (p - y))
        db = (1/n) * np.sum(p - y)
        
        # Обновление параметров
        w -= lr * dw
        b -= lr * db
        
        # Лосс
        loss = log_loss(y, p)
        losses.append(loss)
        
    return w, b, losses

# Применение
X_with_bias = np.c_[np.ones(len(X)), X]  # Добавляем столбец для bias
w, b, losses = logistic_gd(X_with_bias, y)
```

### 2.3 Визуализация обучения
```python
plt.plot(losses)
plt.title('Сходимость Log Loss')
plt.xlabel('Итерация')
plt.ylabel('Log Loss')
plt.show()
```

---

## 🔴 Экспертный уровень (Продвинутые техники)

### 3.1 Регуляризация
```python
# L2-регуляризация
def logistic_gd_reg(X, y, lr=0.01, epochs=1000, lambda_=0.1):
    w = np.zeros(X.shape[1])
    b = 0
    n = len(X)
    
    for epoch in range(epochs):
        z = np.dot(X, w) + b
        p = sigmoid(z)
        
        dw = (1/n) * (np.dot(X.T, (p - y)) + (lambda_/n) * w
        db = (1/n) * np.sum(p - y)
        
        w -= lr * dw
        b -= lr * db
        
    return w, b

# Сравнение с sklearn
model = LogisticRegression(penalty='l2', C=1/lambda_)
```

### 3.2 Подбор порога по ROC-кривой
```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_true, y_probs)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
```

### 3.3 Мультиклассовая классификация
```python
# Стратегии: one-vs-rest (OvR) или multinomial
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Пример для 3 классов
X_multi = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
y_multi = np.array([0, 0, 1, 1, 2, 2])

model.fit(X_multi, y_multi)
print("Предсказания:", model.predict([[2.5]]))
```

---

## 📊 Чеклист по уровням

| Уровень | Навыки |
|---------|--------|
| 🟢 | Сигмоида, бинарная классификация, predict_proba |
| 🟡 | Log Loss, градиентный спуск, визуализация сходимости |
| 🔴 | Регуляризация, подбор порога, мультикласс |

---

## ⚠️ Антипаттерны
1. **Игнорирование дисбаланса классов** (точность 99% при 99% одного класса)
2. **Использование линейной регрессии** для классификации
3. **Отсутствие масштабирования** при градиентном спуске
4. **Выбор порога 0.5 без анализа** ROC-кривой

---

## 🚀 Продвинутые советы
1. **Калибровка вероятностей:**
```python
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(model, cv=3, method='sigmoid')
calibrated.fit(X_train, y_train)
```

2. **Анализ коэффициентов:**
```python
coef = model.coef_[0]
features = X.columns
plt.barh(features, coef)
plt.title('Важность признаков')
```

3. **Интерпретация через odds ratio:**
```python
odds_ratio = np.exp(coef)
print(f"Шансы увеличиваются в {odds_ratio[0]:.2f} раз при изменении признака на 1 единицу")
```

---

## 📌 Тренировочные задания

### 🟢 Базовый уровень
1. Обучите логистическую регрессию на данных: X=[[1], [2], [3], [4]], y=[0, 0, 1, 1].
2. Предскажите вероятность для X=2.5.

### 🟡 Продвинутый уровень
1. Реализуйте градиентный спуск для тех же данных.
2. Сравните Log Loss с sklearn.

### 🔴 Экспертный уровень
1. Добавьте L2-регуляризацию в свою реализацию.
2. Найдите оптимальный порог классификации по ROC-AUC.

---

```python
# Пример решения 🟢 Задания 1
X = np.array([[1], [2], [3], [4]])
y = np.array([0, 0, 1, 1])
model = LogisticRegression()
model.fit(X, y)
print(f"Вероятность для 2.5: {model.predict_proba([[2.5]])[0][1]:.2f}")
```

---

## 📌 Заключение
Ключевые принципы:
1. **Сигмоида преобразует линейную комбинацию в вероятность**
2. **Log Loss лучше MSE для классификации**
3. **Регуляризация критически важна** для предотвращения переобучения
4. **Порог классификации можно оптимизировать** под задачу

Логистическая регрессия — мощный инструмент, несмотря на название "регрессия". Она позволяет:
- Интерпретировать коэффициенты через odds ratio
- Работать с вероятностями, а не "черно-белыми" предсказаниями
- Легко масштабироваться на многоклассовые задачи
