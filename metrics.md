```python
# Глава 22: Метрики качества – MAE, Accuracy, F1
# Уровни сложности: 🟢 Базовый | 🟡 Продвинутый | 🔴 Экспертный

import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# !pip install scikit-learn
```

---

## 🟢 Базовый уровень (Основные метрики)

### 1.1 MAE (Mean Absolute Error)
**Для регрессии:** Средняя абсолютная ошибка  
$MAE = \frac{1}{N}\sum|y_i - \hat{y}_i|$

```python
y_true = np.array([3.0, 5.0, 2.5, 7.0])
y_pred = np.array([2.5, 5.0, 3.0, 8.0])
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.2f}")  # (0.5+0+0.5+1)/4 = 0.5
```

### 1.2 Accuracy (Точность)
**Для классификации:** Доля правильных ответов  
$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$

```python
y_true = [0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1]
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.2f}")  # 4/5 = 0.8
```

### 1.3 F1-Score
**Гармоническое среднее precision и recall:**  
$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$

```python
f1 = f1_score(y_true, y_pred)
print(f"F1-Score: {f1:.2f}")  # Precision=0.67, Recall=0.67 → F1=0.67
```

---

## 🟡 Продвинутый уровень (Выбор метрики)

### 2.1 Когда использовать метрики?
| **Задача**               | **Рекомендуемые метрики**       |
|--------------------------|---------------------------------|
| Бинарная классификация   | F1, AUC-ROC, Precision, Recall  |
| Мультикласс классификация| F1-macro, Accuracy              |
| Регрессия                | MAE, MSE, R²                    |
| Ранжирование             | NDCG, MAP                       |

### 2.2 Матрица ошибок
```python
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказание')
plt.ylabel('Истина')
plt.title('Матрица ошибок')
plt.show()
```

### 2.3 Precision-Recall Tradeoff
```python
# Расчет для разных порогов
thresholds = np.linspace(0, 1, 50)
precisions = []
recalls = []

y_probs = [0.1, 0.9, 0.4, 0.6, 0.7]  # Вероятности класса 1

for thresh in thresholds:
    y_pred_thresh = [1 if prob > thresh else 0 for prob in y_probs]
    precisions.append(precision_score(y_true, y_pred_thresh))
    recalls.append(recall_score(y_true, y_pred_thresh))

# Кривая Precision-Recall
plt.plot(recalls, precisions)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Кривая Precision-Recall')
plt.show()
```

---

## 🔴 Экспертный уровень (Продвинутые техники)

### 3.1 Кастомные метрики
```python
def weighted_accuracy(y_true, y_pred, weights=[0.5, 2.0]):
    """Вес ошибок для разных классов"""
    correct = np.zeros(len(y_true))
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true == pred:
            correct[i] = weights[true]  # Вес за правильный ответ класса
    return np.mean(correct)

print(f"Weighted Accuracy: {weighted_accuracy(y_true, y_pred):.2f}")
```

### 3.2 Оптимизация бизнес-метрик
```python
def profit_metric(y_true, y_pred):
    """Пример: FP стоит 1$, FN стоит 5$"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tp*10 - fp*1 - fn*5  # Прибыль

print(f"Business Profit: ${profit_metric(y_true, y_pred)}")
```

### 3.3 Кросс-энтропия и логарифмические потери
```python
from sklearn.metrics import log_loss

y_true_proba = [0, 1, 0, 1]  # Для бинарной
y_pred_proba = [0.1, 0.9, 0.2, 0.8]
logloss = log_loss(y_true_proba, y_pred_proba)
print(f"LogLoss: {logloss:.4f}")  # Чем меньше, тем лучше
```

### 3.4 Кривые валидации
```python
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier

param_range = [10, 50, 100, 200]  # n_estimators
train_scores, test_scores = validation_curve(
    RandomForestClassifier(),
    X, y,
    param_name="n_estimators",
    param_range=param_range,
    cv=5,
    scoring='f1'
)

plt.plot(param_range, np.mean(train_scores, axis=1), label='Train')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Validation')
plt.legend()
plt.title('Валидационная кривая F1-Score')
plt.xlabel('n_estimators')
plt.ylabel('F1')
plt.show()
```

---

## 📊 Чеклист по уровням

| Уровень | Навыки |
|---------|--------|
| 🟢 | Расчет MAE, Accuracy, F1 для простых случаев |
| 🟡 | Анализ матрицы ошибок, кривые PR, выбор метрик |
| 🔴 | Кастомные метрики, оптимизация бизнес-показателей |

---

## ⚠️ Антипаттерны
1. **Использование Accuracy при дисбалансе** (99% класса 0 → Accuracy 99%)
2. **Оптимизация F1 без учета business context** (разная цена FP/FN)
3. **Сравнение моделей по разным метрикам** (F1 vs AUC-ROC)
4. **Игнорирование доверительных интервалов** для метрик

---

## 🚀 Продвинутые советы
1. **Выбор порога классификации:**
```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Оптимальный порог: {best_threshold:.2f}")
```

2. **Бутстрэп оценок:**
```python
def bootstrap_metric(y_true, y_pred, metric, n_bootstrap=1000):
    scores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        scores.append(metric(y_true[indices], y_pred[indices]))
    return np.percentile(scores, [2.5, 97.5])

print(f"95% CI для F1: {bootstrap_metric(y_true, y_pred, f1_score)}")
```

3. **Калибровка вероятностей:**
```python
from sklearn.calibration import CalibratedClassifierCV

model = RandomForestClassifier()
calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
calibrated.fit(X_train, y_train)
calibrated_probs = calibrated.predict_proba(X_test)[:, 1]
```

---

## 📌 Тренировочные задания

### 🟢 Базовый уровень
1. Рассчитайте MAE для y_true=[10, 20, 30], y_pred=[12, 18, 28].
2. Вычислите Accuracy для y_true=[1,0,1,0], y_pred=[1,1,1,0].

### 🟡 Продвинутый уровень
1. Для бинарной классификации с y_true=[1,0,1,1,0], y_pred=[1,0,0,1,0]:
   - Постройте матрицу ошибок
   - Рассчитайте Precision, Recall, F1
2. Объясните, почему F1 лучше Accuracy при дисбалансе.

### 🔴 Экспертный уровень
1. Реализуйте метрику Cohen's Kappa.
2. Создайте кастомную метрику, где ошибка класса 1 в 5 раз важнее класса 0.

---

```python
# Пример решения 🟢 Задания 1
y_true = np.array([10, 20, 30])
y_pred = np.array([12, 18, 28])
mae = np.mean(np.abs(y_true - y_pred))
print(f"MAE: {mae:.1f}")  # (2+2+2)/3 = 2.0
```

---

## 📌 Заключение
Ключевые принципы:
1. **Выбор метрики зависит от задачи**:
   - Бизнес-требования > статистические метрики
   - Дисбаланс → F1, AUC-ROC
   - Регрессия → MAE (интерпретируемость), MSE (штраф за большие ошибки)
2. **Всегда анализируйте ошибки**:
   - Матрица ошибок > Accuracy
   - Кривые PR/ROC > F1
3. **Статистическая надежность**:
   - Доверительные интервалы
   - Бутстрэп оценки
4. **Калибровка вероятностей** для задач с порогами

Помните: нет "лучшей" метрики — есть наиболее подходящая для вашей конкретной задачи!
