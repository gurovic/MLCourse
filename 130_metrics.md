# Метрики качества – MAE, Accuracy, F1 и др.

## 🟢 Базовый уровень (Основные метрики)

### 1.1 MAE (Mean Absolute Error)
**Для регрессии:** Средняя абсолютная ошибка
```math
\text{MAE} = \frac{1}{N}\sum_{i=1}^{N} |\hat{y}_i - y_i| 
```
```python
y_true = np.array([3.0, 5.0, 2.5, 7.0])
y_pred = np.array([2.5, 5.0, 3.0, 8.0])
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.2f}")  # (0.5+0+0.5+1)/4 = 0.5
```

> ⚠️ **Примечание:** MAE не чувствителен к большим ошибкам (выбросам), поэтому лучше подходит, если выбросов много.

---

### 1.2 Accuracy (Точность)
**Для классификации:** Доля правильных ответов  
```math
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
```

```python
y_true = [0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1]
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.2f}")  # 4/5 = 0.8
```

> ⚠️ **Важно:** Accuracy может быть обманчив при дисбалансе классов.

#### Пример:
```python
y_true = [0, 0, 0, 0, 1]
y_pred = [0, 0, 0, 0, 0]
print(accuracy_score(y_true, y_pred))  # 0.8 → но модель не видит класс 1!
```

---

### 1.3 F1-Score
**Гармоническое среднее precision и recall:**  
```math
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
```

```python
f1 = f1_score(y_true, y_pred)
print(f"F1-Score: {f1:.2f}")  # Precision=0.67, Recall=0.67 → F1=0.67
```

> ⚠️ Почему гармоническое? Чтобы штрафовать случаи, где один из показателей очень низкий.

---

## 🟡 Продвинутый уровень (Выбор метрики)

### 2.1 Когда использовать метрики?

| Задача                     | Рекомендуемые метрики                             |
|---------------------------|---------------------------------------------------|
| Бинарная классификация    | F1, AUC-ROC, Precision, Recall                    |
| Мультикласс классификация | F1-macro, Accuracy                                |
| Регрессия                 | MAE, MSE, R²                                      |
| Ранжирование             | NDCG, MAP                                         |

---

### 2.2 Матрица ошибок

```python
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказание')
plt.ylabel('Истина')
plt.title('Матрица ошибок')
plt.show()
```

#### Расчёт отдельных компонент:
```python
tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
```

#### Производные метрики:
- **Sensitivity (Recall)**: ```math \frac{TP}{TP + FN} ```
- **Specificity**: ```math \frac{TN}{TN + FP} ```
- **FPR (False Positive Rate)**: ```math \frac{FP}{FP + TN} ```

---

### 2.3 Precision-Recall Tradeoff

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

# Визуализация
plt.plot(recalls, precisions)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Кривая Precision-Recall')
plt.show()
```

> ✅ Используй `precision_recall_curve()` вместо ручного перебора порогов.

---

## 🔴 Экспертный уровень (Продвинутые техники)

### 3.1 Кастомные метрики

#### Weighted Accuracy:

```python
def weighted_accuracy(y_true, y_pred, weights=[0.5, 2.0]):
    correct = np.zeros(len(y_true))
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true == pred:
            correct[i] = weights[true]
    return np.mean(correct)

print(f"Weighted Accuracy: {weighted_accuracy(y_true, y_pred):.2f}")
```

#### Cohen's Kappa:

```python
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(y_true, y_pred)
print(f"Cohen's Kappa: {kappa:.2f}")
```

#### Matthews Correlation Coefficient (MCC):

```python
from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(y_true, y_pred)
print(f"MCC: {mcc:.2f}")
```

---

### 3.2 Оптимизация бизнес-метрик

```python
def profit_metric(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tp*10 - fp*1 - fn*5  # Прибыль

print(f"Business Profit: ${profit_metric(y_true, y_pred)}")
```

> 💡 Это позволяет оптимизировать модели под реальные бизнес-цели.

---

### 3.3 LogLoss (Logarithmic Loss)

```python
from sklearn.metrics import log_loss

y_true_proba = [0, 1, 0, 1]
y_pred_proba = [0.1, 0.9, 0.2, 0.8]
logloss = log_loss(y_true_proba, y_pred_proba)
print(f"LogLoss: {logloss:.4f}")  # Чем меньше, тем лучше
```

> ⚠️ LogLoss штрафует уверенность в неверных предсказаниях. Например, если модель уверена на 99%, что это класс 1, а на самом деле это 0 — большой штраф.

---

### 3.4 Кривые валидации

```python
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier

train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(),
    X, y,
    cv=5,
    scoring='f1',
    train_sizes=np.linspace(0.1, 1.0, 5)
)

plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Train')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation')
plt.legend()
plt.title('Learning Curve F1-Score')
plt.xlabel('Training Size')
plt.ylabel('F1')
plt.show()
```

> ✅ Позволяет понять, переобучается ли модель или недообучается.

---

## 📊 Чеклист по уровням

| Уровень | Навыки |
|---------|--------|
| 🟢 | Расчет MAE, Accuracy, F1 для простых случаев |
| 🟡 | Анализ матрицы ошибок, кривые PR, выбор метрик |
| 🔴 | Кастомные метрики, оптимизация бизнес-показателей, интерпретация доверительных интервалов |

---

## ⚠️ Антипаттерны

1. **Использование Accuracy при дисбалансе**  
   *Пример:* 99% класса 0 → Accuracy 99%, но модель бесполезна.
2. **Оптимизация F1 без учета business context**  
   *Разная цена FP/FN.*
3. **Сравнение моделей по разным метрикам**  
   *Нельзя сравнивать F1 и AUC-ROC напрямую.*
4. **Игнорирование доверительных интервалов**  
   *Не все метрики стабильны.*

---

## 🚀 Продвинутые советы

### Выбор порога классификации:
```python
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Оптимальный порог: {best_threshold:.2f}")
```

### Бутстрэп оценок:
```python
def bootstrap_metric(y_true, y_pred, metric, n_bootstrap=1000):
    scores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        scores.append(metric(y_true[indices], y_pred[indices]))
    return np.percentile(scores, [2.5, 97.5])

print(f"95% CI для F1: {bootstrap_metric(y_true, y_pred, f1_score)}")
```

### Калибровка вероятностей:
```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

model = RandomForestClassifier()
calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
calibrated.fit(X_train, y_train)
probs = calibrated.predict_proba(X_test)[:, 1]

# График калибровки
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probs, n_bins=10)
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibrated")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.legend()
plt.show()
```

---

## 📌 Тренировочные задания

### 🟢 Базовый уровень
1. Рассчитайте MAE для `y_true=[10, 20, 30]`, `y_pred=[12, 18, 28]`.
2. Вычислите Accuracy для `y_true=[1,0,1,0]`, `y_pred=[1,1,1,0]`.

### 🟡 Продвинутый уровень
1. Для `y_true=[1,0,1,1,0]`, `y_pred=[1,0,0,1,0]`:
   - Постройте матрицу ошибок
   - Рассчитайте Precision, Recall, F1
2. Объясните, почему F1 лучше Accuracy при дисбалансе.

### 🔴 Экспертный уровень
1. Реализуйте метрику Cohen's Kappa.
2. Создайте кастомную метрику, где ошибка класса 1 в 5 раз важнее класса 0.

---

## 📌 Заключение

### Ключевые принципы:
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

> 🎯 Помните: нет "лучшей" метрики — есть наиболее подходящая для вашей конкретной задачи!

