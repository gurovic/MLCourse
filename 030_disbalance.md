```python
# Дисбаланс классов

# !pip install pandas numpy scikit-learn imbalanced-learn tensorflow
```

---

## 🟢 Базовый уровень (Основные подходы)

### 1.1 Понимание дисбаланса
```python
from sklearn.datasets import make_classification

# Генерация несбалансированных данных
X, y = make_classification(
    n_samples=1000, 
    weights=[0.95, 0.05],  # 95% negative class
    random_state=42
)

# Анализ
print("Распределение классов:", {0: (y == 0).sum(), 1: (y == 1).sum()})
```

### 1.2 Случайное передискретизирование
```python
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Пересэмплирование
oversampler = RandomOverSampler(sampling_strategy=0.5, random_state=42)
X_over, y_over = oversampler.fit_resample(X, y)

# Недоcэмплирование
undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_under, y_under = undersampler.fit_resample(X, y)
```

### 1.3 Взвешивание классов
```python
from sklearn.linear_model import LogisticRegression

# Автоматическое взвешивание
model = LogisticRegression(class_weight='balanced')

# Ручное задание весов
weights = {0: 1, 1: 10}  # Увеличиваем вес миноритарного класса
model = LogisticRegression(class_weight=weights)
```

---

## 🟡 Продвинутый уровень (SMOTE и ансамбли)

### 2.1 SMOTE (Synthetic Minority Oversampling)
```python
from imblearn.over_sampling import SMOTE

# Создание синтетических примеров
smote = SMOTE(
    sampling_strategy=0.3,
    k_neighbors=5,
    random_state=42
)
X_smote, y_smote = smote.fit_resample(X, y)
```

### 2.2 Ансамблевые методы
```python
from imblearn.ensemble import BalancedRandomForestClassifier

# Модель с балансировкой
brf = BalancedRandomForestClassifier(
    n_estimators=100,
    sampling_strategy=0.5,
    replacement=True,
    random_state=42
)
brf.fit(X, y)
```

### 2.3 Метрики для оценки
```python
from sklearn.metrics import classification_report

# Использование F1 вместо accuracy
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
```

---

## 🔴 Экспертный уровень (Продвинутые техники)

### 3.1 Генеративные модели (GAN)
```python
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import numpy as np

# Генератор для создания синтетических данных
def build_generator():
    input = Input(shape=(10,))
    x = Dense(128, activation='relu')(input)
    output = Dense(X.shape[1], activation='tanh')(x)
    return Model(input, output)

# Пример использования GAN для генерации данных
# (Полная реализация требует отдельного пайплайна)
```

### 3.2 Динамическое взвешивание классов
```python
class DynamicWeightedLoss:
    """Кастомная функция потерь с динамическими весами"""
    def __init__(self, alpha=0.5):
        self.alpha = alpha  # Контроль баланса
        
    def __call__(self, y_true, y_pred):
        pos_mask = (y_true == 1)
        pos_weight = (1 - self.alpha) / K.sum(pos_mask)
        neg_weight = self.alpha / K.sum(1 - pos_mask)
        weights = pos_weight * pos_mask + neg_weight * (1 - pos_mask)
        return K.mean(weights * K.binary_crossentropy(y_true, y_pred))
```

### 3.3 Оптимизация порога классификации
```python
from sklearn.metrics import precision_recall_curve

# Поиск оптимального порога
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]

# Применение порога
y_pred_custom = (y_probs >= best_threshold).astype(int)
```

---

## 📊 Чеклист по уровням

| Уровень | Навыки |
|---------|--------|
| 🟢 | RandomOverSampler, class_weight, анализ распределения |
| 🟡 | SMOTE, BalancedRandomForest, метрики F1/ROC-AUC |
| 🔴 | GAN-синтез, кастомные функции потерь, оптимизация порога |

---

## ⚠️ Антипаттерны
### Для всех уровней:
- **Использование accuracy как метрики** для несбалансированных данных
- **Слепое применение SMOTE** без анализа природы данных
- **Полное устранение дисбаланса** (может ухудшить качество)

### 🔴 Эксперты:
- **Переобучение на синтетических данных** (тестируйте на реальных данных)
- **Игнорирование costs-sensitive анализа** (разная цена ошибок)

---

## 🚀 Советы
1. **Анализ перед балансировкой:**
```python
from yellowbrick.classifier import ClassBalance
visualizer = ClassBalance(labels=['Class 0', 'Class 1'])
visualizer.fit(y_train, y_test)
visualizer.show()
```

2. **Комбинируйте методы:**
```python
from imblearn.pipeline import Pipeline

pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy=0.3)),
    ('undersample', RandomUnderSampler(sampling_strategy=0.5)),
    ('model', LogisticRegression())
])
```

3. **Анализ ошибок через матрицу:**
```python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize='true')
```

---

## 📈 Практический пример: Кредитное мошенничество
```python
# 🟡 Продвинутый подход
from imblearn.ensemble import EasyEnsembleClassifier

# EasyEnsemble: ансамбль из балансированных подвыборок
eec = EasyEnsembleClassifier(
    n_estimators=10,
    sampling_strategy=0.5,
    random_state=42
)
eec.fit(X_train, y_train)

# 🔴 Экспертный подход
# Генерация данных с помощью CTGAN
from ctgan import CTGANSynthesizer

ctgan = CTGANSynthesizer()
ctgan.fit(df_minority, ['fraud'])  # Обучаем только на миноритарном классе
synthetic_data = ctgan.sample(1000)
```

---

## � Тренировочные задания

### 🟢 Базовый уровень
**Задача 1:** Примените RandomUnderSampler к датасету кредитного мошенничества (fraud_detection.csv). Сравните F1-score до/после.

### 🟡 Продвинутый уровень
**Задача 2:** Используя SMOTE, сбалансируйте классы в датасете медицинских диагнозов. Постройте ROC-кривые для моделей LogisticRegression и BalancedRandomForest.

### 🔴 Экспертный уровень
**Задача 3:** Реализуйте кастомную функцию потерь в PyTorch, которая динамически рассчитывает веса классов на каждом батче. Проверьте на датасете с соотношением классов 1:100.

---

```python
# Пример решения для 🟢 Задачи 1
import pandas as pd
from sklearn.model_selection import train_test_split

# Загрузка данных
df = pd.read_csv('fraud_detection.csv')
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Применение Undersampling
undersampler = RandomUnderSampler(sampling_strategy=0.5)
X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

# Обучение и оценка
model = LogisticRegression(max_iter=1000)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 📌 Заключение
Ключевые идеи:
1. **Не всегда нужно балансировать классы** — зависит от задачи.
2. **Сочетайте методы** (например, SMOTE + Undersampling).
3. **Экспериментируйте с метриками** — Precision/Recall Tradeoff.
4. **Учитывайте стоимость ошибок** (ложные положительные vs. отрицательные).
