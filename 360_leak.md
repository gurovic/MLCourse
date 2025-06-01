# Утечки данных (Data Leakage)  

---

## 🟢 Базовый уровень (Что такое утечка и как ее избежать)

### 1.1 Определение утечки данных
**Утечка данных** - когда информация из тестового набора или будущих данных **непреднамеренно используется** при обучении модели, что приводит к:
- Нереалистично высоким показателям
- Провалу модели в реальных условиях

### 1.2 Основные типы утечек
```python
# Пример таргетной утечки
import pandas as pd

# Данные о кредитах
data = pd.DataFrame({
    'income': [50000, 60000, 45000, 70000],
    'credit_score': [650, 700, 600, 750],
    'loan_default': [0, 1, 0, 1],
    'late_payment': [0, 1, 0, 1]  # УТЕЧКА! Этот признак станет известен ПОСЛЕ выдачи кредита
})
```

### 1.3 Простые правила предотвращения
1. **Всегда сначала делайте train-test split**
   ```python
   # ПРАВИЛЬНО:
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   
   # Потом предобработка
   scaler.fit(X_train)
   X_train_scaled = scaler.transform(X_train)
   X_test_scaled = scaler.transform(X_test)  # Только transform!
   ```
   
2. **Не используйте будущую информацию**
   - Проверяйте: "Будет ли этот признак доступен В МОМЕНТ предсказания?"

3. **Удаляйте уникальные идентификаторы**
   ```python
   # Удаление ID клиентов
   data = data.drop(columns=['client_id', 'transaction_id'])
   ```

---

## 🟡 Продвинутый уровень (Сложные случаи утечек)

### 2.1 Утечки во временных данных
**Проблема:** Использование будущих данных для предсказания прошлого  
**Решение:** Жесткое разделение по времени:
```python
train = data[data['date'] < '2023-01-01']
test = data[data['date'] >= '2023-01-01']
```

### 2.2 Утечки в кросс-валидации
**Ошибка:**
```python
# УТЕЧКА! Предобработка до CV
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Утечка информации между фолдами
cross_val_score(model, X_scaled, y, cv=5)
```

**Решение:** Используйте Pipeline:
```python
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    StandardScaler(),
    PCA(),
    RandomForestClassifier()
)
cross_val_score(pipeline, X, y, cv=5)  # Безопасно
```

### 2.3 Утечки через агрегацию
**Пример ошибки:**
```python
# УТЕЧКА! Использование глобального среднего
global_mean = data['income'].mean()
data['income'] = data['income'].fillna(global_mean)  # До разделения
```

**Решение:** Рассчитывайте статистики только на трейне:
```python
train_mean = X_train['income'].mean()
X_train_filled = X_train['income'].fillna(train_mean)
X_test_filled = X_test['income'].fillna(train_mean)  # Используем train_mean!
```

### 2.4 Детектирование утечек
**Метод 1:** Анализ корреляций
```python
# Поиск признаков с подозрительно высокой корреляцией
correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
suspicious_features = correlations[correlations > 0.9].index
```

**Метод 2:** Проверка важности признаков
```python
model.fit(X_train, y_train)
if model.feature_importances_[0] > 0.5:  # Один признак доминирует
    print("Потенциальная утечка!")
```

---

## 🔴 Экспертный уровень (Сложные превентивные стратегии)

### 3.1 Безопасная работа с временными рядами
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    # Для каждого фолда учитываем только исторические данные
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Важно: предобработка должна быть внутри цикла!
    imputer.fit(X_train)
    X_train_imp = imputer.transform(X_train)
    X_test_imp = imputer.transform(X_test)
```

### 3.2 Расширенное обнаружение утечек
**Метод пермутационного тестирования:**
```python
def detect_leakage(model, X_test, y_test):
    base_score = model.score(X_test, y_test)
    
    leakage_scores = {}
    for col in X_test.columns:
        X_test_perm = X_test.copy()
        X_test_perm[col] = np.random.permutation(X_test_perm[col])  # Разрушаем связь
        
        perm_score = model.score(X_test_perm, y_test)
        leakage = base_score - perm_score  # Насколько важна колонка
        
        if leakage > 0.3 * base_score:  # Порог 30%
            leakage_scores[col] = leakage
    
    return leakage_scores
```

### 3.3 Защищенные стратегии для feature engineering
**Принципы:**
1. Разделяйте признаки на 3 категории:
   - **Безопасные:** Известны в момент предсказания (возраст, пол)
   - **Опасные:** Требуют проверки (баланс счета, последняя транзакция)
   - **Запрещенные:** Связаны с таргетом (флаг мошенничества)

2. Используйте временные ограничения в feature engineering:
```python
# Пример с FeatureTools
import featuretools as ft

es = ft.EntitySet()
es = es.add_dataframe(dataframe=transactions, 
                     index="id", 
                     time_index="transaction_time")

features, defs = ft.dfs(entityset=es,
                        target_dataframe_name="customers",
                        cutoff_time=cutoff_times,  # Критично для безопасности
                        agg_primitives=["sum", "mean"])
```

### 3.4 Автоматизированные системы защиты
**Паттерн "Двойного разделения":**
```python
# Шаг 1: Исходное разделение
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)

# Шаг 2: Создание валидационного набора для feature engineering
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)

# Шаг 3: Разработка признаков ТОЛЬКО на X_train
feature_engineer.fit(X_train, y_train)

# Шаг 4: Трансформация всех данных
X_val_transformed = feature_engineer.transform(X_val)
X_test_transformed = feature_engineer.transform(X_test)  # Без утечек!
```

### 3.5 Реальные кейсы катастроф
1. **Медицинская диагностика:** Использование ID сканера, который коррелировал с диагнозом
   - Решение: Удаление всех технических идентификаторов

2. **Кредитный скоринг:** Признак "количество запросов в БКИ" (формируется ПОСЛЕ решения)
   - Решение: Использовать только исторические запросы

3. **Прогнозирование продаж:** Включение данных о доставке (известно после заказа)
   - Решение: Использовать только признаки, известные до покупки

---

## 🛡️ Итоговая таблица защиты по уровням

| **Уровень** | **Риски** | **Защитные меры** |
|-------------|-----------|-------------------|
| 🟢 Базовый | Таргетная утечка, Неправильный split | Train-test split, Проверка доступности признаков |
| 🟡 Продвинутый | Временные утечки, Агрегация, CV-утечки | Pipeline, Разделение по времени, Расчет статистик на трейне |
| 🔴 Экспертный | Feature engineering утечки, Скрытые зависимости | Двойное разделение, Пермутационное тестирование, FeatureTools с cutoff |

---

## 📚 Ресурсы для углубленного изучения
1. [Google ML Crash Course: Data Leakage](https://developers.google.com/machine-learning/crash-course/training-and-test-sets/data-leakage)
2. [Kaggle: Data Leakage Tutorial](https://www.kaggle.com/code/alexisbcook/data-leakage)
3. [Weights & Biases: Data Leakage Detection](https://wandb.ai/site/articles/data-leakage-detection-in-machine-learning)

> "Самая опасная утечка - та, о которой вы не подозреваете. Всегда спрашивайте: 
> 'Откуда модель ЭТО знает?' и 'Будет ли эта информация доступна в момент предсказания?'"  
> - Практическое правило борьбы с утечками данных
