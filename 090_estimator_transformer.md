### Глава: Estimator и Transformer в scikit-learn  
*Основные строительные блоки для предобработки данных и моделирования*

---

#### 🧩 Что такое Estimator?  
**Estimator (Оценщик)** - любой объект в scikit-learn, который:  
1. Изучает параметры из данных через `.fit()`  
2. Применяет полученные знания через `.transform()` или `.predict()`  

```python
from sklearn.ensemble import RandomForestClassifier

# Создание оценщика
estimator = RandomForestClassifier(n_estimators=100)

# Обучение параметров
estimator.fit(X_train, y_train)

# Применение обученной модели
predictions = estimator.predict(X_test)
```

**Ключевые категории:**  
- Классификаторы (`ClassifierMixin`)  
- Регрессоры (`RegressorMixin`)  
- Кластеризаторы (`ClusterMixin`)  

---

#### 🔄 Что такое Transformer?  
**Transformer (Преобразователь)** - специальный тип оценщика, который:  
1. Извлекает параметры преобразования через `.fit()`  
2. Применяет преобразование к данным через `.transform()`  

```python
from sklearn.preprocessing import StandardScaler

# Создание преобразователя
transformer = StandardScaler()

# Расчет среднего и std
transformer.fit(X_train)

# Применение стандартизации
X_scaled = transformer.transform(X_train)
```

**Распространенные трансформеры:**  
- `StandardScaler`/`MinMaxScaler` - масштабирование  
- `OneHotEncoder` - кодирование категорий  
- `SimpleImputer` - заполнение пропусков  
- `PCA` - уменьшение размерности  

---

#### 🧪 Особенности работы  
1. **Немезоидность (Non-statefulness)**  
   После `.fit()` трансформер сохраняет параметры в атрибутах с нижним подчеркиванием:  
   ```python
   print(transformer.mean_)  # Средние значения
   print(transformer.scale_)  # Стандартные отклонения
   ```

2. **Проверка входных данных**  
   Трансформеры автоматически проверяют:  
   - Отсутствие NaN в `fit()`  
   - Совпадение размерности в `transform()`  

3. **Метод `.fit_transform()`**  
   Оптимизированная комбинация для одновременного обучения и преобразования:  
   ```python
   X_encoded = OneHotEncoder().fit_transform(X_categorical)
   ```

---

#### 🧩 Композиция: Pipeline  
Трансформеры и оценщики объединяются в цепочки обработки:  
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Создание пайплайна
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Трансформер
    ('scaler', StandardScaler()),                   # Трансформер
    ('classifier', LogisticRegression())            # Оценщик
])

# Единый интерфейс для всей цепочки
pipe.fit(X_train, y_train)  # Обучение всех компонентов
pipe.predict(X_test)        # Автоматическое преобразование + предсказание
```

**Преимущества пайплайнов:**  
- Предотвращение data leakage  
- Упрощение deployment  
- Совместная настройка гиперпараметров  

---

#### 🛠️ Создание кастомного трансформера  
```python
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, add_const=1e-6):
        self.add_const = add_const
        
    def fit(self, X, y=None):
        return self  # Ничего не вычисляем в fit
    
    def transform(self, X):
        return np.log(X + self.add_const)
        
# Использование
preprocessor = Pipeline([
    ('imputer', SimpleImputer()),
    ('log', LogTransformer())
])
```

**Обязательные элементы:**  
1. Наследование от `BaseEstimator` и `TransformerMixin`  
2. Реализация методов `.fit()` и `.transform()`  
3. `TransformerMixin` автоматически добавляет `.fit_transform()`  

---

#### 💡 Лучшие практики  
1. **Разделение данных до преобразований**  
   Всегда делайте `train_test_split()` перед вызовом `.fit()` трансформеров  

2. **ColumnTransformer для разных признаков**  
   ```python
   from sklearn.compose import ColumnTransformer
   
   preprocessor = ColumnTransformer([
       ('num', StandardScaler(), num_cols),
       ('cat', OneHotEncoder(), cat_cols)
   ])
   ```

3. **Сохранение состояния**  
   Всегда сохраняйте обученные трансформеры:  
   ```python
   import joblib
   joblib.dump(transformer, 'scaler.pkl')
   ```

4. **Проверка выходных данных**  
   После `transform()` используйте `.get_feature_names_out()` для отслеживания колонок  

---

**Golden Rule:**  
> "Все, что учится на данных - Estimator, все, что меняет данные - Transformer,  
> а Pipeline - это клей, который их объединяет в воспроизводимый workflow."
