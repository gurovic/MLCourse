# **Bias-Variance Tradeoff**

## **🟢 Базовый уровень: Основные понятия на примере недвижимости**

### **1. Что такое смещение (Bias)?**
**Смещение** — когда модель **слишком простая** и не может уловить реальные закономерности.

**Пример с ценами на дома:**  
- 🤖 *Модель:* "Цена дома = 100 000 + 1000 * площадь" (линейная)  
- ❌ *Проблема:* Не учитывает район, год постройки, этаж  
- 📉 *Результат:* Всегда занижает цены в центре, завышает на окраинах  

<img src="https://i.imgur.com/3Wl5s9G.png" width="500">

### **2. Что такое разброс (Variance)?**
**Разброс** — когда модель **слишком сложная** и подстраивается под случайные колебания.

**Пример с ценами на дома:**  
- 🤖 *Модель:* Учитывает каждый чих - шум стройки рядом, цвет двери, имя продавца  
- ❌ *Проблема:* На новых данных ошибается случайным образом  
- 📈 *Результат:* Вчера предсказала 10 млн, сегодня на таком же доме - 8 млн  

<img src="https://i.imgur.com/5g5LZ7e.png" width="500">

### **3. Идеальный баланс**
**Хорошая модель:**  
- Учитывает ключевые факторы: площадь, район, этажность ✅  
- Игнорирует случайные колебания: "вчера продавец был в плохом настроении" ❌  
- Стабильно работает на новых данных 📊  

```python
# Псевдокод хорошей модели
def predict_price(house):
    base_price = 100_000
    price_per_sqm = 1_000 * house['district_coef']
    return base_price + price_per_sqm * house['area']
```

---

## **🟡 Продвинутый уровень: Визуализируем на данных**

### **1. Генерируем реалистичные данные**
```python
import numpy as np
import matplotlib.pyplot as plt

# Истинная зависимость: цена = 50 + 10 * √площадь
areas = np.linspace(20, 200, 100)
true_prices = 50 + 10 * np.sqrt(areas)

# Добавляем реалистичный шум (цена зависит от многих факторов)
np.random.seed(42)
noise = 20 * np.random.randn(100)
observed_prices = true_prices + noise

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(areas, observed_prices, alpha=0.6, label='Реальные сделки')
plt.plot(areas, true_prices, 'g-', lw=3, label='Истинная зависимость')
plt.xlabel('Площадь (м²)')
plt.ylabel('Цена ($1000)')
plt.title('Рынок недвижимости: реальные данные')
plt.legend()
plt.grid(True)
plt.show()
```

### **2. Сравним три модели**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Простая модель (линейная)
model_simple = LinearRegression()
model_simple.fit(areas.reshape(-1,1), observed_prices)
pred_simple = model_simple.predict(areas.reshape(-1,1))

# Хорошая модель (квадратный корень)
class SqrtTransformer:
    def fit(self, X, y=None): return self
    def transform(self, X): return np.sqrt(X)

model_good = make_pipeline(SqrtTransformer(), LinearRegression())
model_good.fit(areas.reshape(-1,1), observed_prices)
pred_good = model_good.predict(areas.reshape(-1,1))

# Сложная модель (полином 10й степени)
model_complex = make_pipeline(PolynomialFeatures(10), LinearRegression())
model_complex.fit(areas.reshape(-1,1), observed_prices)
pred_complex = model_complex.predict(areas.reshape(-1,1))

# Визуализация
plt.figure(figsize=(12, 8))
plt.scatter(areas, observed_prices, alpha=0.3, label='Данные')
plt.plot(areas, true_prices, 'g-', lw=2, label='Истинная зависимость')
plt.plot(areas, pred_simple, 'r--', lw=2, label='Линейная (high bias)')
plt.plot(areas, pred_good, 'b-', lw=2, label='Хорошая модель (√area)')
plt.plot(areas, pred_complex, 'm--', lw=2, label='Полином 10й степени (high variance)')
plt.xlabel('Площадь (м²)')
plt.ylabel('Цена ($1000)')
plt.legend()
plt.title('Bias-Variance Tradeoff в предсказании цен')
plt.grid(True)
plt.show()
```

**Что видим:**  
- 🔴 **Красная линия (high bias):** Слишком прямая, не учитывает убывающую отдачу от площади  
- 🔵 **Синяя линия (баланс):** Хорошо повторяет истинную зависимость  
- 🟣 **Фиолетовая линия (high variance):** Пытается пройти через все точки, включая случайные выбросы  

---

## **🔴 Экспертный уровень: Как найти золотую середину?**

### **1. Диагностика по ошибкам**
```python
from sklearn.metrics import mean_squared_error

# Разделим данные на тренировочные и тестовые
from sklearn.model_selection import train_test_split
X = areas.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(
    X, observed_prices, test_size=0.3, random_state=42
)

# Обучим модели
model_simple.fit(X_train, y_train)
model_good.fit(X_train, y_train)
model_complex.fit(X_train, y_train)

# Сравним ошибки
print("Ошибки на тренировочных данных:")
print(f"Линейная: {mean_squared_error(y_train, model_simple.predict(X_train)):.1f}")
print(f"Хорошая: {mean_squared_error(y_train, model_good.predict(X_train)):.1f}")
print(f"Полином: {mean_squared_error(y_train, model_complex.predict(X_train)):.1f}")

print("\nОшибки на тестовых данных:")
print(f"Линейная: {mean_squared_error(y_test, model_simple.predict(X_test)):.1f}")
print(f"Хорошая: {mean_squared_error(y_test, model_good.predict(X_test)):.1f}")
print(f"Полином: {mean_squared_error(y_test, model_complex.predict(X_test)):.1f}")
```

**Результаты:**  
```
Ошибки на тренировочных данных:
Линейная: 350.4
Хорошая: 302.5
Полином: 250.1

Ошибки на тестовых данных:
Линейная: 390.7
Хорошая: 315.8
Полином: 450.3
```

**Анализ:**  
- 📉 **Линейная модель:** Высокие ошибки везде (high bias)  
- ✅ **Хорошая модель:** Баланс - ошибки близки  
- 📈 **Полином:** Малая ошибка на тренировке, большая на тесте (high variance)  

### **2. Как улучшить модели?**
**Для линейной модели (high bias):**  
- Добавить новые признаки: √площадь, район, год постройки  
- Использовать нелинейную модель: дерево решений  

**Для полиномиальной модели (high variance):**  
- Упростить модель: уменьшить степень полинома  
- Добавить регуляризацию  
- Увеличить набор данных  

```python
# Пример улучшения: Ridge регрессия с регуляризацией
from sklearn.linear_model import Ridge

model_improved = make_pipeline(
    PolynomialFeatures(5),  # Уменьшили сложность
    Ridge(alpha=10.0)       # Добавили регуляризацию
)
model_improved.fit(X_train, y_train)

print(f"Улучшенная модель тест MSE: {mean_squared_error(y_test, model_improved.predict(X_test)):.1f}")
```

### **3. Кросс-валидация для подбора сложности**
```python
from sklearn.model_selection import cross_val_score

degrees = range(1, 15)
cv_scores = []

for d in degrees:
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    scores = cross_val_score(model, X, observed_prices, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-np.mean(scores))

# Найдем оптимальную степень
optimal_degree = degrees[np.argmin(cv_scores)]

plt.figure(figsize=(10, 6))
plt.plot(degrees, cv_scores, 'o-')
plt.axvline(optimal_degree, color='r', linestyle='--')
plt.xlabel('Степень полинома')
plt.ylabel('Ошибка (MSE)')
plt.title('Подбор оптимальной сложности')
plt.grid(True)
plt.show()
```

---

## **📊 Практические рекомендации**

### **Когда что делать:**
| **Симптомы** | **Диагноз** | **Лечение** |
|--------------|-------------|-------------|
| Высокая ошибка на тренировке и тесте | High bias | Увеличить сложность модели, добавить признаки |
| Низкая ошибка на тренировке, высокая на тесте | High variance | Упростить модель, добавить регуляризацию |
| Ошибки близки, но высоки | Недостаточная емкость | Увеличить сложность модели |

### **Золотые правила:**
1. Всегда начинайте с простой модели (линейная регрессия)
2. Используйте кросс-валидацию для оценки
3. Постепенно увеличивайте сложность
4. Останавливайтесь, когда тестовая ошибка начинает расти
5. Регуляризация — ваш друг против переобучения

> "Хорошая модель как хороший риелтор: учитывает важные факторы (площадь, район), игнорирует случайное (погода в день просмотра), и дает стабильные результаты"

**Примеры регуляризации:**  
```python
# Ridge регрессия (L2 регуляризация)
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)  # Чем больше alpha, тем сильнее упрощаем

# Lasso регрессия (L1 регуляризация)
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)  # Может обнулять неважные признаки
```
