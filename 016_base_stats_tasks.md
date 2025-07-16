### **Кейс: Анализ данных о чаевых в ресторане**  
**Датасет:** [Tips Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv) из библиотеки Seaborn  
**Описание:**  
- 244 записи о чаевых в ресторане  
- 7 признаков:  
  - `total_bill`: общий счет (доллары)  
  - `tip`: чаевые (доллары)  
  - `sex`: пол официанта  
  - `smoker`: курящая компания?  
  - `day`: день недели  
  - `time`: время (ланч/ужин)  
  - `size`: размер компании  

---

### **🟢 Базовый уровень: Описательная статистика**  
**Задача:** Исследуйте основные характеристики данных  
**Действия:**  
1. Рассчитайте базовые статистики для `total_bill` и `tip`:  
   - Среднее, медиану, стандартное отклонение  
   - Квартили (25%, 50%, 75%)  
2. Постройте boxplot для `tip` по дням недели  
3. Проверьте гипотезу: "Средние чаевые на ланче и ужине равны" (t-тест)  

---

### **🟡 Продвинутый уровень: Анализ зависимостей**  
**Задача:** Исследуйте связи между переменными  
**Действия:**  
1. Рассчитайте корреляцию Пирсона между `total_bill` и `tip`  
2. Постройте scatter plot `total_bill` vs `tip` с разделением по времени  
3. Проверьте гипотезы:  
   - "Мужчины оставляют больше чаевых, чем женщины" (U-тест Манна-Уитни)  
   - "Размер компании влияет на процент чаевых" (ANOVA)  

---

### **🔴 Экспертный уровень: Многомерный анализ**  
**Задача:** Выявите сложные закономерности  
**Действия:**  
1. Постройте линейную регрессию: `tip ~ total_bill + size + time`  
   - Интерпретируйте коэффициенты  
   - Рассчитайте R²  
2. Проведите bootstrap для оценки 95% доверительного интервала средних чаевых  
3. Проверьте гипотезу: "Курящие компании оставляют меньший процент чаевых"  
   - Создайте признак `tip_percent` = tip / total_bill  
   - Используйте bootstrap для сравнения групп  

---

### **Примеры решений**

#### 🟢 Базовое описание данных
```python
import seaborn as sns
tips = sns.load_dataset('tips')

# Базовые статистики
print(tips[['total_bill', 'tip']].describe())

# Boxplot по дням
sns.boxplot(x='day', y='tip', data=tips)
plt.title('Распределение чаевых по дням недели')
```

#### 🟡 Корреляционный анализ
```python
# Матрица корреляций
corr_matrix = tips[['total_bill', 'tip', 'size']].corr()
sns.heatmap(corr_matrix, annot=True)

# Scatter plot
sns.scatterplot(x='total_bill', y='tip', hue='time', data=tips)
```

#### 🔴 Линейная регрессия
```python
from sklearn.linear_model import LinearRegression

# Преобразование категориальных признаков
tips_encoded = pd.get_dummies(tips, columns=['time'], drop_first=True)

# Обучение модели
model = LinearRegression()
model.fit(tips_encoded[['total_bill', 'size', 'time_Dinner']], tips['tip'])

print(f"Коэффициенты: {model.coef_}")
print(f"R²: {model.score():.3f}")
```

---

### **Проверка решений**

#### Для 🟢
```python
# Проверка статистик
assert 15 < tips['total_bill'].mean() < 25
assert tips.groupby('time')['tip'].mean().diff().abs().iloc[-1] > 0.5

# Проверка t-теста
from scipy import stats
t_stat, p_val = stats.ttest_ind(
    tips[tips['time']=='Lunch']['tip'],
    tips[tips['time']=='Dinner']['tip']
)
assert p_val < 0.05
```

#### Для 🟡
```python
# Проверка корреляции
assert tips['total_bill'].corr(tips['tip']) > 0.5

# Проверка U-теста
u_stat, p_val = stats.mannwhitneyu(
    tips[tips['sex']=='Male']['tip'],
    tips[tips['sex']=='Female']['tip']
)
assert p_val < 0.1
```

#### Для 🔴
```python
# Проверка R²
assert model.score() > 0.4

# Проверка bootstrap
from sklearn.utils import resample
boot_means = [resample(tips['tip']).mean() for _ in range(1000)]
ci = np.percentile(boot_means, [2.5, 97.5])
assert 2.7 < ci[0] < 3.0 < ci[1] < 3.4
```

---

### **Дополнительные задания**
1. **🟢**: Рассчитайте процент чаевых и постройте его распределение  
2. **🟡**: Проверьте гипотезу о нормальности распределения `total_bill`  
3. **🔴**: Создайте модель предсказания чаевых с учетом взаимодействия признаков  

---

### **Теоретическая справка**
1. **T-тест**: Сравнение средних двух независимых выборок  
2. **U-тест**: Непараметрический аналог t-теста  
3. **ANOVA**: Сравнение средних нескольких групп  
4. **Bootstrap**: Оценка доверительных интервалов через ресэмплинг  
5. **R²**: Доля объясненной дисперсии в регрессии  

> **Философия главы:**  
> "Статистика — это язык, на котором данные рассказывают свои истории. Научитесь слушать."  

**Для загрузки данных:**
```python
import seaborn as sns
tips = sns.load_dataset('tips')
```
