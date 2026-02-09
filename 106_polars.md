# Polars — современная альтернатива pandas

## Введение: Зачем нужен Polars?

Polars — это современная библиотека для работы с данными в Python и Rust, разработанная как высокопроизводительная альтернатива pandas. Если pandas стал стандартом де-факто для анализа данных, то Polars предлагает новый подход, построенный с нуля с учётом современных вычислительных архитектур.

### Основные преимущества Polars:

1. **Скорость**: Написан на Rust, использует Apache Arrow в качестве внутреннего представления данных
2. **Эффективность памяти**: Ленивые вычисления (lazy evaluation) и оптимизация запросов
3. **Параллелизм**: Автоматическая параллелизация операций на всех доступных ядрах CPU
4. **Предсказуемость**: Строгая типизация и явный API без неоднозначностей
5. **Масштабируемость**: Работает с данными, которые не помещаются в память

## 1. Два режима работы: Eager vs Lazy

Polars предлагает два API для работы с данными:

### Eager API — немедленное выполнение

```python
import polars as pl

# Создание DataFrame и немедленное выполнение операций
df = pl.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 70000, 55000]
})

# Каждая операция выполняется сразу
result = df.filter(pl.col('age') > 27).select(['name', 'salary'])
print(result)
```

### Lazy API — ленивое выполнение с оптимизацией

```python
# Создание LazyFrame
lf = pl.scan_csv('large_dataset.csv')  # Файл не читается сразу

# Построение плана запроса
query = (lf
    .filter(pl.col('age') > 27)
    .select(['name', 'salary', 'department'])
    .groupby('department')
    .agg(pl.col('salary').mean())
)

# Ничего не выполнено до вызова .collect()
# Можно посмотреть оптимизированный план запроса
print(query.explain())

# Теперь выполнение с оптимизациями
result = query.collect()
```

**Ключевое отличие**: Lazy API анализирует весь запрос целиком и оптимизирует его перед выполнением — как SQL-движок. Это позволяет:
- Исключить ненужные столбцы на этапе чтения
- Применить фильтры до других операций
- Переупорядочить операции для эффективности

## 2. Выражения (Expressions) — сердце Polars

В отличие от pandas, где много способов сделать одно и то же, Polars использует единообразный подход через **выражения**.

### Основные компоненты выражений

```python
import polars as pl

df = pl.DataFrame({
    'product': ['A', 'B', 'A', 'C', 'B', 'A'],
    'revenue': [100, 150, 120, 200, 180, 110],
    'costs': [60, 90, 70, 120, 100, 65]
})

# Базовый синтаксис: pl.col('название_столбца')
# Все операции строятся как цепочки методов

result = df.select([
    pl.col('product'),
    pl.col('revenue'),
    (pl.col('revenue') - pl.col('costs')).alias('profit'),
    (pl.col('revenue') / pl.col('costs')).alias('margin')
])
```

### Мощь выражений: автоматическая векторизация

```python
# Условная логика внутри выражений
df.with_columns([
    pl.when(pl.col('revenue') > 150)
      .then(pl.lit('high'))
      .when(pl.col('revenue') > 100)
      .then(pl.lit('medium'))
      .otherwise(pl.lit('low'))
      .alias('revenue_category')
])
```

### Агрегация с выражениями

```python
# Группировка с множественными агрегациями
grouped = df.groupby('product').agg([
    pl.col('revenue').sum().alias('total_revenue'),
    pl.col('revenue').mean().alias('avg_revenue'),
    pl.col('revenue').max().alias('max_revenue'),
    pl.col('costs').sum().alias('total_costs'),
    pl.count().alias('count')
])
```

## 3. Контекстные операции: select, with_columns и filter

Polars различает три вида трансформаций в зависимости от контекста:

### select — выбор и создание столбцов

```python
df.select([
    pl.col('revenue'),
    (pl.col('revenue') * 1.1).alias('revenue_with_tax')
])
```

### with_columns — добавление новых столбцов

```python
df.with_columns([
    (pl.col('revenue') - pl.col('costs')).alias('profit')
])
```

### filter — фильтрация строк

```python
df.filter(
    (pl.col('revenue') > 100) & (pl.col('costs') < 100)
)
```

## 4. Работа с типами данных

Polars имеет строгую систему типов, более продуманную, чем у pandas:

```python
import polars as pl
from datetime import date

# Явное указание схемы данных
df = pl.DataFrame({
    'id': pl.Series([1, 2, 3], dtype=pl.Int32),
    'price': pl.Series([10.5, 20.3, 15.7], dtype=pl.Float64),
    'date': pl.Series([date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)]),
    'active': pl.Series([True, False, True], dtype=pl.Boolean)
})

# Приведение типов
df = df.with_columns([
    pl.col('id').cast(pl.Int64),
    pl.col('price').cast(pl.Int32)  # Приведение к целому (усечение дробной части)
])
```

### Специальные типы

```python
# Categorical для экономии памяти при повторяющихся строках
df = df.with_columns([
    pl.col('category').cast(pl.Categorical)
])

# Даты и время
df = df.with_columns([
    pl.col('timestamp').str.strptime(pl.Datetime, fmt='%Y-%m-%d %H:%M:%S')
])
```

## 5. Оконные функции (Window Functions)

Polars имеет мощную поддержку оконных функций:

```python
df = pl.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'B'],
    'value': [10, 20, 15, 25, 30]
})

# Ранжирование внутри групп
df.with_columns([
    pl.col('value').rank().over('group').alias('rank_in_group'),
    pl.col('value').sum().over('group').alias('group_total'),
    pl.col('value').shift(1).over('group').alias('previous_value')
])
```

## 6. Объединение данных: join

```python
df1 = pl.DataFrame({
    'user_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

df2 = pl.DataFrame({
    'user_id': [1, 1, 2, 4],
    'purchase': [100, 150, 200, 120]
})

# Inner join
result = df1.join(df2, on='user_id', how='inner')

# Left join с суффиксами для одинаковых имён
result = df1.join(df2, on='user_id', how='left', suffix='_right')

# Более сложные условия join (по нескольким столбцам)
result = df1.join(df2, left_on=['user_id'], right_on=['user_id'], how='outer')
```

## 7. Работа с большими данными: streaming mode

```python
# Для данных, не помещающихся в память
q = (
    pl.scan_csv('huge_file.csv')
    .filter(pl.col('year') == 2023)
    .groupby('category')
    .agg([
        pl.col('amount').sum(),
        pl.col('amount').count()
    ])
)

# Streaming режим обрабатывает данные порциями
result = q.collect(streaming=True)
```

## 8. Чтение и запись данных

Polars поддерживает множество форматов:

```python
# CSV
df = pl.read_csv('data.csv', 
                 dtypes={'id': pl.Int32, 'amount': pl.Float64},
                 null_values=['NA', 'NULL'])

# Parquet (рекомендуемый формат для больших данных)
df = pl.read_parquet('data.parquet')
df.write_parquet('output.parquet', compression='snappy')

# JSON
df = pl.read_json('data.json')

# Excel
df = pl.read_excel('data.xlsx', sheet_name='Sheet1')

# SQL базы данных
import polars as pl
import connectorx as cx

df = pl.read_database("SELECT * FROM table", connection_uri="postgresql://...")
```

## 9. Сравнение с pandas

| Особенность | Pandas | Polars |
|------------|--------|--------|
| Язык реализации | C/Python | Rust |
| API | Множество способов | Единообразные выражения |
| Параллелизм | Ограничен | Автоматический |
| Ленивые вычисления | Нет | Да (Lazy API) |
| Изменяемость | DataFrame изменяемые | Иммутабельные по умолчанию |
| Скорость | Базовая | 2-10x быстрее |
| Память | Может быть избыточной | Оптимизирована |

### Миграция с pandas на Polars

```python
# Pandas
import pandas as pd
df_pandas = pd.DataFrame({'a': [1, 2, 3]})

# Конвертация
df_polars = pl.from_pandas(df_pandas)

# Обратно в pandas, если нужно
df_pandas_back = df_polars.to_pandas()
```

## 10. Когда использовать Polars?

### Используйте Polars когда:
- Работаете с большими датасетами (>1GB)
- Нужна максимальная производительность
- Важна предсказуемость и отсутствие неявного поведения
- Готовы изучить новый API
- Пишете новый проект с нуля

### Оставайтесь с pandas когда:
- Работаете с legacy кодом
- Нужна совместимость с экосистемой (seaborn, scikit-learn и т.д.)
- Датасеты маленькие (<100MB)
- Команда не готова к изменениям

## Заключение

Polars — это не просто "быстрый pandas". Это переосмысление того, как должна выглядеть библиотека для анализа данных в современную эпоху. Его архитектура, основанная на выражениях, ленивых вычислениях и параллелизме, делает работу с данными не только быстрее, но и более предсказуемой.

Основные принципы работы с Polars:
1. **Думайте выражениями** — используйте `pl.col()` и цепочки методов
2. **Используйте Lazy API** для больших данных — планирование запросов даёт огромный выигрыш
3. **Доверяйте типам** — строгая типизация помогает избежать ошибок
4. **Пишите декларативно** — описывайте *что* нужно сделать, а не *как*

С ростом объёмов данных и требований к производительности, Polars становится всё более важным инструментом в арсенале специалиста по данным.
