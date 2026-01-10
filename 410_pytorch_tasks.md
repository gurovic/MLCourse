# **Задачи: PyTorch Basics - Тензоры и Autograd**

## **⚙️ Подготовка**

Перед началом убедитесь, что у вас установлен PyTorch:
```python
import torch
print(f"PyTorch версия: {torch.__version__}")
# Рекомендуемая версия: 2.0 или выше
```

---

## **🟢 Базовый уровень**

### **Задача 1: Создание и операции с тензорами**
Создайте следующие тензоры и выполните указанные операции:

1. Создайте тензор `a` из списка `[1, 2, 3, 4, 5]`
2. Создайте тензор `b` из нулей размера 3x3
3. Создайте тензор `c` из случайных чисел размера 2x4
4. Выполните поэлементное умножение тензора `a` на 2
5. Найдите сумму всех элементов в тензоре `c`

**Подсказка:** Используйте `torch.tensor()`, `torch.zeros()`, `torch.rand()`, операторы `*` и метод `.sum()`

<details>
<summary>Решение</summary>

```python
import torch

# 1. Создание тензора из списка
a = torch.tensor([1, 2, 3, 4, 5])
print(f"a = {a}")

# 2. Тензор из нулей
b = torch.zeros(3, 3)
print(f"b = {b}")

# 3. Случайный тензор
c = torch.rand(2, 4)
print(f"c = {c}")

# 4. Умножение на скаляр
a_doubled = a * 2
print(f"a * 2 = {a_doubled}")

# 5. Сумма элементов
c_sum = c.sum()
print(f"Сумма элементов c: {c_sum.item():.4f}")
```
</details>

---

### **Задача 2: Изменение формы тензоров**
Дан тензор `x = torch.arange(24)` (числа от 0 до 23).

1. Преобразуйте его в матрицу размера 4x6
2. Преобразуйте его в трехмерный тензор размера 2x3x4
3. Извлеките вторую строку из матрицы 4x6
4. Транспонируйте матрицу 4x6
5. "Сплющите" трехмерный тензор в одномерный вектор

**Подсказка:** Используйте `.reshape()`, `.view()`, индексацию `[i]`, `.T`, `.flatten()`

<details>
<summary>Решение</summary>

```python
import torch

x = torch.arange(24)

# 1. Матрица 4x6
matrix_4x6 = x.reshape(4, 6)
print(f"Матрица 4x6:\n{matrix_4x6}")

# 2. Тензор 2x3x4
tensor_3d = x.reshape(2, 3, 4)
print(f"Тензор 2x3x4:\n{tensor_3d}")

# 3. Вторая строка (индекс 1)
second_row = matrix_4x6[1]
print(f"Вторая строка: {second_row}")

# 4. Транспонирование
transposed = matrix_4x6.T
print(f"Транспонированная матрица 6x4:\n{transposed}")

# 5. Сплющивание
flattened = tensor_3d.flatten()
print(f"Сплющенный тензор: {flattened}")
```
</details>

---

### **Задача 3: Матричные операции**
1. Создайте две матрицы: `A` размера 3x2 и `B` размера 2x3 со случайными значениями
2. Выполните матричное умножение `A @ B` и `B @ A`
3. Какие размеры у получившихся матриц?
4. Создайте матрицу `C = A @ B` и найдите её след (сумму диагональных элементов)

**Подсказка:** Используйте оператор `@` или `torch.matmul()`, `torch.trace()`

<details>
<summary>Решение</summary>

```python
import torch

# 1. Создание матриц
A = torch.randn(3, 2)
B = torch.randn(2, 3)
print(f"A (3x2):\n{A}")
print(f"B (2x3):\n{B}")

# 2. Матричные умножения
C = A @ B  # (3x2) @ (2x3) = (3x3)
D = B @ A  # (2x3) @ (3x2) = (2x2)
print(f"\nA @ B (3x3):\n{C}")
print(f"\nB @ A (2x2):\n{D}")

# 3. Размеры
print(f"\nРазмер C: {C.shape}")
print(f"Размер D: {D.shape}")

# 4. След матрицы C
trace_C = torch.trace(C)
print(f"\nСлед C: {trace_C.item():.4f}")
```
</details>

---

## **🟡 Продвинутый уровень**

### **Задача 4: Автоматическое дифференцирование**
Вычислите градиенты функции `f(x, y) = x^2 + 2xy + y^2` в точке (x=3, y=2).

1. Создайте тензоры `x` и `y` с `requires_grad=True`
2. Вычислите функцию `f`
3. Вызовите `.backward()` для вычисления градиентов
4. Выведите значения `∂f/∂x` и `∂f/∂y`
5. Проверьте результаты вручную (∂f/∂x = 2x + 2y, ∂f/∂y = 2x + 2y)

<details>
<summary>Решение</summary>

```python
import torch

# 1. Создание тензоров с градиентами
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

# 2. Вычисление функции
f = x**2 + 2*x*y + y**2
print(f"f(3, 2) = {f.item()}")

# 3. Вычисление градиентов
f.backward()

# 4. Вывод градиентов
print(f"∂f/∂x = {x.grad.item()}")  # Должно быть 2*3 + 2*2 = 10
print(f"∂f/∂y = {y.grad.item()}")  # Должно быть 2*3 + 2*2 = 10

# 5. Проверка
print(f"\nПроверка:")
print(f"2x + 2y = 2*{x.item()} + 2*{y.item()} = {2*x.item() + 2*y.item()}")
```
</details>

---

### **Задача 5: Градиентный спуск вручную**
Реализуйте один шаг градиентного спуска для функции `f(w) = (w - 5)^2`.

1. Начните с `w = 0.0` с отслеживанием градиента
2. Вычислите функцию `f(w)`
3. Вычислите градиент
4. Обновите `w` по формуле: `w_new = w - learning_rate * grad`
5. Используйте `learning_rate = 0.1`
6. Повторите 50 итераций и выведите финальное значение `w`

**Подсказка:** Используйте `with torch.no_grad():` для обновления весов и `.zero_()` для обнуления градиентов

<details>
<summary>Решение</summary>

```python
import torch

# Параметры
w = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.1

# Градиентный спуск
for i in range(50):
    # Forward pass
    f = (w - 5)**2
    
    # Backward pass
    if w.grad is not None:
        w.grad.zero_()
    f.backward()
    
    # Update weights (без отслеживания градиента)
    with torch.no_grad():
        w -= learning_rate * w.grad
    
    if i % 10 == 0:
        print(f"Итерация {i}: w = {w.item():.4f}, f(w) = {f.item():.4f}")

print(f"\nФинальное значение w = {w.item():.4f} (должно быть близко к 5.0)")
```
</details>

---

### **Задача 6: Работа с torch.no_grad()**
Объясните и продемонстрируйте разницу между вычислениями с градиентами и без.

1. Создайте тензор `x` со значением 2.0 и `requires_grad=True`
2. Вычислите `y1 = x ** 2` (с градиентами)
3. Вычислите `y2 = x ** 2` внутри `torch.no_grad()` (без градиентов)
4. Выведите `.requires_grad` для обоих результатов
5. Попробуйте вызвать `.backward()` для обоих (что произойдет?)

<details>
<summary>Решение</summary>

```python
import torch

# 1. Создание тензора
x = torch.tensor(2.0, requires_grad=True)

# 2. Вычисление с градиентами
y1 = x ** 2
print(f"y1 = {y1.item()}")
print(f"y1.requires_grad = {y1.requires_grad}")  # True

# 3. Вычисление без градиентов
with torch.no_grad():
    y2 = x ** 2
print(f"\ny2 = {y2.item()}")
print(f"y2.requires_grad = {y2.requires_grad}")  # False

# 4. Backward для y1
y1.backward()
print(f"\nГрадиент x после y1.backward(): {x.grad.item()}")  # 4.0

# 5. Попытка backward для y2
try:
    y2.backward()
except RuntimeError as e:
    print(f"\nОшибка при y2.backward(): {str(e)[:50]}...")
    print("y2 не отслеживает градиенты!")
```
</details>

---

## **🔴 Экспертный уровень**

### **Задача 7: Пользовательская функция активации**
Создайте пользовательскую функцию активации Swish: `f(x) = x * sigmoid(x)`.

1. Реализуйте класс `SwishFunction`, наследующийся от `torch.autograd.Function`
2. Реализуйте метод `forward`: сохраните входной тензор и верните `x * sigmoid(x)`
3. Реализуйте метод `backward`: вычислите градиент Swish
   - Градиент: `f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))`
4. Примените функцию к тензору и вычислите градиенты

**Подсказка:** Используйте `torch.sigmoid()`, `ctx.save_for_backward()`, `ctx.saved_tensors`

<details>
<summary>Решение</summary>

```python
import torch

class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Сохраняем входной тензор для backward
        ctx.save_for_backward(input)
        
        # Вычисляем Swish: x * sigmoid(x)
        sigmoid_x = torch.sigmoid(input)
        return input * sigmoid_x
    
    @staticmethod
    def backward(ctx, grad_output):
        # Восстанавливаем входной тензор
        input, = ctx.saved_tensors
        
        # Вычисляем sigmoid(x)
        sigmoid_x = torch.sigmoid(input)
        
        # Градиент Swish: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        grad_swish = sigmoid_x + input * sigmoid_x * (1 - sigmoid_x)
        
        # Применяем правило цепочки
        grad_input = grad_output * grad_swish
        
        return grad_input

# Применение
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
y = SwishFunction.apply(x)

print(f"x = {x}")
print(f"Swish(x) = {y}")

# Вычисляем градиенты
y.backward(torch.ones_like(y))
print(f"\nГрадиенты: {x.grad}")
```
</details>

---

### **Задача 8: Градиенты второго порядка**
Вычислите первую и вторую производные функции `f(x) = x^3 - 3x^2 + 2x` в точке x=2.

1. Создайте тензор `x = 2.0` с `requires_grad=True`
2. Вычислите `f(x)`
3. Вычислите первую производную используя `torch.autograd.grad()` с `create_graph=True`
4. Вычислите вторую производную
5. Проверьте результаты: f'(x) = 3x^2 - 6x + 2, f''(x) = 6x - 6

<details>
<summary>Решение</summary>

```python
import torch

# 1. Создание тензора
x = torch.tensor(2.0, requires_grad=True)

# 2. Вычисление функции
f = x**3 - 3*x**2 + 2*x
print(f"f(2) = {f.item()}")

# 3. Первая производная
df_dx = torch.autograd.grad(f, x, create_graph=True)[0]
print(f"f'(2) = {df_dx.item()}")

# 4. Вторая производная
d2f_dx2 = torch.autograd.grad(df_dx, x)[0]
print(f"f''(2) = {d2f_dx2.item()}")

# 5. Проверка
x_val = 2.0
expected_first = 3*x_val**2 - 6*x_val + 2
expected_second = 6*x_val - 6
print(f"\nПроверка:")
print(f"Ожидаемая f'(2) = 3*4 - 12 + 2 = {expected_first}")
print(f"Ожидаемая f''(2) = 12 - 6 = {expected_second}")
```
</details>

---

### **Задача 9: Градиентный спуск для линейной регрессии**
Реализуйте обучение линейной регрессии с использованием оптимизатора.

1. Сгенерируйте синтетические данные: `y = 2.5*x + 1.3 + шум`
2. Инициализируйте параметры `w` и `b` случайными значениями
3. Используйте `torch.optim.SGD` для обновления параметров, выберите learning rate = 0.1.
4. Обучите модель в течение 100 эпох
5. Выведите финальные значения `w` и `b`
6. Подберите количество эпох, чтобы w и b стали равны 2.5 и 1.3 соответственно

<details>
<summary>Решение</summary>

```python
import torch
import matplotlib.pyplot as plt

# 1. Генерация синтетических данных
torch.manual_seed(42)
n_samples = 100
x = torch.linspace(0, 10, n_samples)
y_true = 2.5 * x + 1.3
y_noisy = y_true + 0.5 * torch.randn(n_samples)

# 2. Инициализация параметров
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# 3. Создание оптимизатора
optimizer = torch.optim.SGD([w, b], lr=0.01)

# 4. Обучение
losses = []
for epoch in range(100):
    # Forward pass
    y_pred = w * x + b
    loss = torch.mean((y_pred - y_noisy)**2)
    losses.append(loss.item())
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")

# 5. Финальные результаты
print(f"\nФинальные параметры:")
print(f"w = {w.item():.4f} (истинное значение: 2.5)")
print(f"b = {b.item():.4f} (истинное значение: 1.3)")

# Визуализация
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(x.numpy(), y_noisy.numpy(), alpha=0.5, label='Данные')
plt.plot(x.numpy(), y_pred.detach().numpy(), 'r-', linewidth=2, label='Прогноз')
plt.plot(x.numpy(), y_true.numpy(), 'g--', linewidth=2, label='Истинная зависимость')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Линейная регрессия')

plt.subplot(1, 2, 2)
plt.plot(losses)
plt.xlabel('Эпоха')
plt.ylabel('MSE Loss')
plt.title('Кривая обучения')
plt.grid(True)

plt.tight_layout()
plt.savefig('linear_regression_result.png', dpi=100)
print("\nГрафик сохранен в 'linear_regression_result.png'")
```
</details>

---

## **💡 Дополнительные задачи**

### **Задача 10: Отладка градиентов**
Найдите ошибку в следующем коде:

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = torch.mean(y)

z.backward()
print(f"Градиент после первого backward: {x.grad}")

z.backward()  # Вторая попытка
print(f"Градиент после второго backward: {x.grad}")
```

Что произойдет? Как это исправить?

<details>
<summary>Решение</summary>

**Проблема:** При втором вызове `.backward()` произойдет ошибка, так как граф вычислений уже был освобожден.

**Решение 1:** Обнулить градиенты и пересчитать:
```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = torch.mean(y)

z.backward()
print(f"Градиент после первого backward: {x.grad}")

# Обнуляем градиенты
x.grad.zero_()

# Пересчитываем функцию
y = x ** 2
z = torch.mean(y)
z.backward()
print(f"Градиент после второго backward: {x.grad}")
```

**Решение 2:** Использовать `retain_graph=True` (не рекомендуется для обучения):
```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = torch.mean(y)

z.backward(retain_graph=True)
print(f"Градиент после первого backward: {x.grad}")

x.grad.zero_()
z.backward()  
print(f"Градиент после второго backward: {x.grad}")
```
</details>

---

### **Задача 11: Broadcasting в ML — Реалистичные примеры**
Изучите применение broadcasting в типичных задачах машинного обучения.

#### **Пример 1: Добавление bias к батчу данных**

В нейронных сетях bias добавляется ко всем примерам в батче одновременно.

```python
import torch

# Батч из 32 примеров, каждый с 10 признаками
batch_size = 32
features = 10
X = torch.randn(batch_size, features)

# Веса и bias для одного нейрона
w = torch.randn(features, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)  # Один bias для всех примеров

# Forward pass: X @ w дает (32, 1), b (1,) broadcasting до (32, 1)
output = X @ w + b  # Broadcasting: (32, 1) + (1,) -> (32, 1)

print(f"X.shape: {X.shape}")
print(f"w.shape: {w.shape}")
print(f"b.shape: {b.shape}")
print(f"output.shape: {output.shape}")

# Вычисление градиентов
loss = output.sum()
loss.backward()

print(f"\nb.grad: {b.grad}")  # Сумма градиентов по всему батчу
print("Bias получает градиент = сумма по всем 32 примерам")
```

#### **Пример 2: Batch Normalization**

Нормализация батча использует broadcasting для вычитания среднего и деления на std.

```python
import torch

# Батч из 64 примеров, каждый с 128 признаками
batch_size = 64
features = 128
X = torch.randn(batch_size, features)

# Вычисление статистик по батчу (для каждого признака отдельно)
mean = X.mean(dim=0, keepdim=True)  # (1, 128)
std = X.std(dim=0, keepdim=True)    # (1, 128)

# Нормализация: broadcasting mean и std на весь батч
X_normalized = (X - mean) / (std + 1e-5)

print(f"X.shape: {X.shape}")
print(f"mean.shape: {mean.shape}")
print(f"X_normalized.shape: {X_normalized.shape}")

# Проверка: среднее должно быть ~0, std ~1 для каждого признака
print(f"\nСреднее после нормализации: {X_normalized.mean(dim=0).abs().max().item():.6f}")
print(f"Std после нормализации: {X_normalized.std(dim=0).mean().item():.4f}")
```

#### **Пример 3: Attention Mask**

В трансформерах маска внимания применяется ко всем батчам и головам одновременно.

```python
import torch

# Attention scores: (batch_size, num_heads, seq_len, seq_len)
batch_size = 8
num_heads = 12
seq_len = 20

attention_scores = torch.randn(batch_size, num_heads, seq_len, seq_len)

# Создаем causal mask (нижнетреугольная матрица)
# Форма (seq_len, seq_len), будет broadcast на (batch_size, num_heads, seq_len, seq_len)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

# Применяем маску: broadcasting (20, 20) -> (8, 12, 20, 20)
attention_scores_masked = attention_scores.masked_fill(mask, float('-inf'))

print(f"attention_scores.shape: {attention_scores.shape}")
print(f"mask.shape: {mask.shape}")
print(f"attention_scores_masked.shape: {attention_scores_masked.shape}")
print(f"\nМаска применяется ко всем батчам и головам одновременно!")
```

#### **Пример 4: Вычисление попарных расстояний**

В k-NN или кластеризации нужно вычислить расстояния между всеми парами точек.

```python
import torch

# Два набора точек: X (100 точек в 3D) и Y (50 точек в 3D)
X = torch.randn(100, 3)
Y = torch.randn(50, 3)

# Эффективное вычисление попарных расстояний через broadcasting
# X: (100, 1, 3), Y: (1, 50, 3) -> разница: (100, 50, 3)
distances = torch.sqrt(((X.unsqueeze(1) - Y.unsqueeze(0)) ** 2).sum(dim=2))

print(f"X.shape: {X.shape}")
print(f"Y.shape: {Y.shape}")
print(f"distances.shape: {distances.shape}")  # (100, 50)
print(f"\ndistances[i, j] = расстояние от X[i] до Y[j]")

# Найдем ближайшего соседа для каждой точки из X
nearest_neighbors = distances.argmin(dim=1)
print(f"\nБлижайшие соседи для первых 5 точек X: {nearest_neighbors[:5]}")
```

#### **Пример 5: Применение dropout по признакам**

В некоторых архитектурах (например, DropConnect) dropout применяется по-разному к разным измерениям.

```python
import torch

# Батч из 32 примеров, каждый с 256 признаками
batch_size = 32
features = 256
X = torch.randn(batch_size, features)

# Создаем маску dropout только для признаков (одинаковую для всего батча)
dropout_rate = 0.5
mask = (torch.rand(1, features) > dropout_rate).float()  # (1, 256)

# Применяем маску через broadcasting
X_dropped = X * mask / (1 - dropout_rate)  # Broadcasting (32, 256) * (1, 256)

print(f"X.shape: {X.shape}")
print(f"mask.shape: {mask.shape}")
print(f"X_dropped.shape: {X_dropped.shape}")

# Проверяем, сколько признаков обнулились
num_dropped = (mask == 0).sum().item()
print(f"\nОбнулено признаков: {num_dropped} из {features}")
print("Одинаковые признаки обнулены для всех примеров в батче")
```

**Задания для практики:**
1. Реализуйте Layer Normalization используя broadcasting (нормализация по признакам для каждого примера)
2. Создайте функцию для вычисления косинусного сходства между всеми парами векторов в двух батчах
3. Реализуйте positional encoding для трансформера с broadcasting

---

### **Задача 12: Train/Val Split и Early Stopping**
Реализуйте обучение с валидацией и ранней остановкой.

1. Создайте синтетический датасет из 500 примеров
2. Разделите на train (80%) и validation (20%)
3. Реализуйте early stopping: останавливайте обучение, если val_loss не улучшается 10 эпох
4. Сохраните лучшие веса модели

<details>
<summary>Решение</summary>

```python
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

torch.manual_seed(42)

# Генерация данных: y = 3x1 - 2x2 + x3 + шум
n_samples = 500
X = torch.randn(n_samples, 3)
y = 3*X[:, 0] - 2*X[:, 1] + X[:, 2] + 0.5*torch.randn(n_samples)

# Создание датасета и разделение
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Модель
w = torch.randn(3, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.Adam([w, b], lr=0.01)

# Early stopping параметры
best_val_loss = float('inf')
patience = 10
patience_counter = 0
best_w = None
best_b = None

# Обучение
for epoch in range(200):
    # Training
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        y_pred = batch_X @ w + b
        loss = torch.mean((y_pred.squeeze() - batch_y)**2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            y_pred = batch_X @ w + b
            loss = torch.mean((y_pred.squeeze() - batch_y)**2)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_w = w.clone().detach()
        best_b = b.clone().detach()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, Patience={patience_counter}")
    
    # Остановка при превышении patience
    if patience_counter >= patience:
        print(f"\nEarly stopping на эпохе {epoch}")
        break

# Восстановление лучших весов
w.data = best_w
b.data = best_b

print(f"\nЛучший Val Loss: {best_val_loss:.4f}")
print(f"Финальные веса: {w.squeeze().detach()}")
print(f"Истинные веса: [3.0, -2.0, 1.0]")
```
</details>

---

## **📚 Рекомендации для самостоятельного изучения**

1. **Практикуйте ежедневно:** Решайте хотя бы одну задачу в день
2. **Экспериментируйте:** Меняйте параметры и смотрите на результаты
3. **Визуализируйте:** Используйте matplotlib для понимания процессов
4. **Читайте документацию:** [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
5. **Пробуйте свои задачи:** Придумывайте собственные функции и вычисляйте их градиенты

**Дополнительные ресурсы:**
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [PyTorch Examples](https://github.com/pytorch/examples)
