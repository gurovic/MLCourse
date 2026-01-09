# **PyTorch Basics: Тензоры и Autograd**  

## **Введение в PyTorch**  
PyTorch — фреймворк для глубокого обучения с фокусом на гибкость и скорость.  
**Ключевые особенности:**  
- ⚡ **Тензоры** — многомерные массивы для эффективных вычислений  
- 🔄 **Autograd** — автоматическое вычисление градиентов  
- 🧩 **Динамический граф вычислений** — изменение структуры сети на лету  

**Зачем учить PyTorch?**  
- Интуитивный Python-like API  
- Широкое применение в исследованиях (90% статей на NeurIPS)  
- Поддержка промышленного развертывания (TorchServe, TorchScript)  

---

## **🟢 Базовый уровень: Работа с тензорами**  

### **1.1 Создание тензоров**  
```python
import torch

# Создание из списка
tensor_a = torch.tensor([1, 2, 3])  # вектор [1, 2, 3]

# Специальные тензоры
zeros = torch.zeros(2, 3)       # матрица 2x3 из нулей  
rand_matrix = torch.rand(3, 3)  # случайные значения 0-1

# Явное указание типа данных (рекомендуется)
tensor_float = torch.tensor([1.0, 2.0], dtype=torch.float32)
tensor_int = torch.tensor([1, 2], dtype=torch.int64)
```

### **1.2 Операции с тензорами**  
```python
# requires_grad=True включает автоматическое отслеживание операций для вычисления градиентов
a = torch.tensor([1.0, 2.0], requires_grad=True)
b = torch.tensor([3.0, 4.0], requires_grad=True)

# Базовые операции
c = a + b            # поэлементное сложение [4.0, 6.0]
d = torch.dot(a, b)  # скалярное произведение 1*3 + 2*4 = 11.0

# Матричные операции
mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 2)
mat_mul = mat1 @ mat2  # матричное умножение (современный синтаксис)
```

### **1.3 Индексация и изменение формы**  
```python
tensor = torch.arange(12).reshape(3, 4)  # матрица 3x4

# Индексация
row = tensor[1]        # вторая строка [4, 5, 6, 7]
element = tensor[0, 2] # элемент (1,3) → 2

# Изменение формы
flattened = tensor.flatten()  # вектор из 12 элементов
transposed = tensor.T         # транспонированная матрица 4x3
```

---

## **🟡 Продвинутый уровень: Autograd в действии**  

### **2.1 Как работает автоматическое дифференцирование**  
```python
# Создаем тензоры с флагом отслеживания градиентов
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(0.7, requires_grad=True)

# Вычисляем функцию
y = w * x + b  # линейная функция

# Вычисляем градиенты
y.backward()  # автоматическое дифференцирование

print(f"dy/dw = {w.grad}")  # 2.0 (x)
print(f"dy/db = {b.grad}")  # 1.0
```

### **2.2 Вычисление градиентов для сложных функций**  
```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
z = torch.prod(x)  # z = x1 * x2 = 2.0

z.backward()  # dz/dx1 = x2 = 2.0, dz/dx2 = x1 = 1.0
print(x.grad)  # [2.0, 1.0]
```

### **2.3 Контроль потока вычислений**  
```python
# Отключаем отслеживание градиентов
with torch.no_grad():
    y = x * 2  # операции не будут записаны в граф

# Ручное управление памятью градиентов (используется с оптимизатором)
# optimizer.zero_grad()  # обнуляем градиенты перед новым backward()
# См. раздел "Практический пример: Линейная регрессия" для полного примера
```

---

## **🔴 Экспертный уровень: Динамические графы**  

### **3.1 Пользовательские функции с autograd**  
```python
class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Использование
x = torch.randn(4, requires_grad=True)
y = CustomReLU.apply(x)
y.backward(torch.ones_like(y))
```

### **3.2 Градиенты второго порядка**  
```python
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x

# Первая производная
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]

# Вторая производная
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(f"d²y/dx² = {d2y_dx2.item()}")  # 2.0
```

### **3.3 Отладка графа вычислений**  
```python
# Визуализация графа (требуется torchviz)
from torchviz import make_dot

x = torch.tensor(2.0, requires_grad=True)
y = x**3 + torch.sin(x)
make_dot(y).render("graph", format="png")  # сохраняет граф в PNG
```

---

## **🚀 Практический пример: Линейная регрессия**  

```python
import torch
import matplotlib.pyplot as plt

# Установка seed для воспроизводимости
torch.manual_seed(42)

# Данные: y = 1.5*x + 0.8 + шум
x = torch.linspace(0, 1, 100)
y_true = 1.5 * x + 0.8
y_noisy = y_true + 0.1 * torch.randn_like(x)

# Параметры модели
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Обучение
optimizer = torch.optim.SGD([w, b], lr=0.1)
for epoch in range(100):
    y_pred = w * x + b
    loss = torch.mean((y_pred - y_noisy)**2)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: w={w.item():.3f}, b={b.item():.3f}")

# Результат
plt.scatter(x, y_noisy, label='Данные')
plt.plot(x, y_pred.detach(), 'r-', label='Прогноз')
plt.legend()
plt.show()
```

---

## **🔄 Расширенный пример: Обучение с DataLoader**

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Установка seed
torch.manual_seed(42)

# Генерация большего датасета
n_samples = 1000
x_data = torch.randn(n_samples, 5)  # 5 признаков
weights_true = torch.tensor([[2.0], [-1.5], [0.5], [3.0], [-2.5]])
y_data = x_data @ weights_true + 0.5 * torch.randn(n_samples, 1)

# Создание DataLoader для батч-обработки
dataset = TensorDataset(x_data, y_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Модель с несколькими параметрами
w = torch.randn(5, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Оптимизатор
optimizer = torch.optim.Adam([w, b], lr=0.01)

# Обучение с батчами
for epoch in range(20):
    epoch_loss = 0.0
    for batch_x, batch_y in dataloader:
        # Forward pass
        y_pred = batch_x @ w + b
        loss = torch.mean((y_pred - batch_y)**2)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if epoch % 5 == 0:
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}: Avg Loss={avg_loss:.4f}")

print(f"\nИтоговые веса:\n{w.detach()}")
print(f"Истинные веса:\n{weights_true}")
```

---

## **📊 Пример с валидацией: Train/Val split**

```python
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.manual_seed(42)

# Генерация данных
n_samples = 1000
x_data = torch.randn(n_samples, 3)
y_data = 2*x_data[:, 0] - 3*x_data[:, 1] + x_data[:, 2] + torch.randn(n_samples)

# Создание датасета и разделение на train/val (80/20)
dataset = TensorDataset(x_data, y_data)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader для обоих наборов
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Простая модель
w = torch.randn(3, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.Adam([w, b], lr=0.01)

# Обучение с валидацией
best_val_loss = float('inf')
for epoch in range(50):
    # Training
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        y_pred = batch_x @ w + b
        loss = torch.mean((y_pred - batch_y.unsqueeze(1))**2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            y_pred = batch_x @ w + b
            loss = torch.mean((y_pred - batch_y.unsqueeze(1))**2)
            val_loss += loss.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    # Сохранение лучшей модели
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_w = w.clone().detach()
        best_b = b.clone().detach()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

print(f"\nЛучший Val Loss: {best_val_loss:.4f}")
```

---

## **💎 Заключение**  
**Ключевые концепции PyTorch:**  
1. **Тензоры** — основа всех вычислений  
2. **Autograd** — автоматический расчет градиентов для оптимизации  
3. **Динамический граф** — гибкость в построении архитектур  

**Лучшие практики:**  
- Используйте `.detach()` для блокировки градиентов  
- Регулярно вызывайте `.zero_grad()` при обучении  
- Визуализируйте графы для сложных моделей  

**⚠️ Типичные ошибки новичков:**

1. **Забыли обнулить градиенты:**
```python
# ❌ Неправильно
for epoch in range(100):
    loss.backward()
    optimizer.step()
# Градиенты накапливаются!

# ✅ Правильно
for epoch in range(100):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

2. **Использование тензора с графом для логирования:**
```python
# ❌ Неправильно - утечка памяти
losses = []
for epoch in range(100):
    loss = compute_loss()
    losses.append(loss)  # Сохраняет весь граф!

# ✅ Правильно
losses = []
for epoch in range(100):
    loss = compute_loss()
    losses.append(loss.item())  # Только значение
```

3. **Изменение тензора in-place во время backward:**
```python
# ❌ Неправильно
x = torch.tensor([1.0], requires_grad=True)
y = x * 2
y.backward()
x.data.zero_()  # Модифицирует тензор после backward

# ✅ Правильно
with torch.no_grad():
    x -= learning_rate * x.grad
```

> **"PyTorch дает исследователям и инженерам свободу творчества, сочетая простоту Python с мощью вычислительных ресурсов."**  

**Дальнейшее изучение:**  
- [Официальные туториалы PyTorch](https://pytorch.org/tutorials/)  
- [Deep Learning с PyTorch](https://practicaldeeplearning.ai/)  
- [Интерактивный курс Kaggle](https://www.kaggle.com/learn/pytorch)

**🎯 Дополнительные техники:**
- **Градиентный клиппинг:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- **Learning Rate Scheduling:** `torch.optim.lr_scheduler.StepLR`
- **Early Stopping:** Останавливайте обучение, когда val_loss перестает улучшаться
- **Checkpointing:** Сохраняйте модель с `torch.save(model.state_dict(), 'model.pth')`

