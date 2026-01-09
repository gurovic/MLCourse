# **PyTorch Basics: Тензоры и Autograd**  

## **Введение в PyTorch**  
PyTorch — фреймворк для глубокого обучения с фокусом на гибкость и скорость.  
**Ключевые особенности:**  
- ⚡ **Тензоры** — многомерные массивы с GPU-ускорением  
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

# На GPU (ускорение ~10-100x)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor_gpu = rand_matrix.to(device) # создать копию тензора на GPU, если доступно
```

### **1.2 Операции с тензорами**  
```python
a = torch.tensor([1.0, 2.0], requires_grad=True) # Когда вы вызываете .backward(), PyTorch автоматически вычисляет градиенты для всех тензоров с requires_grad=True.
b = torch.tensor([3.0, 4.0], requires_grad=True)

# Базовые операции
c = a + b            # поэлементное сложение [4.0, 6.0]
d = torch.dot(a, b)  # скалярное произведение 1*3 + 2*4 = 11.0

# Матричные операции
mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 2)
mat_mul = torch.mm(mat1, mat2)  # матричное умножение
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

# Ручное управление памятью градиентов
model.zero_grad()  # обнуляем градиенты перед новым backward()
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
import matplotlib.pyplot as plt

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

## **💎 Заключение**  
**Ключевые концепции PyTorch:**  
1. **Тензоры** — основа всех вычислений, поддерживают GPU-ускорение  
2. **Autograd** — автоматический расчет градиентов для оптимизации  
3. **Динамический граф** — гибкость в построении архитектур  

**Лучшие практики:**  
- Используйте `.detach()` для блокировки градиентов  
- Регулярно вызывайте `.zero_grad()` при обучении  
- Визуализируйте графы для сложных моделей  

> **"PyTorch дает исследователям и инженерам свободу творчества, сочетая простоту Python с мощью GPU."**  

**Дальнейшее изучение:**  
- [Официальные туториалы PyTorch](https://pytorch.org/tutorials/)  
- [Deep Learning с PyTorch](https://practicaldeeplearning.ai/)  
- [Интерактивный курс Kaggle](https://www.kaggle.com/learn/pytorch)
