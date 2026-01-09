# **GPU в PyTorch: Ускорение вычислений**

## **Введение**
GPU (Graphics Processing Unit) обеспечивает значительное ускорение обучения нейронных сетей благодаря параллельной обработке данных. PyTorch предоставляет простой и гибкий API для работы с GPU.

**Почему GPU важны для глубокого обучения?**
- ⚡ **Параллелизм:** Тысячи ядер для одновременной обработки
- 🚀 **Ускорение:** 10-100x быстрее по сравнению с CPU для больших моделей
- 💾 **Пропускная способность памяти:** Быстрый доступ к данным

---

## **🟢 Базовый уровень: Работа с GPU**

### **1.1 Проверка доступности GPU**
```python
import torch

# Проверка доступности CUDA (NVIDIA GPU)
print(f"CUDA доступна: {torch.cuda.is_available()}")

# Количество доступных GPU
print(f"Количество GPU: {torch.cuda.device_count()}")

# Название GPU
if torch.cuda.is_available():
    print(f"Текущий GPU: {torch.cuda.get_device_name(0)}")
```

### **1.2 Создание тензоров на GPU**
```python
# Способ 1: Создание тензора напрямую на GPU
tensor_gpu = torch.tensor([1.0, 2.0, 3.0], device='cuda')

# Способ 2: Перенос существующего тензора на GPU
tensor_cpu = torch.tensor([1.0, 2.0, 3.0])
tensor_gpu = tensor_cpu.to('cuda')

# Способ 3: Использование device для гибкости
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = torch.randn(3, 3).to(device)
```

### **1.3 Базовые операции на GPU**
```python
# Создаем тензоры на GPU
a = torch.randn(1000, 1000, device='cuda')
b = torch.randn(1000, 1000, device='cuda')

# Все операции выполняются на GPU
c = a @ b  # матричное умножение
d = torch.sum(c)  # суммирование
e = torch.relu(c)  # функция активации

# Проверка, на каком устройстве находится тензор
print(f"Тензор c находится на: {c.device}")
```

---

## **🟡 Продвинутый уровень: Оптимизация работы с GPU**

### **2.1 Перенос данных между CPU и GPU**
```python
# CPU → GPU
tensor_cpu = torch.randn(100, 100)
tensor_gpu = tensor_cpu.to('cuda')  # копирование данных на GPU

# GPU → CPU
tensor_back = tensor_gpu.to('cpu')  # копирование данных на CPU
# или
tensor_back = tensor_gpu.cpu()  # альтернативный синтаксис

# Важно: .to() создает копию, если устройства разные
x = torch.tensor([1, 2, 3], device='cuda')
y = x.to('cuda')  # не создает копию, возвращает x
z = x.to('cpu')   # создает копию на CPU
```

### **2.2 Смешанные вычисления (CPU + GPU)**
```python
# ❌ ОШИБКА: тензоры на разных устройствах
a = torch.tensor([1.0, 2.0], device='cuda')
b = torch.tensor([3.0, 4.0], device='cpu')
# c = a + b  # RuntimeError: Expected all tensors to be on the same device

# ✅ ПРАВИЛЬНО: переносим на одно устройство
c = a + b.to('cuda')
# или
c = a.cpu() + b
```

### **2.3 Контекстный менеджер для устройства**
```python
# Установка устройства по умолчанию для блока кода
with torch.cuda.device(0):  # использовать GPU 0
    x = torch.randn(100, 100)  # создается на CPU
    y = x.cuda()  # переносится на текущий GPU

# Альтернатива: явное указание устройства
device = torch.device('cuda:0')  # первый GPU
x = torch.randn(100, 100, device=device)
```

### **2.4 Работа с несколькими GPU**
```python
# Проверка наличия нескольких GPU
if torch.cuda.device_count() > 1:
    print(f"Доступно {torch.cuda.device_count()} GPU")
    
    # Создание тензоров на разных GPU
    tensor_gpu0 = torch.randn(100, 100, device='cuda:0')
    tensor_gpu1 = torch.randn(100, 100, device='cuda:1')
    
    # Перенос между GPU
    tensor_gpu0_copy = tensor_gpu1.to('cuda:0')
```

---

## **🔴 Экспертный уровень: Продвинутые техники**

### **3.1 Профилирование производительности GPU**
```python
import time

# CPU
start = time.time()
x_cpu = torch.randn(5000, 5000)
y_cpu = torch.randn(5000, 5000)
z_cpu = x_cpu @ y_cpu
cpu_time = time.time() - start

# GPU
if torch.cuda.is_available():
    start = time.time()
    x_gpu = torch.randn(5000, 5000, device='cuda')
    y_gpu = torch.randn(5000, 5000, device='cuda')
    torch.cuda.synchronize()  # ждем завершения операций на GPU
    z_gpu = x_gpu @ y_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f"CPU: {cpu_time:.4f}s")
    print(f"GPU: {gpu_time:.4f}s")
    print(f"Ускорение: {cpu_time/gpu_time:.2f}x")
```

### **3.2 Управление памятью GPU**
```python
# Проверка использования памяти GPU
if torch.cuda.is_available():
    print(f"Выделено памяти: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Зарезервировано памяти: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Очистка кэша GPU
    torch.cuda.empty_cache()
    
    # Освобождение памяти
    # del удаляет Python-ссылку на объект
    # Память освобождается только когда удалены все ссылки на тензор
    del tensor_gpu  # удаление ссылки
    torch.cuda.empty_cache()  # очистка кэша
```

### **3.3 Асинхронные операции**
```python
# GPU операции асинхронные относительно CPU (выполняются на отдельном потоке)
# Операции на GPU выполняются в порядке очереди в своем потоке (stream)
x = torch.randn(1000, 1000, device='cuda')
y = torch.randn(1000, 1000, device='cuda')

# Эта операция добавляется в очередь GPU
z = x @ y

# Результат доступен автоматически (PyTorch синхронизируется при доступе)
# Для точного измерения времени нужна явная синхронизация
torch.cuda.synchronize()  # ждем завершения всех операций в очереди GPU

print(z.sum().item())
```

### **3.4 Pinned Memory для ускорения переноса**
```python
# Pinned (page-locked) память ускоряет перенос CPU ↔ GPU
tensor_cpu = torch.randn(1000, 1000)
tensor_pinned = torch.randn(1000, 1000).pin_memory()

# Перенос pinned памяти на GPU быстрее
tensor_gpu = tensor_pinned.to('cuda', non_blocking=True)
```

---

## **🚀 Практический пример: Обучение модели на GPU**

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Установка устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

# Генерация данных
x = torch.linspace(0, 1, 100).unsqueeze(1).to(device)
y_true = 1.5 * x + 0.8
y_noisy = y_true + 0.1 * torch.randn_like(x)

# Определение модели
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

# Создание модели и перенос на GPU
model = LinearModel().to(device)

# Оптимизатор и функция потерь
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

# Обучение на GPU
for epoch in range(100):
    # Forward pass
    y_pred = model(x)
    loss = criterion(y_pred, y_noisy)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.4f}")

# Визуализация (переносим данные на CPU для matplotlib)
x_cpu = x.cpu()
y_pred_cpu = model(x).detach().cpu()
y_noisy_cpu = y_noisy.cpu()

plt.scatter(x_cpu, y_noisy_cpu, label='Данные', alpha=0.5)
plt.plot(x_cpu, y_pred_cpu, 'r-', label='Прогноз', linewidth=2)
plt.legend()
plt.title('Линейная регрессия на GPU')
plt.show()
```

---

## **⚡ Продвинутая оптимизация: Mixed Precision Training**

### **Автоматическое смешанное точность (AMP)**
Mixed Precision Training использует float16 для ускорения вычислений при сохранении точности.

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# Проверка доступности
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Модель
model = nn.Sequential(
    nn.Linear(1000, 500),
    nn.ReLU(),
    nn.Linear(500, 10)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Scaler для автоматического масштабирования градиентов
scaler = GradScaler()

# Обучение с Mixed Precision
x = torch.randn(128, 1000, device=device)
y = torch.randint(0, 10, (128,), device=device)

for epoch in range(10):
    optimizer.zero_grad()
    
    # autocast автоматически использует float16 где возможно
    with autocast():
        outputs = model(x)
        loss = criterion(outputs, y)
    
    # Масштабирование градиентов для предотвращения underflow
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.4f}")

print("\n✅ Mixed Precision ускоряет обучение в 2-3 раза!")
```

### **PyTorch 2.0+: torch.compile()**
```python
# Компиляция модели для дополнительного ускорения (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    model = torch.compile(model)
    print("✅ Модель скомпилирована с torch.compile()")
```

### **DataParallel: Простое multi-GPU обучение**
```python
import torch
import torch.nn as nn

# Проверка наличия нескольких GPU
if torch.cuda.device_count() > 1:
    print(f"Используем {torch.cuda.device_count()} GPU!")
    
    # Создание модели
    model = nn.Sequential(
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 10)
    )
    
    # Обертывание модели для параллельного выполнения
    model = nn.DataParallel(model)
    model = model.cuda()
    
    # Обучение автоматически распределяется по GPU
    x = torch.randn(128, 1000).cuda()
    output = model(x)
    print(f"Output shape: {output.shape}")
else:
    print("Доступен только 1 GPU или GPU недоступен")
```

**Примечание:** Для production используйте `DistributedDataParallel` (более эффективно).

---

## **💡 Лучшие практики**

### **Общие рекомендации:**
1. **Проверяйте доступность GPU:**
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

2. **Переносите модель и данные на одно устройство:**
   ```python
   model = model.to(device)
   data = data.to(device)
   ```

3. **Используйте `.detach()` и `.cpu()` перед визуализацией:**
   ```python
   plt.plot(x.cpu().numpy(), y.detach().cpu().numpy())
   ```

4. **Синхронизируйте при измерении времени:**
   ```python
   torch.cuda.synchronize()
   start = time.time()
   # ... операции ...
   torch.cuda.synchronize()
   elapsed = time.time() - start
   ```

5. **Очищайте память при нехватке:**
   ```python
   del large_tensor
   torch.cuda.empty_cache()
   ```

### **Типичные ошибки:**

❌ **Ошибка 1: Операции с тензорами на разных устройствах**
```python
x = torch.tensor([1, 2], device='cuda')
y = torch.tensor([3, 4], device='cpu')
z = x + y  # RuntimeError!
```

✅ **Решение:**
```python
z = x + y.to('cuda')  # или x.cpu() + y
```

❌ **Ошибка 2: Забыли перенести модель на GPU**
```python
model = LinearModel()  # модель на CPU
data = data.to('cuda')
output = model(data)  # RuntimeError!
```

✅ **Решение:**
```python
model = LinearModel().to('cuda')
data = data.to('cuda')
output = model(data)  # Работает!
```

❌ **Ошибка 3: Утечка памяти GPU**
```python
# Сохранение тензоров с вычислительным графом накапливает память
for epoch in range(1000):
    loss = compute_loss()
    losses.append(loss)  # сохраняем тензор с графом!
```

✅ **Решение:**
```python
for epoch in range(1000):
    loss = compute_loss()
    # .item() извлекает Python-значение (число), которое не содержит граф
    # Сам тензор loss с графом удаляется автоматически в конце итерации
    losses.append(loss.item())  # сохраняем только число
```

---

## **📊 Сравнение производительности: CPU vs GPU**

```python
import time
import torch

def benchmark(device, size=5000, iterations=10):
    """Бенчмарк матричного умножения"""
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)
    
    # Прогрев (для GPU)
    for _ in range(5):
        _ = x @ y
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Измерение
    start = time.time()
    for _ in range(iterations):
        z = x @ y
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    return elapsed / iterations

# Тестирование
cpu_time = benchmark('cpu', size=2000)
print(f"CPU: {cpu_time:.4f}s")

if torch.cuda.is_available():
    gpu_time = benchmark('cuda', size=2000)
    print(f"GPU: {gpu_time:.4f}s")
    print(f"Ускорение: {cpu_time/gpu_time:.1f}x")
```

**Типичные результаты:**
- Малые модели (< 1000 параметров): GPU может быть медленнее из-за overhead
- Средние модели (1000-100k параметров): Ускорение 5-20x
- Большие модели (> 100k параметров): Ускорение 20-100x

---

## **🔧 Отладка GPU-кода**

### **Проверка устройства тензоров:**
```python
def check_device(tensor, name="tensor"):
    """Вывод информации об устройстве тензора"""
    print(f"{name}: device={tensor.device}, dtype={tensor.dtype}, shape={tensor.shape}")

# Пример использования
x = torch.randn(3, 3, device='cuda')
check_device(x, "x")
```

### **Отладка утечек памяти:**
```python
# Мониторинг памяти в процессе обучения
for epoch in range(100):
    # ... обучение ...
    
    if epoch % 10 == 0:
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"Epoch {epoch}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
```

---

## **💎 Заключение**

**Ключевые концепции работы с GPU в PyTorch:**
1. **Устройства** — проверка доступности и выбор устройства (`'cuda'` или `'cpu'`)
2. **Перенос данных** — `.to(device)` для моделей и тензоров
3. **Синхронизация** — `torch.cuda.synchronize()` для точных измерений
4. **Управление памятью** — мониторинг и очистка кэша

**Правила работы с GPU:**
- ✅ Всегда проверяйте доступность GPU перед использованием
- ✅ Переносите модель и данные на одно устройство
- ✅ Используйте `.item()` для извлечения скалярных значений
- ✅ Очищайте память при работе с большими моделями
- ✅ Синхронизируйте GPU при профилировании

> **"GPU — это не просто ускорение, это возможность обучать модели, которые на CPU обучались бы неделями."**

**Дальнейшее изучение:**
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [Multi-GPU Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
