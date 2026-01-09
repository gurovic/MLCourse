# **Задачи: GPU в PyTorch**

## **⚙️ Подготовка**

Перед началом проверьте доступность GPU:
```python
import torch
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA версия: {torch.version.cuda}")
# Если GPU недоступен, используйте Google Colab с GPU runtime
```

---

## **🟢 Базовый уровень**

### **Задача 1: Проверка доступности GPU**
Напишите скрипт, который:
1. Проверяет, доступна ли CUDA
2. Выводит количество доступных GPU
3. Для каждого GPU выводит его название и объем памяти
4. Определяет текущее устройство по умолчанию

**Подсказка:** Используйте `torch.cuda.is_available()`, `torch.cuda.device_count()`, `torch.cuda.get_device_name()`, `torch.cuda.get_device_properties()`

<details>
<summary>Решение</summary>

```python
import torch

print("=" * 50)
print("Информация о GPU")
print("=" * 50)

# 1. Проверка доступности CUDA
cuda_available = torch.cuda.is_available()
print(f"\nCUDA доступна: {cuda_available}")

if cuda_available:
    # 2. Количество GPU
    gpu_count = torch.cuda.device_count()
    print(f"Количество GPU: {gpu_count}")
    
    # 3. Информация о каждом GPU
    for i in range(gpu_count):
        print(f"\n--- GPU {i} ---")
        print(f"Название: {torch.cuda.get_device_name(i)}")
        
        props = torch.cuda.get_device_properties(i)
        print(f"Память: {props.total_memory / 1e9:.2f} GB")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Мультипроцессоры: {props.multi_processor_count}")
    
    # 4. Текущее устройство
    current_device = torch.cuda.current_device()
    print(f"\nТекущее устройство: GPU {current_device}")
else:
    print("\nGPU не доступен. Будет использоваться CPU.")
```
</details>

---

### **Задача 2: Создание тензоров на GPU**
Создайте тензоры тремя разными способами:
1. Создайте тензор напрямую на GPU
2. Создайте тензор на CPU и перенесите на GPU
3. Используйте переменную `device` для гибкости

Для каждого тензора выведите его устройство.

<details>
<summary>Решение</summary>

```python
import torch

# Проверка доступности GPU
if not torch.cuda.is_available():
    print("GPU недоступен. Используйте Google Colab с GPU runtime.")
    device = 'cpu'
else:
    device = 'cuda'

print(f"Используется устройство: {device}\n")

# 1. Создание напрямую на GPU
tensor1 = torch.randn(3, 3, device=device)
print(f"Способ 1 - Напрямую на {device}:")
print(f"Устройство: {tensor1.device}")
print(f"Тензор:\n{tensor1}\n")

# 2. Создание на CPU и перенос
tensor2 = torch.randn(3, 3)  # по умолчанию на CPU
tensor2 = tensor2.to(device)
print(f"Способ 2 - Перенос с CPU на {device}:")
print(f"Устройство: {tensor2.device}")
print(f"Тензор:\n{tensor2}\n")

# 3. Использование переменной device
device_var = torch.device(device)
tensor3 = torch.randn(3, 3).to(device_var)
print(f"Способ 3 - С переменной device:")
print(f"Устройство: {tensor3.device}")
print(f"Тензор:\n{tensor3}")
```
</details>

---

### **Задача 3: Базовые операции на GPU**
Выполните следующие операции на GPU:
1. Создайте две матрицы 1000x1000 на GPU
2. Выполните матричное умножение
3. Найдите максимальный элемент результата
4. Примените функцию ReLU к результату
5. Вычислите среднее значение

**Подсказка:** Все операции должны выполняться на GPU

<details>
<summary>Решение</summary>

```python
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используется: {device}\n")

# 1. Создание матриц на GPU
A = torch.randn(1000, 1000, device=device)
B = torch.randn(1000, 1000, device=device)
print(f"Матрицы A и B созданы на {device}")
print(f"A.device: {A.device}")
print(f"B.device: {B.device}")

# 2. Матричное умножение
C = A @ B
print(f"\nМатричное умножение выполнено")
print(f"C.shape: {C.shape}")
print(f"C.device: {C.device}")

# 3. Максимальный элемент
max_val = torch.max(C)
print(f"\nМаксимальный элемент: {max_val.item():.4f}")

# 4. ReLU
C_relu = torch.relu(C)
print(f"ReLU применен")
print(f"Количество положительных элементов: {(C_relu > 0).sum().item()}")

# 5. Среднее значение
mean_val = torch.mean(C_relu)
print(f"\nСреднее значение после ReLU: {mean_val.item():.4f}")
```
</details>

---

## **🟡 Продвинутый уровень**

### **Задача 4: Сравнение производительности CPU vs GPU**
Сравните скорость выполнения матричного умножения на CPU и GPU.

1. Создайте функцию `benchmark_matmul(device, size, iterations)`
2. Измерьте время для размеров матриц: 100, 500, 1000, 2000, 5000
3. Постройте график зависимости времени от размера
4. Вычислите ускорение (speedup) для каждого размера

**Важно:** Не забывайте использовать `torch.cuda.synchronize()` для GPU!

<details>
<summary>Решение</summary>

```python
import torch
import time
import matplotlib.pyplot as plt

def benchmark_matmul(device, size, iterations=10):
    """Бенчмарк матричного умножения"""
    # Создание тензоров
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)
    
    # Прогрев (для GPU)
    for _ in range(3):
        _ = A @ B
    
    # Синхронизация для GPU
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Измерение времени
    start = time.time()
    for _ in range(iterations):
        C = A @ B
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    return elapsed / iterations

# Размеры матриц для тестирования
sizes = [100, 500, 1000, 2000, 5000]
cpu_times = []
gpu_times = []
speedups = []

print("Бенчмаркинг матричного умножения...")
print("-" * 60)

for size in sizes:
    # CPU
    cpu_time = benchmark_matmul('cpu', size, iterations=5)
    cpu_times.append(cpu_time)
    
    # GPU
    if torch.cuda.is_available():
        gpu_time = benchmark_matmul('cuda', size, iterations=5)
        gpu_times.append(gpu_time)
        speedup = cpu_time / gpu_time
        speedups.append(speedup)
        
        print(f"Размер: {size}x{size}")
        print(f"  CPU: {cpu_time:.4f}s")
        print(f"  GPU: {gpu_time:.4f}s")
        print(f"  Ускорение: {speedup:.2f}x\n")
    else:
        print(f"Размер: {size}x{size}")
        print(f"  CPU: {cpu_time:.4f}s")
        print(f"  GPU: недоступен\n")

# Визуализация
if torch.cuda.is_available():
    plt.figure(figsize=(14, 5))
    
    # График времени
    plt.subplot(1, 2, 1)
    plt.plot(sizes, cpu_times, 'o-', label='CPU', linewidth=2)
    plt.plot(sizes, gpu_times, 's-', label='GPU', linewidth=2)
    plt.xlabel('Размер матрицы')
    plt.ylabel('Время (секунды)')
    plt.title('Сравнение производительности CPU vs GPU')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # График ускорения
    plt.subplot(1, 2, 2)
    plt.plot(sizes, speedups, 'o-', color='green', linewidth=2)
    plt.xlabel('Размер матрицы')
    plt.ylabel('Ускорение (раз)')
    plt.title('Ускорение GPU относительно CPU')
    plt.grid(True)
    plt.axhline(y=1, color='r', linestyle='--', label='Без ускорения')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('cpu_vs_gpu_benchmark.png', dpi=100)
    print("График сохранен в 'cpu_vs_gpu_benchmark.png'")
```
</details>

---

### **Задача 5: Обработка ошибок с разными устройствами**
Напишите код, который корректно обрабатывает операции между тензорами на разных устройствах.

1. Создайте функцию `safe_add(a, b)`, которая складывает тензоры
2. Функция должна автоматически переносить тензоры на одно устройство
3. Приоритет: если один на GPU, результат тоже на GPU
4. Протестируйте функцию на различных комбинациях

<details>
<summary>Решение</summary>

```python
import torch

def safe_add(a, b):
    """
    Безопасное сложение тензоров с автоматическим переносом на одно устройство.
    Приоритет: GPU > CPU
    """
    # Получаем устройства тензоров
    device_a = a.device
    device_b = b.device
    
    print(f"Тензор a на: {device_a}")
    print(f"Тензор b на: {device_b}")
    
    # Если устройства одинаковые
    if device_a == device_b:
        print("Устройства совпадают, выполняем сложение")
        return a + b
    
    # Определяем целевое устройство (приоритет GPU)
    if device_a.type == 'cuda':
        target_device = device_a
        print(f"Переносим b на {target_device}")
        return a + b.to(target_device)
    elif device_b.type == 'cuda':
        target_device = device_b
        print(f"Переносим a на {target_device}")
        return a.to(target_device) + b
    else:
        # Оба на CPU
        print("Оба тензора на CPU")
        return a + b

# Тестирование
print("=" * 60)
print("Тест 1: Оба на CPU")
print("=" * 60)
a_cpu = torch.tensor([1, 2, 3])
b_cpu = torch.tensor([4, 5, 6])
result = safe_add(a_cpu, b_cpu)
print(f"Результат: {result}")
print(f"Устройство результата: {result.device}\n")

if torch.cuda.is_available():
    print("=" * 60)
    print("Тест 2: Один на GPU, другой на CPU")
    print("=" * 60)
    a_gpu = torch.tensor([1, 2, 3], device='cuda')
    b_cpu = torch.tensor([4, 5, 6])
    result = safe_add(a_gpu, b_cpu)
    print(f"Результат: {result}")
    print(f"Устройство результата: {result.device}\n")
    
    print("=" * 60)
    print("Тест 3: Оба на GPU")
    print("=" * 60)
    a_gpu = torch.tensor([1, 2, 3], device='cuda')
    b_gpu = torch.tensor([4, 5, 6], device='cuda')
    result = safe_add(a_gpu, b_gpu)
    print(f"Результат: {result}")
    print(f"Устройство результата: {result.device}")
```
</details>

---

### **Задача 6: Мониторинг памяти GPU**
Создайте функцию для мониторинга использования памяти GPU в процессе вычислений.

1. Напишите функцию `print_gpu_memory(message="")` 
2. Создайте большие тензоры и отслеживайте память
3. Продемонстрируйте утечку памяти и её исправление
4. Покажите эффект от `torch.cuda.empty_cache()`

<details>
<summary>Решение</summary>

```python
import torch

def print_gpu_memory(message=""):
    """Вывод информации об использовании памяти GPU"""
    if not torch.cuda.is_available():
        print("GPU недоступен")
        return
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"{message}")
    print(f"  Выделено: {allocated:.2f} GB")
    print(f"  Зарезервировано: {reserved:.2f} GB")
    print(f"  Максимум выделено: {max_allocated:.2f} GB")

if torch.cuda.is_available():
    # Сброс статистики
    torch.cuda.reset_peak_memory_stats()
    
    print("=" * 60)
    print("1. Начальное состояние")
    print("=" * 60)
    print_gpu_memory("Начало:")
    
    print("\n" + "=" * 60)
    print("2. Создание больших тензоров")
    print("=" * 60)
    
    # Создаем большие тензоры
    tensors = []
    for i in range(5):
        t = torch.randn(1000, 1000, device='cuda')
        tensors.append(t)
        print_gpu_memory(f"После создания тензора {i+1}:")
    
    print("\n" + "=" * 60)
    print("3. Удаление тензоров (утечка памяти?)")
    print("=" * 60)
    
    # Удаляем ссылки
    del tensors
    print_gpu_memory("После del tensors:")
    
    print("\n" + "=" * 60)
    print("4. Очистка кэша")
    print("=" * 60)
    
    torch.cuda.empty_cache()
    print_gpu_memory("После empty_cache():")
    
    print("\n" + "=" * 60)
    print("5. Демонстрация правильной очистки")
    print("=" * 60)
    
    # Создаем новые тензоры
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    print_gpu_memory("Создали x и y:")
    
    # Выполняем операцию
    z = x @ y
    print_gpu_memory("После матричного умножения:")
    
    # Правильная очистка
    del x, y, z
    torch.cuda.empty_cache()
    print_gpu_memory("После полной очистки:")
    
else:
    print("GPU недоступен. Используйте Google Colab с GPU runtime.")
```
</details>

---

## **🔴 Экспертный уровень**

### **Задача 7: Обучение модели на GPU с профилированием**
Реализуйте обучение нейронной сети на GPU с детальным профилированием.

1. Создайте простую MLP (многослойный перцептрон)
2. Обучите на синтетических данных
3. Измерьте время каждого этапа: forward, backward, optimizer step
4. Сравните общее время с CPU
5. Постройте графики использования времени

<details>
<summary>Решение</summary>

```python
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train_with_profiling(device, epochs=50):
    """Обучение с профилированием"""
    # Гиперпараметры
    input_size = 1000
    hidden_size = 500
    output_size = 10
    batch_size = 128
    
    # Создание модели
    model = SimpleMLP(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Синтетические данные
    X = torch.randn(1000, input_size).to(device)
    y = torch.randint(0, output_size, (1000,)).to(device)
    
    # Профилирование
    forward_times = []
    backward_times = []
    optimizer_times = []
    total_times = []
    
    print(f"\nОбучение на {device}...")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Выбираем батч
        indices = torch.randperm(X.size(0))[:batch_size]
        X_batch = X[indices]
        y_batch = y[indices]
        
        # Forward pass
        if device == 'cuda':
            torch.cuda.synchronize()
        forward_start = time.time()
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        forward_time = time.time() - forward_start
        
        # Backward pass
        if device == 'cuda':
            torch.cuda.synchronize()
        backward_start = time.time()
        
        optimizer.zero_grad()
        loss.backward()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        backward_time = time.time() - backward_start
        
        # Optimizer step
        if device == 'cuda':
            torch.cuda.synchronize()
        optimizer_start = time.time()
        
        optimizer.step()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        optimizer_time = time.time() - optimizer_start
        
        epoch_time = time.time() - epoch_start
        
        # Сохраняем времена
        forward_times.append(forward_time)
        backward_times.append(backward_time)
        optimizer_times.append(optimizer_time)
        total_times.append(epoch_time)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
                  f"Time: {epoch_time:.4f}s")
    
    return {
        'forward': forward_times,
        'backward': backward_times,
        'optimizer': optimizer_times,
        'total': total_times
    }

# Обучение на CPU
print("=" * 60)
cpu_times = train_with_profiling('cpu', epochs=50)

# Обучение на GPU (если доступен)
if torch.cuda.is_available():
    print("=" * 60)
    gpu_times = train_with_profiling('cuda', epochs=50)
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # График 1: Сравнение общего времени
    axes[0, 0].plot(cpu_times['total'], label='CPU', alpha=0.7)
    axes[0, 0].plot(gpu_times['total'], label='GPU', alpha=0.7)
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('Время (с)')
    axes[0, 0].set_title('Общее время на эпоху')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # График 2: Разбивка времени для CPU
    epochs = list(range(1, 51))
    axes[0, 1].stackplot(epochs, 
                         cpu_times['forward'],
                         cpu_times['backward'],
                         cpu_times['optimizer'],
                         labels=['Forward', 'Backward', 'Optimizer'],
                         alpha=0.7)
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('Время (с)')
    axes[0, 1].set_title('Разбивка времени - CPU')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True)
    
    # График 3: Разбивка времени для GPU
    axes[1, 0].stackplot(epochs,
                         gpu_times['forward'],
                         gpu_times['backward'],
                         gpu_times['optimizer'],
                         labels=['Forward', 'Backward', 'Optimizer'],
                         alpha=0.7)
    axes[1, 0].set_xlabel('Эпоха')
    axes[1, 0].set_ylabel('Время (с)')
    axes[1, 0].set_title('Разбивка времени - GPU')
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True)
    
    # График 4: Ускорение
    speedup = [c/g for c, g in zip(cpu_times['total'], gpu_times['total'])]
    axes[1, 1].plot(speedup, color='green', linewidth=2)
    axes[1, 1].axhline(y=1, color='r', linestyle='--', label='Без ускорения')
    axes[1, 1].set_xlabel('Эпоха')
    axes[1, 1].set_ylabel('Ускорение (раз)')
    axes[1, 1].set_title('Ускорение GPU относительно CPU')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_profiling.png', dpi=100)
    print("\nГрафик сохранен в 'training_profiling.png'")
    
    # Статистика
    avg_cpu = sum(cpu_times['total']) / len(cpu_times['total'])
    avg_gpu = sum(gpu_times['total']) / len(gpu_times['total'])
    avg_speedup = avg_cpu / avg_gpu
    
    print(f"\n{'='*60}")
    print("Итоговая статистика:")
    print(f"{'='*60}")
    print(f"Среднее время CPU: {avg_cpu:.4f}s")
    print(f"Среднее время GPU: {avg_gpu:.4f}s")
    print(f"Среднее ускорение: {avg_speedup:.2f}x")
else:
    print("\nGPU недоступен. Используйте Google Colab с GPU runtime.")
```
</details>

---

### **Задача 8: Работа с несколькими GPU**
Реализуйте параллельное вычисление на нескольких GPU (если доступны).

1. Проверьте количество доступных GPU
2. Разделите вычисления между GPU
3. Соберите результаты на одном устройстве
4. Сравните с вычислением на одном GPU

**Примечание:** Задача требует наличия нескольких GPU

<details>
<summary>Решение</summary>

```python
import torch
import time

def check_multi_gpu():
    """Проверка наличия нескольких GPU"""
    if not torch.cuda.is_available():
        print("GPU недоступен")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"Доступно GPU: {gpu_count}")
    
    if gpu_count < 2:
        print("Для этой задачи нужно минимум 2 GPU")
        return False
    
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    return True

def compute_on_single_gpu(size=5000):
    """Вычисление на одном GPU"""
    device = 'cuda:0'
    
    # Создание тензоров
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)
    
    # Прогрев
    _ = A @ B
    torch.cuda.synchronize()
    
    # Вычисление
    start = time.time()
    C = A @ B
    D = torch.relu(C)
    result = torch.sum(D)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return result, elapsed

def compute_on_multi_gpu(size=5000, num_gpus=2):
    """Вычисление на нескольких GPU"""
    # Разделяем вычисления
    split_size = size // num_gpus
    
    # Создание тензоров на разных GPU
    A_parts = []
    B_parts = []
    
    for i in range(num_gpus):
        device = f'cuda:{i}'
        A_part = torch.randn(split_size, size, device=device)
        B_part = torch.randn(size, split_size, device=device)
        A_parts.append(A_part)
        B_parts.append(B_part)
    
    # Прогрев
    for i in range(num_gpus):
        _ = A_parts[i] @ B_parts[i]
    torch.cuda.synchronize()
    
    # Параллельное вычисление
    start = time.time()
    
    results = []
    for i in range(num_gpus):
        C_part = A_parts[i] @ B_parts[i]
        D_part = torch.relu(C_part)
        result_part = torch.sum(D_part)
        results.append(result_part)
    
    # Синхронизация всех GPU
    torch.cuda.synchronize()
    
    # Сбор результатов на GPU 0
    total_result = sum([r.to('cuda:0') for r in results])
    
    elapsed = time.time() - start
    
    return total_result, elapsed

# Основная программа
print("=" * 60)
print("Тестирование вычислений на нескольких GPU")
print("=" * 60)

if check_multi_gpu():
    num_gpus = torch.cuda.device_count()
    
    print(f"\n{'='*60}")
    print("Вычисление на одном GPU")
    print("=" * 60)
    result_single, time_single = compute_on_single_gpu()
    print(f"Результат: {result_single.item():.4f}")
    print(f"Время: {time_single:.4f}s")
    
    print(f"\n{'='*60}")
    print(f"Вычисление на {num_gpus} GPU")
    print("=" * 60)
    result_multi, time_multi = compute_on_multi_gpu(num_gpus=num_gpus)
    print(f"Результат: {result_multi.item():.4f}")
    print(f"Время: {time_multi:.4f}s")
    
    print(f"\n{'='*60}")
    print("Сравнение")
    print("=" * 60)
    if time_single > time_multi:
        speedup = time_single / time_multi
        print(f"Ускорение: {speedup:.2f}x")
    else:
        slowdown = time_multi / time_single
        print(f"Замедление: {slowdown:.2f}x (overhead от коммуникации)")
else:
    print("\nНедостаточно GPU для выполнения задачи")
    print("Рекомендация: используйте облачные платформы с multi-GPU")
```
</details>

---

### **Задача 9: Отладка и оптимизация GPU-кода**
Найдите и исправьте проблемы в следующем коде:

```python
import torch

def train_model_buggy():
    model = torch.nn.Linear(100, 10).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    losses = []
    for epoch in range(100):
        x = torch.randn(32, 100)  # Данные на CPU!
        y = torch.randint(0, 10, (32,)).cuda()
        
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss)  # Утечка памяти!
    
    return losses
```

Найдите и исправьте:
1. Несоответствие устройств
2. Утечку памяти
3. Отсутствие `zero_grad()`
4. Добавьте профилирование времени

<details>
<summary>Решение</summary>

```python
import torch
import time

def train_model_buggy():
    """ИСХОДНАЯ ВЕРСИЯ С ОШИБКАМИ"""
    model = torch.nn.Linear(100, 10).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    losses = []
    for epoch in range(100):
        x = torch.randn(32, 100)  # ❌ Данные на CPU!
        y = torch.randint(0, 10, (32,)).cuda()
        
        output = model(x)  # ❌ RuntimeError: expected input and model on same device
        loss = torch.nn.functional.cross_entropy(output, y)
        
        loss.backward()
        optimizer.step()  # ❌ Нет zero_grad(), градиенты накапливаются!
        
        losses.append(loss)  # ❌ Утечка памяти! Сохраняем тензор с графом
    
    return losses

def train_model_fixed():
    """ИСПРАВЛЕННАЯ ВЕРСИЯ"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ✅ Модель на GPU
    model = torch.nn.Linear(100, 10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    losses = []
    epoch_times = []
    
    print(f"Обучение на {device}")
    print("-" * 60)
    
    for epoch in range(100):
        start_time = time.time()
        
        # ✅ Данные на том же устройстве, что и модель
        x = torch.randn(32, 100, device=device)
        y = torch.randint(0, 10, (32,), device=device)
        
        # Forward pass
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        
        # ✅ Обнуление градиентов
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # ✅ Сохраняем только значение (без графа)
        losses.append(loss.item())
        
        # Измерение времени
        if device == 'cuda':
            torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/100: Loss={loss.item():.4f}, Time={epoch_time:.4f}s")
    
    return losses, epoch_times

# Демонстрация ошибки
print("=" * 60)
print("1. ПОПЫТКА ЗАПУСКА БАГОВАННОГО КОДА")
print("=" * 60)
try:
    train_model_buggy()
    print("Код выполнился (но с утечкой памяти)")
except RuntimeError as e:
    print(f"❌ Ошибка: {str(e)[:80]}...")
    print("   Причина: данные на CPU, модель на GPU")

# Исправленная версия
print("\n" + "=" * 60)
print("2. ИСПРАВЛЕННЫЙ КОД")
print("=" * 60)
losses, times = train_model_fixed()

# Статистика
print("\n" + "=" * 60)
print("СТАТИСТИКА")
print("=" * 60)
print(f"Финальная функция потерь: {losses[-1]:.4f}")
print(f"Среднее время на эпоху: {sum(times)/len(times):.4f}s")
print(f"Общее время обучения: {sum(times):.2f}s")

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e6
    print(f"Использовано памяти GPU: {allocated:.2f} MB")

# Визуализация
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.title('Кривая обучения')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(times)
plt.xlabel('Эпоха')
plt.ylabel('Время (с)')
plt.title('Время на эпоху')
plt.grid(True)

plt.tight_layout()
plt.savefig('debug_training.png', dpi=100)
print("\nГрафик сохранен в 'debug_training.png'")
```

**Объяснение исправлений:**

1. **Несоответствие устройств**: Данные `x` создавались на CPU, а модель была на GPU
   - Исправление: `.to(device)` для данных

2. **Утечка памяти**: Сохранение тензора `loss` с вычислительным графом
   - Исправление: Используем `.item()` для извлечения только числа

3. **Отсутствие `zero_grad()`**: Градиенты накапливались между итерациями
   - Исправление: Добавлен `optimizer.zero_grad()` перед `backward()`

4. **Профилирование**: Добавлено измерение времени с `torch.cuda.synchronize()`
</details>

---

### **Задача 10: Mixed Precision Training**
Реализуйте обучение с использованием автоматического смешанного точности (AMP).

1. Создайте простую нейронную сеть для классификации
2. Реализуйте обычное обучение
3. Реализуйте обучение с Mixed Precision (torch.cuda.amp)
4. Сравните скорость и использование памяти

**Требуется:** PyTorch 1.6+ и GPU с Compute Capability >= 7.0 (Volta или новее)

<details>
<summary>Решение</summary>

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time

if not torch.cuda.is_available():
    print("GPU недоступен. Используйте Google Colab с GPU.")
else:
    # Проверка поддержки Mixed Precision
    device_capability = torch.cuda.get_device_capability()
    print(f"GPU Compute Capability: {device_capability}")
    if device_capability[0] >= 7:
        print("✅ GPU поддерживает Mixed Precision Training")
    
    # Модель
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)
    
    # Синтетические данные
    x = torch.randn(1000, 1024).cuda()
    y = torch.randint(0, 10, (1000,)).cuda()
    
    def train_normal(epochs=20):
        """Обычное обучение (FP32)"""
        model = SimpleNet().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        
        return elapsed, peak_memory
    
    def train_mixed_precision(epochs=20):
        """Обучение с Mixed Precision (FP16)"""
        model = SimpleNet().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()
        
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Автоматическое использование FP16
            with autocast():
                outputs = model(x)
                loss = criterion(outputs, y)
            
            # Масштабирование и обратное распространение
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        
        return elapsed, peak_memory
    
    # Сравнение
    print("\n" + "="*60)
    print("ОБЫЧНОЕ ОБУЧЕНИЕ (FP32)")
    print("="*60)
    time_normal, mem_normal = train_normal()
    print(f"Время: {time_normal:.2f}s")
    print(f"Пик памяти: {mem_normal:.2f} GB")
    
    print("\n" + "="*60)
    print("MIXED PRECISION TRAINING (FP16)")
    print("="*60)
    time_amp, mem_amp = train_mixed_precision()
    print(f"Время: {time_amp:.2f}s")
    print(f"Пик памяти: {mem_amp:.2f} GB")
    
    # Результаты
    print("\n" + "="*60)
    print("СРАВНЕНИЕ")
    print("="*60)
    speedup = time_normal / time_amp
    memory_saving = (1 - mem_amp / mem_normal) * 100
    print(f"Ускорение: {speedup:.2f}x")
    print(f"Экономия памяти: {memory_saving:.1f}%")
    print("\n✅ Mixed Precision обеспечивает существенное ускорение!")
```
</details>

---

## **💡 Дополнительные задачи**

### **Задача 11: Pinned Memory для ускорения**
Продемонстрируйте разницу между обычным и pinned memory при переносе данных на GPU.

<details>
<summary>Решение</summary>

```python
import torch
import time

if not torch.cuda.is_available():
    print("GPU недоступен")
else:
    size = 10000
    iterations = 100
    
    # Обычная память
    tensor_regular = torch.randn(size, size)
    
    # Pinned память
    tensor_pinned = torch.randn(size, size).pin_memory()
    
    # Бенчмарк для обычной памяти
    start = time.time()
    for _ in range(iterations):
        _ = tensor_regular.to('cuda')
    torch.cuda.synchronize()
    time_regular = time.time() - start
    
    # Бенчмарк для pinned памяти
    start = time.time()
    for _ in range(iterations):
        _ = tensor_pinned.to('cuda', non_blocking=True)
    torch.cuda.synchronize()
    time_pinned = time.time() - start
    
    print(f"Размер тензора: {size}x{size}")
    print(f"Итераций: {iterations}")
    print(f"\nОбычная память: {time_regular:.4f}s")
    print(f"Pinned память: {time_pinned:.4f}s")
    print(f"Ускорение: {time_regular/time_pinned:.2f}x")
```
</details>

---

## **📚 Рекомендации**

1. **Всегда проверяйте доступность GPU** перед использованием
2. **Переносите модель и данные** на одно устройство
3. **Используйте `torch.cuda.synchronize()`** при измерении времени
4. **Мониторьте память GPU** при работе с большими моделями
5. **Очищайте память** с помощью `del` и `torch.cuda.empty_cache()`

**Дополнительные ресурсы:**
- [CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Multi-GPU Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
