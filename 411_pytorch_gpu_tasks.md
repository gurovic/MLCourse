# **Задачи: GPU в PyTorch**

## **⚙️ Подготовка**

Перед началом проверьте доступность GPU:
```python
import torch
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA версия: {torch.version.cuda}")
    print(f"Название GPU: {torch.cuda.get_device_name(0)}")
# Если GPU недоступен, используйте Google Colab с GPU runtime
```

## **⚠️ Важные правила работы с GPU**

1. **Всегда проверяйте доступность GPU** перед использованием `cuda`
2. **Синхронизируйте GPU** при измерении времени: `torch.cuda.synchronize()`
3. **Используйте `.item()`** для извлечения скалярных значений (избегайте утечек памяти)
4. **Переносите модель и данные** на одно устройство
5. **Очищайте память** при работе с большими моделями: `del tensor; torch.cuda.empty_cache()`

---

## **🟢 Базовый уровень**

### **Задача 1: Умный выбор устройства**
Напишите функцию `smart_device_selection(model_size_mb, batch_size, available_memory_fraction=0.8)`, которая:
1. Анализирует доступные вычислительные ресурсы (CPU и GPU)
2. Оценивает, поместится ли модель + данные в память GPU
3. Принимает решение, использовать GPU или CPU
4. Возвращает объект `device` и краткое объяснение выбора

**Требования:**
- Учитывайте доступную память GPU (используйте `available_memory_fraction` как запас)
- Если GPU недоступен или памяти недостаточно, выбирайте CPU
- Для очень маленьких моделей (< 10 MB) и маленьких батчей (< 16) CPU может быть предпочтительнее из-за overhead

**Подсказка:** Оцените необходимую память как: `model_size_mb + batch_size * average_sample_size_mb * 4` (множитель 4 учитывает промежуточные активации и градиенты)

<details>
<summary>Решение</summary>

```python
import torch

def smart_device_selection(model_size_mb, batch_size, average_sample_size_mb=1.0, available_memory_fraction=0.8):
    """
    Интеллектуальный выбор устройства на основе анализа ресурсов
    
    Args:
        model_size_mb: Размер модели в мегабайтах
        batch_size: Размер батча
        average_sample_size_mb: Средний размер одного образца в MB (по умолчанию 1 MB)
        available_memory_fraction: Доля памяти GPU, которую безопасно использовать
    
    Returns:
        (device, explanation): Выбранное устройство и объяснение
    """
    # Проверка доступности GPU
    if not torch.cuda.is_available():
        return torch.device('cpu'), "GPU недоступен - используем CPU"
    
    # Получаем информацию о GPU
    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # в MB
    gpu_memory_available = gpu_memory_total * available_memory_fraction
    
    # Оценка необходимой памяти
    # Множитель 4: модель + данные + градиенты + промежуточные активации
    estimated_memory = model_size_mb + (batch_size * average_sample_size_mb * 4)
    
    # Эвристика: для очень маленьких моделей и батчей CPU может быть быстрее
    is_tiny = model_size_mb < 10 and batch_size < 16
    
    # Принятие решения
    if is_tiny:
        return torch.device('cpu'), (
            f"Модель маленькая ({model_size_mb:.1f}MB) и batch_size={batch_size} - "
            f"overhead GPU может быть больше выигрыша. Используем CPU."
        )
    elif estimated_memory > gpu_memory_available:
        return torch.device('cpu'), (
            f"Нужно ~{estimated_memory:.1f}MB, доступно {gpu_memory_available:.1f}MB на GPU. "
            f"Недостаточно памяти - используем CPU."
        )
    else:
        gpu_name = torch.cuda.get_device_name(0)
        return torch.device('cuda'), (
            f"GPU ({gpu_name}) подходит: нужно ~{estimated_memory:.1f}MB, "
            f"доступно {gpu_memory_available:.1f}MB. Используем GPU."
        )

# Тестирование различных сценариев
print("=" * 70)
print("Тестирование умного выбора устройства")
print("=" * 70)

# Сценарий 1: Маленькая модель, маленький батч
device1, reason1 = smart_device_selection(model_size_mb=5, batch_size=8)
print(f"\n1. Маленькая модель (5MB), batch_size=8")
print(f"   Выбор: {device1}")
print(f"   Причина: {reason1}")

# Сценарий 2: Средняя модель, средний батч
device2, reason2 = smart_device_selection(model_size_mb=100, batch_size=32)
print(f"\n2. Средняя модель (100MB), batch_size=32")
print(f"   Выбор: {device2}")
print(f"   Причина: {reason2}")

# Сценарий 3: Большая модель, большой батч
device3, reason3 = smart_device_selection(model_size_mb=500, batch_size=128, average_sample_size_mb=2.0)
print(f"\n3. Большая модель (500MB), batch_size=128, samples=2MB")
print(f"   Выбор: {device3}")
print(f"   Причина: {reason3}")

# Сценарий 4: Очень большая модель
device4, reason4 = smart_device_selection(model_size_mb=5000, batch_size=64)
print(f"\n4. Очень большая модель (5GB), batch_size=64")
print(f"   Выбор: {device4}")
print(f"   Причина: {reason4}")
```
</details>

---

### **Задача 2: Адаптивная обработка батча**
Создайте функцию `process_batch(data, model, device='auto')`, которая:
1. Принимает батч данных (может быть на CPU или GPU) и модель (может быть на CPU или GPU)
2. Автоматически определяет оптимальную стратегию обработки:
   - Если `device='auto'`, анализирует текущее расположение модели и данных
   - Минимизирует количество переносов данных между устройствами
   - Корректно обрабатывает смешанные ситуации
3. Возвращает результат на том же устройстве, где были входные данные

**Требования:**
- Функция должна работать во всех комбинациях: (модель на CPU, данные на CPU), (модель на GPU, данные на GPU), (модель на CPU, данные на GPU), (модель на GPU, данные на CPU)
- При необходимости переноса выводить предупреждение
- Не должно быть лишних переносов (например, если модель и данные уже на GPU, не переносить на CPU)

<details>
<summary>Решение</summary>

```python
import torch
import torch.nn as nn

def process_batch(data, model, device='auto'):
    """
    Адаптивная обработка батча с минимизацией переносов данных
    
    Args:
        data: входной тензор (может быть на CPU или GPU)
        model: модель PyTorch (может быть на CPU или GPU)
        device: 'auto', 'cpu', 'cuda' или torch.device объект
    
    Returns:
        output: результат на том же устройстве, где были входные данные
    """
    # Определяем устройства
    data_device = data.device
    model_device = next(model.parameters()).device
    
    print(f"Данные на: {data_device}, Модель на: {model_device}")
    
    # Если device='auto', определяем оптимальную стратегию
    if device == 'auto':
        # Приоритет: если модель на GPU, используем GPU
        if model_device.type == 'cuda':
            compute_device = model_device
            if data_device.type != 'cuda':
                print(f"⚠️  Перенос данных с {data_device} на {compute_device}")
                data_to_process = data.to(compute_device)
            else:
                data_to_process = data
        # Иначе используем устройство данных
        else:
            compute_device = data_device
            if model_device != data_device:
                print(f"⚠️  Модель на {model_device}, данные на {data_device}")
                print(f"    Рекомендация: перенесите модель на {data_device} перед циклом обучения")
                data_to_process = data.to(model_device)
                compute_device = model_device
            else:
                data_to_process = data
    else:
        # Явное указание устройства
        compute_device = torch.device(device)
        if model_device != compute_device:
            print(f"⚠️  Модель будет временно использована на {compute_device}")
        data_to_process = data.to(compute_device)
    
    # Выполняем вычисление
    with torch.no_grad():
        output = model(data_to_process)
    
    # Возвращаем результат на устройство входных данных
    if output.device != data_device:
        print(f"    Возврат результата на {data_device}")
        output = output.to(data_device)
    
    return output

# Создаем простую модель для тестирования
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, output_size=5):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.fc(x)

# Тестирование различных комбинаций
print("=" * 70)
print("Тестирование адаптивной обработки батча")
print("=" * 70)

# Случай 1: Модель на CPU, данные на CPU
print("\n" + "=" * 70)
print("Случай 1: Модель на CPU, данные на CPU")
print("=" * 70)
model_cpu = SimpleModel()
data_cpu = torch.randn(4, 10)
result1 = process_batch(data_cpu, model_cpu, device='auto')
print(f"Результат на: {result1.device}")

# Случай 2: Модель на GPU, данные на CPU
if torch.cuda.is_available():
    print("\n" + "=" * 70)
    print("Случай 2: Модель на GPU, данные на CPU")
    print("=" * 70)
    model_gpu = SimpleModel().cuda()
    data_cpu = torch.randn(4, 10)
    result2 = process_batch(data_cpu, model_gpu, device='auto')
    print(f"Результат на: {result2.device}")
    
    # Случай 3: Модель на GPU, данные на GPU
    print("\n" + "=" * 70)
    print("Случай 3: Модель на GPU, данные на GPU (оптимально!)")
    print("=" * 70)
    data_gpu = torch.randn(4, 10).cuda()
    result3 = process_batch(data_gpu, model_gpu, device='auto')
    print(f"Результат на: {result3.device}")
    
    # Случай 4: Модель на CPU, данные на GPU
    print("\n" + "=" * 70)
    print("Случай 4: Модель на CPU, данные на GPU")
    print("=" * 70)
    model_cpu = SimpleModel()
    data_gpu = torch.randn(4, 10).cuda()
    result4 = process_batch(data_gpu, model_cpu, device='auto')
    print(f"Результат на: {result4.device}")
else:
    print("\n⚠️  GPU недоступен - тестируем только CPU варианты")

print("\n💡 Вывод: Всегда переносите модель и данные на одно устройство")
print("   перед циклом обучения, чтобы избежать постоянных переносов!")
```
</details>

---

### **Задача 3: Вычисления с ограничением памяти**
У вас есть большой датасет (10000 образцов) и модель. GPU имеет ограниченную память, и полный батч не помещается. Реализуйте функцию `compute_with_memory_limit(model, data, max_batch_size, device)`, которая:

1. Автоматически разбивает данные на под-батчи, помещающиеся в память
2. Обрабатывает каждый под-батч на GPU
3. Собирает результаты обратно
4. Отслеживает использование памяти GPU и выводит статистику

**Дополнительное условие:** После обработки каждого под-батча освобождайте неиспользуемую память GPU.

**Подсказка:** Используйте `torch.cuda.memory_allocated()`, `torch.cuda.empty_cache()`, и `del` для управления памятью.

<details>
<summary>Решение</summary>

```python
import torch
import torch.nn as nn

def compute_with_memory_limit(model, data, max_batch_size, device):
    """
    Обработка данных с автоматическим разбиением на под-батчи при ограничении памяти
    
    Args:
        model: модель PyTorch
        data: входные данные (N, ...)
        max_batch_size: максимальный размер батча для GPU
        device: устройство для вычислений
    
    Returns:
        results: объединенные результаты всех под-батчей
    """
    model = model.to(device)
    model.eval()
    
    total_samples = data.shape[0]
    results = []
    
    print(f"Обработка {total_samples} образцов с max_batch_size={max_batch_size}")
    print(f"Устройство: {device}")
    print("-" * 70)
    
    if device == 'cuda':
        # Сброс пиковой статистики памяти
        torch.cuda.reset_peak_memory_stats()
    
    # Разбиваем на под-батчи
    num_batches = (total_samples + max_batch_size - 1) // max_batch_size
    
    for i in range(num_batches):
        start_idx = i * max_batch_size
        end_idx = min((i + 1) * max_batch_size, total_samples)
        batch_data = data[start_idx:end_idx].to(device)
        
        # Вычисление
        with torch.no_grad():
            batch_result = model(batch_data)
        
        # Переносим результат обратно на CPU для сохранения
        results.append(batch_result.cpu())
        
        # Освобождаем память
        del batch_data, batch_result
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # Статистика памяти
        if device == 'cuda':
            current_mem = torch.cuda.memory_allocated() / 1e6
            peak_mem = torch.cuda.max_memory_allocated() / 1e6
            print(f"Батч {i+1}/{num_batches} ({end_idx-start_idx} образцов): "
                  f"Текущая память: {current_mem:.1f}MB, "
                  f"Пик: {peak_mem:.1f}MB")
        else:
            print(f"Батч {i+1}/{num_batches} обработан на CPU")
    
    # Объединяем результаты
    final_result = torch.cat(results, dim=0)
    
    print("-" * 70)
    print(f"Завершено! Обработано {total_samples} образцов")
    if device == 'cuda':
        final_peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"Пиковое использование памяти GPU: {final_peak:.1f}MB")
    
    return final_result

# Создаем тестовую модель и данные
class TestModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=50, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Генерируем большой датасет
print("=" * 70)
print("Создание тестовых данных")
print("=" * 70)
data = torch.randn(10000, 100)  # 10000 образцов
model = TestModel()
print(f"Данные: {data.shape}")
print(f"Модель: {sum(p.numel() for p in model.parameters())} параметров")

# Тест 1: Обработка на CPU (для сравнения)
print("\n" + "=" * 70)
print("Тест 1: Обработка на CPU")
print("=" * 70)
results_cpu = compute_with_memory_limit(model, data, max_batch_size=500, device='cpu')
print(f"Результат: {results_cpu.shape}")

# Тест 2: Обработка на GPU с разными размерами батча
if torch.cuda.is_available():
    print("\n" + "=" * 70)
    print("Тест 2: Обработка на GPU с большим батчем")
    print("=" * 70)
    results_gpu1 = compute_with_memory_limit(model, data, max_batch_size=2000, device='cuda')
    
    print("\n" + "=" * 70)
    print("Тест 3: Обработка на GPU с маленьким батчем (экономия памяти)")
    print("=" * 70)
    results_gpu2 = compute_with_memory_limit(model, data, max_batch_size=500, device='cuda')
    
    # Проверка корректности
    print("\n" + "=" * 70)
    print("Проверка корректности")
    print("=" * 70)
    print(f"Разница между результатами (должна быть ~0): {torch.max(torch.abs(results_gpu1 - results_gpu2)).item():.6f}")
else:
    print("\n⚠️  GPU недоступен - запустите в Google Colab с GPU")

print("\n💡 Вывод: Разбиение на батчи позволяет обрабатывать большие данные")
print("   даже при ограниченной памяти GPU!")
```
</details>

---

## **🟡 Продвинутый уровень**

### **Задача 4: Анализ точки перелома CPU/GPU**
Проведите эксперимент для определения, при каком размере задачи GPU начинает быть эффективнее CPU.

1. Создайте функцию `find_gpu_breakeven_point(operation_type)` для разных типов операций:
   - Матричное умножение
   - Поэлементные операции (сложение, умножение)
   - Свертки (convolution)
2. Для каждой операции найдите минимальный размер данных, при котором GPU быстрее CPU
3. Объясните, почему точки перелома разные для разных операций
4. Постройте график зависимости ускорения от размера данных

**Вопросы для размышления:**
- Почему для матричного умножения точка перелома может быть при меньших размерах, чем для поэлементных операций?
- Как overhead на перенос данных влияет на точку перелома?
- Что происходит с ускорением при очень больших размерах данных?

<details>
<summary>Решение</summary>

```python
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np

def benchmark_operation(operation_type, size, device, iterations=10):
    """
    Бенчмарк различных типов операций
    
    Args:
        operation_type: 'matmul', 'elementwise', 'conv'
        size: размер данных
        device: 'cpu' или 'cuda'
        iterations: количество повторений
    """
    # Создаем данные в зависимости от типа операции
    if operation_type == 'matmul':
        A = torch.randn(size, size, device=device)
        B = torch.randn(size, size, device=device)
        operation = lambda: A @ B
    
    elif operation_type == 'elementwise':
        A = torch.randn(size, size, device=device)
        B = torch.randn(size, size, device=device)
        operation = lambda: A * B + A / (B + 1)
    
    elif operation_type == 'conv':
        # Для свертки используем batch для честного сравнения
        batch_size = max(1, 32 // max(1, size // 100))  # адаптивный batch
        x = torch.randn(batch_size, 3, size, size, device=device)
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1).to(device)
        operation = lambda: conv(x)
    
    # Прогрев
    for _ in range(3):
        _ = operation()
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Измерение
    start = time.time()
    for _ in range(iterations):
        _ = operation()
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    return elapsed / iterations

def find_gpu_breakeven_point(operation_type):
    """
    Находит точку перелома, где GPU становится быстрее CPU
    """
    if not torch.cuda.is_available():
        print("GPU недоступен")
        return None
    
    print(f"\n{'='*70}")
    print(f"Анализ точки перелома для операции: {operation_type}")
    print('='*70)
    
    # Диапазон размеров для тестирования (логарифмическая шкала)
    if operation_type == 'conv':
        sizes = [16, 32, 64, 128, 256]  # размеры изображения
    else:
        sizes = [10, 50, 100, 200, 500, 1000, 2000, 3000]
    
    cpu_times = []
    gpu_times = []
    speedups = []
    
    for size in sizes:
        try:
            cpu_time = benchmark_operation(operation_type, size, 'cpu', iterations=5)
            gpu_time = benchmark_operation(operation_type, size, 'cuda', iterations=5)
            
            cpu_times.append(cpu_time)
            gpu_times.append(gpu_time)
            speedup = cpu_time / gpu_time
            speedups.append(speedup)
            
            status = "🚀 GPU быстрее" if speedup > 1 else "🐌 CPU быстрее"
            print(f"Размер: {size:4d} | CPU: {cpu_time:.5f}s | GPU: {gpu_time:.5f}s | "
                  f"Ускорение: {speedup:.2f}x | {status}")
        except RuntimeError as e:
            print(f"Размер {size}: Ошибка памяти - {e}")
            break
    
    # Находим точку перелома
    breakeven_idx = None
    for i, speedup in enumerate(speedups):
        if speedup > 1.0:
            breakeven_idx = i
            break
    
    if breakeven_idx is not None:
        print(f"\n💡 Точка перелома: размер ≈ {sizes[breakeven_idx]}")
        print(f"   При этом размере ускорение: {speedups[breakeven_idx]:.2f}x")
    else:
        print("\n⚠️  GPU медленнее CPU на всех размерах (overhead слишком велик)")
    
    return {
        'sizes': sizes[:len(speedups)],
        'cpu_times': cpu_times,
        'gpu_times': gpu_times,
        'speedups': speedups,
        'breakeven_idx': breakeven_idx
    }

# Анализируем все типы операций
operation_types = ['matmul', 'elementwise', 'conv']
results = {}

for op_type in operation_types:
    results[op_type] = find_gpu_breakeven_point(op_type)

# Визуализация результатов
if torch.cuda.is_available() and len(results) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # График 1: Сравнение времени выполнения
    ax = axes[0, 0]
    for op_type, data in results.items():
        if data:
            ax.plot(data['sizes'], data['cpu_times'], 'o-', label=f'{op_type} (CPU)', alpha=0.7)
            ax.plot(data['sizes'], data['gpu_times'], 's-', label=f'{op_type} (GPU)', alpha=0.7)
    ax.set_xlabel('Размер данных')
    ax.set_ylabel('Время (сек)')
    ax.set_title('Время выполнения: CPU vs GPU')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    # График 2: Ускорение для каждой операции
    ax = axes[0, 1]
    for op_type, data in results.items():
        if data:
            ax.plot(data['sizes'], data['speedups'], 'o-', label=op_type, linewidth=2)
            # Отмечаем точку перелома
            if data['breakeven_idx'] is not None:
                idx = data['breakeven_idx']
                ax.plot(data['sizes'][idx], data['speedups'][idx], 'r*', markersize=15)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Без ускорения')
    ax.set_xlabel('Размер данных')
    ax.set_ylabel('Ускорение (GPU/CPU)')
    ax.set_title('Ускорение GPU относительно CPU')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # График 3: Только матричное умножение (детально)
    ax = axes[1, 0]
    if 'matmul' in results and results['matmul']:
        data = results['matmul']
        ax.plot(data['sizes'], data['speedups'], 'go-', linewidth=2, markersize=8)
        if data['breakeven_idx'] is not None:
            idx = data['breakeven_idx']
            ax.plot(data['sizes'][idx], data['speedups'][idx], 'r*', markersize=20,
                   label=f'Точка перелома: {data["sizes"][idx]}')
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Размер матрицы')
        ax.set_ylabel('Ускорение')
        ax.set_title('Матричное умножение: детальный анализ')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # График 4: Сравнение точек перелома
    ax = axes[1, 1]
    breakeven_points = []
    labels = []
    for op_type, data in results.items():
        if data and data['breakeven_idx'] is not None:
            breakeven_points.append(data['sizes'][data['breakeven_idx']])
            labels.append(op_type)
    
    if breakeven_points:
        colors = ['blue', 'green', 'orange'][:len(breakeven_points)]
        ax.barh(labels, breakeven_points, color=colors, alpha=0.7)
        ax.set_xlabel('Размер данных (точка перелома)')
        ax.set_title('Сравнение точек перелома')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('gpu_breakeven_analysis.png', dpi=100)
    print("\n" + "="*70)
    print("График сохранен в 'gpu_breakeven_analysis.png'")
    print("="*70)

# Итоговый анализ
print("\n" + "="*70)
print("ИТОГОВЫЙ АНАЛИЗ")
print("="*70)
print("\n💡 Выводы:")
print("1. Матричное умножение: GPU эффективен раньше всего")
print("   - Причина: высокая арифметическая интенсивность (много операций на элемент)")
print("   - Хорошее соотношение вычислений к переносу данных")
print("\n2. Поэлементные операции: GPU эффективен при больших размерах")
print("   - Причина: низкая арифметическая интенсивность")
print("   - Overhead на перенос данных относительно большой")
print("\n3. Свертки: GPU эффективен при средних размерах")
print("   - Причина: баланс между вычислениями и памятью")
print("   - Оптимизированные CUDA kernels для сверток")
print("\n4. Overhead на перенос данных:")
print("   - Включает: копирование CPU→GPU, синхронизацию, инициализацию")
print("   - Для маленьких данных может превысить выигрыш от параллелизма")
print("\n5. При очень больших размерах:")
print("   - Ускорение стабилизируется (достигает максимальной утилизации GPU)")
print("   - Может появиться ограничение по пропускной способности памяти")
```
</details>

---

### **Задача 5: Динамическое управление устройствами в приложении**
Создайте класс `DeviceManager`, который интеллектуально управляет вычислениями на CPU и GPU в зависимости от условий:

1. Отслеживает доступность и загруженность GPU
2. Автоматически переключается между CPU и GPU если:
   - Память GPU заканчивается (> 90% использования)
   - Размер батча слишком маленький для эффективного использования GPU
   - GPU занят другими процессами
3. Кэширует решение о выборе устройства для однотипных операций
4. Предоставляет метод `get_optimal_device(operation_profile)` с параметрами операции

**Дополнительно:** Реализуйте метод `reset_if_memory_critical()`, который при критичном уровне памяти GPU автоматически очищает кэш и переносит менее важные вычисления на CPU.

**Подсказка:** `operation_profile` может содержать: `{'type': 'matmul', 'size': 1000, 'batch_size': 32, 'priority': 'high'}`

<details>
<summary>Решение</summary>

```python
import torch
import time
from typing import Dict, Any

class DeviceManager:
    """
    Интеллектуальный менеджер вычислительных устройств
    """
    def __init__(self, memory_threshold=0.9, min_batch_size_for_gpu=16):
        """
        Args:
            memory_threshold: порог использования памяти GPU (0-1)
            min_batch_size_for_gpu: минимальный batch_size для использования GPU
        """
        self.memory_threshold = memory_threshold
        self.min_batch_size_for_gpu = min_batch_size_for_gpu
        self.gpu_available = torch.cuda.is_available()
        self.device_cache = {}  # кэш решений
        self.operation_history = []  # история операций
        
        if self.gpu_available:
            self.total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
            print(f"✅ DeviceManager инициализирован")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Память: {self.total_gpu_memory / 1e9:.2f} GB")
        else:
            print("⚠️  GPU недоступен, будет использоваться только CPU")
    
    def get_gpu_memory_usage(self):
        """Возвращает текущее использование памяти GPU (0-1)"""
        if not self.gpu_available:
            return 0.0
        allocated = torch.cuda.memory_allocated()
        return allocated / self.total_gpu_memory
    
    def is_gpu_memory_critical(self):
        """Проверяет, критичен ли уровень памяти GPU"""
        return self.get_gpu_memory_usage() > self.memory_threshold
    
    def reset_if_memory_critical(self):
        """Очищает память GPU если использование критичное"""
        if not self.gpu_available:
            return False
        
        if self.is_gpu_memory_critical():
            usage_before = self.get_gpu_memory_usage()
            torch.cuda.empty_cache()
            usage_after = self.get_gpu_memory_usage()
            
            print(f"⚠️  Критичный уровень памяти GPU!")
            print(f"   До очистки: {usage_before*100:.1f}%")
            print(f"   После очистки: {usage_after*100:.1f}%")
            
            # Очищаем кэш решений
            self.device_cache.clear()
            return True
        return False
    
    def get_optimal_device(self, operation_profile: Dict[str, Any]):
        """
        Определяет оптимальное устройство для операции
        
        Args:
            operation_profile: словарь с параметрами операции
                {'type': str, 'size': int, 'batch_size': int, 'priority': str}
        
        Returns:
            torch.device
        """
        # Создаем ключ для кэша
        cache_key = (
            operation_profile.get('type'),
            operation_profile.get('size'),
            operation_profile.get('batch_size')
        )
        
        # Проверяем кэш (если операция не высокого приоритета)
        if operation_profile.get('priority') != 'high' and cache_key in self.device_cache:
            return self.device_cache[cache_key]
        
        # Проверяем критичность памяти
        if self.reset_if_memory_critical():
            # После очистки повторно проверяем
            if self.is_gpu_memory_critical():
                return torch.device('cpu')  # Переключаемся на CPU
        
        # Принимаем решение
        decision = self._make_device_decision(operation_profile)
        
        # Сохраняем в кэш
        self.device_cache[cache_key] = decision
        
        # Записываем в историю
        self.operation_history.append({
            'profile': operation_profile,
            'device': str(decision),
            'time': time.time()
        })
        
        return decision
    
    def _make_device_decision(self, profile: Dict[str, Any]):
        """Внутренняя логика принятия решения"""
        if not self.gpu_available:
            return torch.device('cpu')
        
        # Извлекаем параметры
        op_type = profile.get('type', 'unknown')
        size = profile.get('size', 0)
        batch_size = profile.get('batch_size', 1)
        priority = profile.get('priority', 'normal')
        
        # Правило 1: Высокий приоритет → всегда GPU (если доступен)
        if priority == 'high' and not self.is_gpu_memory_critical():
            return torch.device('cuda')
        
        # Правило 2: Маленький батч → CPU
        if batch_size < self.min_batch_size_for_gpu:
            return torch.device('cpu')
        
        # Правило 3: Очень маленькие данные → CPU (overhead)
        if size < 100:
            return torch.device('cpu')
        
        # Правило 4: Критичная память → CPU
        if self.is_gpu_memory_critical():
            return torch.device('cpu')
        
        # Правило 5: В зависимости от типа операции
        if op_type == 'matmul' and size >= 200:
            return torch.device('cuda')
        elif op_type == 'elementwise' and size >= 1000:
            return torch.device('cuda')
        elif op_type == 'conv':
            return torch.device('cuda')
        
        # По умолчанию CPU для безопасности
        return torch.device('cpu')
    
    def get_statistics(self):
        """Выводит статистику использования"""
        if not self.operation_history:
            print("Нет истории операций")
            return
        
        total = len(self.operation_history)
        gpu_count = sum(1 for op in self.operation_history if 'cuda' in op['device'])
        cpu_count = total - gpu_count
        
        print(f"\n{'='*70}")
        print("Статистика DeviceManager")
        print('='*70)
        print(f"Всего операций: {total}")
        print(f"На GPU: {gpu_count} ({gpu_count/total*100:.1f}%)")
        print(f"На CPU: {cpu_count} ({cpu_count/total*100:.1f}%)")
        print(f"Размер кэша: {len(self.device_cache)}")
        if self.gpu_available:
            print(f"Текущее использование GPU памяти: {self.get_gpu_memory_usage()*100:.1f}%")

# Демонстрация работы DeviceManager
print("=" * 70)
print("Демонстрация DeviceManager")
print("=" * 70)

manager = DeviceManager(memory_threshold=0.7, min_batch_size_for_gpu=16)

# Тестовые сценарии
test_profiles = [
    {'type': 'matmul', 'size': 50, 'batch_size': 8, 'priority': 'normal'},
    {'type': 'matmul', 'size': 500, 'batch_size': 32, 'priority': 'normal'},
    {'type': 'elementwise', 'size': 100, 'batch_size': 32, 'priority': 'normal'},
    {'type': 'elementwise', 'size': 2000, 'batch_size': 64, 'priority': 'normal'},
    {'type': 'conv', 'size': 128, 'batch_size': 16, 'priority': 'high'},
    {'type': 'matmul', 'size': 5000, 'batch_size': 128, 'priority': 'high'},
]

print("\n" + "=" * 70)
print("Тестирование различных профилей операций")
print("=" * 70)

for i, profile in enumerate(test_profiles, 1):
    device = manager.get_optimal_device(profile)
    print(f"\n{i}. Профиль: {profile}")
    print(f"   Выбрано устройство: {device}")

# Симуляция загрузки GPU памяти
if torch.cuda.is_available():
    print("\n" + "=" * 70)
    print("Симуляция заполнения памяти GPU")
    print("=" * 70)
    
    # Создаем большие тензоры на GPU
    tensors = []
    for i in range(3):
        try:
            t = torch.randn(2000, 2000, device='cuda')
            tensors.append(t)
            usage = manager.get_gpu_memory_usage()
            print(f"Создан тензор {i+1}: использование памяти {usage*100:.1f}%")
        except RuntimeError:
            print(f"Недостаточно памяти для тензора {i+1}")
            break
    
    # Теперь пробуем получить устройство при заполненной памяти
    print("\nПопытка выбора устройства при заполненной памяти:")
    profile_high_priority = {'type': 'matmul', 'size': 1000, 'batch_size': 32, 'priority': 'high'}
    device = manager.get_optimal_device(profile_high_priority)
    print(f"Для high priority операции выбрано: {device}")
    
    # Очистка
    del tensors
    torch.cuda.empty_cache()

# Итоговая статистика
manager.get_statistics()

print("\n" + "=" * 70)
print("💡 Преимущества DeviceManager:")
print("=" * 70)
print("1. Автоматический выбор оптимального устройства")
print("2. Защита от переполнения памяти GPU")
print("3. Кэширование решений для однотипных операций")
print("4. Учет приоритета операций")
print("5. Адаптация к размеру батча и данных")
```
</details>

---

### **Задача 6: Детектор утечек памяти GPU**
Создайте инструмент для обнаружения утечек памяти в процессе обучения нейросети.

Реализуйте класс `GPUMemoryProfiler`, который:
1. Отслеживает использование памяти на каждой итерации обучения
2. Детектирует аномальный рост памяти (утечки)
3. Идентифицирует потенциальные причины утечек
4. Предоставляет рекомендации по исправлению

**Требования:**
- Метод `start_monitoring()` - начать мониторинг
- Метод `log_iteration(iteration_num)` - записать состояние памяти
- Метод `detect_leaks()` - анализ и обнаружение утечек
- Метод `get_recommendations()` - рекомендации по исправлению

**Подсказки для детектирования:**
- Если память растет линейно с каждой итерацией → вероятно утечка
- Проверьте: сохранение тензоров с `.requires_grad=True`, накопление списков тензоров, отсутствие `.detach()`

<details>
<summary>Решение</summary>

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class GPUMemoryProfiler:
    """
    Профайлер для детектирования утечек памяти GPU
    """
    def __init__(self):
        self.memory_log = []
        self.is_monitoring = False
        self.baseline_memory = 0
        
        if not torch.cuda.is_available():
            print("⚠️  GPU недоступен")
    
    def start_monitoring(self):
        """Начать мониторинг памяти"""
        if not torch.cuda.is_available():
            return
        
        torch.cuda.reset_peak_memory_stats()
        self.memory_log = []
        self.baseline_memory = torch.cuda.memory_allocated()
        self.is_monitoring = True
        print(f"✅ Мониторинг начат. Базовая память: {self.baseline_memory/1e6:.2f} MB")
    
    def log_iteration(self, iteration_num):
        """Записать состояние памяти на текущей итерации"""
        if not self.is_monitoring or not torch.cuda.is_available():
            return
        
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak = torch.cuda.max_memory_allocated()
        
        self.memory_log.append({
            'iteration': iteration_num,
            'allocated': allocated,
            'reserved': reserved,
            'peak': peak,
            'growth': allocated - self.baseline_memory
        })
    
    def detect_leaks(self, threshold=1e6):
        """
        Детектирование утечек памяти
        
        Args:
            threshold: порог роста памяти (байт) для считывания как утечка
        
        Returns:
            dict с результатами анализа
        """
        if not self.memory_log:
            return {'has_leak': False, 'message': 'Нет данных для анализа'}
        
        # Анализ тренда роста памяти
        iterations = [log['iteration'] for log in self.memory_log]
        growth = [log['growth'] for log in self.memory_log]
        
        # Линейная регрессия для определения тренда
        if len(iterations) > 2:
            coeffs = np.polyfit(iterations, growth, 1)
            slope = coeffs[0]  # коэффициент наклона
            
            # Если наклон положительный и значительный → утечка
            is_leaking = slope > threshold / len(iterations)
            
            total_growth = growth[-1] - growth[0] if len(growth) > 1 else 0
            
            result = {
                'has_leak': is_leaking,
                'slope': slope,
                'total_growth': total_growth,
                'growth_per_iteration': slope,
                'final_memory': self.memory_log[-1]['allocated']
            }
            
            return result
        else:
            return {'has_leak': False, 'message': 'Недостаточно данных'}
    
    def get_recommendations(self, leak_info):
        """Получить рекомендации по устранению утечек"""
        if not leak_info.get('has_leak', False):
            return ["✅ Утечек памяти не обнаружено!"]
        
        recommendations = [
            "⚠️  Обнаружена вероятная утечка памяти!",
            "",
            "🔍 Проверьте следующее:",
            "",
            "1. Сохранение loss/метрик:",
            "   ❌ losses.append(loss)  # сохраняет граф вычислений!",
            "   ✅ losses.append(loss.item())  # сохраняет только значение",
            "",
            "2. Накопление тензоров в списках:",
            "   ❌ all_predictions.append(predictions)  # если predictions на GPU",
            "   ✅ all_predictions.append(predictions.detach().cpu())",
            "",
            "3. Отсутствие .detach() при сохранении:",
            "   ❌ saved_tensor = intermediate_result",
            "   ✅ saved_tensor = intermediate_result.detach()",
            "",
            "4. Не вызывается optimizer.zero_grad():",
            "   ✅ Всегда вызывайте optimizer.zero_grad() перед backward()",
            "",
            "5. Промежуточные вычисления вне with torch.no_grad():",
            "   ✅ Используйте torch.no_grad() для inference",
            "",
            f"📊 Статистика:",
            f"   Рост памяти за итерацию: {leak_info.get('growth_per_iteration', 0)/1e6:.2f} MB",
            f"   Общий рост: {leak_info.get('total_growth', 0)/1e6:.2f} MB"
        ]
        
        return recommendations
    
    def visualize(self):
        """Визуализация использования памяти"""
        if not self.memory_log:
            print("Нет данных для визуализации")
            return
        
        iterations = [log['iteration'] for log in self.memory_log]
        allocated = [log['allocated'] / 1e6 for log in self.memory_log]
        reserved = [log['reserved'] / 1e6 for log in self.memory_log]
        growth = [log['growth'] / 1e6 for log in self.memory_log]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # График 1: Абсолютное использование памяти
        ax = axes[0]
        ax.plot(iterations, allocated, 'b-', label='Allocated', linewidth=2)
        ax.plot(iterations, reserved, 'r--', label='Reserved', linewidth=2, alpha=0.7)
        ax.set_xlabel('Итерация')
        ax.set_ylabel('Память (MB)')
        ax.set_title('Использование памяти GPU')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # График 2: Рост памяти относительно базовой
        ax = axes[1]
        ax.plot(iterations, growth, 'g-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Итерация')
        ax.set_ylabel('Рост памяти (MB)')
        ax.set_title('Рост памяти относительно начала')
        ax.grid(True, alpha=0.3)
        
        # Добавляем линию тренда
        if len(iterations) > 2:
            z = np.polyfit(iterations, growth, 1)
            p = np.poly1d(z)
            ax.plot(iterations, p(iterations), "r--", alpha=0.8, label=f'Тренд: {z[0]:.3f} MB/iter')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('memory_profiling.png', dpi=100)
        print("\n📊 График сохранен в 'memory_profiling.png'")

# Демонстрация детектирования утечек
def train_with_leak(epochs=20):
    """Пример обучения С утечкой памяти"""
    model = nn.Linear(100, 10).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    losses = []  # ОШИБКА: будем сохранять тензоры с графом!
    
    profiler = GPUMemoryProfiler()
    profiler.start_monitoring()
    
    for epoch in range(epochs):
        x = torch.randn(32, 100, device='cuda')
        y = torch.randn(32, 10, device='cuda')
        
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        
        # ОШИБКА: сохраняем тензор с графом вычислений!
        losses.append(loss)
        
        loss.backward()
        optimizer.step()
        
        profiler.log_iteration(epoch)
    
    return profiler, losses

def train_without_leak(epochs=20):
    """Пример обучения БЕЗ утечки памяти"""
    model = nn.Linear(100, 10).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    losses = []
    
    profiler = GPUMemoryProfiler()
    profiler.start_monitoring()
    
    for epoch in range(epochs):
        x = torch.randn(32, 100, device='cuda')
        y = torch.randn(32, 10, device='cuda')
        
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        
        # ПРАВИЛЬНО: сохраняем только значение!
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        profiler.log_iteration(epoch)
    
    return profiler, losses

# Тестирование
if torch.cuda.is_available():
    print("=" * 70)
    print("Тест 1: Обучение С утечкой памяти")
    print("=" * 70)
    profiler_leak, losses_leak = train_with_leak(epochs=30)
    leak_info = profiler_leak.detect_leaks(threshold=0.5e6)
    
    print(f"\n{'='*70}")
    print("РЕЗУЛЬТАТЫ АНАЛИЗА (С утечкой)")
    print('='*70)
    for rec in profiler_leak.get_recommendations(leak_info):
        print(rec)
    
    profiler_leak.visualize()
    
    # Очищаем память
    del losses_leak
    torch.cuda.empty_cache()
    
    print("\n\n" + "=" * 70)
    print("Тест 2: Обучение БЕЗ утечки памяти")
    print("=" * 70)
    profiler_clean, losses_clean = train_without_leak(epochs=30)
    clean_info = profiler_clean.detect_leaks(threshold=0.5e6)
    
    print(f"\n{'='*70}")
    print("РЕЗУЛЬТАТЫ АНАЛИЗА (БЕЗ утечки)")
    print('='*70)
    for rec in profiler_clean.get_recommendations(clean_info):
        print(rec)
    
    print("\n" + "=" * 70)
    print("💡 Итоговое сравнение")
    print("=" * 70)
    print(f"С утечкой - рост памяти: {leak_info.get('total_growth', 0)/1e6:.2f} MB")
    print(f"Без утечки - рост памяти: {clean_info.get('total_growth', 0)/1e6:.2f} MB")
    print(f"\nРазница: {(leak_info.get('total_growth', 0) - clean_info.get('total_growth', 0))/1e6:.2f} MB")
else:
    print("⚠️  GPU недоступен. Запустите в Google Colab с GPU.")
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
2. Реализуйте обычное обучение (FP32)
3. Реализуйте обучение с Mixed Precision (FP16 + FP32)
4. Сравните скорость и использование памяти

**Что такое Mixed Precision?**
- Использует FP16 (16-bit) для большинства операций → быстрее и меньше памяти
- Использует FP32 (32-bit) для критичных операций → сохраняет точность
- GradScaler предотвращает underflow градиентов в FP16

**Требуется:** PyTorch 1.6+ и GPU с Compute Capability >= 7.0 (Volta или новее, например T4, V100, A100)

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
        """Обучение с Mixed Precision (FP16 + FP32)"""
        model = SimpleNet().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # GradScaler масштабирует градиенты для предотвращения underflow в FP16
        scaler = GradScaler()
        
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # autocast() автоматически выбирает FP16 или FP32 для каждой операции
            with autocast():
                outputs = model(x)
                loss = criterion(outputs, y)
            
            # Масштабирование градиентов (умножение на большое число)
            scaler.scale(loss).backward()
            
            # Демасштабирование и обновление весов
            scaler.step(optimizer)
            
            # Обновление масштабирующего фактора для следующей итерации
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

**Что такое Pinned Memory?**
- Обычная память может быть перемещена ОС (pageable memory)
- Pinned (page-locked) память зафиксирована и не может быть перемещена
- Перенос pinned памяти на GPU быстрее, т.к. нет копирования в промежуточный буфер
- Используйте с `non_blocking=True` для асинхронного переноса

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
    
    # Обычная память (pageable) - может быть перемещена ОС
    tensor_regular = torch.randn(size, size)
    
    # Pinned память (page-locked) - зафиксирована в RAM
    tensor_pinned = torch.randn(size, size).pin_memory()
    
    print("💡 Pinned memory ускоряет перенос CPU → GPU")
    print(f"Размер тензора: {size}x{size}")
    print(f"Итераций: {iterations}\n")
    
    # Бенчмарк для обычной памяти
    start = time.time()
    for _ in range(iterations):
        _ = tensor_regular.to('cuda')
    torch.cuda.synchronize()
    time_regular = time.time() - start
    
    # Бенчмарк для pinned памяти с асинхронным переносом
    start = time.time()
    for _ in range(iterations):
        _ = tensor_pinned.to('cuda', non_blocking=True)
    torch.cuda.synchronize()
    time_pinned = time.time() - start
    
    print(f"Обычная память: {time_regular:.4f}s")
    print(f"Pinned память:  {time_pinned:.4f}s")
    print(f"Ускорение: {time_regular/time_pinned:.2f}x")
    
    print("\n💡 Используйте pinned memory для:")
    print("   - DataLoader с pin_memory=True")
    print("   - Частых переносов данных CPU → GPU")
    print("   ⚠️ Не злоупотребляйте - pinned память ограничена!")
```
</details>

---

## **🔧 Troubleshooting: Частые проблемы и решения**

### **Проблема 1: RuntimeError: Expected all tensors to be on the same device**
```python
# ❌ Ошибка
model = MyModel().cuda()
data = torch.randn(10, 5)  # на CPU
output = model(data)  # RuntimeError!
```
**Решение:** Убедитесь, что модель и данные на одном устройстве
```python
# ✅ Правильно
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MyModel().to(device)
data = torch.randn(10, 5).to(device)
output = model(data)
```

### **Проблема 2: CUDA out of memory**
**Причины:**
- Слишком большой размер батча
- Утечка памяти (сохранение тензоров с вычислительным графом)
- Накопление промежуточных результатов

**Решения:**
```python
# 1. Уменьшите размер батча
batch_size = 32  # вместо 128

# 2. Используйте .item() для скалярных значений
losses = []
for epoch in range(100):
    loss = compute_loss()
    losses.append(loss.item())  # ✅ вместо losses.append(loss)

# 3. Очищайте память
del large_tensor
torch.cuda.empty_cache()

# 4. Используйте gradient accumulation для больших батчей
accumulation_steps = 4
for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### **Проблема 3: Медленная работа с GPU (медленнее CPU)**
**Причины:**
- Маленький размер данных (overhead переноса > выигрыш от GPU)
- Частые переносы CPU ↔ GPU
- Не используется `torch.cuda.synchronize()` при измерении времени

**Решения:**
```python
# 1. Увеличьте размер данных/батчей
# 2. Минимизируйте переносы между устройствами
# 3. Правильно измеряйте время:
torch.cuda.synchronize()  # перед началом
start = time.time()
# ... ваш код ...
torch.cuda.synchronize()  # перед замером
elapsed = time.time() - start
```

### **Проблема 4: Забыли обнулить градиенты**
```python
# ❌ Ошибка - градиенты накапливаются
for epoch in range(100):
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()  # Градиенты накапливаются!
```
**Решение:**
```python
# ✅ Правильно
for epoch in range(100):
    optimizer.zero_grad()  # Обнуляем перед backward
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
```

### **Проблема 5: Утечка памяти при сохранении loss**
```python
# ❌ Ошибка - сохраняем тензор с вычислительным графом
losses = []
for epoch in range(1000):
    loss = compute_loss()
    losses.append(loss)  # Утечка памяти!
```
**Решение:**
```python
# ✅ Правильно - сохраняем только значение
losses = []
for epoch in range(1000):
    loss = compute_loss()
    losses.append(loss.item())  # Сохраняем Python float
    # или
    losses.append(loss.detach().cpu())  # Если нужен тензор без графа
```

### **Проблема 6: Нет ускорения от GPU**
**Чек-лист проверки:**
```python
# 1. Проверьте, что модель на GPU
print(f"Модель на GPU: {next(model.parameters()).is_cuda}")

# 2. Проверьте, что данные на GPU
print(f"Данные на GPU: {x.is_cuda}")

# 3. Проверьте размер данных (для маленьких данных GPU медленнее)
print(f"Размер данных: {x.shape}")

# 4. Используйте подходящий batch size (32-256 обычно оптимально)
# 5. Убедитесь, что используете GPU-оптимизированные операции
```

### **💡 Советы по оптимизации**

1. **Используйте Mixed Precision** (на GPU с Compute Capability >= 7.0):
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for data, target in dataloader:
    optimizer.zero_grad()
    with autocast():  # Автоматическое использование FP16
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

2. **Включите cuDNN autotuner** для фиксированных размеров входа:
```python
torch.backends.cudnn.benchmark = True
```

3. **Используйте DataLoader с num_workers** для асинхронной загрузки:
```python
dataloader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
```

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
