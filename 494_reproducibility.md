# Повторяемость экспериментов (Reproducibility)

## 🟢 Основы воспроизводимости

### Введение

**Reproducibility (воспроизводимость)** — способность получить идентичные результаты при повторном запуске эксперимента с теми же параметрами.

**Почему это важно:**
- Отладка моделей — невозможно найти баг, если результаты меняются каждый раз
- Научная работа — результаты должны быть проверяемыми
- Production — предсказуемое поведение модели критично
- Командная работа — коллеги должны получать те же результаты

**Источники недетерминизма в ML:**
1. Инициализация весов нейросети (случайная)
2. Перемешивание данных (shuffle)
3. Dropout и другая стохастическая регуляризация
4. Аугментация данных (случайные трансформации)
5. Параллельные операции на GPU (недетерминированные алгоритмы)

### Основная концепция: Random Seed

**Random seed** — начальное значение для генератора псевдослучайных чисел (PRNG).

```python
import numpy as np

# Без фиксации seed - разные результаты
print(np.random.rand(3))  # [0.417 0.720 0.000]
print(np.random.rand(3))  # [0.302 0.147 0.092]

# С фиксацией seed - одинаковые результаты
np.random.seed(42)
print(np.random.rand(3))  # [0.374 0.950 0.731]

np.random.seed(42)
print(np.random.rand(3))  # [0.374 0.950 0.731] - те же самые!
```

**Важно:** seed нужно устанавливать в начале скрипта/ноутбука, до любых случайных операций.

---

## 🟡 Воспроизводимость в NumPy

### Установка seed

```python
import numpy as np

# Глобальный seed для всех операций NumPy
np.random.seed(42)

# Альтернативный способ (рекомендуемый с NumPy 1.17+)
rng = np.random.default_rng(seed=42)
```

### Примеры случайных операций

```python
import numpy as np

np.random.seed(42)

# Случайные числа
random_numbers = np.random.rand(5)
print(f"Random numbers: {random_numbers}")

# Случайные целые числа
random_ints = np.random.randint(0, 100, size=5)
print(f"Random integers: {random_ints}")

# Перемешивание массива
arr = np.array([1, 2, 3, 4, 5])
np.random.shuffle(arr)
print(f"Shuffled array: {arr}")

# Выбор случайных элементов
choices = np.random.choice([1, 2, 3, 4, 5], size=3, replace=False)
print(f"Random choices: {choices}")

# Нормальное распределение
normal = np.random.normal(loc=0, scale=1, size=1000)
print(f"Normal distribution mean: {normal.mean():.4f}")
```

### Современный подход: Generator API

```python
import numpy as np

# Создаем генератор с фиксированным seed
rng = np.random.default_rng(seed=42)

# Используем методы генератора
random_numbers = rng.random(5)
random_ints = rng.integers(0, 100, size=5)
normal = rng.normal(loc=0, scale=1, size=1000)

print(f"Random numbers: {random_numbers}")
print(f"Random integers: {random_ints}")
print(f"Normal mean: {normal.mean():.4f}")
```

**Преимущества Generator API:**
- Изолированное состояние (можно иметь несколько генераторов)
- Более современный и гибкий интерфейс
- Лучшие алгоритмы генерации случайных чисел

---

## 🟡 Воспроизводимость в PyTorch (CPU)

### Базовая настройка

```python
import torch
import random
import numpy as np

def set_seed(seed=42):
    """Устанавливает seed для воспроизводимости на CPU"""
    random.seed(seed)           # Python random
    np.random.seed(seed)        # NumPy
    torch.manual_seed(seed)     # PyTorch CPU
    
set_seed(42)
```

### Влияние seed на разные операции

```python
import torch

# Устанавливаем seed
torch.manual_seed(42)

# 1. Инициализация весов
model = torch.nn.Linear(10, 5)
print(f"Weights shape: {model.weight.shape}")
print(f"First 3 weights: {model.weight.data[0, :3]}")

# 2. Случайные тензоры
random_tensor = torch.randn(3, 3)
print(f"\nRandom tensor:\n{random_tensor}")

# 3. Dropout (стохастическая регуляризация)
dropout = torch.nn.Dropout(p=0.5)
x = torch.ones(10)
dropped = dropout(x)
print(f"\nAfter dropout: {dropped}")

# 4. Перемешивание данных в DataLoader
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
loader = DataLoader(dataset, batch_size=10, shuffle=True)

# Первый батч будет одинаковым при одном seed
first_batch = next(iter(loader))
print(f"\nFirst batch shape: {first_batch[0].shape}")
```

### Детерминированные алгоритмы

```python
import torch

# Полная воспроизводимость (может замедлить обучение)
torch.use_deterministic_algorithms(True)

# Или более мягкий вариант (для большинства операций)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Важно:** `use_deterministic_algorithms(True)` может сильно замедлить обучение, так как отключает оптимизированные недетерминированные алгоритмы.

---

## 🔴 Воспроизводимость с CUDA (GPU)

### Базовая настройка для GPU

```python
import torch
import random
import numpy as np
import os

def set_seed_gpu(seed=42):
    """Устанавливает seed для воспроизводимости на GPU"""
    # Python, NumPy, PyTorch CPU
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # PyTorch GPU (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # для multi-GPU
    
    # Детерминированные алгоритмы в cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Дополнительно для полной воспроизводимости
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
set_seed_gpu(42)
```

### Понимание cuDNN и недетерминизм

**cuDNN** — библиотека NVIDIA для оптимизации операций нейросетей на GPU.

```python
import torch

# Режим 1: Максимальная скорость (недетерминировано)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# → Автоматически выбирает fastest алгоритмы
# → Результаты могут различаться между запусками

# Режим 2: Детерминированность (медленнее)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# → Использует детерминированные алгоритмы
# → Результаты всегда одинаковые

# Режим 3: Быстрый детерминизм (компромисс)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# Используем конкретные алгоритмы для convolution
```

### Проверка воспроизводимости на GPU

```python
import torch

def test_reproducibility(seed=42, device='cuda'):
    """Проверяем воспроизводимость результатов"""
    
    results = []
    
    for run in range(3):
        # Устанавливаем seed перед каждым запуском
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)
        
        # Простая модель
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1)
        ).to(device)
        
        # Случайные данные
        x = torch.randn(100, 10, device=device)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        results.append(output.sum().item())
    
    print(f"Run 1: {results[0]:.10f}")
    print(f"Run 2: {results[1]:.10f}")
    print(f"Run 3: {results[2]:.10f}")
    
    # Проверяем идентичность
    if results[0] == results[1] == results[2]:
        print("✓ Полная воспроизводимость!")
    else:
        print("✗ Результаты различаются")
        print(f"Max diff: {max(results) - min(results):.10f}")

# Проверка на GPU
if torch.cuda.is_available():
    print("Testing on GPU:")
    set_seed_gpu(42)
    test_reproducibility(seed=42, device='cuda')
else:
    print("CUDA not available, testing on CPU:")
    test_reproducibility(seed=42, device='cpu')
```

### Multi-GPU воспроизводимость

```python
import torch
import torch.distributed as dist

def set_seed_multigpu(seed=42, rank=0):
    """Seed для multi-GPU training"""
    # Базовый seed
    torch.manual_seed(seed)
    
    # Каждая GPU получает свой seed (детерминированный)
    # Это важно для DataParallel и DistributedDataParallel
    torch.cuda.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    
    # Детерминированность
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Пример для DistributedDataParallel
def setup_distributed(rank, world_size, seed=42):
    """Настройка для распределенного обучения"""
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Установка seed с учетом rank
    set_seed_multigpu(seed, rank)
    
    # Каждый процесс работает на своей GPU
    torch.cuda.set_device(rank)
```

---

## 🔴 Полный пример воспроизводимого обучения

### Комплексная функция настройки

```python
import torch
import numpy as np
import random
import os

def set_all_seeds(seed=42, use_deterministic=True):
    """
    Полная настройка воспроизводимости для deep learning.
    
    Args:
        seed: значение seed
        use_deterministic: использовать ли полностью детерминированные алгоритмы
                          (может замедлить обучение на 10-50%)
    """
    print(f"Setting seed: {seed}")
    
    # 1. Python built-in random
    random.seed(seed)
    
    # 2. NumPy
    np.random.seed(seed)
    
    # 3. PyTorch CPU
    torch.manual_seed(seed)
    
    # 4. PyTorch GPU (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU
        
        # cuDNN settings
        if use_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("✓ Using deterministic cuDNN algorithms")
        else:
            torch.backends.cudnn.benchmark = True
            print("⚠ Using fast non-deterministic cuDNN algorithms")
    
    # 5. Environment variables для полной детерминированности
    if use_deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Включаем детерминированные алгоритмы PyTorch
        try:
            torch.use_deterministic_algorithms(True)
            print("✓ Using deterministic PyTorch algorithms")
        except Exception as e:
            print(f"⚠ Could not enable deterministic algorithms: {e}")
    
    print("✓ Seed configuration complete\n")

# Использование
set_all_seeds(seed=42, use_deterministic=True)
```

### Воспроизводимый training loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def create_reproducible_dataloader(X, y, batch_size=32, seed=42):
    """Создает DataLoader с воспроизводимым shuffle"""
    dataset = TensorDataset(X, y)
    
    # Generator для DataLoader - обеспечивает воспроизводимость shuffle
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,  # Важно!
        num_workers=0  # Для воспроизводимости лучше 0
    )
    
    return loader

def train_reproducible(seed=42):
    """Полностью воспроизводимое обучение"""
    
    # 1. Установка seed
    set_all_seeds(seed, use_deterministic=True)
    
    # 2. Генерация данных (с seed!)
    torch.manual_seed(seed)
    X = torch.randn(1000, 20)
    y = (X.sum(dim=1) > 0).long()
    
    # 3. Создание воспроизводимого DataLoader
    train_loader = create_reproducible_dataloader(X, y, batch_size=32, seed=seed)
    
    # 4. Инициализация модели (seed уже установлен)
    model = nn.Sequential(
        nn.Linear(20, 50),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(50, 2)
    )
    
    # 5. GPU если доступно
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 6. Optimizer и criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 7. Training loop
    print("Starting training...")
    model.train()
    
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        # Логирование
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}, Acc = {accuracy:.2f}%")
    
    return model

# Запуск 1
print("=" * 50)
print("Run 1:")
print("=" * 50)
model1 = train_reproducible(seed=42)

# Запуск 2 (должен дать идентичные результаты)
print("\n" + "=" * 50)
print("Run 2:")
print("=" * 50)
model2 = train_reproducible(seed=42)

# Проверка идентичности весов
weights_match = torch.allclose(
    list(model1.parameters())[0],
    list(model2.parameters())[0]
)
print(f"\n✓ Weights identical: {weights_match}")
```

---

## 🟢 Частые ошибки и решения

### Ошибка 1: Забыли установить seed для DataLoader

```python
# ❌ Неправильно - результаты не воспроизводятся
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ✓ Правильно - с генератором
generator = torch.Generator()
generator.manual_seed(42)
loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=generator)
```

### Ошибка 2: Seed устанавливается слишком поздно

```python
# ❌ Неправильно - операции до set_seed будут случайными
X = torch.randn(100, 10)  # Эти значения случайны!
set_all_seeds(42)
model = nn.Linear(10, 5)

# ✓ Правильно - seed в самом начале
set_all_seeds(42)
X = torch.randn(100, 10)
model = nn.Linear(10, 5)
```

### Ошибка 3: Использование num_workers > 0

```python
# ⚠ Может нарушить воспроизводимость
loader = DataLoader(dataset, num_workers=4, shuffle=True)

# ✓ Для полной воспроизводимости используйте num_workers=0
loader = DataLoader(dataset, num_workers=0, shuffle=True, generator=generator)

# Или настройте worker_init_fn
def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)

loader = DataLoader(
    dataset, 
    num_workers=4, 
    shuffle=True, 
    generator=generator,
    worker_init_fn=worker_init_fn
)
```

### Ошибка 4: Забыли про environment variables

```python
import os

# Важно для некоторых операций (особенно на GPU)
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Эти переменные нужно установить ДО импорта PyTorch!
```

---

## 🟡 Влияние на производительность

### Бенчмарк: детерминизм vs скорость

```python
import torch
import time

def benchmark_training(deterministic=True, epochs=10):
    """Сравнение скорости детерминированного и недетерминированного обучения"""
    
    # Настройка
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    # Модель и данные
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, 3),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(128, 10)
    ).to(device)
    
    X = torch.randn(128, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (128,), device=device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Warmup
    for _ in range(2):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    
    # Benchmark
    start = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    
    elapsed = time.time() - start
    
    return elapsed

# Сравнение
if torch.cuda.is_available():
    print("Benchmarking on GPU:")
    time_det = benchmark_training(deterministic=True)
    time_fast = benchmark_training(deterministic=False)
    
    print(f"Deterministic: {time_det:.3f}s")
    print(f"Fast (non-det): {time_fast:.3f}s")
    print(f"Slowdown: {(time_det/time_fast - 1)*100:.1f}%")
else:
    print("GPU not available, skipping benchmark")
```

**Типичные результаты:**
- CPU: разница незначительна (< 5%)
- GPU: детерминированный режим медленнее на 10-30%
- Сложные операции (conv, RNN): разница больше

### Когда использовать детерминизм

**✓ Используйте детерминистичные настройки:**
- Отладка и разработка моделей
- Научные исследования (публикации)
- A/B тестирование моделей
- Критичные production системы

**✗ Можно не использовать:**
- Production inference (нет обучения)
- Большие модели где скорость критична
- Ансамбли моделей (вариативность полезна)

---

## 🔴 Практические рекомендации

### 1. Структура проекта с воспроизводимостью

```python
# config.py
from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    deterministic: bool = True
    device: str = 'cuda'
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10

# utils.py
import torch
import numpy as np
import random
import os

def setup_environment(config):
    """Полная настройка окружения"""
    # Seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        
        if config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    return device

# main.py
from config import Config
from utils import setup_environment

def main():
    config = Config(seed=42, deterministic=True)
    device = setup_environment(config)
    
    # Теперь можно обучать модель
    # ...

if __name__ == '__main__':
    main()
```

### 2. Логирование для воспроизводимости

```python
import json
from datetime import datetime

def log_experiment_info(config, model, results, filename='experiment_log.json'):
    """Сохраняет всю информацию для воспроизведения эксперимента"""
    
    experiment_info = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'seed': config.seed,
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            'deterministic': config.deterministic
        },
        'environment': {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        },
        'model': {
            'architecture': str(model),
            'num_parameters': sum(p.numel() for p in model.parameters())
        },
        'results': results
    }
    
    with open(filename, 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    print(f"✓ Experiment info saved to {filename}")

# Использование
log_experiment_info(
    config=config,
    model=model,
    results={'train_loss': 0.234, 'val_acc': 0.89}
)
```

### 3. Чеклист воспроизводимости

```python
def check_reproducibility_setup():
    """Проверка настроек воспроизводимости"""
    
    checks = []
    
    # 1. Seed установлен
    try:
        torch.initial_seed()
        checks.append(('✓', 'PyTorch seed is set'))
    except:
        checks.append(('✗', 'PyTorch seed NOT set'))
    
    # 2. cuDNN детерминизм
    if torch.cuda.is_available():
        if torch.backends.cudnn.deterministic:
            checks.append(('✓', 'cuDNN deterministic mode ON'))
        else:
            checks.append(('⚠', 'cuDNN deterministic mode OFF'))
        
        if torch.backends.cudnn.benchmark:
            checks.append(('⚠', 'cuDNN benchmark mode ON (non-deterministic)'))
        else:
            checks.append(('✓', 'cuDNN benchmark mode OFF'))
    
    # 3. Environment variables
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
        checks.append(('✓', 'CUBLAS_WORKSPACE_CONFIG is set'))
    else:
        checks.append(('⚠', 'CUBLAS_WORKSPACE_CONFIG not set'))
    
    if os.environ.get('PYTHONHASHSEED'):
        checks.append(('✓', 'PYTHONHASHSEED is set'))
    else:
        checks.append(('⚠', 'PYTHONHASHSEED not set'))
    
    # Печать результатов
    print("Reproducibility Setup Check:")
    print("=" * 50)
    for status, message in checks:
        print(f"{status} {message}")
    print("=" * 50)

# Использование
set_all_seeds(42, use_deterministic=True)
check_reproducibility_setup()
```

---

## Заключение

### Ключевые выводы

1. **Всегда устанавливайте seed** в начале эксперимента
2. **Используйте Generator** для DataLoader
3. **Логируйте seed** вместе с результатами
4. **Знайте компромиссы** между скоростью и детерминизмом
5. **Тестируйте воспроизводимость** периодически

### Минимальный шаблон

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# В начале каждого скрипта/ноутбука
set_seed(42)
```

### Полезные ссылки

- [PyTorch Reproducibility Guide](https://pytorch.org/docs/stable/notes/randomness.html)
- [NumPy Random Generator](https://numpy.org/doc/stable/reference/random/generator.html)
- [cuDNN Determinism](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html)

---

## Дополнительные материалы

### Воспроизводимость в других библиотеках

```python
# TensorFlow
import tensorflow as tf
tf.random.set_seed(42)

# Scikit-learn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)

# Pandas (для shuffle)
df_shuffled = df.sample(frac=1, random_state=42)
```

### Git commit для воспроизводимости

```bash
# Сохраняйте точную версию кода
git log -1 --format="%H" > experiment_commit.txt

# В лог эксперимента добавляйте git hash
echo "Experiment run with git commit: $(git rev-parse HEAD)"
```

### Docker для полной воспроизводимости

```dockerfile
# Dockerfile для воспроизводимости окружения
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /workspace

# Точные версии зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Фиксируем seed через environment variable
ENV RANDOM_SEED=42

CMD ["python", "train.py"]
```

**Главное правило воспроизводимости:** 
> *Фиксируйте ВСЕ источники случайности и логируйте версии всех зависимостей!*
