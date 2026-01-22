## 🟢 Базовый уровень: Основные концепции

### 1. Что такое Dataset?

`Dataset` — это класс, который инкапсулирует ваши данные. PyTorch предоставляет готовые классы для разных типов данных:

```python
from torch.utils.data import Dataset, TensorDataset
import torch

# Самый простой способ: TensorDataset для данных в виде тензоров
# Данные уже должны быть тензорами
data_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
labels_tensor = torch.tensor([0, 1, 0, 1])

# Создаем датасет из тензоров
dataset = TensorDataset(data_tensor, labels_tensor)

print(f"Размер датасета: {len(dataset)}")
print(f"Элемент 0: данные={dataset[0][0]}, метка={dataset[0][1]}")

# Когда нужен кастомный Dataset?
class CustomDataset(Dataset):
    """Используем, когда данные требуют специальной обработки"""
    
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Сравнение подходов
simple_data = [[1, 2], [3, 4]]
simple_labels = [0, 1]

# Вариант 1: Создаем тензоры и используем TensorDataset
data_ready = torch.tensor(simple_data, dtype=torch.float32)
labels_ready = torch.tensor(simple_labels, dtype=torch.long)
dataset1 = TensorDataset(data_ready, labels_ready)

# Вариант 2: Используем кастомный Dataset
dataset2 = CustomDataset(simple_data, simple_labels)

print("\nСравнение:")
print(f"TensorDataset: {dataset1[0]}")
print(f"CustomDataset: {dataset2[0]}")
```

### 2. Что такое DataLoader?

`DataLoader` — это инструмент для итерации по датасету с дополнительными возможностями:

```python
from torch.utils.data import DataLoader

# Создаем простой DataLoader
loader = DataLoader(
    dataset=dataset,
    batch_size=2,      # Размер батча
    shuffle=True,      # Перемешивать ли данные
    num_workers=0      # Количество процессов для загрузки
)

print("Итерация по батчам:")
for batch_idx, (batch_data, batch_labels) in enumerate(loader):
    print(f"\nБатч {batch_idx}:")
    print(f"  Данные: {batch_data}")
    print(f"  Метки: {batch_labels}")
```

### 3. Зачем нужны батчи?

**Проблема:** 
- Данные не помещаются в память GPU
- Обработка по одному примеру неэффективна

**Решение:** Загрузка данных порциями. Преимущества:
1. **Эффективность GPU** — матричные операции работают быстрее с батчами
2. **Стабильность градиентов** — усреднение по нескольким примерам
3. **Контроль памяти** — загружаем только то, что нужно

```python
# Демонстрация разных размеров батчей
large_data = torch.randn(1000, 10)
large_labels = torch.randint(0, 2, (1000,))
large_dataset = TensorDataset(large_data, large_labels)

print("Размеры батчей:")
for batch_size in [1, 32, 128, 512]:
    loader = DataLoader(large_dataset, batch_size=batch_size, shuffle=False)
    data, labels = next(iter(loader))
    print(f"batch_size={batch_size}: данные {data.shape}, метки {labels.shape}")
```

---

## 🟡 Средний уровень: Практические применения

### 1. Использование стандартных Dataset из torchvision

PyTorch предоставляет готовые датасеты для распространенных задач:

```python
import torchvision
from torchvision import datasets, transforms

# Загрузка стандартного датасета MNIST (рукописные цифры)
# Преобразования для нормализации данных
transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразует PIL Image или numpy.ndarray в тензор
    transforms.Normalize((0.5,), (0.5,))  # Нормализация к диапазону [-1, 1]
])

# Скачивание и загрузка тренировочного набора MNIST
mnist_train = datasets.MNIST(
    root='./data',          # Папка для сохранения данных
    train=True,            # Тренировочный набор
    download=True,         # Скачать если нет на диске
    transform=transform    # Применяемые преобразования
)

# Создание DataLoader для MNIST
mnist_loader = DataLoader(
    mnist_train,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

# Пример использования
print(f"MNIST тренировочный набор: {len(mnist_train)} примеров")
data_batch, labels_batch = next(iter(mnist_loader))
print(f"Размер батча: {data_batch.shape}")  # [64, 1, 28, 28]
print(f"Метки в батче: {labels_batch[:10]}")  # Первые 10 меток
```

### 2. Трансформации данных с torchvision.transforms

```python
from torchvision import transforms

# Стандартные трансформации для изображений
standard_transform = transforms.Compose([
    transforms.Resize(256),           # Изменение размера
    transforms.CenterCrop(224),       # Центральное обрезание
    transforms.ToTensor(),            # Преобразование в тензор
    transforms.Normalize(             # Нормализация
        mean=[0.485, 0.456, 0.406],   # Средние значения ImageNet
        std=[0.229, 0.224, 0.225]     # Стандартные отклонения
    )
])

# Пример с CIFAR-10 (цветные изображения 32x32)
cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar_train = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=cifar_transform
)

cifar_loader = DataLoader(cifar_train, batch_size=32, shuffle=True)
```

### 3. Разделение данных на тренировочные и тестовые

```python
from torch.utils.data import random_split, Subset
import numpy as np

# Способ 1: random_split (рекомендуется)
full_dataset = TensorDataset(
    torch.randn(1000, 10),  # 1000 примеров, 10 признаков
    torch.randint(0, 2, (1000,))  # Бинарные метки
)

# Разделяем 80% на обучение, 20% на тест
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)  # Для воспроизводимости
)

print(f"Всего данных: {len(full_dataset)}")
print(f"Тренировочные: {len(train_dataset)} ({len(train_dataset)/len(full_dataset):.0%})")
print(f"Тестовые: {len(test_dataset)} ({len(test_dataset)/len(full_dataset):.0%})")

# Способ 2: Subset (когда нужны конкретные индексы)
indices = list(range(len(full_dataset)))
np.random.seed(42)
np.random.shuffle(indices)

split_idx = int(0.8 * len(indices))
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

train_subset = Subset(full_dataset, train_indices)
test_subset = Subset(full_dataset, test_indices)
```

---

## 🔴 Продвинутый уровень: Оптимизация и специальные случаи

### 1. Оптимизация производительности с pin_memory

```python
# Настройки для максимальной производительности на GPU
optimized_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,          # Параллельная загрузка
    pin_memory=True,        # Критически важно для GPU!
    persistent_workers=True,  # Сохранять workers между эпохами
    drop_last=False         # Не отбрасывать последний батч
)

# Когда использовать num_workers:
# - Многоядерный процессор
# - Данные загружаются медленно (с диска, по сети)
# - Предобработка данных требует времени

# Важно: при num_workers > 0 код должен быть в if __name__ == '__main__'
# или запущен в отдельных процессах
```

### 2. Работа с несбалансированными данными

**Проблема:** В реальных задачах классы часто представлены неравномерно. Например:
- Обнаружение болезней: здоровых пациентов больше, чем больных
- Обнаружение мошенничества: нормальных транзакций больше, чем мошеннических
- Распознавание редких объектов: обычных объектов больше, чем редких

**Почему это проблема:** Если один класс встречается в 100 раз чаще другого, модель может просто всегда предсказывать частый класс и достигать 99% точности, но быть бесполезной на практике.

```python
from torch.utils.data import WeightedRandomSampler

# Создаем несбалансированный датасет
# 950 примеров класса 0, 50 примеров класса 1
n_samples = 1000
n_class_0 = 950  # 95%
n_class_1 = 50   # 5%

# Генерируем данные
data = torch.randn(n_samples, 10)
labels = torch.cat([
    torch.zeros(n_class_0),  # Класс 0
    torch.ones(n_class_1)    # Класс 1
]).long()

dataset = TensorDataset(data, labels)

print(f"Исходное распределение:")
print(f"Класс 0: {n_class_0} примеров ({n_class_0/n_samples:.1%})")
print(f"Класс 1: {n_class_1} примеров ({n_class_1/n_samples:.1%})")

# 1. Обычный DataLoader (проблема!)
normal_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Смотрим распределение в первом батче
first_batch = next(iter(normal_loader))
_, batch_labels = first_batch
class_0_count = (batch_labels == 0).sum().item()
class_1_count = (batch_labels == 1).sum().item()

print(f"\nПервый батч в обычном загрузчике:")
print(f"Класс 0: {class_0_count} примеров ({class_0_count/len(batch_labels):.1%})")
print(f"Класс 1: {class_1_count} примеров ({class_1_count/len(batch_labels):.1%})")

# 2. Решение: WeightedRandomSampler
# Вычисляем веса для балансировки
class_counts = torch.bincount(labels)  # [950, 50]
class_weights = 1.0 / class_counts.float()  # [0.00105, 0.02]

# Назначаем вес каждому примеру
sample_weights = class_weights[labels]

# Создаем сэмплер (выборщик примеров)
sampler = WeightedRandomSampler(
    weights=sample_weights,    # Примеры редкого класса имеют больший вес
    num_samples=len(dataset),  # Сколько примеров выбирать
    replacement=True           # Разрешаем выбирать один пример несколько раз
)

# Создаем DataLoader с сэмплером
balanced_loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,    # Используем сэмплер для выборки
    shuffle=False       # Не перемешиваем, так как сэмплер уже делает случайную выборку
)

# Проверяем результат
first_batch_balanced = next(iter(balanced_loader))
_, batch_labels_balanced = first_batch_balanced
class_0_balanced = (batch_labels_balanced == 0).sum().item()
class_1_balanced = (batch_labels_balanced == 1).sum().item()

print(f"\nПервый батч в сбалансированном загрузчике:")
print(f"Класс 0: {class_0_balanced} примеров ({class_0_balanced/len(batch_labels_balanced):.1%})")
print(f"Класс 1: {class_1_balanced} примеров ({class_1_balanced/len(batch_labels_balanced):.1%})")
```

**Что произошло:**
1. **WeightedRandomSampler** дает редкому классу (класс 1) больший вес
2. Каждый пример класса 1 имеет вес 0.02, а класса 0 — 0.00105
3. При выборке примеры класса 1 выбираются примерно в 20 раз чаще
4. В результате в каждом батче получается примерно равное количество примеров каждого класса

**Когда использовать:**
- Когда один класс значительно преобладает над другим
- Когда важна метрика F1-score или precision/recall, а не просто accuracy
- В задачах обнаружения аномалий, мошенничества, редких заболеваний

**Практический совет:** Всегда анализируйте распределение классов в ваших данных перед началом обучения!

### 3. Работа с последовательностями и текстом

```python
from torch.utils.data import Dataset

# Пример для работы с текстовыми данными
class TextDataset(Dataset):
    """Dataset для текстовых данных"""
    
    def __init__(self, texts, labels, max_length=128):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Простая токенизация для примера
        tokens = text.split()[:self.max_length]
        
        return tokens, label

# Пример с коллацией (объединением) последовательностей разной длины
def collate_fn(batch):
    """Специальная функция для объединения последовательностей разной длины"""
    texts, labels = zip(*batch)
    
    # Находим максимальную длину в батче
    max_len = max(len(text) for text in texts)
    
    # Паддинг (дополнение) до одинаковой длины
    padded_texts = []
    for text in texts:
        if len(text) < max_len:
            padded = text + ['<PAD>'] * (max_len - len(text))
        else:
            padded = text
        padded_texts.append(padded)
    
    return padded_texts, torch.stack(labels)

# Пример создания и использования TextDataset
sample_texts = ["Привет мир", "Как дела", "Сегодня хорошая погода"]
sample_labels = [0, 1, 0]

text_dataset = TextDataset(sample_texts, sample_labels, max_length=10)

# Создание DataLoader с кастомной collate_fn
text_loader = DataLoader(
    text_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn  # Используем нашу функцию
)

# Проверяем работу
for batch_texts, batch_labels in text_loader:
    print(f"\nТексты в батче: {batch_texts}")
    print(f"Метки в батче: {batch_labels}")
    break
```

---

## 📊 Рекомендации по использованию

### Стандартные настройки для разных задач:

```python
# Для изображений (MNIST, CIFAR, ImageNet)
image_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Для текста (короткие последовательности)
text_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=custom_collate_fn  # Для паддинга
)

# Для табличных данных
tabular_loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2
)
```

### Когда что использовать:

- **TensorDataset** — данные уже в тензорах, простая задача
- **torchvision.datasets** — работа с изображениями
- **Кастомный Dataset** — сложная логика загрузки, предобработки
- **WeightedRandomSampler** — несбалансированные классы
- **num_workers > 0** — большие данные, медленная загрузка
- **pin_memory=True** — обучение на GPU

---

## ✅ Ключевые выводы

1. **Используйте стандартные датасеты** когда возможно (torchvision.datasets)
2. **TensorDataset** — простейший вариант для данных в тензорах
3. **WeightedRandomSampler решает проблему несбалансированности**, давая редким классам больше "веса"
4. **pin_memory критически важен для производительности GPU**
5. **Всегда разделяйте данные** на тренировочные и тестовые наборы
6. **num_workers ускоряет загрузку**, но требует правильной настройки

### Пример полного рабочего процесса:

```python
# 1. Загрузка стандартного датасета
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)

# 2. Создание DataLoader
train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# 3. Использование в цикле обучения
def train_model(model, train_loader, test_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # data, target уже на правильном устройстве (GPU если pin_memory=True)
            # ... обучение модели ...
            pass
        
        # Валидация
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                # ... оценка модели ...
                pass
```

**Итог:** PyTorch предоставляет мощные, но простые в использовании инструменты для работы с данными. Начните со стандартных решений и переходите к кастомным только когда это действительно необходимо. Правильная настройка DataLoader может ускорить обучение в несколько раз!

