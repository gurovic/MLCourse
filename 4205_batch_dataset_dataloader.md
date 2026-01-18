# Работа с данными в PyTorch: Dataset, DataLoader и Batch

## 🎯 Почему нужны батчи? Мотивирующий пример

Представьте, что у вас есть 100,000 изображений (например, MNIST) и вы хотите обучить на них нейронную сеть. Если попытаться загрузить все данные сразу:

```python
import torch
import numpy as np

# Представьте, что это ваши 100,000 изображений 28x28
num_images = 100000
image_size = 28 * 28  # 784 пикселя

# Попытка загрузить всё сразу
try:
    all_images = torch.randn(num_images, image_size)
    all_labels = torch.randint(0, 10, (num_images,))
    
    print(f"Память для всех изображений: {all_images.element_size() * all_images.nelement() / 1024**2:.1f} MB")
    print(f"Память для всех меток: {all_labels.element_size() * all_labels.nelement() / 1024**2:.1f} MB")
    
except Exception as e:
    print(f"Проблема: {e}")
```

**Проблема:** Большие данные не помещаются в память GPU!

**Решение:** Обрабатывать данные **порциями (батчами)**. Именно для этого в PyTorch есть `Dataset`, `DataLoader` и работа с батчами.

---

## 🟢 Базовый уровень: Основные понятия

### 1. Что такое Dataset?

`Dataset` — это абстракция, которая:
- Хранит ваши данные
- Знает, как получить один элемент данных по индексу
- Знает, сколько всего элементов

```python
from torch.utils.data import Dataset
import torch

class SimpleDataset(Dataset):
    """Простейший Dataset для понимания концепции"""
    
    def __init__(self, data, labels):
        """
        Args:
            data: тензор с признаками
            labels: тензор с метками
        """
        self.data = data
        self.labels = labels
    
    def __len__(self):
        """Возвращает количество элементов в датасете"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Возвращает один элемент по индексу"""
        return self.data[idx], self.labels[idx]

# Создаём простой датасет
simple_data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
simple_labels = torch.tensor([0, 1, 0])

simple_dataset = SimpleDataset(simple_data, simple_labels)

print(f"Размер датасета: {len(simple_dataset)}")
print(f"Первый элемент: {simple_dataset[0]}")
print(f"Последний элемент: {simple_dataset[-1]}")
```

### 2. Что такое DataLoader?

`DataLoader` — это инструмент, который:
- Берет `Dataset` и разбивает его на батчи
- Перемешивает данные (опционально)
- Загружает данные параллельно (опционально)

```python
from torch.utils.data import DataLoader

# Создаём DataLoader для нашего датасета
simple_loader = DataLoader(
    dataset=simple_dataset,
    batch_size=2,      # Размер батча
    shuffle=True,      # Перемешивать ли данные
    num_workers=0      # Число процессов для загрузки (0 для простоты)
)

print("Батчи из DataLoader:")
for batch_idx, (batch_data, batch_labels) in enumerate(simple_loader):
    print(f"Батч {batch_idx}:")
    print(f"  Данные: {batch_data}")
    print(f"  Метки: {batch_labels}")
    print()
```

### 3. Что такое Batch (пакет)?

**Batch** — это группа примеров, которые обрабатываются вместе за один шаг обучения.

```python
def understand_batch_concept():
    """Понимаем концепцию батча на простом примере"""
    
    # Создаём "большой" датасет
    big_data = torch.randn(100, 3)  # 100 примеров, 3 признака
    big_labels = torch.randint(0, 2, (100,))
    
    dataset = SimpleDataset(big_data, big_labels)
    
    # DataLoader с разными размерами батча
    batch_sizes = [1, 10, 50, 100]
    
    for batch_size in batch_sizes:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\nРазмер батча: {batch_size}")
        print(f"Всего батчей: {len(loader)}")
        
        # Смотрим на первый батч
        first_batch_data, first_batch_labels = next(iter(loader))
        print(f"Форма данных в батче: {first_batch_data.shape}")
        print(f"Форма меток в батче: {first_batch_labels.shape}")

understand_batch_concept()
```

---

## 🔄 Попробуйте сами 🟢 (Базовый уровень)

```python
# ИЗМЕНИТЕ ЭТИ ПАРАМЕТРЫ И НАБЛЮДАЙТЕ:

TOTAL_SAMPLES = 20      # Сколько всего примеров?
BATCH_SIZE = 4          # Какой размер батча?
SHUFFLE_DATA = True     # Перемешивать данные?

# Создаём синтетические данные
your_data = torch.randn(TOTAL_SAMPLES, 5)  # 5 признаков
your_labels = torch.randint(0, 3, (TOTAL_SAMPLES,))  # 3 класса

# Создаём свой Dataset
class YourDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Создаём Dataset
your_dataset = YourDataset(your_data, your_labels)

print("=" * 50)
print("ВАШ DATASET:")
print("=" * 50)
print(f"Всего примеров: {len(your_dataset)}")
print(f"Размер одного примера: {your_dataset[0][0].shape}")
print(f"Метка первого примера: {your_dataset[0][1]}")

# Создаём DataLoader
your_loader = DataLoader(
    dataset=your_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE_DATA,
    num_workers=0
)

print(f"\nВАШ DATALOADER:")
print(f"Размер батча: {BATCH_SIZE}")
print(f"Перемешивание: {'ВКЛ' if SHUFFLE_DATA else 'ВЫКЛ'}")
print(f"Всего батчей: {len(your_loader)}")

# Просматриваем батчи
print("\nБАТЧИ:")
for batch_idx, (batch_data, batch_labels) in enumerate(your_loader):
    print(f"\nБатч #{batch_idx}:")
    print(f"  Примеров в батче: {len(batch_data)}")
    print(f"  Форма данных: {batch_data.shape}")
    print(f"  Форма меток: {batch_labels.shape}")
    print(f"  Метки в батче: {batch_labels}")

# Рассчитываем количество батчей вручную
total_batches = (TOTAL_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
print(f"\nПРОВЕРКА:")
print(f"Рассчитано батчей: {total_batches}")
print(f"Получено батчей: {len(your_loader)}")

if total_batches == len(your_loader):
    print("✅ Расчёт верный!")
else:
    print("❌ Что-то не так!")

# Вопросы для размышления:
# 1. Что происходит, когда BATCH_SIZE > TOTAL_SAMPLES?
# 2. Как SHUFFLE_DATA влияет на порядок данных?
# 3. Сколько будет батчей при TOTAL_SAMPLES=17 и BATCH_SIZE=5?
```

---

## 🟡 Средний уровень: Реальные примеры и трансформации

### 1. Реальный Dataset для изображений

```python
from torchvision import transforms
from PIL import Image
import os

class ImageDataset(Dataset):
    """Dataset для работы с изображениями из папки"""
    
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir: путь к папке с изображениями
            transform: трансформации для изображений
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Собираем все изображения
        self.image_paths = []
        for file_name in os.listdir(image_dir):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(image_dir, file_name))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Загружаем изображение
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Применяем трансформации
        if self.transform:
            image = self.transform(image)
        
        # Для примера - создаём фиктивные метки
        label = idx % 3  # 3 класса
        
        return image, label

# Определяем трансформации
image_transforms = transforms.Compose([
    transforms.Resize((64, 64)),      # Изменяем размер
    transforms.ToTensor(),            # Конвертируем в тензор
    transforms.Normalize(             # Нормализуем
        mean=[0.485, 0.456, 0.406],   # Средние значения ImageNet
        std=[0.229, 0.224, 0.225]     # Стандартные отклонения ImageNet
    )
])

# Пример создания датасета (закомментировано, так как нужны реальные изображения)
# image_dataset = ImageDataset("path/to/images", transform=image_transforms)
# print(f"Dataset содержит {len(image_dataset)} изображений")
```

### 2. Сложные трансформации и аугментации

```python
def demonstrate_transformations():
    """Демонстрация различных трансформаций"""
    
    # Трансформации для обучения (с аугментацией)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),      # Случайное кадрирование
        transforms.RandomHorizontalFlip(),      # Случайное отражение
        transforms.ColorJitter(                 # Изменение цвета
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Трансформации для валидации (без аугментации)
    val_transforms = transforms.Compose([
        transforms.Resize(256),                 # Фиксированный размер
        transforms.CenterCrop(224),             # Центральное кадрирование
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("Трансформации для обучения:")
    print(train_transforms)
    print("\nТрансформации для валидации:")
    print(val_transforms)
    
    return train_transforms, val_transforms

train_transforms, val_transforms = demonstrate_transformations()
```

### 3. Разделение данных на train/val/test

```python
from torch.utils.data import random_split

class SplitDataset(Dataset):
    """Dataset с возможностью разделения"""
    
    def __init__(self, data_tensor, labels_tensor):
        self.data = data_tensor
        self.labels = labels_tensor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_train_val_test_split():
    """Создание разделения данных на train/val/test"""
    
    # Создаём синтетические данные
    total_samples = 1000
    data = torch.randn(total_samples, 10)
    labels = torch.randint(0, 5, (total_samples,))
    
    # Создаём полный датасет
    full_dataset = SplitDataset(data, labels)
    
    # Определяем размеры разделов
    train_size = int(0.7 * total_samples)  # 70% для обучения
    val_size = int(0.15 * total_samples)   # 15% для валидации
    test_size = total_samples - train_size - val_size  # 15% для теста
    
    print(f"Всего примеров: {total_samples}")
    print(f"Train: {train_size} ({train_size/total_samples:.0%})")
    print(f"Val: {val_size} ({val_size/total_samples:.0%})")
    print(f"Test: {test_size} ({test_size/total_samples:.0%})")
    
    # Разделяем датасет
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Для воспроизводимости
    )
    
    # Создаём DataLoader для каждого раздела
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"\nБатчей в train loader: {len(train_loader)}")
    print(f"Батчей в val loader: {len(val_loader)}")
    print(f"Батчей в test loader: {len(test_loader)}")
    
    # Проверяем, что данные не пересекаются
    train_indices = set(train_dataset.indices)
    val_indices = set(val_dataset.indices)
    test_indices = set(test_dataset.indices)
    
    print(f"\nПроверка пересечений:")
    print(f"Train ∩ Val: {len(train_indices & val_indices)}")
    print(f"Train ∩ Test: {len(train_indices & test_indices)}")
    print(f"Val ∩ Test: {len(val_indices & test_indices)}")
    
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = create_train_val_test_split()
```

---

## 🔄 Попробуйте сами 🟡 (Средний уровень)

```python
# ЭКСПЕРИМЕНТ: СОЗДАЙТЕ СВОЙ ПОЛНЫЙ ПАЙПЛАЙН ДАННЫХ

# Параметры эксперимента
TOTAL_EXAMPLES = 1000      # Всего примеров
FEATURES = 8               # Число признаков
CLASSES = 5                # Число классов
BATCH_SIZE = 64            # Размер батча
SEED = 123                 # Seed для воспроизводимости

# 1. Создаём синтетические данные с разными распределениями
torch.manual_seed(SEED)

# Данные для разных классов имеют разные распределения
data_list = []
labels_list = []

for class_idx in range(CLASSES):
    # Каждый класс имеет своё среднее значение
    mean = class_idx * 2.0
    class_data = torch.randn(TOTAL_EXAMPLES // CLASSES, FEATURES) + mean
    class_labels = torch.full((TOTAL_EXAMPLES // CLASSES,), class_idx)
    
    data_list.append(class_data)
    labels_list.append(class_labels)

# Объединяем все данные
all_data = torch.cat(data_list, dim=0)
all_labels = torch.cat(labels_list, dim=0)

print("=" * 60)
print("СОЗДАНИЕ ДАННЫХ")
print("=" * 60)
print(f"Всего примеров: {len(all_data)}")
print(f"Признаков на пример: {FEATURES}")
print(f"Классов: {CLASSES}")

# Показываем статистику по классам
print("\nРаспределение по классам:")
for class_idx in range(CLASSES):
    class_mask = all_labels == class_idx
    count = class_mask.sum().item()
    mean_features = all_data[class_mask].mean(dim=0)
    print(f"  Класс {class_idx}: {count} примеров, среднее признаков: {mean_features[0]:.2f}...")

# 2. Создаём Dataset с дополнительной логикой
class AdvancedDataset(Dataset):
    """Продвинутый Dataset с предобработкой"""
    
    def __init__(self, data, labels, normalize=True):
        self.data = data
        self.labels = labels
        self.normalize = normalize
        
        if normalize:
            # Сохраняем параметры нормализации
            self.data_mean = self.data.mean(dim=0)
            self.data_std = self.data.std(dim=0)
            
            # Нормализуем данные
            self.data = (self.data - self.data_mean) / (self.data_std + 1e-8)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.data[idx]
        label = self.labels[idx]
        
        # Можно добавить дополнительные преобразования здесь
        # Например, добавление шума для аугментации
        if torch.rand(1) < 0.1:  # 10% chance
            features = features + torch.randn_like(features) * 0.1
        
        return features, label
    
    def get_original_data(self, idx):
        """Получить оригинальные (ненормализованные) данные"""
        if self.normalize:
            original = self.data[idx] * self.data_std + self.data_mean
        else:
            original = self.data[idx]
        return original, self.labels[idx]

# 3. Создаём и проверяем Dataset
full_dataset = AdvancedDataset(all_data, all_labels, normalize=True)

print(f"\nDataset создан:")
print(f"  Нормализация: {'ВКЛ' if full_dataset.normalize else 'ВЫКЛ'}")
print(f"  Размер: {len(full_dataset)}")

# Проверяем нормализацию
sample_idx = 0
normalized_sample, label = full_dataset[sample_idx]
original_sample, _ = full_dataset.get_original_data(sample_idx)

print(f"\nПроверка нормализации (пример #{sample_idx}):")
print(f"  Оригинальные признаки: {original_sample[:3].tolist()}...")
print(f"  Нормализованные признаки: {normalized_sample[:3].tolist()}...")
print(f"  Метка: {label}")

# 4. Разделяем на train/val/test
from torch.utils.data import random_split

train_size = int(0.6 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

print(f"\nРАЗДЕЛЕНИЕ ДАННЫХ:")
print(f"  Train: {len(train_dataset)} ({len(train_dataset)/len(full_dataset):.0%})")
print(f"  Val: {len(val_dataset)} ({len(val_dataset)/len(full_dataset):.0%})")
print(f"  Test: {len(test_dataset)} ({len(test_dataset)/len(full_dataset):.0%})")

# 5. Создаём DataLoader для каждого раздела
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True  # Отбрасываем последний неполный батч
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"\nDATALOADER НАСТРОЙКИ:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# 6. Анализируем батчи
print("\nАНАЛИЗ БАТЧЕЙ:")
for loader_name, loader in [("Train", train_loader), ("Val", val_loader)]:
    # Берем первый батч
    batch_data, batch_labels = next(iter(loader))
    
    print(f"\n{loader_name} loader - первый батч:")
    print(f"  Форма данных: {batch_data.shape}")
    print(f"  Форма меток: {batch_labels.shape}")
    print(f"  Уникальные метки в батче: {torch.unique(batch_labels).tolist()}")
    
    # Статистика по батчу
    print(f"  Статистика данных:")
    print(f"    Минимум: {batch_data.min():.3f}")
    print(f"    Максимум: {batch_data.max():.3f}")
    print(f"    Среднее: {batch_data.mean():.3f}")
    print(f"    Стандартное отклонение: {batch_data.std():.3f}")

# Вопросы для анализа:
# 1. Что делает параметр drop_last=True?
# 2. Почему мы shuffle только train данные?
# 3. Как нормализация влияет на обучение?
# 4. Что происходит при изменении BATCH_SIZE?
```

---

## 🔴 Продвинутый уровень: Кастомные DataLoader и оптимизация

### 1. Кастомный DataLoader с кэшированием

```python
class CachedDataset(Dataset):
    """Dataset с кэшированием загруженных данных"""
    
    def __init__(self, base_dataset, cache_size=100):
        """
        Args:
            base_dataset: базовый dataset
            cache_size: размер кэша в элементах
        """
        self.base_dataset = base_dataset
        self.cache_size = cache_size
        self.cache = {}
        self.access_count = {}
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Проверяем, есть ли в кэше
        if idx in self.cache:
            # Обновляем счётчик доступа
            self.access_count[idx] += 1
            return self.cache[idx]
        
        # Загружаем из базового датасета
        data = self.base_dataset[idx]
        
        # Добавляем в кэш
        if len(self.cache) >= self.cache_size:
            # Находим наименее используемый элемент
            lru_idx = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_idx]
            del self.access_count[lru_idx]
        
        self.cache[idx] = data
        self.access_count[idx] = 1
        
        return data
    
    def get_cache_stats(self):
        """Получить статистику кэша"""
        cache_hits = sum(self.access_count.values()) - len(self.access_count)
        total_accesses = sum(self.access_count.values())
        hit_rate = cache_hits / total_accesses if total_accesses > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size,
            'cache_hits': cache_hits,
            'total_accesses': total_accesses,
            'hit_rate': hit_rate
        }

def test_cached_dataset():
    """Тестирование Dataset с кэшированием"""
    
    # Создаём базовый dataset
    base_data = torch.randn(1000, 10)
    base_labels = torch.randint(0, 5, (1000,))
    base_dataset = SimpleDataset(base_data, base_labels)
    
    # Создаём кэшированную версию
    cached_dataset = CachedDataset(base_dataset, cache_size=50)
    
    # Симулируем доступ к данным
    print("Тестирование кэширования...")
    
    # Первый проход - кэш пустой
    indices = list(range(100))
    for idx in indices:
        _ = cached_dataset[idx]
    
    stats1 = cached_dataset.get_cache_stats()
    print(f"\nПосле первого прохода (100 элементов):")
    print(f"  Попаданий в кэш: {stats1['cache_hits']}")
    print(f"  Общие обращения: {stats1['total_accesses']}")
    print(f"  Hit rate: {stats1['hit_rate']:.1%}")
    
    # Второй проход - некоторые данные в кэше
    for idx in indices[:50]:  # Первые 50 уже в кэше
        _ = cached_dataset[idx]
    
    stats2 = cached_dataset.get_cache_stats()
    print(f"\nПосле второго прохода (первые 50 элементов):")
    print(f"  Попаданий в кэш: {stats2['cache_hits']}")
    print(f"  Общие обращения: {stats2['total_accesses']}")
    print(f"  Hit rate: {stats2['hit_rate']:.1%}")
    
    return cached_dataset

cached_dataset = test_cached_dataset()
```

### 2. DataLoader с балансировкой классов

```python
from torch.utils.data import WeightedRandomSampler

class BalancedDataLoader:
    """DataLoader с балансировкой классов"""
    
    def __init__(self, dataset, batch_size=32, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Вычисляем веса для сэмплера
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        
        labels = torch.tensor(labels)
        class_counts = torch.bincount(labels)
        
        # Вес для каждого класса обратно пропорционален его частоте
        class_weights = 1. / class_counts.float()
        sample_weights = class_weights[labels]
        
        # Создаём WeightedRandomSampler
        self.sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True
        )
        
        # Создаём DataLoader с сэмплером
        self.loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=num_workers
        )
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)

def test_balanced_loader():
    """Тестирование DataLoader с балансировкой"""
    
    # Создаём несбалансированный dataset
    torch.manual_seed(42)
    
    # Классы с разным количеством примеров
    class_counts = [100, 20, 5]  # Очень несбалансированные
    all_data = []
    all_labels = []
    
    for class_idx, count in enumerate(class_counts):
        class_data = torch.randn(count, 5) + class_idx  # Разные средние
        class_labels = torch.full((count,), class_idx)
        all_data.append(class_data)
        all_labels.append(class_labels)
    
    data = torch.cat(all_data, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    dataset = SimpleDataset(data, labels)
    
    print("Исходное распределение классов:")
    for class_idx in range(len(class_counts)):
        count = (labels == class_idx).sum().item()
        print(f"  Класс {class_idx}: {count} примеров ({count/len(labels):.1%})")
    
    # Обычный DataLoader
    regular_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Балансированный DataLoader
    balanced_loader = BalancedDataLoader(dataset, batch_size=16)
    
    # Анализируем распределение в батчах
    print("\nАнализ распределения в батчах:")
    
    for loader_name, loader in [("Обычный", regular_loader), ("Балансированный", balanced_loader)]:
        # Собираем статистику по нескольким батчам
        class_distribution = {0: 0, 1: 0, 2: 0}
        total_samples = 0
        
        for batch_idx, (_, batch_labels) in enumerate(loader):
            if batch_idx >= 10:  # Анализируем первые 10 батчей
                break
            
            for class_idx in range(3):
                count = (batch_labels == class_idx).sum().item()
                class_distribution[class_idx] += count
                total_samples += count
        
        print(f"\n{loader_name} loader:")
        for class_idx in range(3):
            percentage = class_distribution[class_idx] / total_samples
            print(f"  Класс {class_idx}: {percentage:.1%}")
    
    return regular_loader, balanced_loader

regular_loader, balanced_loader = test_balanced_loader()
```

### 3. Пайплайн с многопроцессорной загрузкой

```python
class OptimizedDataPipeline:
    """Оптимизированный пайплайн загрузки данных"""
    
    def __init__(self, dataset, batch_size=64, num_workers=4, 
                 pin_memory=True, prefetch_factor=2):
        """
        Args:
            dataset: исходный dataset
            batch_size: размер батча
            num_workers: число процессов для загрузки
            pin_memory: копировать ли данные в pinned memory
            prefetch_factor: сколько батчей загружать заранее
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Создаём DataLoader с оптимизациями
        self.loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=num_workers > 0
        )
    
    def benchmark_performance(self, num_batches=100):
        """Бенчмарк производительности пайплайна"""
        import time
        
        print(f"Бенчмарк производительности:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Num workers: {self.num_workers}")
        print(f"  Pin memory: {'Yes' if self.loader.pin_memory else 'No'}")
        
        # Прогрев
        warmup_batches = 10
        for i, _ in enumerate(self.loader):
            if i >= warmup_batches:
                break
        
        # Измерение времени
        start_time = time.time()
        batch_count = 0
        
        for batch_idx, batch in enumerate(self.loader):
            batch_count += 1
            if batch_count >= num_batches:
                break
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nРезультаты:")
        print(f"  Обработано батчей: {batch_count}")
        print(f"  Общее время: {total_time:.2f} сек")
        print(f"  Время на батч: {total_time/batch_count:.4f} сек")
        print(f"  Батчей в секунду: {batch_count/total_time:.1f}")
        
        return total_time

def compare_pipeline_configs():
    """Сравнение разных конфигураций пайплайна"""
    
    # Создаём тестовый dataset
    test_data = torch.randn(10000, 10)
    test_labels = torch.randint(0, 5, (10000,))
    test_dataset = SimpleDataset(test_data, test_labels)
    
    # Тестируемые конфигурации
    configs = [
        {'batch_size': 32, 'num_workers': 0, 'pin_memory': False},
        {'batch_size': 32, 'num_workers': 2, 'pin_memory': False},
        {'batch_size': 32, 'num_workers': 4, 'pin_memory': True},
        {'batch_size': 64, 'num_workers': 4, 'pin_memory': True},
        {'batch_size': 128, 'num_workers': 4, 'pin_memory': True},
    ]
    
    results = []
    
    print("Сравнение конфигураций пайплайна")
    print("=" * 60)
    
    for config in configs:
        print(f"\nТестируем конфигурацию:")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Num workers: {config['num_workers']}")
        print(f"  Pin memory: {config['pin_memory']}")
        
        pipeline = OptimizedDataPipeline(
            test_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )
        
        time_taken = pipeline.benchmark_performance(num_batches=50)
        
        results.append({
            **config,
            'time_per_batch': time_taken / 50
        })
    
    # Анализ результатов
    print("\n" + "=" * 60)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
    print("=" * 60)
    
    best_config = min(results, key=lambda x: x['time_per_batch'])
    
    for idx, result in enumerate(results):
        speedup = best_config['time_per_batch'] / result['time_per_batch']
        print(f"\nКонфигурация {idx+1}:")
        print(f"  Batch: {result['batch_size']}, Workers: {result['num_workers']}, Pin: {result['pin_memory']}")
        print(f"  Время на батч: {result['time_per_batch']:.4f} сек")
        print(f"  Скорость относительно лучшей: {speedup:.1f}x")
    
    print(f"\nЛучшая конфигурация:")
    print(f"  Batch size: {best_config['batch_size']}")
    print(f"  Num workers: {best_config['num_workers']}")
    print(f"  Pin memory: {best_config['pin_memory']}")
    print(f"  Время на батч: {best_config['time_per_batch']:.4f} сек")
    
    return results

# Запускаем сравнение (закомментировано для скорости)
# pipeline_results = compare_pipeline_configs()
```

---

## 🔄 Попробуйте сами 🔴 (Продвинутый уровень)

```python
# ВЫЗОВ: СОЗДАЙТЕ ПОЛНЫЙ ПРОИЗВОДСТВЕННЫЙ ПАЙПЛАЙН ДАННЫХ

# 1. Создайте сложный Dataset с несколькими модальностями
class MultimodalDataset(Dataset):
    """
    Dataset с несколькими типами данных:
    - Числовые признаки
    - Изображения (симулированные)
    - Текст (симулированные эмбеддинги)
    """
    
    def __init__(self, num_samples=1000, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        
        # Генерируем разные типы данных
        self.numeric_data = self._generate_numeric_data()
        self.image_data = self._generate_image_data()
        self.text_embeddings = self._generate_text_embeddings()
        self.labels = self._generate_labels()
        
        # Кэш для быстрого доступа
        self.cache = {}
        self.cache_hits = 0
        self.total_accesses = 0
    
    def _generate_numeric_data(self):
        """Генерация числовых признаков (10 признаков)"""
        numeric = torch.randn(self.num_samples, 10)
        # Добавляем зависимости между признаками
        numeric[:, 2] = numeric[:, 0] * 0.5 + numeric[:, 1] * 0.3 + torch.randn(self.num_samples) * 0.2
        return numeric
    
    def _generate_image_data(self):
        """Генерация симулированных изображений (1x28x28)"""
        images = torch.randn(self.num_samples, 1, 28, 28)
        # Добавляем структуру: разные классы имеют разные паттерны
        for i in range(self.num_samples):
            class_pattern = (i % 3) * 0.5
            images[i] += class_pattern
        return images
    
    def _generate_text_embeddings(self):
        """Генерация симулированных текстовых эмбеддингов (50-мерных)"""
        embeddings = torch.randn(self.num_samples, 50)
        # Делаем эмбеддинги кластеризованными
        cluster_centers = torch.randn(3, 50)
        for i in range(self.num_samples):
            cluster_idx = i % 3
            embeddings[i] = cluster_centers[cluster_idx] + torch.randn(50) * 0.3
        return embeddings
    
    def _generate_labels(self):
        """Генерация сложных мульти-лейблов"""
        labels = torch.zeros(self.num_samples, 3)  # 3 независимых задачи
        
        # Задача 1: бинарная классификация на основе числовых данных
        labels[:, 0] = (self.numeric_data[:, 0] > 0).float()
        
        # Задача 2: мультикласс на основе изображений
        labels[:, 1] = torch.randint(0, 3, (self.num_samples,)).float()
        
        # Задача 3: регрессия на основе текста
        labels[:, 2] = self.text_embeddings.mean(dim=1)
        
        return labels
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        self.total_accesses += 1
        
        # Проверяем кэш
        if idx in self.cache:
            self.cache_hits += 1
            return self.cache[idx]
        
        # Получаем данные
        numeric = self.numeric_data[idx]
        image = self.image_data[idx]
        text = self.text_embeddings[idx]
        label = self.labels[idx]
        
        # Применяем трансформации если есть
        if self.transform:
            image = self.transform(image)
        
        # Собираем в словарь
        sample = {
            'numeric': numeric,
            'image': image,
            'text': text,
            'label': label,
            'index': idx
        }
        
        # Сохраняем в кэш (ограничиваем размер)
        if len(self.cache) < 100:
            self.cache[idx] = sample
        
        return sample
    
    def get_cache_stats(self):
        """Статистика кэша"""
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'total_accesses': self.total_accesses,
            'hit_rate': self.cache_hits / self.total_accesses if self.total_accesses > 0 else 0
        }
    
    def get_dataset_stats(self):
        """Статистика датасета"""
        return {
            'total_samples': self.num_samples,
            'numeric_shape': self.numeric_data.shape,
            'image_shape': self.image_data.shape,
            'text_shape': self.text_embeddings.shape,
            'label_shape': self.labels.shape,
            'label_stats': {
                'task1_mean': self.labels[:, 0].mean().item(),
                'task2_distribution': torch.bincount(self.labels[:, 1].long()).tolist(),
                'task3_range': [self.labels[:, 2].min().item(), self.labels[:, 2].max().item()]
            }
        }

# 2. Создайте кастомный DataLoader с продвинутыми фичами
class AdvancedDataLoader:
    """
    Продвинутый DataLoader с:
    - Динамическим батчингом
    - Автоматической балансировкой
    - Мониторингом производительности
    """
    
    def __init__(self, dataset, base_batch_size=32, max_batch_size=128, 
                 num_workers=2, adaptive_batching=True):
        
        self.dataset = dataset
        self.base_batch_size = base_batch_size
        self.max_batch_size = max_batch_size
        self.adaptive_batching = adaptive_batching
        
        # Статистика
        self.batch_times = []
        self.current_batch_size = base_batch_size
        
        # Создаём сэмплер для балансировки по первой задаче
        labels = torch.stack([dataset[i]['label'] for i in range(min(100, len(dataset)))])
        task1_labels = labels[:, 0].long()
        class_counts = torch.bincount(task1_labels)
        class_weights = 1. / class_counts.float()
        sample_weights = class_weights[task1_labels]
        
        self.sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True
        )
        
        # Создаём DataLoader
        self.loader = DataLoader(
            dataset=dataset,
            batch_size=self.current_batch_size,
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=self.custom_collate
        )
    
    def custom_collate(self, batch):
        """Кастомная функция для сборки батча"""
        import time
        start_time = time.time()
        
        # Разделяем данные по типам
        numeric_batch = torch.stack([item['numeric'] for item in batch])
        image_batch = torch.stack([item['image'] for item in batch])
        text_batch = torch.stack([item['text'] for item in batch])
        label_batch = torch.stack([item['label'] for item in batch])
        indices = torch.tensor([item['index'] for item in batch])
        
        # Создаём структурированный батч
        collated_batch = {
            'numeric': numeric_batch,
            'image': image_batch,
            'text': text_batch,
            'label': {
                'task1': label_batch[:, 0],  # Бинарная классификация
                'task2': label_batch[:, 1].long(),  # Мультикласс
                'task3': label_batch[:, 2]  # Регрессия
            },
            'indices': indices,
            'batch_size': len(batch)
        }
        
        # Измеряем время обработки
        process_time = time.time() - start_time
        self.batch_times.append(process_time)
        
        # Адаптивный батчинг (если включен)
        if self.adaptive_batching and len(self.batch_times) > 10:
            avg_time = sum(self.batch_times[-10:]) / 10
            if avg_time < 0.01 and self.current_batch_size < self.max_batch_size:
                # Увеличиваем размер батча
                self.current_batch_size = min(
                    self.current_batch_size * 2,
                    self.max_batch_size
                )
                print(f"Увеличиваем batch size до {self.current_batch_size}")
            
            elif avg_time > 0.05 and self.current_batch_size > self.base_batch_size:
                # Уменьшаем размер батча
                self.current_batch_size = max(
                    self.current_batch_size // 2,
                    self.base_batch_size
                )
                print(f"Уменьшаем batch size до {self.current_batch_size}")
        
        return collated_batch
    
    def __iter__(self):
        # Обновляем DataLoader с новым batch_size
        if self.adaptive_batching:
            self.loader = DataLoader(
                dataset=self.dataset,
                batch_size=self.current_batch_size,
                sampler=self.sampler,
                num_workers=self.loader.num_workers,
                pin_memory=self.loader.pin_memory,
                prefetch_factor=self.loader.prefetch_factor,
                collate_fn=self.custom_collate
            )
        
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)
    
    def get_performance_stats(self):
        """Статистика производительности"""
        if not self.batch_times:
            return {}
        
        return {
            'avg_batch_time': sum(self.batch_times) / len(self.batch_times),
            'total_batches': len(self.batch_times),
            'current_batch_size': self.current_batch_size,
            'min_batch_time': min(self.batch_times),
            'max_batch_time': max(self.batch_times)
        }

# 3. Создайте и протестируйте пайплайн
print("=" * 70)
print("СОЗДАНИЕ ПРОИЗВОДСТВЕННОГО ПАЙПЛАЙНА ДАННЫХ")
print("=" * 70)

# Создаём мультимодальный датасет
multimodal_dataset = MultimodalDataset(num_samples=5000)

# Анализируем датасет
stats = multimodal_dataset.get_dataset_stats()
print("\nСТАТИСТИКА DATASET:")
print(f"Всего примеров: {stats['total_samples']}")
print(f"Числовые данные: {stats['numeric_shape']}")
print(f"Изображения: {stats['image_shape']}")
print(f"Текстовые эмбеддинги: {stats['text_shape']}")
print(f"Метки: {stats['label_shape']}")
print(f"\nСтатистика меток:")
print(f"  Задача 1 (бинарная): среднее = {stats['label_stats']['task1_mean']:.3f}")
print(f"  Задача 2 (мультикласс): распределение = {stats['label_stats']['task2_distribution']}")
print(f"  Задача 3 (регрессия): диапазон = [{stats['label_stats']['task3_range'][0]:.3f}, {stats['label_stats']['task3_range'][1]:.3f}]")

# Создаём продвинутый DataLoader
advanced_loader = AdvancedDataLoader(
    dataset=multimodal_dataset,
    base_batch_size=16,
    max_batch_size=64,
    num_workers=0,  # Для демонстрации
    adaptive_batching=True
)

print(f"\nADVANCED DATALOADER:")
print(f"Base batch size: {advanced_loader.base_batch_size}")
print(f"Max batch size: {advanced_loader.max_batch_size}")
print(f"Adaptive batching: {'ВКЛ' if advanced_loader.adaptive_batching else 'ВЫКЛ'}")
print(f"Sampler type: {type(advanced_loader.sampler).__name__}")

# Тестируем пайплайн
print("\n" + "=" * 70)
print("ТЕСТИРОВАНИЕ ПАЙПЛАЙНА")
print("=" * 70)

# Проходим по нескольким батчам
num_test_batches = 5
batch_samples = []

for batch_idx, batch in enumerate(advanced_loader):
    if batch_idx >= num_test_batches:
        break
    
    print(f"\nБатч #{batch_idx}:")
    print(f"  Размер батча: {batch['batch_size']}")
    print(f"  Числовые данные: {batch['numeric'].shape}")
    print(f"  Изображения: {batch['image'].shape}")
    print(f"  Текст: {batch['text'].shape}")
    
    # Анализируем метки
    task1_labels = batch['label']['task1']
    task2_labels = batch['label']['task2']
    task3_labels = batch['label']['task3']
    
    print(f"  Метки задача 1 (0/1): {(task1_labels == 0).sum()}/{ (task1_labels == 1).sum()}")
    print(f"  Метки задача 2 (0/1/2): "
          f"{(task2_labels == 0).sum()}/{(task2_labels == 1).sum()}/{(task2_labels == 2).sum()}")
    print(f"  Метки задача 3: среднее = {task3_labels.mean():.3f}")
    
    batch_samples.append(batch)

# Анализ производительности
print("\n" + "=" * 70)
print("АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ")
print("=" * 70)

# Статистика кэша датасета
cache_stats = multimodal_dataset.get_cache_stats()
print(f"\nСтатистика кэша Dataset:")
print(f"  Размер кэша: {cache_stats['cache_size']}")
print(f"  Попадания в кэш: {cache_stats['cache_hits']}")
print(f"  Всего обращений: {cache_stats['total_accesses']}")
print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")

# Статистика производительности DataLoader
perf_stats = advanced_loader.get_performance_stats()
if perf_stats:
    print(f"\nСтатистика производительности DataLoader:")
    print(f"  Среднее время обработки батча: {perf_stats['avg_batch_time']:.4f} сек")
    print(f"  Минимальное время: {perf_stats['min_batch_time']:.4f} сек")
    print(f"  Максимальное время: {perf_stats['max_batch_time']:.4f} сек")
    print(f"  Всего обработанных батчей: {perf_stats['total_batches']}")
    print(f"  Текущий размер батча: {perf_stats['current_batch_size']}")

# 4. Создайте систему мониторинга
class DataPipelineMonitor:
    """Мониторинг пайплайна данных"""
    
    def __init__(self, data_loader):
        self.loader = data_loader
        self.metrics = {
            'batch_times': [],
            'batch_sizes': [],
            'data_shapes': [],
            'label_distributions': []
        }
    
    def monitor_epoch(self, num_batches=None):
        """Мониторинг одной эпохи"""
        batch_count = 0
        
        for batch in self.loader:
            # Сохраняем метрики
            self.metrics['batch_times'].append(
                advanced_loader.batch_times[-1] if advanced_loader.batch_times else 0
            )
            self.metrics['batch_sizes'].append(batch['batch_size'])
            self.metrics['data_shapes'].append({
                'numeric': batch['numeric'].shape,
                'image': batch['image'].shape,
                'text': batch['text'].shape
            })
            
            # Распределение меток
            task1_dist = torch.bincount(batch['label']['task1'].long()).tolist()
            task2_dist = torch.bincount(batch['label']['task2']).tolist()
            self.metrics['label_distributions'].append({
                'task1': task1_dist,
                'task2': task2_dist
            })
            
            batch_count += 1
            if num_batches and batch_count >= num_batches:
                break
    
    def generate_report(self):
        """Генерация отчёта"""
        print("\n" + "=" * 70)
        print("ОТЧЁТ МОНИТОРИНГА ПАЙПЛАЙНА")
        print("=" * 70)
        
        if not self.metrics['batch_times']:
            print("Нет данных для анализа")
            return
        
        # Анализ времени
        avg_time = sum(self.metrics['batch_times']) / len(self.metrics['batch_times'])
        print(f"\nПроизводительность:")
        print(f"  Среднее время батча: {avg_time:.4f} сек")
        print(f"  Всего батчей: {len(self.metrics['batch_times'])}")
        print(f"  Общее время: {sum(self.metrics['batch_times']):.2f} сек")
        
        # Анализ размеров батчей
        unique_sizes = set(self.metrics['batch_sizes'])
        print(f"\nРазмеры батчей:")
        for size in sorted(unique_sizes):
            count = self.metrics['batch_sizes'].count(size)
            print(f"  {size}: {count} батчей ({count/len(self.metrics['batch_sizes']):.1%})")
        
        # Анализ распределения меток
        print(f"\nРаспределение меток (усреднённое):")
        
        # Задача 1
        task1_total = [0, 0]
        for dist in self.metrics['label_distributions']:
            task1_dist = dist['task1']
            if len(task1_dist) > 0:
                task1_total[0] += task1_dist[0]
            if len(task1_dist) > 1:
                task1_total[1] += task1_dist[1]
        
        task1_sum = sum(task1_total)
        if task1_sum > 0:
            print(f"  Задача 1 (бинарная):")
            print(f"    Класс 0: {task1_total[0]} ({task1_total[0]/task1_sum:.1%})")
            print(f"    Класс 1: {task1_total[1]} ({task1_total[1]/task1_sum:.1%})")
        
        # Задача 2
        task2_total = [0, 0, 0]
        for dist in self.metrics['label_distributions']:
            task2_dist = dist['task2']
            for i in range(min(len(task2_dist), 3)):
                task2_total[i] += task2_dist[i]
        
        task2_sum = sum(task2_total)
        if task2_sum > 0:
            print(f"  Задача 2 (мультикласс):")
            for i, count in enumerate(task2_total):
                print(f"    Класс {i}: {count} ({count/task2_sum:.1%})")

# Тестируем мониторинг
print("\n" + "=" * 70)
print("ТЕСТИРОВАНИЕ МОНИТОРИНГА")
print("=" * 70)

monitor = DataPipelineMonitor(advanced_loader)
monitor.monitor_epoch(num_batches=10)
monitor.generate_report()

# Вопросы для анализа:
# 1. Как кэширование влияет на производительность?
# 2. В чём преимущества адаптивного батчинга?
# 3. Как балансировка классов влияет на распределение в батчах?
# 4. Какие метрики мониторинга самые важные?
```

---

## 📚 Резюме по уровням сложности

### 🟢 Базовый уровень (Вы должны уметь):
- Создавать простые Dataset классы
- Использовать стандартный DataLoader
- Понимать концепцию батчей
- Различать параметры shuffle и batch_size

### 🟡 Средний уровень (Вы должны уметь):
- Создавать сложные Dataset с трансформациями
- Разделять данные на train/val/test
- Работать с несбалансированными данными
- Использовать WeightedRandomSampler

### 🔴 Продвинутый уровень (Вы должны уметь):
- Создавать кастомные DataLoader с оптимизациями
- Реализовывать кэширование и предзагрузку
- Оптимизировать пайплайн для production
- Мониторить производительность пайплайна
- Работать с многомодальными данными

---

## 🎯 Ключевые выводы

### Почему батчи важны?
1. **Эффективность GPU** — GPU лучше работают с батчами
2. **Стабильность градиентов** — усреднение по батчу дает более стабильные градиенты
3. **Экономия памяти** — не нужно хранить все данные сразу

### Лучшие практики:
- **Dataset**: Должен быть легковесным, хранить только индексы/пути
- **DataLoader**: Используйте num_workers > 0 для загрузки в фоне
- **Batch size**: Начинайте с 32/64, увеличивайте пока есть память
- **Transforms**: Применяйте аугментации только к train данным

### Производительность:
```python
# Хорошая конфигурация для начала:
loader = DataLoader(
    dataset=dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,      # Использовать несколько процессов
    pin_memory=True,    # Быстрее копирование на GPU
    prefetch_factor=2   # Предзагрузка батчей
)
```

---

**Следующий шаг:** Интегрируйте ваш пайплайн данных с обучением модели! Попробуйте использовать созданные DataLoader в реальном тренировочном цикле.
