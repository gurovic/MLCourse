# Data Augmentation

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# !pip install torch torchvision pillow matplotlib
```

---

## 🟢 Базовый уровень: Зачем аугментация?

### 1.1 Проблема ограниченных данных

**Data Augmentation** — искусственное увеличение датасета через трансформации.

**Зачем?**
- Больше данных → лучше обобщение
- Уменьшает переобучение
- Модель становится инвариантной к трансформациям

```python
# Демонстрация: без аугментации vs с аугментацией
def visualize_augmentation_effect():
    # Загружаем одно изображение
    transform_simple = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=True, download=True, 
                            transform=transform_simple)
    img, label = dataset[0]
    
    # Различные аугментации
    augmentations = [
        ('Оригинал', transforms.ToTensor()),
        ('Поворот', transforms.Compose([
            transforms.RandomRotation(30),
            transforms.ToTensor()
        ])),
        ('Обрезка', transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor()
        ])),
        ('Отражение', transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor()
        ])),
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    
    for idx, (name, transform) in enumerate(augmentations):
        dataset_aug = datasets.MNIST('./data', train=True, 
                                    transform=transform)
        img_aug, _ = dataset_aug[0]
        
        axes[idx].imshow(img_aug.squeeze(), cmap='gray')
        axes[idx].set_title(name)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_augmentation_effect()
```

### 1.2 Базовые трансформации для изображений

```python
# Простые трансформации
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Горизонтальное отражение
    transforms.RandomRotation(10),            # Поворот ±10 градусов
    transforms.ToTensor(),                    # Преобразование в тензор
])

# Для валидации/теста аугментация НЕ нужна!
transform_test = transforms.Compose([
    transforms.ToTensor()
])

# Применение
train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                transform=transform_train)
test_dataset = datasets.CIFAR10('./data', train=False, 
                               transform=transform_test)

# Визуализация батча с аугментациями
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, 
                                           shuffle=True)

def show_augmented_batch():
    images, labels = next(iter(train_loader))
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for idx, ax in enumerate(axes.flat):
        img = images[idx].permute(1, 2, 0)  # CHW -> HWC
        ax.imshow(img)
        ax.axis('off')
    
    plt.suptitle('Батч с аугментацией (каждый раз разный!)')
    plt.tight_layout()
    plt.show()

show_augmented_batch()
```

### 1.3 Стандартный набор для CIFAR-10

```python
# Типичная аугментация для CIFAR-10
transform_cifar_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),     # Обрезка с паддингом
    transforms.RandomHorizontalFlip(),         # Отражение
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),  # Нормализация
                        (0.2023, 0.1994, 0.2010))
])

transform_cifar_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010))
])
```

---

## 🟡 Продвинутый уровень: Продвинутые техники

### 2.1 ColorJitter — изменение цвета

```python
transform_color = transforms.Compose([
    transforms.ColorJitter(
        brightness=0.2,   # Яркость ±20%
        contrast=0.2,     # Контраст ±20%
        saturation=0.2,   # Насыщенность ±20%
        hue=0.1          # Оттенок ±10%
    ),
    transforms.ToTensor()
])

# Демонстрация
dataset = datasets.CIFAR10('./data', train=True, transform=transform_color)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i in range(10):
    img, _ = dataset[0]  # Одно и то же изображение
    ax = axes[i // 5, i % 5]
    ax.imshow(img.permute(1, 2, 0))
    ax.axis('off')

plt.suptitle('ColorJitter: случайные изменения цвета')
plt.tight_layout()
plt.show()
```

### 2.2 RandAugment — автоматическая аугментация

```python
from torchvision.transforms import RandAugment

# RandAugment применяет N случайных трансформаций с magnitude M
transform_randaug = transforms.Compose([
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor()
])

# Сравнение с базовой аугментацией
def compare_augmentations():
    transforms_list = [
        ('Без аугментации', transforms.ToTensor()),
        ('Базовая', transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])),
        ('RandAugment', transform_randaug)
    ]
    
    fig, axes = plt.subplots(3, 5, figsize=(12, 7))
    
    for row, (name, transform) in enumerate(transforms_list):
        dataset = datasets.CIFAR10('./data', train=True, transform=transform)
        
        for col in range(5):
            img, _ = dataset[0]
            axes[row, col].imshow(img.permute(1, 2, 0) if img.dim() == 3 
                                 else img.squeeze(), cmap='gray')
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(name, rotation=90, size=12)
    
    plt.tight_layout()
    plt.show()

compare_augmentations()
```

### 2.3 Cutout — вырезание случайных областей

```python
class Cutout:
    """Вырезает случайный квадрат из изображения"""
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):
        """
        img: Tensor размера (C, H, W)
        """
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img

transform_cutout = transforms.Compose([
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16)
])

# Визуализация
dataset = datasets.CIFAR10('./data', train=True, transform=transform_cutout)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i in range(10):
    img, _ = dataset[0]
    ax = axes[i // 5, i % 5]
    ax.imshow(img.permute(1, 2, 0))
    ax.axis('off')

plt.suptitle('Cutout: случайное вырезание областей')
plt.tight_layout()
plt.show()
```

---

## 🔴 Экспертный уровень: Современные техники

### 3.1 MixUp — смешивание изображений

```python
def mixup_data(x, y, alpha=1.0):
    """MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Использование в обучении
for epoch in range(num_epochs):
    for data, target in train_loader:
        # Применяем MixUp
        data, targets_a, targets_b, lam = mixup_data(data, target, alpha=1.0)
        
        optimizer.zero_grad()
        output = model(data)
        loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

# Визуализация MixUp
def visualize_mixup():
    images, labels = next(iter(train_loader))
    mixed_images, labels_a, labels_b, lam = mixup_data(images, labels)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    for i in range(4):
        # Оригиналы
        axes[0, i].imshow(images[i].permute(1, 2, 0))
        axes[0, i].set_title(f'Label: {labels[i]}')
        axes[0, i].axis('off')
        
        # MixUp
        axes[1, i].imshow(mixed_images[i].permute(1, 2, 0))
        axes[1, i].set_title(f'{lam:.2f}*{labels_a[i]} + {1-lam:.2f}*{labels_b[i]}')
        axes[1, i].axis('off')
    
    plt.suptitle('MixUp: Смешивание изображений')
    plt.tight_layout()
    plt.show()

visualize_mixup()
```

### 3.2 CutMix — вырезание и вставка

```python
def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    # Вычисляем размер и позицию вырезаемой области
    W = x.size(2)
    H = x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Случайная позиция
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Копируем и вставляем
    x_mixed = x.clone()
    x_mixed[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Корректируем lambda на основе реальной площади
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    return x_mixed, y_a, y_b, lam

# Визуализация CutMix
def visualize_cutmix():
    images, labels = next(iter(train_loader))
    mixed_images, labels_a, labels_b, lam = cutmix_data(images, labels)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    for i in range(4):
        axes[0, i].imshow(images[i].permute(1, 2, 0))
        axes[0, i].set_title(f'Label: {labels[i]}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mixed_images[i].permute(1, 2, 0))
        axes[1, i].set_title(f'CutMix: {lam:.2f}*{labels_a[i]} + {1-lam:.2f}*{labels_b[i]}')
        axes[1, i].axis('off')
    
    plt.suptitle('CutMix: Вырезание и вставка')
    plt.tight_layout()
    plt.show()

visualize_cutmix()
```

### 3.3 AutoAugment — найденная через поиск

```python
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# AutoAugment с политикой для CIFAR-10
transform_autoaug = transforms.Compose([
    AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor()
])

# Сравнение производительности разных аугментаций
def compare_augmentation_strategies():
    strategies = {
        'Без аугментации': transforms.ToTensor(),
        'Базовая': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        'RandAugment': RandAugment(num_ops=2, magnitude=9),
        'AutoAugment': AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
    }
    
    # Здесь можно обучить модели с каждой стратегией
    # и сравнить accuracy (пропускаем для краткости)
```

### 3.4 Аугментация для разных типов данных

```python
# Для табличных данных
class TabularAugmentation:
    """Аугментация для табличных данных"""
    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std
    
    def __call__(self, x):
        # Добавляем небольшой гауссов шум
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

# Для временных рядов
class TimeSeriesAugmentation:
    """Аугментация для временных рядов"""
    def __init__(self):
        pass
    
    def jitter(self, x, sigma=0.03):
        """Добавление шума"""
        return x + np.random.normal(0, sigma, x.shape)
    
    def scaling(self, x, sigma=0.1):
        """Масштабирование"""
        factor = np.random.normal(1, sigma, (x.shape[0], 1))
        return x * factor
    
    def time_warp(self, x, sigma=0.2):
        """Деформация по времени"""
        # Упрощенная версия
        return x

# Для текста (токены)
class TextAugmentation:
    """Аугментация для текста"""
    def __init__(self):
        pass
    
    def random_deletion(self, tokens, p=0.1):
        """Случайное удаление токенов"""
        mask = np.random.random(len(tokens)) > p
        return [t for t, m in zip(tokens, mask) if m]
    
    def random_swap(self, tokens, n=1):
        """Случайная перестановка"""
        tokens = tokens.copy()
        for _ in range(n):
            idx1, idx2 = np.random.choice(len(tokens), 2, replace=False)
            tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
        return tokens
```

---

## 💎 Заключение

**Сравнение техник аугментации:**

| Техника | Сложность | Эффект | Применение |
|---------|-----------|--------|-----------|
| **Flip/Rotate** | Низкая | Средний | Базовая аугментация |
| **Crop/Pad** | Низкая | Средний | Computer Vision |
| **ColorJitter** | Низкая | Средний | Естественные изображения |
| **Cutout** | Средняя | Хороший | Classification |
| **MixUp** | Средняя | Отличный | Сильная регуляризация |
| **CutMix** | Средняя | Отличный | Computer Vision |
| **RandAugment** | Средняя | Отличный | Универсальная |
| **AutoAugment** | Высокая | Отличный | SOTA, но медленно |

**Рекомендации по выбору:**

1. **MNIST/FashionMNIST:**
   ```python
   transforms.Compose([
       transforms.RandomRotation(10),
       transforms.RandomAffine(0, translate=(0.1, 0.1)),
       transforms.ToTensor()
   ])
   ```

2. **CIFAR-10/100:**
   ```python
   transforms.Compose([
       transforms.RandomCrop(32, padding=4),
       transforms.RandomHorizontalFlip(),
       RandAugment(num_ops=2, magnitude=9),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)
   ])
   ```

3. **ImageNet:**
   ```python
   transforms.Compose([
       transforms.RandomResizedCrop(224),
       transforms.RandomHorizontalFlip(),
       AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)
   ])
   ```

4. **Маленькие датасеты:**
   - Агрессивная аугментация
   - MixUp + Cutout
   - RandAugment с высоким magnitude

**Лучшие практики:**
- ✅ Аугментация ТОЛЬКО на train, НЕ на val/test
- ✅ Начинайте с простых трансформаций
- ✅ Для маленьких датасетов → больше аугментации
- ✅ MixUp/CutMix дают +2-3% accuracy
- ✅ Нормализация ПОСЛЕ аугментации
- ✅ Проверяйте, что аугментация не меняет семантику

**Типичные ошибки:**
- ❌ Аугментация на test/validation (завышение метрик!)
- ❌ Слишком агрессивная аугментация (искажает изображения)
- ❌ Неправильный порядок: нормализация ДО аугментации
- ❌ Отражение для асимметричных объектов (цифры 6/9)
- ❌ Не проверять визуально результаты аугментации

**Когда НЕ использовать:**
- ❌ Медицинские изображения (нужна осторожность)
- ❌ OCR задачи (можно испортить текст)
- ❌ Задачи где ориентация важна

> **"Data Augmentation — это бесплатный lunch в машинном обучении. Правильная аугментация может дать +5-10% accuracy без изменения архитектуры."**

**Дальнейшее изучение:**
- [AutoAugment Paper](https://arxiv.org/abs/1805.09501)
- [RandAugment Paper](https://arxiv.org/abs/1909.13719)
- [MixUp Paper](https://arxiv.org/abs/1710.09412)
- [CutMix Paper](https://arxiv.org/abs/1905.04899)

---

## 📝 Задачи

**[Перейти к задачам по Data Augmentation →](434_augmentation_tasks.md)**

Практические задания для закрепления материала:
- 🟢 Базовый уровень: базовые трансформации, ColorJitter
- 🟡 Продвинутый уровень: RandAugment, Cutout, Mixup
- 🔴 Экспертный уровень: CutMix, TTA, AutoAugment, domain-specific augmentation
