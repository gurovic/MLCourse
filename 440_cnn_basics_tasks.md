### **Задачи: Основы CNN (Convolutional Neural Networks)**

**Цель:** Понять архитектуру сверточных нейронных сетей, научиться строить и обучать CNN для задач компьютерного зрения.

---

## 🟢 Базовый уровень

### **Задача 1: Понимание свертки**

**Условие:** Реализуйте операцию свертки вручную и сравните с PyTorch.

**Требования:**
1. Напишите функцию, выполняющую 2D свертку без использования PyTorch:
   ```python
   def manual_conv2d(image, kernel):
       """Простая 2D свертка"""
       # TODO: реализуйте
       pass
   ```
2. Создайте различные kernels и примените к изображению:
   - Вертикальные границы: `[[1, 0, -1], [1, 0, -1], [1, 0, -1]]`
   - Горизонтальные границы: `[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]`
   - Blur: `[[1/9]*3]*3`
   - Sharpen: `[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]`
3. Визуализируйте результаты для каждого kernel
4. Сравните с `F.conv2d` из PyTorch

**Ожидаемый результат:** Ваша реализация дает те же результаты, что и PyTorch.

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Загрузите простое изображение
img = np.array(Image.open('test_image.jpg').convert('L'))

# Тестируйте разные kernels
kernels = {
    'Vertical edges': np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
    'Horizontal edges': np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
    'Blur': np.ones((3, 3)) / 9,
    'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
}
```

---

### **Задача 2: Первая CNN для MNIST**

**Условие:** Создайте и обучите простую CNN на MNIST.

**Требования:**
1. Архитектура:
   - Conv2d(1, 32, kernel_size=3) + ReLU + MaxPool2d(2)
   - Conv2d(32, 64, kernel_size=3) + ReLU + MaxPool2d(2)
   - Flatten
   - Linear(64*5*5, 128) + ReLU
   - Linear(128, 10)
2. Обучите 5 эпох
3. Достигните accuracy > 98%
4. Подсчитайте количество параметров
5. Сравните с MLP той же размерности

**Ожидаемый результат:** CNN быстрее сходится и имеет меньше параметров, чем MLP.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # TODO: определите слои
        
    def forward(self, x):
        # TODO: реализуйте forward pass
        # Не забудьте про activations и pooling!
        pass

# Подсчет параметров
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

---

### **Задача 3: Влияние stride и padding**

**Условие:** Исследуйте, как stride и padding влияют на размер выходного тензора.

**Требования:**
1. Создайте входной тензор размера [1, 1, 28, 28]
2. Примените различные комбинации Conv2d:
   - kernel_size=3, stride=1, padding=0
   - kernel_size=3, stride=1, padding=1 (same padding)
   - kernel_size=3, stride=2, padding=0
   - kernel_size=3, stride=2, padding=1
   - kernel_size=5, stride=1, padding=2
3. Для каждой комбинации выведите размер выхода
4. Проверьте формулу: `output_size = (input_size - kernel_size + 2*padding) / stride + 1`
5. Визуализируйте, как разные параметры влияют на receptive field

**Вопрос:** Когда использовать stride > 1 вместо pooling?

```python
x = torch.randn(1, 1, 28, 28)

configs = [
    {'kernel_size': 3, 'stride': 1, 'padding': 0},
    {'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'kernel_size': 3, 'stride': 2, 'padding': 0},
    # TODO: добавьте остальные
]

for config in configs:
    conv = nn.Conv2d(1, 16, **config)
    out = conv(x)
    print(f"Config {config}: {x.shape} -> {out.shape}")
```

---

## 🟡 Продвинутый уровень

### **Задача 4: Сравнение Max Pooling vs Average Pooling**

**Условие:** Сравните влияние разных типов pooling на качество модели.

**Требования:**
1. Обучите три CNN на CIFAR-10:
   - С MaxPool2d
   - С AvgPool2d
   - Со Strided Convolution (stride=2 вместо pooling)
2. Сравните:
   - Test accuracy
   - Скорость обучения
   - Robustness к noise (добавьте Gaussian noise на test)
3. Визуализируйте feature maps после каждого pooling слоя

**Ожидаемый результат:** MaxPool обычно лучше для классификации.

```python
class CNN_MaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # TODO: остальные слои

class CNN_AvgPool(nn.Module):
    # TODO: то же, но с AvgPool2d

class CNN_StridedConv(nn.Module):
    # TODO: Conv2d со stride=2 вместо pooling
```

---

### **Задача 5: Визуализация feature maps**

**Условие:** Визуализируйте, что изучают сверточные слои.

**Требования:**
1. Обучите CNN на MNIST или CIFAR-10
2. Для тестового изображения извлеките активации после каждого conv слоя
3. Визуализируйте:
   - Все feature maps первого слоя (32 канала)
   - Несколько feature maps второго слоя
   - Несколько feature maps третьего слоя
4. Проанализируйте:
   - Что детектирует первый слой? (края, текстуры)
   - Что детектирует второй слой? (паттерны)
   - Что детектирует третий слой? (сложные структуры)

```python
def visualize_feature_maps(model, image, layer_name):
    """Визуализирует feature maps определенного слоя"""
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Регистрируем hook
    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(get_activation(layer_name))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(image.unsqueeze(0))
    
    # Визуализируем
    feature_maps = activation[layer_name].squeeze()
    
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for idx, ax in enumerate(axes.flat):
        if idx < feature_maps.size(0):
            ax.imshow(feature_maps[idx].cpu(), cmap='viridis')
            ax.axis('off')
    
    plt.suptitle(f'Feature maps from {layer_name}')
    plt.show()
    
    handle.remove()
```

---

### **Задача 6: Receptive Field Analysis**

**Условие:** Вычислите и визуализируйте receptive field вашей сети.

**Требования:**
1. Создайте функцию для вычисления receptive field:
   ```python
   def calculate_receptive_field(layers):
       """
       layers: список (kernel_size, stride) для каждого conv/pool слоя
       """
       rf = 1
       stride_prod = 1
       
       for k, s in layers:
           rf = rf + (k - 1) * stride_prod
           stride_prod *= s
       
       return rf
   ```
2. Вычислите receptive field для вашей CNN
3. Создайте несколько архитектур с одинаковым receptive field, но разным количеством слоев
4. Сравните их производительность
5. Визуализируйте receptive field на изображении

**Вопрос:** Почему глубокие сети с малыми kernel'ами часто лучше мелких с большими?

---

## 🔴 Экспертный уровень

### **Задача 7: Dilated (Atrous) Convolutions**

**Условие:** Реализуйте и исследуйте dilated convolutions.

**Требования:**
1. Создайте CNN с dilated convolutions:
   ```python
   class DilatedCNN(nn.Module):
       def __init__(self):
           super().__init__()
           self.conv1 = nn.Conv2d(3, 32, 3, padding=1, dilation=1)
           self.conv2 = nn.Conv2d(32, 64, 3, padding=2, dilation=2)
           self.conv3 = nn.Conv2d(64, 128, 3, padding=4, dilation=4)
           # Receptive field растет экспоненциально!
   ```
2. Сравните с обычной CNN:
   - Receptive field при одинаковом количестве параметров
   - Качество на CIFAR-10
   - Скорость обучения
3. Визуализируйте, как dilation влияет на receptive field

**Применение:** Dilated conv популярны в semantic segmentation.

---

### **Задача 8: Depthwise Separable Convolutions**

**Условие:** Реализуйте MobileNet-style separable convolutions.

**Требования:**
1. Реализуйте Depthwise Separable Convolution:
   ```python
   class SeparableConv2d(nn.Module):
       def __init__(self, in_channels, out_channels, kernel_size):
           super().__init__()
           # Depthwise: каждый канал отдельно
           self.depthwise = nn.Conv2d(
               in_channels, in_channels, kernel_size,
               padding=kernel_size//2, groups=in_channels
           )
           # Pointwise: 1x1 conv для смешивания каналов
           self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
       
       def forward(self, x):
           x = self.depthwise(x)
           x = self.pointwise(x)
           return x
   ```
2. Создайте две CNN:
   - С обычными Conv2d
   - С SeparableConv2d
3. Сравните:
   - Количество параметров
   - FLOPs (вычислительная сложность)
   - Test accuracy на CIFAR-10
   - Скорость inference

**Ожидаемый результат:** Separable conv ~8x меньше параметров с небольшой потерей accuracy.

---

### **Задача 9: 1x1 Convolutions**

**Условие:** Исследуйте роль 1x1 convolutions в архитектурах.

**Требования:**
1. Создайте "bottleneck" блок (как в ResNet):
   ```python
   class BottleneckBlock(nn.Module):
       def __init__(self, in_channels, mid_channels, out_channels):
           super().__init__()
           # 1x1 conv для уменьшения каналов
           self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
           self.bn1 = nn.BatchNorm2d(mid_channels)
           
           # 3x3 conv
           self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
           self.bn2 = nn.BatchNorm2d(mid_channels)
           
           # 1x1 conv для восстановления каналов
           self.conv3 = nn.Conv2d(mid_channels, out_channels, 1)
           self.bn3 = nn.BatchNorm2d(out_channels)
   ```
2. Сравните с обычным блоком без bottleneck:
   - Количество параметров
   - FLOPs
   - Качество
3. Объясните, зачем нужны 1x1 convolutions

**Применения 1x1 conv:**
- Уменьшение размерности (bottleneck)
- Увеличение размерности
- Добавление нелинейности
- Cross-channel interactions

---

### **Задача 10: Global Average Pooling vs Flatten + FC**

**Условие:** Сравните два подхода к финальным слоям CNN.

**Требования:**
1. Создайте две версии CNN:
   ```python
   class CNN_FC(nn.Module):
       # ... conv layers ...
       def forward(self, x):
           x = self.conv_layers(x)
           x = x.view(x.size(0), -1)  # Flatten
           x = self.fc(x)
           return x
   
   class CNN_GAP(nn.Module):
       # ... conv layers ...
       def forward(self, x):
           x = self.conv_layers(x)
           x = F.adaptive_avg_pool2d(x, (1, 1))  # Global Average Pooling
           x = x.view(x.size(0), -1)
           x = self.fc(x)
           return x
   ```
2. Обучите обе на CIFAR-10
3. Сравните:
   - Количество параметров
   - Склонность к переобучению
   - Робастность к разным размерам входа
4. Протестируйте на изображениях разных размеров

**Вопрос:** Почему GAP менее склонен к переобучению?

---

### **Задача 11: CNN для разных разрешений**

**Условие:** Создайте CNN, работающую с входами разных размеров.

**Требования:**
1. Используйте только conv + pooling (без fully connected)
2. Финальный слой: Global Average Pooling + 1x1 Conv
3. Протестируйте на входах разных размеров:
   - 32x32 (CIFAR-10)
   - 64x64 (upsampled)
   - 224x224 (ImageNet size)
4. Визуализируйте feature maps для разных входов
5. Сравните accuracy

```python
class FlexibleCNN(nn.Module):
    """CNN, работающая с входами любого размера"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # TODO: добавьте слои
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # GAP - работает с любым размером!
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Тестирование
model = FlexibleCNN()
for size in [32, 64, 128, 224]:
    x = torch.randn(1, 3, size, size)
    out = model(x)
    print(f"Input {size}x{size}: Output shape {out.shape}")
```

---

## 💎 Заключение

### **Архитектурные принципы CNN:**

✅ **Базовые компоненты:**
- **Conv2d**: Извлечение локальных паттернов
- **ReLU**: Нелинейность
- **Pooling**: Уменьшение размерности и инвариантность
- **BatchNorm**: Стабилизация обучения
- **Dropout**: Регуляризация

✅ **Структурные паттерны:**
- Постепенное увеличение каналов: 32 → 64 → 128 → 256
- Постепенное уменьшение spatial size: 32 → 16 → 8 → 4
- Правило: spatial size ↓ → channels ↑

✅ **Выбор гиперпараметров:**
- **Kernel size**: 3x3 (стандарт), 5x5 (реже), 7x7 (только первый слой)
- **Stride**: 1 (с pooling) или 2 (вместо pooling)
- **Padding**: 'same' (сохраняет размер) или 'valid' (уменьшает)
- **Pooling size**: 2x2 (стандарт)

### **Рекомендации по архитектуре:**

| Задача | Рекомендуемая архитектура | Особенности |
|--------|--------------------------|-------------|
| **MNIST (28x28, grayscale)** | 2-3 conv слоя, 32-64 каналов | Простая архитектура |
| **CIFAR-10 (32x32, RGB)** | 3-4 conv слоя, 64-128 каналов | ResNet-style |
| **ImageNet (224x224, RGB)** | Deep CNN (ResNet, EfficientNet) | Transfer learning |
| **High-res images (>512x512)** | Dilated conv, Global pooling | Memory-efficient |

### **Типичная архитектура:**

```python
class TypicalCNN(nn.Module):
    """Типичная CNN архитектура для классификации"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Block 1: 32x32 -> 16x16
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        
        # Block 2: 16x16 -> 8x8
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        
        # Block 3: 8x8 -> 4x4
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Classifier
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
```

### **Оптимизация производительности:**

**Уменьшение параметров:**
- Depthwise Separable Convolutions (MobileNet)
- Bottleneck blocks (ResNet)
- Global Average Pooling вместо FC

**Увеличение receptive field:**
- Dilated convolutions
- Больше слоев с малыми kernels
- Pooling с большим stride

**Уменьшение overfitting:**
- Batch Normalization
- Dropout
- Data Augmentation
- L2 regularization

### **Debugging CNN:**

```python
def debug_cnn(model, input_size=(1, 3, 32, 32)):
    """Отладка CNN: проверка размеров и параметров"""
    x = torch.randn(input_size)
    
    print("=" * 60)
    print(f"Input: {x.shape}")
    print("=" * 60)
    
    total_params = 0
    
    for name, module in model.named_children():
        x = module(x)
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        
        print(f"{name:20s} {str(x.shape):30s} {params:>10,} params")
    
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print("=" * 60)

# Использование
model = TypicalCNN()
debug_cnn(model)
```

### **Дополнительные ресурсы:**

1. **Классические статьи:**
   - LeNet (1998) — первая успешная CNN
   - AlexNet (2012) — революция в ImageNet
   - VGG (2014) — простота и глубина
   - ResNet (2015) — skip connections
   - MobileNet (2017) — efficient CNN

2. **Инструменты:**
   - `torchvision.models` — предобученные модели
   - `torchsummary` — визуализация архитектуры
   - `netron` — визуализация графа модели

3. **Практика:**
   - Начните с простых архитектур
   - Постепенно добавляйте сложность
   - Используйте transfer learning для реальных задач
   - Всегда визуализируйте feature maps

> **"CNN revolutionized computer vision. Понимание базовых принципов CNN — это foundation для современного deep learning!"**
