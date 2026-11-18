### **Задачи: Архитектуры CNN**

**Цель:** Понять эволюцию CNN архитектур, научиться использовать и модифицировать классические архитектуры.

---

## 🟢 Базовый уровень

### **Задача 1: Реализация LeNet-5**

**Условие:** Реализуйте LeNet-5 с нуля и обучите на MNIST, а затем на CIFAR10

**Требования:**
1. Реализуйте классическую архитектуру LeNet-5
2. Обучите на MNIST/CIFAR10 (10/100/... эпох)
3. Достигните accuracy > 98%
4. Сравните с современной версией (ReLU вместо Tanh, MaxPool вместо AvgPool)
5. Визуализируйте feature maps первого conv слоя

**Ожидаемый результат:** Современная версия быстрее сходится и дает лучший результат.

```python
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: реализуйте слои согласно оригинальной архитектуре
        
    def forward(self, x):
        # TODO: реализуйте forward pass
        pass

class ModernLeNet(nn.Module):
    """LeNet с ReLU и MaxPool"""
    # TODO: реализуйте современную версию
    pass
```

---

### **Задача 2: Использование предобученных моделей**

**Условие:** Используйте предобученные ResNet для классификации изображений.

**Требования:**
1. Загрузите предобученный ResNet18 из torchvision
2. Реализуйте функцию inference для одного изображения
3. Загрузите ImageNet labels
4. Протестируйте на 5-10 изображениях
5. Визуализируйте top-5 предсказаний с вероятностями

```python
import torchvision.models as models
from torchvision import transforms
from PIL import Image

def predict_image(model, image_path):
    """Классифицирует изображение"""
    # TODO: реализуйте preprocessing
    # TODO: сделайте forward pass
    # TODO: верните top-5 predictions
    pass

# Использование
model = models.resnet18(pretrained=True)
model.eval()
predictions = predict_image(model, 'test_image.jpg')
```

**Вопрос:** Почему нужна нормализация с mean=[0.485, 0.456, 0.406]?

---

### **Задача 3: Сравнение размеров моделей**

**Условие:** Сравните количество параметров и вычислительную сложность разных архитектур.

**Требования:**
1. Для каждой архитектуры вычислите:
   - Количество параметров
   - Размер в памяти (MB)
   - Время inference (100 изображений)
2. Архитектуры: AlexNet, VGG16, ResNet18, ResNet50, ResNet152
3. Создайте сравнительную таблицу
4. Постройте bar plot для визуализации

**Ожидаемый результат:** VGG16 — самая тяжелая, ResNet18 — оптимальный баланс.

```python
def analyze_model(model, name):
    """Анализирует модель"""
    params = sum(p.numel() for p in model.parameters())
    size_mb = params * 4 / (1024**2)  # float32
    
    # TODO: измерьте время inference
    
    return {
        'name': name,
        'parameters': params,
        'size_mb': size_mb,
        'inference_time_ms': ...
    }
```

---

## 🟡 Продвинутый уровень

### **Задача 4: Реализация ResNet Residual Block**

**Условие:** Реализуйте базовый и bottleneck блоки ResNet.

**Требования:**
1. Реализуйте BasicBlock (для ResNet18/34):
   - 3x3 conv → BN → ReLU → 3x3 conv → BN → + residual → ReLU
2. Реализуйте Bottleneck (для ResNet50/101/152):
   - 1x1 conv → 3x3 conv → 1x1 conv с skip connection
3. Обучите маленький ResNet на CIFAR-10
4. Сравните BasicBlock vs Bottleneck:
   - Количество параметров
   - Скорость обучения
   - Test accuracy

```python
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # TODO: реализуйте два conv слоя + shortcut
        
    def forward(self, x):
        residual = x
        # TODO: реализуйте forward с skip connection
        return out

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # TODO: 1x1 → 3x3 → 1x1 conv + shortcut
        
    def forward(self, x):
        # TODO: реализуйте bottleneck forward
        pass
```

**Вопрос:** Почему Bottleneck эффективнее для глубоких сетей?

---

### **Задача 5: VGG vs ResNet на CIFAR-10**

**Условие:** Сравните обучение VGG-style и ResNet-style архитектур.

**Требования:**
1. Создайте VGG-11 для CIFAR-10 (32x32 вместо 224x224)
2. Создайте ResNet-18 для CIFAR-10
3. Обучите обе модели 100 эпох
4. Сравните:
   - Train/val accuracy curves
   - Скорость сходимости
   - Финальную accuracy
   - Устойчивость к overfitting
5. Визуализируйте gradient flow в обеих сетях

**Ожидаемый результат:** ResNet быстрее сходится и лучше генерализует.

---

### **Задача 6: Inception Module реализация**

**Условие:** Реализуйте Inception module и протестируйте на классификации.

**Требования:**
1. Реализуйте Inception module с 4 ветками:
   - 1x1 conv
   - 1x1 → 3x3 conv
   - 1x1 → 5x5 conv (или два 3x3)
   - 3x3 maxpool → 1x1 conv
2. Создайте простую сеть из нескольких Inception modules
3. Обучите на CIFAR-10
4. Сравните с обычной CNN той же глубины

```python
class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, 
                 ch5x5red, ch5x5, pool_proj):
        super().__init__()
        # TODO: создайте 4 ветки
        
    def forward(self, x):
        # TODO: конкатенируйте выходы всех веток
        pass
```

**Вопрос:** Зачем 1x1 convolutions перед 3x3 и 5x5?

---

## 🔴 Экспертный уровень

### **Задача 7: Transfer Learning с fine-tuning**

**Условие:** Используйте предобученный ResNet для custom dataset.

**Требования:**
1. Загрузите предобученный ResNet50 (ImageNet)
2. Замените последний FC layer для вашего количества классов
3. Экспериментируйте со стратегиями fine-tuning:
   - Заморозить все слои кроме последнего
   - Заморозить только early layers
   - Fine-tune все слои с разными LR (discriminative learning rates)
4. Используйте собственный датасет (или Caltech-101, Food-101)
5. Сравните результаты всех стратегий

```python
# Стратегия 1: Freeze all except FC
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)

# Стратегия 2: Discriminative LR
optimizer = optim.SGD([
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-3},
    {'params': model.layer4.parameters(), 'lr': 1e-3},
    {'params': model.fc.parameters(), 'lr': 1e-2}
])
```

**Вопрос:** Когда fine-tune все слои, а когда только последние?

---

### **Задача 8: Ensemble из разных архитектур**

**Условие:** Создайте ensemble из нескольких CNN архитектур.

**Требования:**
1. Обучите несколько моделей:
   - ResNet18
   - VGG11
   - DenseNet (если хватает ресурсов)
2. Реализуйте разные стратегии ансамблирования:
   - Voting (argmax каждой модели)
   - Average probabilities
   - Weighted average (оптимизируйте веса на validation)
3. Сравните accuracy:
   - Каждой отдельной модели
   - Всех ensemble стратегий
4. Проанализируйте diversity моделей (correlation of errors)

```python
class EnsembleModel(nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0] * len(models)
    
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        # TODO: weighted average of probabilities
        pass
```

---

### **Задача 9: Neural Architecture Search (упрощенный)**

**Условие:** Автоматически найдите лучшую архитектуру из пространства вариантов.

**Требования:**
1. Определите search space:
   - Количество блоков: 2-4
   - Каналы в блоке: [32, 64, 128]
   - Тип блока: [BasicBlock, Bottleneck, InceptionModule]
2. Реализуйте random search (50-100 архитектур)
3. Быстрая оценка: обучение 5 эпох на subset данных
4. Полное обучение топ-5 архитектур
5. Визуализируйте зависимость accuracy от:
   - Количества параметров
   - Глубины сети
   - Типа блоков

```python
import random

def generate_random_architecture():
    """Генерирует случайную архитектуру"""
    num_blocks = random.randint(2, 4)
    channels = random.choice([32, 64, 128])
    block_type = random.choice(['basic', 'bottleneck', 'inception'])
    
    return {
        'num_blocks': num_blocks,
        'channels': channels,
        'block_type': block_type
    }

def quick_eval(architecture, train_loader, val_loader):
    """Быстрая оценка архитектуры"""
    model = build_model(architecture)
    # TODO: обучите 5 эпох, верните val accuracy
    pass

# Random search
results = []
for _ in range(100):
    arch = generate_random_architecture()
    score = quick_eval(arch, train_loader, val_loader)
    results.append({'arch': arch, 'score': score})

# Топ-5 архитектур
top5 = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
```

---

### **Задача 10: Создание custom архитектуры**

**Условие:** Создайте свою уникальную архитектуру, комбинируя идеи из разных моделей.

**Требования:**
1. Используйте компоненты из разных архитектур:
   - Residual connections (ResNet)
   - Inception modules (GoogLeNet)
   - Dense connections (DenseNet, если знакомы)
2. Добавьте Squeeze-and-Excitation blocks для attention
3. Используйте modern techniques:
   - BatchNorm
   - Dropout
   - Label smoothing
   - Mixup augmentation
4. Обучите на CIFAR-10 или CIFAR-100
5. Сравните с baseline (ResNet18)

```python
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CustomBlock(nn.Module):
    """Ваш кастомный блок"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO: комбинируйте residual + inception + SE
        
    def forward(self, x):
        # TODO: реализуйте forward
        pass
```

---

### **Задача 11: Pruning и quantization**

**Условие:** Оптимизируйте обученную модель для deployment.

**Требования:**
1. Обучите ResNet18 на CIFAR-10
2. Примените magnitude-based pruning:
   - Удалите 50% весов с наименьшей magnitude
   - Fine-tune pruned model
3. Примените quantization:
   - Конвертируйте FP32 → INT8
   - Измерьте accuracy drop
4. Сравните:
   - Размер модели (MB)
   - Inference speed
   - Accuracy
5. Визуализируйте sparsity паттерны

```python
import torch.nn.utils.prune as prune

def apply_pruning(model, amount=0.5):
    """Применяет magnitude-based pruning"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model

def apply_quantization(model):
    """Квантизация FP32 -> INT8"""
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
    )
    return quantized_model
```

**Вопрос:** Какой метод дает лучший trade-off: pruning или quantization?

---

## 💎 Заключение

### **Выбор архитектуры:**

| Критерий | Рекомендация |
|----------|--------------|
| **Прототипирование** | ResNet18 — быстро, хорошее качество |
| **Лучшая accuracy** | ResNet50/101 или EfficientNet |
| **Мало данных** | Transfer learning с ImageNet |
| **Edge devices** | MobileNet, EfficientNet-B0 |
| **Interpretability** | VGG (простые features) |

### **Лучшие практики:**

✅ **Transfer Learning:**
- Всегда начинайте с предобученной модели
- Fine-tune с меньшим learning rate
- Используйте discriminative learning rates

✅ **Architecture Design:**
- Residual connections для глубоких сетей
- BatchNorm после каждого conv
- Global Average Pooling вместо FC

✅ **Training:**
- Data augmentation критична
- Cosine annealing LR
- Label smoothing + mixup

### **Эволюция:**

1998: LeNet → 2012: AlexNet → 2014: VGG, Inception → 2015: ResNet → 2019: EfficientNet → 2020: Vision Transformers

**Ключевые инновации:**
- **2012:** GPU, ReLU, Dropout
- **2014:** Deeper networks, Inception
- **2015:** Skip connections (ResNet)
- **2019:** Compound scaling (EfficientNet)
- **2020:** Attention mechanisms (ViT)

> **"ResNet изменил paradigm. Skip connections позволяют обучать сети любой глубины — это фундаментальное открытие для deep learning."**

**Дополнительные ресурсы:**
- [PyTorch Hub](https://pytorch.org/hub/) — предобученные модели
- [Papers with Code](https://paperswithcode.com/) — benchmarks
- [Netron](https://netron.app/) — визуализация архитектур
