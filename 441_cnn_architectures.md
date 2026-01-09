# Архитектуры CNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt

# !pip install torch torchvision matplotlib
```

---

## 🟢 Базовый уровень: Эволюция CNN архитектур

### 1.1 LeNet-5 (1998) — первая успешная CNN

**LeNet-5** — пионер сверточных нейросетей, созданная Yann LeCun для распознавания рукописных цифр.

**Архитектура:**
- Input: 32x32 grayscale
- Conv1: 6 filters, 5x5
- Pool1: 2x2 average pooling
- Conv2: 16 filters, 5x5
- Pool2: 2x2 average pooling
- FC1: 120 neurons
- FC2: 84 neurons
- Output: 10 classes

```python
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = F.avg_pool2d(F.tanh(self.conv1(x)), 2)
        x = F.avg_pool2d(F.tanh(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# Создание модели
model = LeNet5()
print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")

# Тест
x = torch.randn(1, 1, 32, 32)
output = model(x)
print(f"Вход: {x.shape}, Выход: {output.shape}")
```

**Особенности LeNet-5:**
- ✅ Доказала эффективность CNN
- ✅ Малое количество параметров (~60K)
- ❌ Tanh активации (устарело, сейчас ReLU)
- ❌ Average pooling (сейчас max pooling)

---

### 1.2 AlexNet (2012) — ImageNet революция

**AlexNet** выиграла ImageNet 2012 с огромным отрывом, запустив эпоху глубокого обучения.

**Ключевые инновации:**
- ReLU вместо Tanh
- Dropout для регуляризации
- Data augmentation
- GPU вычисления

**Архитектура:**
```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # Conv1: 224x224x3 -> 55x55x96
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2: 55x55x96 -> 27x27x256
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3: 27x27x256 -> 13x13x384
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 13x13x384 -> 13x13x384
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 13x13x384 -> 13x13x256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Использование предобученной модели
model = models.alexnet(pretrained=True)
print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")
```

**Достижения:**
- Top-5 error: 15.3% (vs 26.2% у второго места)
- ~60M параметров
- Первое практическое применение GPU

---

## 🟡 Продвинутый уровень: VGG и принцип простоты

### 2.1 VGG (2014) — deeper is better

**VGG** показала, что глубина критична: 16-19 слоев vs 8 в AlexNet.

**Принципы VGG:**
- Только 3x3 convolutions
- Только 2x2 max pooling
- Постепенное удвоение каналов: 64 → 128 → 256 → 512
- Простая и регулярная архитектура

```python
class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 224x224x3 -> 112x112x64
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 112x112x64 -> 56x56x128
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 56x56x128 -> 28x28x256
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 28x28x256 -> 14x14x512
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5: 14x14x512 -> 7x7x512
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Предобученная модель
model = models.vgg16(pretrained=True)
print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")  # ~138M
```

**Преимущества VGG:**
- ✅ Простая и понятная архитектура
- ✅ Хорошо работает как feature extractor
- ❌ Очень много параметров (~138M)
- ❌ Медленное обучение и inference

**Почему 3x3 convolutions?**
- Два слоя 3x3 = receptive field 5x5
- Три слоя 3x3 = receptive field 7x7
- Меньше параметров + больше нелинейности

---

### 2.2 ResNet (2015) — революция skip connections

**ResNet** решила проблему деградации очень глубоких сетей через residual connections.

**Проблема:** С глубиной >20 слоев качество ухудшалось (не из-за overfitting!)

**Решение:** Skip connections позволяют gradient'ам проходить напрямую.

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut для изменения размерности
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)  # Skip connection!
        out = F.relu(out)
        
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # 4 groups of residual blocks
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Предобученные модели доступны
model = models.resnet18(pretrained=True)  # ~11M параметров
# model = models.resnet50(pretrained=True)  # ~25M параметров
# model = models.resnet152(pretrained=True) # ~60M параметров
```

**Преимущества ResNet:**
- ✅ Можно обучать очень глубокие сети (100+ слоев)
- ✅ Нет проблемы деградации
- ✅ Легко градиенты проходят через сеть
- ✅ Меньше параметров, чем VGG

---

## 🔴 Экспертный уровень: Inception и современные архитектуры

### 3.1 Inception (GoogLeNet, 2014)

**Идея:** Вместо выбора одного размера kernel, используем несколько параллельно!

```python
class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 conv -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 conv -> 5x5 conv branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 pool -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concatenate along channel dimension
        outputs = torch.cat([branch1, branch2, branch3, branch4], 1)
        return outputs

# Inception используется в GoogLeNet
model = models.googlenet(pretrained=True)
```

**Ключевые идеи:**
- Параллельные пути с разными receptive fields
- 1x1 convolutions для уменьшения вычислений (bottleneck)
- Global Average Pooling вместо FC слоев

---

### 3.2 Сравнение архитектур

```python
def compare_architectures():
    """Сравнение популярных архитектур"""
    
    architectures = {
        'AlexNet': models.alexnet(pretrained=False),
        'VGG16': models.vgg16(pretrained=False),
        'ResNet18': models.resnet18(pretrained=False),
        'ResNet50': models.resnet50(pretrained=False),
        'GoogLeNet': models.googlenet(pretrained=False),
    }
    
    comparison = []
    x = torch.randn(1, 3, 224, 224)
    
    for name, model in architectures.items():
        model.eval()
        
        # Параметры
        params = sum(p.numel() for p in model.parameters())
        
        # FLOPs (приблизительно)
        import time
        with torch.no_grad():
            start = time.time()
            for _ in range(100):
                _ = model(x)
            elapsed = time.time() - start
        
        comparison.append({
            'Architecture': name,
            'Parameters (M)': params / 1e6,
            'Time (ms)': elapsed * 10,  # per image
        })
    
    import pandas as pd
    df = pd.DataFrame(comparison)
    print(df.to_string(index=False))

compare_architectures()
```

**Таблица сравнения:**

| Architecture | Params (M) | Top-1 Acc | Top-5 Acc | Year |
|--------------|------------|-----------|-----------|------|
| LeNet-5      | 0.06       | -         | -         | 1998 |
| AlexNet      | 61.1       | 56.5%     | 79.1%     | 2012 |
| VGG16        | 138.4      | 71.6%     | 90.6%     | 2014 |
| GoogLeNet    | 6.8        | 69.8%     | 89.9%     | 2014 |
| ResNet50     | 25.6       | 76.1%     | 92.9%     | 2015 |
| ResNet152    | 60.2       | 78.3%     | 94.1%     | 2015 |

---

### 3.3 Использование предобученных моделей

```python
from torchvision import models, transforms
from PIL import Image

# Загрузка предобученной модели
model = models.resnet50(pretrained=True)
model.eval()

# Preprocessing для ImageNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

# Inference
img = Image.open('image.jpg')
img_tensor = preprocess(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_tensor)
    probs = F.softmax(output, dim=1)
    top5_prob, top5_idx = torch.topk(probs, 5)

print(f"Top 5 predictions:")
for i in range(5):
    print(f"{i+1}. Class {top5_idx[0][i].item()}: {top5_prob[0][i].item():.4f}")
```

---

## 💎 Заключение

### **Эволюция CNN:**

**1998-2012: Early days**
- LeNet: proof of concept
- Ограничения вычислительных ресурсов

**2012: Deep Learning Renaissance**
- AlexNet: GPU + ReLU + Dropout + Data Augmentation
- ImageNet победа запустила революцию

**2014: Deeper networks**
- VGG: простота и глубина
- GoogLeNet: Inception modules, эффективность

**2015-present: Skip connections**
- ResNet: очень глубокие сети (100+ слоев)
- Решена проблема деградации

### **Рекомендации по выбору:**

| Задача | Рекомендуемая архитектура | Почему |
|--------|--------------------------|--------|
| **Прототипирование** | ResNet18/34 | Быстро, хорошее качество |
| **Лучшее качество** | ResNet50/101 | State-of-the-art на ImageNet |
| **Feature extraction** | VGG16 | Классика, понятные features |
| **Мало ресурсов** | MobileNet, EfficientNet | Оптимизированы для edge |
| **Research** | ResNet + modifications | Хорошая база для экспериментов |

### **Лучшие практики:**

✅ **Transfer Learning:**
- Используйте предобученные модели
- Fine-tune для своей задачи
- Заморозьте ранние слои

✅ **Architecture choice:**
- ResNet — универсальный выбор
- VGG — если нужны interpretable features
- Inception/EfficientNet — для production

✅ **Training tips:**
- Data augmentation критична
- Cosine annealing LR schedule
- Label smoothing для регуляризации

### **Современные тренды:**

- **Vision Transformers (ViT)** — альтернатива CNN
- **EfficientNet** — оптимальный scaling
- **Neural Architecture Search (NAS)** — автоматический дизайн
- **Self-supervised learning** — обучение без разметки

> **"ResNet изменил paradigm: теперь мы можем обучать сети любой глубины. Skip connections — это must-have в современных архитектурах."**

**Дальнейшее изучение:**
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385) (ResNet paper)
- [Very Deep Convolutional Networks](https://arxiv.org/abs/1409.1556) (VGG paper)
- [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) (Inception paper)
- [ImageNet Classification with Deep CNNs](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) (AlexNet paper)

---

## 📝 Задачи

**[Перейти к задачам по архитектурам CNN →](441_cnn_architectures_tasks.md)**

Практические задания для закрепления материала:
- 🟢 Базовый уровень: реализация LeNet, использование предобученных моделей
- 🟡 Продвинутый уровень: сравнение VGG vs ResNet, визуализация feature maps
- 🔴 Экспертный уровень: создание гибридных архитектур, NAS, оптимизация производительности
