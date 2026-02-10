# Современные архитектуры CNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt

# !pip install torch torchvision timm matplotlib
```

---

## 🟢 Базовый уровень: От ResNet к современным архитектурам

### 1.1 Проблемы классических CNN

**ResNet** (2015) был прорывом, но имеет ограничения:
- ❌ Фиксированная глубина → сложно масштабировать
- ❌ Одинаковый kernel size (3x3) → не оптимально для всех признаков
- ❌ Много параметров при увеличении точности
- ❌ Медленный inference на больших разрешениях

**Современные архитектуры решают эти проблемы:**
- ✅ EfficientNet — оптимальное масштабирование (глубина + ширина + разрешение)
- ✅ MobileNet — эффективность для мобильных устройств
- ✅ Vision Transformer — альтернатива сверткам через self-attention

---

## 🟡 Продвинутый уровень: EfficientNet

### 2.1 Главная идея: Compound Scaling

**Проблема:** Как правильно увеличивать модель для повышения точности?

**Наивные подходы:**
1. **Больше слоев** (depth) → ResNet18 → ResNet50 → ResNet152
2. **Шире слои** (width) → больше фильтров
3. **Больше разрешение** (resolution) → 224x224 → 384x384

**Проблема наивных подходов:** 
- Масштабирование только по одному измерению неэффективно
- Требуется ручной подбор гиперпараметров

**Решение EfficientNet: Compound Scaling**

Масштабируем **одновременно** глубину, ширину и разрешение по формулам:

```
depth: d = α^φ
width: w = β^φ
resolution: r = γ^φ

где α, β, γ — константы (подбираются один раз)
φ — compound coefficient (1, 2, 3, ...)
```

```python
# Пример: EfficientNet-B0 → EfficientNet-B7
# φ увеличивается от 0 до 7

models_family = {
    'B0': {'depth': 1.0, 'width': 1.0, 'resolution': 224},  # baseline
    'B1': {'depth': 1.1, 'width': 1.0, 'resolution': 240},
    'B2': {'depth': 1.2, 'width': 1.1, 'resolution': 260},
    'B3': {'depth': 1.4, 'width': 1.2, 'resolution': 300},
    'B4': {'depth': 1.8, 'width': 1.4, 'resolution': 380},
    'B5': {'depth': 2.2, 'width': 1.6, 'resolution': 456},
    'B6': {'depth': 2.6, 'width': 1.8, 'resolution': 528},
    'B7': {'depth': 3.1, 'width': 2.0, 'resolution': 600},
}
```

**Результат:** 
- EfficientNet-B7: 84.3% top-1 accuracy на ImageNet
- В **8.4 раза меньше параметров** и **6.1 раза быстрее**, чем лучшая альтернатива (GPipe)

---

### 2.2 MBConv: Mobile Inverted Bottleneck Convolution

EfficientNet использует **MBConv block** вместо обычных сверток.

**Структура MBConv:**
1. **Expansion:** 1x1 conv увеличивает каналы в 6 раз
2. **Depthwise conv:** 3x3 или 5x5 (разделяемая свертка)
3. **Squeeze-and-Excitation:** recalibration каналов
4. **Projection:** 1x1 conv уменьшает каналы обратно
5. **Skip connection** (если input = output size)

```python
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6, kernel_size=3, stride=1):
        super().__init__()
        
        expanded = in_channels * expand_ratio
        
        # 1. Expansion phase
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded, 1, bias=False),
            nn.BatchNorm2d(expanded),
            nn.SiLU()  # Swish activation
        )
        
        # 2. Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded, expanded, kernel_size, stride, 
                     padding=kernel_size//2, groups=expanded, bias=False),
            nn.BatchNorm2d(expanded),
            nn.SiLU()
        )
        
        # 3. Squeeze-and-Excitation
        se_channels = max(1, in_channels // 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded, se_channels, 1),
            nn.SiLU(),
            nn.Conv2d(se_channels, expanded, 1),
            nn.Sigmoid()
        )
        
        # 4. Projection phase
        self.project = nn.Sequential(
            nn.Conv2d(expanded, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection
        self.use_residual = (stride == 1 and in_channels == out_channels)
    
    def forward(self, x):
        identity = x
        
        x = self.expand(x)
        x = self.depthwise(x)
        
        # SE block
        se = self.se(x)
        x = x * se
        
        x = self.project(x)
        
        if self.use_residual:
            x = x + identity
        
        return x

# Тест
block = MBConvBlock(32, 64, expand_ratio=6, kernel_size=3)
x = torch.randn(1, 32, 56, 56)
output = block(x)
print(f"Input: {x.shape}, Output: {output.shape}")
```

**Преимущества MBConv:**
- ✅ Меньше параметров (depthwise conv)
- ✅ SE block адаптивно взвешивает каналы
- ✅ Swish activation лучше ReLU

---

### 2.3 Использование EfficientNet в PyTorch

```python
import torchvision.models as models

# Загрузка предобученных моделей
model_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model_b7 = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)

# Адаптация под свою задачу (10 классов)
num_features = model_b0.classifier[1].in_features
model_b0.classifier[1] = nn.Linear(num_features, 10)

# Сравнение размеров
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

print(f"EfficientNet-B0: {count_parameters(model_b0):,} параметров")
print(f"EfficientNet-B7: {count_parameters(model_b7):,} параметров")
print(f"ResNet50: {count_parameters(models.resnet50()):,} параметров")

# EfficientNet-B0: 5,288,548 параметров
# EfficientNet-B7: 66,347,960 параметров
# ResNet50: 25,557,032 параметров
```

**Рекомендации по выбору:**
- **B0/B1:** Быстрые эксперименты, мало памяти
- **B2/B3:** Баланс качества и скорости
- **B4/B5:** Соревнования Kaggle
- **B6/B7:** Максимальное качество, есть мощное GPU

---

## 🟡 Продвинутый уровень: MobileNet

### 3.1 Главная идея: Depthwise Separable Convolutions

**Обычная свертка:**
```
Input: H x W x C_in
Kernel: K x K x C_in x C_out
Output: H x W x C_out

Вычислений: H * W * K * K * C_in * C_out
```

**Depthwise Separable Convolution = Depthwise + Pointwise:**

**Шаг 1: Depthwise Convolution** (свертка по каналам отдельно)
```
Input: H x W x C_in
Kernel: K x K x 1 (для каждого канала)
Output: H x W x C_in

Вычислений: H * W * K * K * C_in
```

**Шаг 2: Pointwise Convolution** (1x1 свертка)
```
Input: H x W x C_in
Kernel: 1 x 1 x C_in x C_out
Output: H x W x C_out

Вычислений: H * W * C_in * C_out
```

**Итого вычислений:**
```
Обычная:    H * W * K * K * C_in * C_out
Separable:  H * W * K * K * C_in  +  H * W * C_in * C_out
          = H * W * C_in * (K * K + C_out)

Ускорение: (K * K * C_out) / (K * K + C_out) ≈ K * K  (для больших C_out)
Для K=3: ускорение ~ 8-9 раз!
```

---

### 3.2 MobileNetV2: Inverted Residual Block

**Классический residual block (ResNet):**
```
Wide → Narrow → Wide
[256] → [64] → [256]  (bottleneck)
```

**Inverted residual block (MobileNetV2):**
```
Narrow → Wide → Narrow
[64] → [384] → [64]  (expansion)
```

**Почему "inverted"?**
- В ResNet нелинейность применяется в узком месте → потеря информации
- В MobileNetV2 нелинейность в широкой части → меньше потерь

```python
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        
        hidden = in_channels * expand_ratio
        
        layers = []
        
        # Expansion (если expand_ratio > 1)
        if expand_ratio > 1:
            layers.append(nn.Conv2d(in_channels, hidden, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden))
            layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
        ])
        
        # Pointwise (linear, без activation!)
        layers.extend([
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
        self.use_residual = (stride == 1 and in_channels == out_channels)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

# Тест
block = InvertedResidual(32, 64, stride=1, expand_ratio=6)
x = torch.randn(1, 32, 56, 56)
print(f"Параметров: {sum(p.numel() for p in block.parameters()):,}")
```

---

### 3.3 MobileNetV3: Neural Architecture Search

MobileNetV3 использует **Neural Architecture Search (NAS)** для автоматического поиска оптимальной архитектуры.

**Улучшения в V3:**
- ✅ h-swish activation вместо ReLU6: `h-swish(x) = x * ReLU6(x + 3) / 6`
- ✅ Squeeze-and-Excitation блоки
- ✅ Redesigned head (последние слои)
- ✅ Две версии: Large (качество) и Small (скорость)

```python
# Использование MobileNetV3
model_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
model_small = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

# Адаптация
model_small.classifier[3] = nn.Linear(model_small.classifier[3].in_features, 10)

print(f"MobileNetV3-Large: {count_parameters(model_large):,}")
print(f"MobileNetV3-Small: {count_parameters(model_small):,}")

# MobileNetV3-Large: 5,483,032 параметров
# MobileNetV3-Small: 2,542,856 параметров
```

---

## 🔴 Экспертный уровень: Vision Transformer (ViT)

### 4.1 От CNN к Transformers

**Проблема CNN:**
- Локальный receptive field → нужны глубокие сети для глобального контекста
- Фиксированный kernel size → ограниченная гибкость
- Inductive bias (locality, translation equivariance) → иногда мешает

**Решение: Vision Transformer**
- Изображение разбивается на patches (16x16)
- Patches обрабатываются как tokens (как слова в NLP)
- Self-attention улавливает глобальные зависимости

---

### 4.2 Архитектура ViT

**Шаг 1: Patch Embedding**
```
Изображение 224x224x3
↓ разбиваем на patches 16x16
= 14x14 patches
↓ flatten каждый patch
= 196 patches × 768 features
```

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        
        self.num_patches = (img_size // patch_size) ** 2
        
        # Свертка с kernel_size = patch_size делает patch extraction + linear projection
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: [B, 3, 224, 224]
        x = self.proj(x)  # [B, 768, 14, 14]
        x = x.flatten(2)  # [B, 768, 196]
        x = x.transpose(1, 2)  # [B, 196, 768]
        return x
```

**Шаг 2: Positional Encoding**
```python
# Learnable positional embeddings
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
```

**Шаг 3: Transformer Encoder**
```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x
```

**Полная архитектура:**
```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, 196, 768]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, 768]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 197, 768]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Classification (используем только CLS token)
        cls_output = x[:, 0]
        return self.head(cls_output)

# Создание модели
model = VisionTransformer(img_size=224, patch_size=16, num_classes=1000)
print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")
# Параметров: 86,567,656
```

---

### 4.3 Использование предобученных ViT

```python
# С помощью timm (PyTorch Image Models)
# !pip install timm

import timm

# Доступные модели
available = timm.list_models('vit*', pretrained=True)
print(f"Доступно {len(available)} предобученных ViT моделей")

# Загрузка
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
model.eval()

# Inference
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(f"Output shape: {output.shape}")  # [1, 10]
```

**Популярные варианты ViT:**
- **ViT-Base/16:** 86M параметров, patch 16x16
- **ViT-Large/16:** 307M параметров
- **ViT-Huge/14:** 632M параметров, patch 14x14

---

### 4.4 ViT vs CNN: Когда что использовать?

| Критерий | CNN (ResNet, EfficientNet) | ViT |
|----------|----------------------------|-----|
| **Данные** | Работает с малым датасетом (< 100K) | Нужно много данных (> 1M) |
| **Точность** | 80-85% ImageNet | 85-90% ImageNet |
| **Скорость** | Быстрее | Медленнее |
| **Память** | Меньше | Больше |
| **Интерпретируемость** | Feature maps | Attention maps |
| **Transfer learning** | Отлично работает | Отлично работает (на больших данных) |

**Рекомендации:**
- ✅ **CNN:** Малые датасеты, ограниченные ресурсы, inference на edge devices
- ✅ **ViT:** Большие датасеты, мощное GPU, нужна максимальная точность

---

### 4.5 Hybrid: Комбинация CNN и Transformer

**Swin Transformer** — иерархический vision transformer с shifted windows.

```python
import timm

model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)

print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")
# Swin-Base: 88M параметров, 85.2% top-1 accuracy
```

**Преимущества Swin:**
- ✅ Иерархическая структура (как CNN)
- ✅ Локальный self-attention (shifted windows) → меньше вычислений
- ✅ Лучше на dense prediction tasks (segmentation, detection)

---

## 📊 Сравнительная таблица

| Модель | Параметры | Top-1 Acc | Inference (ms) | Когда использовать |
|--------|-----------|-----------|----------------|-------------------|
| **ResNet50** | 25M | 76.1% | 15 | Baseline, проверенная архитектура |
| **EfficientNet-B0** | 5M | 77.1% | 12 | Мало памяти, быстрые эксперименты |
| **EfficientNet-B7** | 66M | 84.3% | 80 | Максимальное качество на CNN |
| **MobileNetV3-Small** | 2.5M | 67.4% | 5 | Мобильные устройства |
| **MobileNetV3-Large** | 5.5M | 75.2% | 8 | Баланс качества/скорость на mobile |
| **ViT-Base/16** | 86M | 77.9% | 25 | Много данных, нужен transformer |
| **ViT-Large/16** | 307M | 76.5% | 65 | Максимальная емкость модели |
| **Swin-Base** | 88M | 85.2% | 30 | SOTA качество, гибридный подход |

*(Inference на NVIDIA V100, batch=1, image 224x224)*

---

## 🎯 Ключевые выводы

1. **EfficientNet** — лучший выбор для большинства задач (compound scaling, MBConv)

2. **MobileNet** — оптимален для мобильных и embedded устройств (depthwise separable conv)

3. **Vision Transformer** — SOTA на больших датасетах, но требует много данных и ресурсов

4. **Swin Transformer** — hybrid подход, сочетающий преимущества CNN и ViT

5. **Выбор модели:**
   - Ограничены ресурсы → MobileNetV3
   - Универсальная задача → EfficientNet-B0/B2
   - Соревнование Kaggle → EfficientNet-B4/B7
   - Исследования, много данных → ViT или Swin

6. **Transfer learning** работает отлично для всех архитектур!

---

## 📚 Дополнительные материалы

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
- [timm library documentation](https://timm.fast.ai/)
- [Papers With Code - Image Classification](https://paperswithcode.com/task/image-classification)
