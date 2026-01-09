### **Задачи: Современные архитектуры CNN**

**Цель:** Понять принципы работы современных архитектур (EfficientNet, MobileNet, ViT), научиться выбирать оптимальную модель для задачи, сравнить производительность.

---

## 🟢 Базовый уровень

### **Задача 1: Сравнение EfficientNet-B0 и ResNet50**

**Условие:** Сравните EfficientNet-B0 и ResNet50 на датасете CIFAR-100.

**Требования:**
1. Загрузите обе предобученные модели
2. Адаптируйте под CIFAR-100 (100 классов)
3. Feature extraction (5 эпох) для обеих
4. Сравните:
   - Количество параметров
   - Размер модели в MB
   - Accuracy на test set
   - Время обучения одной эпохи
   - Время inference (100 изображений)

5. Постройте bar charts для сравнения

**Ожидаемый результат:** EfficientNet-B0 будет меньше и быстрее при сопоставимом качестве.

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import time

def evaluate_model(model_name, model, train_loader, test_loader):
    """Оценивает модель по всем метрикам"""
    
    # Параметры
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Размер на диске
    torch.save(model.state_dict(), f'{model_name}.pth')
    size_mb = os.path.getsize(f'{model_name}.pth') / (1024 ** 2)
    
    # TODO: обучите модель и измерьте время
    # TODO: оцените accuracy
    # TODO: измерьте время inference
    
    return {
        'model': model_name,
        'params': params,
        'trainable_params': trainable,
        'size_mb': size_mb,
        'train_time_sec': ...,
        'test_acc': ...,
        'inference_time_ms': ...
    }

# TODO: загрузите и оцените обе модели
# TODO: визуализируйте результаты
```

**Вопросы для анализа:**
1. Почему EfficientNet-B0 меньше ResNet50, но показывает сопоставимое качество?
2. Что такое compound scaling и как он применяется?
3. В каких случаях ResNet50 может быть предпочтительнее EfficientNet-B0?

---

### **Задача 2: MobileNet для мобильного deployment**

**Условие:** Сравните разные версии MobileNet для определения оптимальной для мобильного устройства.

**Требования:**
1. Используйте модели:
   - MobileNetV2
   - MobileNetV3-Small
   - MobileNetV3-Large

2. Для каждой модели измерьте:
   - Количество параметров
   - FLOPs (floating point operations)
   - Latency на CPU (100 изображений)
   - Accuracy на CIFAR-10

3. Постройте scatter plot: latency vs accuracy
4. Определите оптимальную модель для:
   - Максимального качества
   - Минимальной latency
   - Лучшего баланса

**Ожидаемый результат:** MobileNetV3-Small — минимальная latency, MobileNetV3-Large — лучший баланс.

```python
from thop import profile  # !pip install thop

def measure_model(model, input_size=(1, 3, 224, 224)):
    """Измеряет параметры и FLOPs"""
    dummy_input = torch.randn(input_size)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    return {
        'params': params,
        'flops': flops,
        'gflops': flops / 1e9
    }

def measure_latency(model, num_samples=100):
    """Измеряет latency на CPU"""
    model.eval()
    model.cpu()
    
    dummy_input = torch.randn(num_samples, 3, 224, 224)
    
    # Warmup
    with torch.no_grad():
        _ = model(dummy_input[:10])
    
    # Измерение
    start = time.time()
    with torch.no_grad():
        _ = model(dummy_input)
    end = time.time()
    
    return (end - start) * 1000 / num_samples  # ms per image

# TODO: оцените все модели
# TODO: визуализируйте trade-off
```

---

### **Задача 3: EfficientNet Family — выбор оптимальной версии**

**Условие:** Исследуйте семейство EfficientNet (B0-B7) и выберите оптимальную версию для вашей задачи.

**Требования:**
1. Используйте модели B0, B2, B4, B6 (пропускаем нечетные для экономии времени)
2. Для каждой модели:
   - Feature extraction на маленьком датасете (5000 изображений, 10 классов)
   - 5 эпох обучения
   
3. Измерьте:
   - Train/Val accuracy
   - Время обучения эпохи
   - Память GPU (peak usage)
   - Inference time

4. Постройте графики:
   - Accuracy vs Model Size (B0-B6)
   - Training Time vs Model Size
   - Memory Usage vs Model Size

**Ожидаемый результат:** Качество растет с размером модели, но после B4 прирост минимален.

```python
models_family = {
    'B0': models.efficientnet_b0(pretrained=True),
    'B2': models.efficientnet_b2(pretrained=True),
    'B4': models.efficientnet_b4(pretrained=True),
    'B6': models.efficientnet_b6(pretrained=True),
}

# TODO: обучите все модели
# TODO: постройте графики зависимостей
```

**Вопросы:**
1. Как compound scaling влияет на performance?
2. Какая версия оптимальна для production?
3. Стоит ли использовать B6/B7 на практике?

---

## 🟡 Продвинутый уровень

### **Задача 4: Реализация MBConv блока**

**Условие:** Реализуйте MBConv block (Mobile Inverted Bottleneck Convolution) с нуля.

**Требования:**
1. Реализуйте полный MBConv с:
   - Expansion phase (1x1 conv)
   - Depthwise convolution (3x3 или 5x5)
   - Squeeze-and-Excitation
   - Projection phase (1x1 conv)
   - Skip connection

2. Создайте маленькую сеть из нескольких MBConv блоков
3. Обучите на MNIST
4. Сравните с обычной CNN (обычные Conv2d)
5. Измерьте:
   - Количество параметров
   - FLOPs
   - Accuracy

**Ожидаемый результат:** MBConv сеть будет легче и быстрее при сопоставимом качестве.

```python
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        # TODO: реализуйте SE block
        pass
    
    def forward(self, x):
        # TODO: реализуйте forward pass
        pass

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution"""
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride):
        super().__init__()
        # TODO: реализуйте все фазы
        pass
    
    def forward(self, x):
        # TODO: реализуйте forward pass
        pass

# TODO: создайте сеть из MBConv блоков
# TODO: сравните с обычной CNN
```

---

### **Задача 5: Vision Transformer с нуля**

**Условие:** Реализуйте упрощенный Vision Transformer и обучите на CIFAR-10.

**Требования:**
1. Реализуйте компоненты:
   - Patch Embedding (разбиение на patches 4x4)
   - Positional Encoding
   - Transformer Encoder Block
   - Classification Head

2. Создайте маленький ViT:
   - Patch size: 4x4
   - Embed dim: 256
   - Depth: 6 layers
   - Num heads: 8

3. Обучите на CIFAR-10 (20 эпох)
4. Визуализируйте attention maps
5. Сравните с CNN (EfficientNet-B0)

**Ожидаемый результат:** ViT будет медленнее обучаться, но достигнет сопоставимого качества.

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        # TODO: реализуйте patch extraction и projection
        pass
    
    def forward(self, x):
        # TODO: x: [B, C, H, W] -> [B, num_patches, embed_dim]
        pass

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        # TODO: реализуйте self-attention + MLP
        pass
    
    def forward(self, x):
        # TODO: реализуйте forward pass
        pass

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, embed_dim, depth, num_heads):
        super().__init__()
        # TODO: соберите все компоненты
        pass
    
    def forward(self, x):
        # TODO: реализуйте forward pass
        pass

# TODO: обучите ViT
# TODO: визуализируйте attention
```

---

### **Задача 6: Визуализация attention в ViT**

**Условие:** Визуализируйте attention maps в Vision Transformer для понимания, на что модель обращает внимание.

**Требования:**
1. Загрузите предобученный ViT (например, vit_base_patch16_224)
2. Реализуйте extraction attention maps из разных слоев
3. Для нескольких тестовых изображений:
   - Визуализируйте attention от CLS token к patches
   - Постройте attention maps для разных слоев (1, 6, 12)
   - Наложите attention на оригинальное изображение

4. Сравните attention patterns для:
   - Простых изображений (один объект)
   - Сложных изображений (несколько объектов)

**Ожидаемый результат:** Ранние слои смотрят на локальные паттерны, поздние — на семантически важные области.

```python
import timm

def get_attention_maps(model, image, layer_idx):
    """Извлекает attention maps из заданного слоя"""
    
    attentions = []
    
    def hook(module, input, output):
        # output: (batch, num_heads, num_patches, num_patches)
        attentions.append(output.detach())
    
    # Регистрируем hook на нужный transformer block
    handle = model.blocks[layer_idx].attn.register_forward_hook(hook)
    
    with torch.no_grad():
        _ = model(image.unsqueeze(0))
    
    handle.remove()
    
    return attentions[0]

def visualize_attention(image, attention, patch_size=16):
    """Визуализирует attention на изображении"""
    # TODO: реализуйте наложение attention map на изображение
    pass

# TODO: загрузите ViT
# TODO: извлеките и визуализируйте attention для разных слоев
```

---

## 🔴 Экспертный уровень

### **Задача 7: Hybrid Architecture — CNN + Transformer**

**Условие:** Создайте гибридную архитектуру, комбинирующую CNN backbone и Transformer head.

**Требования:**
1. **Архитектура:**
   - CNN backbone: ResNet18 (без fc layer)
   - Feature extractor: AdaptiveAvgPool → flatten
   - Reshape в sequence: [B, 512] → [B, 32, 16]
   - Transformer: 4 layers, 8 heads
   - Classification head

2. Обучите на CIFAR-100
3. Сравните с:
   - Pure CNN (ResNet18)
   - Pure ViT
   - Hybrid (ваша модель)

4. Измерьте:
   - Accuracy
   - Inference speed
   - Memory usage

**Ожидаемый результат:** Hybrid будет лучше Pure CNN, но быстрее Pure ViT.

```python
class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # CNN backbone
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # до avgpool
        
        # Transformer
        # TODO: реализуйте transformer encoder
        
        # Classifier
        # TODO: реализуйте classification head
        
    def forward(self, x):
        # TODO: CNN features -> reshape -> transformer -> classify
        pass

# TODO: обучите и сравните три подхода
```

---

### **Задача 8: Knowledge Distillation: EfficientNet-B7 → MobileNetV3**

**Условие:** Перенесите знания из большой модели (EfficientNet-B7) в маленькую (MobileNetV3).

**Требования:**
1. **Teacher:** EfficientNet-B7 (предобученная)
2. **Student:** MobileNetV3-Small (обучаем)
3. Реализуйте distillation с:
   - Hard targets (ground truth labels)
   - Soft targets (teacher predictions)
   - Feature matching (intermediate layers)

4. Экспериментируйте с:
   - Temperature T: [1, 3, 5, 10]
   - Alpha (баланс hard/soft): [0.3, 0.5, 0.7]
   
5. Сравните 4 варианта:
   - Student с нуля
   - Student с transfer learning
   - Student с distillation (logits only)
   - Student с distillation (logits + features)

**Ожидаемый результат:** Distillation даст лучшее качество, чем training с нуля.

```python
def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
    """Combined distillation loss"""
    
    # Soft loss (KL divergence)
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (T ** 2)
    
    # Hard loss (cross entropy)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * soft_loss + (1 - alpha) * hard_loss

def feature_distillation_loss(student_features, teacher_features):
    """MSE loss between intermediate features"""
    # TODO: реализуйте matching intermediate features
    pass

# TODO: обучите student с разными стратегиями distillation
```

---

### **Задача 9: Neural Architecture Search (NAS) упрощенный**

**Условие:** Реализуйте упрощенный NAS для поиска оптимальной архитектуры из набора блоков.

**Требования:**
1. Определите search space:
   - Блоки: Conv3x3, Conv5x5, MBConv, Skip connection
   - Глубина: 4-8 слоев
   - Ширина: 32-128 каналов

2. Реализуйте random search:
   - Генерируйте 20 случайных архитектур
   - Обучайте каждую 5 эпох на CIFAR-10
   - Выбирайте лучшую по val accuracy

3. Для лучшей архитектуры:
   - Обучите полностью (50 эпох)
   - Сравните с baseline (ResNet18)

4. Визуализируйте:
   - Accuracy vs Num Parameters (scatter plot для всех 20 архитектур)
   - Pareto front (accuracy vs speed)

**Ожидаемый результат:** Найдете архитектуру, сопоставимую с ResNet18, но легче.

```python
import random

SEARCH_SPACE = {
    'blocks': ['conv3x3', 'conv5x5', 'mbconv', 'skip'],
    'depth': list(range(4, 9)),
    'channels': [32, 64, 96, 128]
}

def sample_architecture():
    """Генерирует случайную архитектуру"""
    depth = random.choice(SEARCH_SPACE['depth'])
    channels = random.choice(SEARCH_SPACE['channels'])
    
    blocks = []
    for _ in range(depth):
        block_type = random.choice(SEARCH_SPACE['blocks'])
        blocks.append(block_type)
    
    return {'depth': depth, 'channels': channels, 'blocks': blocks}

def build_model(architecture):
    """Строит модель по архитектуре"""
    # TODO: реализуйте построение модели
    pass

# TODO: реализуйте random search
# TODO: визуализируйте результаты
```

---

### **Задача 10: Сравнение на реальной задаче: Medical Imaging**

**Условие:** Сравните современные архитектуры на медицинском датасете (X-ray, CT, MRI).

**Требования:**
1. Используйте датасет: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

2. Тестируйте модели:
   - EfficientNet-B0
   - EfficientNet-B4
   - MobileNetV3-Large
   - ViT-Base
   - Swin-Tiny

3. Для каждой модели:
   - Transfer learning (fine-tuning)
   - 20 эпох обучения
   - Data augmentation
   
4. Оценивайте по метрикам:
   - Accuracy
   - Precision/Recall
   - F1-score
   - ROC-AUC
   - Confusion matrix

5. Визуализируйте:
   - GradCAM для понимания решений модели
   - ROC curves для всех моделей
   - Comparison table

**Ожидаемый результат:** EfficientNet-B4 покажет лучший баланс качества и скорости.

```python
def train_and_evaluate(model_name, model, train_loader, val_loader, epochs=20):
    """Полный пайплайн обучения и оценки"""
    # TODO: реализуйте обучение
    # TODO: рассчитайте все метрики
    pass

def visualize_gradcam(model, image, target_class):
    """Визуализирует GradCAM для интерпретации"""
    # TODO: реализуйте GradCAM
    pass

# TODO: обучите все модели
# TODO: сравните результаты
# TODO: визуализируйте GradCAM для правильных и неправильных предсказаний
```

**Вопросы для анализа:**
1. Какая модель лучше всего подходит для медицинских изображений?
2. Помогает ли transfer learning с ImageNet на медицинских данных?
3. На что обращают внимание разные модели (по GradCAM)?

---

## 📝 Дополнительные вопросы для размышления

1. **Compound Scaling в EfficientNet:**
   - Почему важно масштабировать одновременно глубину, ширину и разрешение?
   - Можно ли улучшить compound scaling?

2. **Depthwise Separable Convolutions:**
   - Почему они эффективнее обычных сверток?
   - Есть ли задачи, где они работают хуже?

3. **Vision Transformers:**
   - Почему ViT требует больше данных, чем CNN?
   - Что такое inductive bias и как он влияет на обучение?
   - Заменят ли Transformers CNN полностью?

4. **Выбор архитектуры:**
   - Как выбрать оптимальную модель для production?
   - Какие метрики важнее: accuracy или latency?
   - Стоит ли использовать самые новые архитектуры?

---

## 🎯 Критерии успешного выполнения

- ✅ Вы понимаете принципы работы EfficientNet (compound scaling, MBConv)
- ✅ Вы знаете, как работают depthwise separable convolutions
- ✅ Вы понимаете основы Vision Transformers (patches, attention)
- ✅ Вы умеете выбирать оптимальную архитектуру под задачу
- ✅ Вы можете сравнивать модели по accuracy, speed, memory
- ✅ Вы умеете визуализировать attention maps в ViT
- ✅ Вы понимаете trade-offs между разными архитектурами

---

## 📚 Полезные ресурсы

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
- [timm library](https://github.com/rwightman/pytorch-image-models)
- [Papers With Code - Image Classification](https://paperswithcode.com/task/image-classification)
- [Model vs Dataset Size](https://arxiv.org/abs/2106.10270)
