# Основы CNN (Convolutional Neural Networks)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# !pip install torch torchvision matplotlib
```

---

## 🟢 Базовый уровень: Что такое CNN?

### 1.1 Проблема полносвязных сетей для изображений

**Проблема:** Изображение 28x28 = 784 параметра. Изображение 224x224x3 = 150,528 параметров!

```python
# MLP для изображения 224x224x3
mlp_input_size = 224 * 224 * 3  # 150,528
mlp_hidden = 1000
mlp_params = mlp_input_size * mlp_hidden  # 150,528,000 параметров!

print(f"MLP параметров только в первом слое: {mlp_params:,}")
# Это огромное количество! CNN решает эту проблему.
```

### 1.2 Сверточный слой (Convolution)

**Идея:** Скользящее окно (kernel/filter) обрабатывает локальные области.

```python
# Простая свертка вручную
def simple_conv2d(image, kernel):
    """Простая 2D свертка без padding"""
    h, w = image.shape
    kh, kw = kernel.shape
    
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            # Элементная операция и сумма
            output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    
    return output

# Пример: обнаружение вертикальных границ
image = np.array([
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0]
])

vertical_edge_kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

result = simple_conv2d(image, vertical_edge_kernel)
print("Результат свертки (вертикальные границы):")
print(result)

# Визуализация
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Исходное изображение')
axes[1].imshow(vertical_edge_kernel, cmap='gray')
axes[1].set_title('Kernel (детектор границ)')
axes[2].imshow(result, cmap='gray')
axes[2].set_title('После свертки')
plt.show()
```

### 1.3 Первая CNN в PyTorch

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Conv2d(in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # x: [batch, 1, 28, 28]
        x = F.relu(self.conv1(x))          # [batch, 32, 28, 28]
        x = F.max_pool2d(x, 2)             # [batch, 32, 14, 14]
        x = F.relu(self.conv2(x))          # [batch, 64, 14, 14]
        x = F.max_pool2d(x, 2)             # [batch, 64, 7, 7]
        x = x.view(x.size(0), -1)          # [batch, 64*7*7]
        x = F.relu(self.fc1(x))            # [batch, 128]
        x = self.fc2(x)                    # [batch, 10]
        return x

model = SimpleCNN()
print(model)

# Тест
x = torch.randn(1, 1, 28, 28)
output = model(x)
print(f"Выход: {output.shape}")  # [1, 10]
```

---

## 🟡 Продвинутый уровень: Параметры свертки

### 2.1 Stride — шаг свертки

**Stride** — на сколько пикселей сдвигается kernel.

```python
# Stride = 1 (по умолчанию)
conv_stride1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
x = torch.randn(1, 1, 28, 28)
out1 = conv_stride1(x)
print(f"Stride=1: {x.shape} -> {out1.shape}")  # [1,1,28,28] -> [1,16,28,28]

# Stride = 2
conv_stride2 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
out2 = conv_stride2(x)
print(f"Stride=2: {x.shape} -> {out2.shape}")  # [1,1,28,28] -> [1,16,14,14]

# Формула размера выхода:
# output_size = (input_size - kernel_size + 2*padding) / stride + 1
```

### 2.2 Padding — дополнение краев

```python
# Без padding
conv_no_pad = nn.Conv2d(1, 16, kernel_size=3, padding=0)
out_no_pad = conv_no_pad(x)
print(f"No padding: {x.shape} -> {out_no_pad.shape}")  # 28 -> 26

# С padding=1 (сохраняет размер)
conv_pad1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
out_pad1 = conv_pad1(x)
print(f"Padding=1: {x.shape} -> {out_pad1.shape}")  # 28 -> 28

# Same padding: padding = (kernel_size - 1) // 2
```

### 2.3 Pooling — уменьшение размера

**Max Pooling** — берет максимум в окне  
**Average Pooling** — берет среднее

```python
# Max Pooling 2x2
x = torch.tensor([[
    [1., 2., 3., 4.],
    [5., 6., 7., 8.],
    [9., 10., 11., 12.],
    [13., 14., 15., 16.]
]]).unsqueeze(0)  # [1, 1, 4, 4]

max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

max_out = max_pool(x)
avg_out = avg_pool(x)

print("Исходное:")
print(x.squeeze())
print("\nMax Pooling:")
print(max_out.squeeze())  # [[6, 8], [14, 16]]
print("\nAverage Pooling:")
print(avg_out.squeeze())  # [[3.5, 5.5], [11.5, 13.5]]
```

### 2.4 Обучение CNN на MNIST

```python
# Полный пример обучения
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                  transform=transforms.ToTensor()),
    batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=1000
)

model = SimpleCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data, target in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return 100. * correct / len(loader.dataset)

# Обучение
for epoch in range(5):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    test_acc = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Accuracy={test_acc:.2f}%")
```

---

## 🔴 Экспертный уровень: Продвинутые концепции

### 3.1 Receptive Field — поле восприятия

**Receptive field** — область входного изображения, влияющая на один выходной нейрон.

```python
def calculate_receptive_field(layers):
    """Вычисление receptive field"""
    rf = 1  # начальное поле
    stride_prod = 1  # произведение stride'ов
    
    for kernel_size, stride in layers:
        rf = rf + (kernel_size - 1) * stride_prod
        stride_prod *= stride
    
    return rf

# Пример: две свертки 3x3 с stride=1
layers = [(3, 1), (3, 1)]
rf = calculate_receptive_field(layers)
print(f"Receptive field после 2 сверток 3x3: {rf}")  # 5x5

# С pooling (stride=2)
layers_with_pool = [(3, 1), (2, 2), (3, 1)]
rf_pool = calculate_receptive_field(layers_with_pool)
print(f"Receptive field с pooling: {rf_pool}")  # 10x10
```

### 3.2 Depthwise Separable Convolution

**Идея:** Разделение свертки по каналам → меньше параметров.

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise: каждый канал отдельно
        self.depthwise = nn.Conv2d(in_channels, in_channels, 
                                  kernel_size, padding=1, groups=in_channels)
        # Pointwise: 1x1 свертка для объединения
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Сравнение параметров
in_ch, out_ch = 32, 64
kernel = 3

# Обычная свертка
standard_conv = nn.Conv2d(in_ch, out_ch, kernel)
standard_params = in_ch * out_ch * kernel * kernel

# Depthwise Separable
sep_conv = DepthwiseSeparableConv(in_ch, out_ch, kernel)
sep_params = in_ch * kernel * kernel + in_ch * out_ch

print(f"Стандартная: {standard_params:,} параметров")
print(f"Separable: {sep_params:,} параметров")
print(f"Экономия: {standard_params / sep_params:.1f}x")
```

### 3.3 Dilated Convolution — расширенная свертка

```python
# Dilation увеличивает receptive field без увеличения параметров
conv_normal = nn.Conv2d(1, 16, kernel_size=3, padding=1)
conv_dilated = nn.Conv2d(1, 16, kernel_size=3, padding=2, dilation=2)

x = torch.randn(1, 1, 28, 28)
out_normal = conv_normal(x)
out_dilated = conv_dilated(x)

print(f"Normal: {out_normal.shape}")    # [1, 16, 28, 28]
print(f"Dilated: {out_dilated.shape}")  # [1, 16, 28, 28]
# Но dilated имеет больший receptive field!
```

### 3.4 Визуализация фильтров CNN

```python
def visualize_filters(model, layer_name='conv1'):
    """Визуализация фильтров первого слоя"""
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            weights = module.weight.data.cpu()
            
            # Берем первые 32 фильтра
            num_filters = min(32, weights.shape[0])
            fig, axes = plt.subplots(4, 8, figsize=(12, 6))
            
            for i, ax in enumerate(axes.flat):
                if i < num_filters:
                    # Нормализуем для визуализации
                    filter_img = weights[i, 0]  # первый канал
                    ax.imshow(filter_img, cmap='gray')
                ax.axis('off')
            
            plt.suptitle(f'Фильтры слоя {layer_name}')
            plt.tight_layout()
            plt.show()
            break

# Визуализация обученных фильтров
visualize_filters(model, 'conv1')
```

### 3.5 Feature Maps — карты признаков

```python
def visualize_feature_maps(model, image, layer_name='conv1'):
    """Визуализация активаций"""
    activation = {}
    
    def hook_fn(module, input, output):
        activation['output'] = output
    
    # Регистрируем hook
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook_fn)
            break
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(image.unsqueeze(0))
    
    # Визуализация
    feature_maps = activation['output'].squeeze().cpu()
    num_maps = min(16, feature_maps.shape[0])
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < num_maps:
            ax.imshow(feature_maps[i], cmap='viridis')
        ax.axis('off')
    
    plt.suptitle(f'Feature Maps: {layer_name}')
    plt.tight_layout()
    plt.show()

# Пример
test_image, _ = next(iter(test_loader))
visualize_feature_maps(model, test_image[0], 'conv1')
```

---

## 💎 Заключение

**Ключевые концепции CNN:**

| Концепция | Описание | Зачем нужно |
|-----------|----------|-------------|
| **Convolution** | Скользящее окно для обработки | Локальные паттерны, shared weights |
| **Padding** | Дополнение краев | Сохранение размера |
| **Stride** | Шаг свертки | Уменьшение размера |
| **Pooling** | Агрегация (max/avg) | Уменьшение размера, инвариантность |
| **Receptive Field** | Область "видимости" | Понимание контекста |

**Преимущества CNN:**
- ✅ Меньше параметров (shared weights)
- ✅ Локальная связность (spatial locality)
- ✅ Инвариантность к сдвигам (через pooling)
- ✅ Иерархия признаков (низкие → высокие)

**Типичная архитектура CNN:**
```
Input → [Conv-ReLU-Pool] × N → Flatten → FC layers → Output
```

**Формулы размеров:**
```python
# Размер после свертки
output_size = (input_size - kernel_size + 2*padding) / stride + 1

# Количество параметров в Conv2d
params = out_channels * (in_channels * kernel_h * kernel_w + 1)
#                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   bias
```

**Лучшие практики:**
- ✅ Начинайте с малых фильтров (3x3)
- ✅ Используйте padding=1 для сохранения размера
- ✅ Max pooling лучше average pooling
- ✅ Batch normalization после каждой свертки
- ✅ ReLU как активация
- ✅ Dropout перед FC слоями

**Типичные ошибки:**
- ❌ Слишком большие фильтры (используйте несколько 3x3 вместо одного 7x7)
- ❌ Забыть flatten перед FC слоями
- ❌ Неправильный расчет размеров после свертки
- ❌ Pooling перед каждой сверткой (слишком быстрое уменьшение)

**Когда использовать CNN:**
- ✅ Изображения
- ✅ Видео
- ✅ Сигналы (1D convolution)
- ✅ Любые данные с пространственной структурой

> **"CNN революционизировали computer vision. Понимание основ свертки критично для работы с изображениями."**

**Дальнейшее изучение:**
- [CS231n: Convolutional Networks](https://cs231n.github.io/convolutional-networks/)
- [Understanding Convolutions](https://colah.github.io/posts/2014-07-Understanding-Convolutions/)
- [A guide to convolution arithmetic](https://arxiv.org/abs/1603.07285)
