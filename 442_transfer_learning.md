# Transfer Learning

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# !pip install torch torchvision matplotlib pillow
```

---

## 🟢 Базовый уровень: Что такое Transfer Learning?

### 1.1 Основная идея

**Transfer Learning** — использование знаний, полученных при решении одной задачи, для решения другой связанной задачи.

**Зачем это нужно?**
- ✅ Экономия времени обучения (часы вместо дней/недель)
- ✅ Меньше требуется данных (сотни вместо миллионов)
- ✅ Лучшее качество на малых датасетах
- ✅ Использование знаний с ImageNet (14M изображений)

**Основной принцип:**
- Нижние слои CNN учат **общие признаки** (края, текстуры, формы)
- Верхние слои учат **специфичные признаки** (глаза, колеса, буквы)

```
ImageNet модель              Ваша задача
(1000 классов)      →        (2 класса: кошки vs собаки)

[Общие признаки]     →       [Общие признаки]  ✓ переиспользуем
[Специфичные]        →       [Специфичные]     ✗ переобучаем
```

---

### 1.2 Два основных подхода

#### **Подход 1: Feature Extraction (Извлечение признаков)**

Используем предобученную сеть как **fixed feature extractor** — замораживаем все слои и обучаем только последний классификатор.

```python
# Загружаем предобученную модель
model = models.resnet18(pretrained=True)

# Замораживаем все параметры
for param in model.parameters():
    param.requires_grad = False

# Меняем последний слой под нашу задачу
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 2 класса: кошки vs собаки

# Проверяем, что обучается только последний слой
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Обучаемых параметров: {trainable_params:,} из {total_params:,}")
# Обучаемых параметров: 1,024 из 11,689,512
```

**Когда использовать:**
- ✅ У вас **мало данных** (< 10K изображений)
- ✅ Ваша задача **похожа** на ImageNet (естественные изображения)
- ✅ **Быстрое** обучение (минуты)
- ❌ Может не хватить гибкости для специфичных задач

---

#### **Подход 2: Fine-Tuning (Дообучение)**

Инициализируем сеть предобученными весами и **дообучаем некоторые/все слои** на наших данных.

```python
# Загружаем предобученную модель
model = models.resnet18(pretrained=True)

# Меняем последний слой
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Все параметры обучаемые (по умолчанию)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Обучаемых параметров: {trainable_params:,}")
# Обучаемых параметров: 11,689,512
```

**Когда использовать:**
- ✅ У вас **достаточно данных** (> 10K изображений)
- ✅ Ваша задача **отличается** от ImageNet (медицинские снимки, спутниковые фото)
- ✅ Нужна **максимальная точность**
- ❌ Медленнее обучение (часы)
- ❌ Риск переобучения на малых данных

---

### 1.3 Стратегии Fine-Tuning

#### **Стратегия 1: Заморозить ранние слои, обучать поздние**

```python
model = models.resnet18(pretrained=True)

# Замораживаем первые N слоев
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

# Обучаются только layer4 и fc
print("Обучаемые слои:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.numel():,} параметров")
```

**Логика:** Ранние слои учат универсальные признаки (края, текстуры), их можно не трогать.

---

#### **Стратегия 2: Differential Learning Rates**

Разные слои обучаем с разными learning rates.

```python
# Разбиваем параметры на группы
params = [
    {'params': model.layer1.parameters(), 'lr': 1e-5},  # Ранние слои — медленно
    {'params': model.layer2.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-3},
    {'params': model.layer4.parameters(), 'lr': 1e-2},
    {'params': model.fc.parameters(), 'lr': 1e-2},      # Последний слой — быстро
]

optimizer = optim.Adam(params)
```

**Логика:** Ранние слои уже хорошо обучены, меняем их осторожно. Последний слой случайный, обучаем агрессивно.

---

## 🟡 Продвинутый уровень: Практическое применение

### 2.1 Полный пример: Dogs vs Cats

```python
# Шаг 1: Подготовка данных
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder('data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Шаг 2: Модель
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Шаг 3: Feature Extraction (сначала обучим только fc)
for param in model.parameters():
    param.requires_grad = False
model.fc.weight.requires_grad = True
model.fc.bias.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Обучение только fc (5 эпох)
for epoch in range(5):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Шаг 4: Fine-tuning (размораживаем все, снижаем lr)
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Обучение всей сети (5 эпох)
for epoch in range(5):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Fine-tuning Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
```

---

### 2.2 Выбор предобученной модели

**Популярные модели для Transfer Learning:**

| Модель | Параметры | Top-1 Accuracy | Когда использовать |
|--------|-----------|----------------|-------------------|
| **ResNet18** | 11M | 69.8% | Быстрые эксперименты, ограниченные ресурсы |
| **ResNet50** | 25M | 76.1% | Баланс качества и скорости |
| **ResNet152** | 60M | 78.3% | Максимальное качество, есть GPU |
| **EfficientNet-B0** | 5M | 77.1% | Мало памяти, нужна скорость |
| **EfficientNet-B7** | 66M | 84.3% | SOTA качество, мощное железо |
| **VGG16** | 138M | 71.6% | Простая архитектура (⚠️ очень тяжелая) |
| **MobileNetV2** | 3.5M | 72.0% | Мобильные устройства |

```python
# Примеры загрузки
model_resnet = models.resnet50(pretrained=True)
model_efficient = models.efficientnet_b0(pretrained=True)
model_mobilenet = models.mobilenet_v2(pretrained=True)
```

---

### 2.3 Аугментация данных при Transfer Learning

Аугментация критична для успешного fine-tuning!

```python
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Для train — агрессивная аугментация
train_dataset = torchvision.datasets.ImageFolder('data/train', transform=train_transform)

# Для val/test — только resize и crop
val_dataset = torchvision.datasets.ImageFolder('data/val', transform=val_transform)
```

---

### 2.4 Визуализация активаций

Посмотрим, что видит сеть на разных слоях:

```python
def visualize_activations(model, image, layer_name):
    """Визуализирует активации заданного слоя"""
    
    # Hook для сохранения активаций
    activations = []
    def hook(module, input, output):
        activations.append(output.detach())
    
    # Регистрируем hook
    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(hook)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(image.unsqueeze(0))
    
    handle.remove()
    
    # Визуализация
    act = activations[0].squeeze().cpu()
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i < act.shape[0]:
            ax.imshow(act[i], cmap='viridis')
            ax.axis('off')
    plt.tight_layout()
    plt.show()

# Пример использования
model = models.resnet18(pretrained=True)
image = transform(Image.open('cat.jpg'))
visualize_activations(model, image, 'layer1.0.conv1')
```

---

## 🔴 Продвинутый уровень: Сложные сценарии

### 3.1 Transfer Learning для специфичных доменов

**Проблема:** ImageNet содержит естественные изображения. Что если ваша задача сильно отличается?

**Решения:**

#### **1. Medical Imaging (рентген, МРТ)**
- Предобучение на ImageNet все равно помогает (базовые признаки)
- Но лучше взять модель, предобученную на медицинских данных:
  - **CheXNet** (рентген грудной клетки)
  - **MedicalNet** (3D МРТ)

```python
# Пример: загрузка специализированной модели
# model = torch.load('chexnet_pretrained.pth')
# Дальше fine-tuning как обычно
```

#### **2. Satellite Imagery (спутниковые снимки)**
- ImageNet модели слабо помогают (другой масштаб, цвета)
- Используйте модели, предобученные на спутниковых данных:
  - **EuroSAT**
  - **Planet: Understanding the Amazon from Space**

#### **3. Microscopy Images (микроскопия)**
- Обычно мало данных (< 1000 изображений)
- Feature extraction работает лучше, чем fine-tuning

---

### 3.2 Multi-task Transfer Learning

Обучаем одну сеть на нескольких задачах одновременно.

```python
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Общий feature extractor
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Убираем последний слой
        
        # Задача 1: классификация (5 классов)
        self.classifier = nn.Linear(512, 5)
        
        # Задача 2: регрессия возраста
        self.age_regressor = nn.Linear(512, 1)
        
        # Задача 3: бинарная классификация пола
        self.gender_classifier = nn.Linear(512, 2)
    
    def forward(self, x):
        features = self.backbone(x)
        
        cls = self.classifier(features)
        age = self.age_regressor(features)
        gender = self.gender_classifier(features)
        
        return cls, age, gender

# Обучение
model = MultiTaskModel()
criterion_cls = nn.CrossEntropyLoss()
criterion_age = nn.MSELoss()
criterion_gender = nn.CrossEntropyLoss()

# В цикле обучения
for inputs, (labels_cls, labels_age, labels_gender) in train_loader:
    outputs_cls, outputs_age, outputs_gender = model(inputs)
    
    loss_cls = criterion_cls(outputs_cls, labels_cls)
    loss_age = criterion_age(outputs_age, labels_age)
    loss_gender = criterion_gender(outputs_gender, labels_gender)
    
    # Взвешенная сумма потерь
    loss = loss_cls + 0.5 * loss_age + 0.3 * loss_gender
    
    loss.backward()
    optimizer.step()
```

---

### 3.3 Knowledge Distillation

Переносим знания из большой модели (teacher) в маленькую (student).

```python
def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
    """
    T: температура для softening probability distribution
    alpha: вес между hard loss (labels) и soft loss (teacher)
    """
    
    # Soft targets от teacher
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (T ** 2)
    
    # Hard targets от ground truth
    hard_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * soft_loss + (1 - alpha) * hard_loss

# Обучение
teacher = models.resnet152(pretrained=True).eval()  # Большая модель
student = models.resnet18(pretrained=False)         # Маленькая модель

for inputs, labels in train_loader:
    # Teacher inference (без градиентов)
    with torch.no_grad():
        teacher_logits = teacher(inputs)
    
    # Student training
    student_logits = student(inputs)
    loss = distillation_loss(student_logits, teacher_logits, labels)
    
    loss.backward()
    optimizer.step()
```

**Результат:** Маленькая модель достигает качества, близкого к большой, но работает в разы быстрее.

---

### 3.4 Domain Adaptation

**Проблема:** Обучили на одном домене (source), применяем к другому (target).

**Пример:** 
- Source: Фото продуктов в студии (яркие, четкие)
- Target: Фото продуктов от пользователей (темные, размытые)

**Метод 1: Adversarial Domain Adaptation**

```python
class DomainAdversarialNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()
        
        # Классификатор задачи
        self.task_classifier = nn.Linear(512, 10)
        
        # Классификатор домена (source=0, target=1)
        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        task_pred = self.task_classifier(features)
        domain_pred = self.domain_classifier(features)
        return task_pred, domain_pred

# Обучение
for inputs_source, labels_source, inputs_target in train_loader:
    # Source данные
    task_pred_source, domain_pred_source = model(inputs_source)
    task_loss = F.cross_entropy(task_pred_source, labels_source)
    domain_loss_source = F.cross_entropy(domain_pred_source, torch.zeros(...))
    
    # Target данные (нет labels)
    _, domain_pred_target = model(inputs_target)
    domain_loss_target = F.cross_entropy(domain_pred_target, torch.ones(...))
    
    # Хотим: хороший task classifier, но domain classifier не может различить домены
    loss = task_loss + domain_loss_source + domain_loss_target
```

---

## 📊 Checklist: Когда какой подход использовать?

```
┌─────────────────────────────────────────────────────────────┐
│ Мало данных (< 1K)         → Feature Extraction             │
│ Средне данных (1K-10K)     → Feature Extraction + Freeze early layers │
│ Много данных (> 10K)       → Full Fine-Tuning               │
│                                                               │
│ Задача похожа на ImageNet  → Higher learning rate           │
│ Задача отличается          → Lower learning rate, больше эпох │
│                                                               │
│ Ограничены ресурсы         → MobileNet, EfficientNet-B0     │
│ Нужна точность             → EfficientNet-B7, ResNet152     │
│ Быстрые эксперименты       → ResNet18                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Ключевые выводы

1. **Transfer Learning экономит время и ресурсы** — используйте его всегда, когда возможно.

2. **Feature Extraction vs Fine-Tuning:**
   - Мало данных → Feature Extraction
   - Много данных → Fine-Tuning

3. **Стратегия обучения:**
   - Сначала обучите только последний слой (5 эпох)
   - Затем разморозьте все и обучите с низким lr (5-10 эпох)

4. **Выбор модели:**
   - ResNet18 — для быстрых экспериментов
   - ResNet50 — золотая середина
   - EfficientNet — лучшее качество/скорость

5. **Нормализация критична** — используйте mean/std из ImageNet: `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`

6. **Аугментация данных** — обязательна при fine-tuning для предотвращения переобучения.

---

## 📚 Дополнительные материалы

- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [CS231n: Transfer Learning](http://cs231n.github.io/transfer-learning/)
- [Paper: How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
