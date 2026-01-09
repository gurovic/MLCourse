### **Задачи: Transfer Learning**

**Цель:** Научиться эффективно использовать предобученные модели, понять разницу между feature extraction и fine-tuning, применить transfer learning на практике.

---

## 🟢 Базовый уровень

### **Задача 1: Feature Extraction на CIFAR-10**

**Условие:** Используйте предобученный ResNet18 как feature extractor для классификации CIFAR-10.

**Требования:**
1. Загрузите предобученный ResNet18
2. Заморозьте все слои кроме последнего
3. Замените последний слой на классификатор для 10 классов
4. Обучите только последний слой (5 эпох)
5. Достигните accuracy > 80% на test set
6. Посчитайте сколько параметров обучается

**Ожидаемый результат:** Обучение займет ~5 минут, accuracy ~82-85%.

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# Подготовка данных
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                             download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                            download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

# TODO: Загрузите ResNet18 (pretrained=True)
# TODO: Заморозьте все параметры
# TODO: Замените последний слой
# TODO: Обучите модель
# TODO: Оцените на test set
```

**Вопросы для анализа:**
1. Почему нужна нормализация с mean=[0.485, 0.456, 0.406]?
2. Сколько параметров обучается? Сколько всего параметров в модели?
3. Почему ResNet18, обученная на ImageNet (1000 классов), хорошо работает на CIFAR-10 (10 классов)?

---

### **Задача 2: Сравнение Feature Extraction и Fine-Tuning**

**Условие:** Сравните два подхода на датасете с цветками (Flowers102 или свой).

**Требования:**
1. **Вариант A: Feature Extraction**
   - Заморозьте все слои кроме fc
   - Обучите 5 эпох, lr=0.001
   
2. **Вариант B: Full Fine-Tuning**
   - Размораживание всех слоев
   - Обучите 5 эпох, lr=0.0001
   
3. Сравните:
   - Accuracy на test set
   - Время обучения одной эпохи
   - Размер модели на диске
   
4. Постройте графики:
   - Train/Val accuracy по эпохам
   - Train/Val loss по эпохам

**Ожидаемый результат:** Fine-tuning даст выше accuracy, но будет медленнее обучаться.

```python
import time
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, epochs, lr):
    """Обучает модель и возвращает историю"""
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [], 'time': []}
    
    for epoch in range(epochs):
        # TODO: реализуйте обучение
        # TODO: считайте метрики
        pass
    
    return history

# TODO: реализуйте оба варианта
# TODO: постройте сравнительные графики
```

---

### **Задача 3: Использование разных предобученных моделей**

**Условие:** Сравните качество разных архитектур на вашем датасете.

**Требования:**
1. Используйте следующие предобученные модели:
   - ResNet18
   - ResNet50
   - EfficientNet-B0
   - MobileNetV2
   
2. Для каждой модели:
   - Feature extraction (5 эпох)
   - Измерьте accuracy, время inference (100 изображений)
   - Посчитайте количество параметров
   
3. Создайте сравнительную таблицу
4. Постройте bar plot для accuracy
5. Постройте scatter plot: параметры vs accuracy

**Ожидаемый результат:** EfficientNet-B0 покажет лучший баланс качества и размера.

```python
models_to_test = {
    'ResNet18': models.resnet18(pretrained=True),
    'ResNet50': models.resnet50(pretrained=True),
    'EfficientNet-B0': models.efficientnet_b0(pretrained=True),
    'MobileNetV2': models.mobilenet_v2(pretrained=True),
}

results = []

for name, model in models_to_test.items():
    # TODO: адаптируйте модель под вашу задачу
    # TODO: обучите
    # TODO: оцените
    # TODO: измерьте время inference
    pass

# TODO: визуализируйте результаты
```

---

## 🟡 Продвинутый уровень

### **Задача 4: Постепенное размораживание слоев**

**Условие:** Реализуйте стратегию постепенного размораживания слоев в ResNet50.

**Требования:**
1. **Этап 1 (2 эпохи):** Обучаем только fc
2. **Этап 2 (2 эпохи):** Размораживаем layer4, обучаем layer4 + fc
3. **Этап 3 (2 эпохи):** Размораживаем layer3, обучаем layer3 + layer4 + fc
4. **Этап 4 (2 эпохи):** Размораживаем layer2, обучаем layer2 + layer3 + layer4 + fc
5. **Этап 5 (2 эпохи):** Размораживаем все, обучаем полностью

Для каждого этапа:
- Используйте learning rate = 0.001 для новых размороженных слоев
- Снижайте lr для уже обученных слоев в 10 раз

**Ожидаемый результат:** Постепенное размораживание предотвращает "катастрофическое забывание" и дает лучший результат.

```python
def unfreeze_layers(model, layers_to_unfreeze):
    """Размораживает указанные слои"""
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_unfreeze):
            param.requires_grad = True

# TODO: реализуйте 5-этапное обучение
# TODO: логируйте accuracy после каждого этапа
# TODO: постройте график accuracy по этапам
```

---

### **Задача 5: Differential Learning Rates**

**Условие:** Реализуйте обучение с разными learning rates для разных слоев.

**Требования:**
1. Разделите ResNet50 на 5 групп:
   - layer1: lr = 1e-5
   - layer2: lr = 1e-4
   - layer3: lr = 1e-3
   - layer4: lr = 1e-2
   - fc: lr = 1e-2

2. Обучите модель 10 эпох
3. Сравните с baseline (uniform lr = 1e-3)
4. Постройте графики loss для каждого слоя отдельно (используя hooks)

**Ожидаемый результат:** Differential learning rates дают быстрее сходимость и выше accuracy.

```python
# TODO: создайте optimizer с разными lr для разных слоев
params_groups = [
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    # TODO: остальные группы
]

optimizer = torch.optim.Adam(params_groups)

# TODO: реализуйте обучение
# TODO: визуализируйте потери для каждого слоя
```

**Вопрос:** Почему для ранних слоев используется меньший learning rate?

---

### **Задача 6: Data Augmentation Impact**

**Условие:** Исследуйте влияние data augmentation на качество fine-tuning.

**Требования:**
1. Подготовьте 3 варианта трансформаций:
   
   **A. Минимальная:**
   ```python
   transforms.Resize(224)
   transforms.ToTensor()
   transforms.Normalize(...)
   ```
   
   **B. Средняя:**
   ```python
   transforms.RandomResizedCrop(224)
   transforms.RandomHorizontalFlip()
   transforms.ToTensor()
   transforms.Normalize(...)
   ```
   
   **C. Агрессивная:**
   ```python
   transforms.RandomResizedCrop(224)
   transforms.RandomHorizontalFlip()
   transforms.RandomRotation(30)
   transforms.ColorJitter(0.3, 0.3, 0.3)
   transforms.RandomGrayscale(p=0.1)
   transforms.ToTensor()
   transforms.Normalize(...)
   ```

2. Для каждого варианта:
   - Fine-tune ResNet18 на маленьком датасете (1000 изображений)
   - Обучите 20 эпох
   - Измерьте train/val accuracy
   
3. Постройте графики train/val accuracy для всех трех вариантов
4. Определите, какой вариант лучше борется с переобучением

**Ожидаемый результат:** Агрессивная аугментация предотвратит переобучение на маленьком датасете.

---

## 🔴 Экспертный уровень

### **Задача 7: Multi-Task Learning**

**Условие:** Обучите одну сеть одновременно на нескольких задачах.

**Требования:**
1. Используйте датасет CelebA (или похожий) с несколькими аннотациями
2. Создайте multi-task архитектуру:
   - Общий backbone (ResNet18)
   - Head 1: классификация пола (2 класса)
   - Head 2: классификация возраста (5 классов: 0-18, 19-25, 26-35, 36-50, 51+)
   - Head 3: классификация эмоций (7 классов)

3. Обучите модель с взвешенными потерями:
   ```python
   total_loss = w1 * loss_gender + w2 * loss_age + w3 * loss_emotion
   ```
   
4. Экспериментируйте с весами w1, w2, w3
5. Сравните с 3 отдельными моделями (по одной на задачу)

**Ожидаемый результат:** Multi-task модель будет компактнее и может показать лучшее качество за счет shared representations.

```python
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: реализуйте архитектуру
        pass
    
    def forward(self, x):
        # TODO: forward pass для всех задач
        pass

# TODO: реализуйте обучение
# TODO: сравните с single-task моделями
```

---

### **Задача 8: Knowledge Distillation**

**Условие:** Перенесите знания из большой модели (teacher) в маленькую (student).

**Требования:**
1. **Teacher:** ResNet152 (предобученная на ImageNet)
2. **Student:** ResNet18 (обучаем с нуля)
3. Реализуйте distillation loss:
   ```python
   loss = alpha * KL_div(student_logits/T, teacher_logits/T) + (1-alpha) * CE(student_logits, labels)
   ```
4. Подберите гиперпараметры T (temperature) и alpha
5. Сравните 3 варианта:
   - Student обучен с нуля (без teacher)
   - Student обучен через distillation
   - Student с transfer learning (pretrained на ImageNet)

**Ожидаемый результат:** Distillation даст quality между "с нуля" и "transfer learning".

```python
def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
    # TODO: реализуйте distillation loss
    pass

# TODO: обучите student с distillation
# TODO: сравните три варианта
```

**Вопросы для анализа:**
1. Что такое temperature T и как она влияет на обучение?
2. Почему student может достичь качества лучше, чем при обучении с нуля?
3. В каких случаях distillation эффективнее transfer learning?

---

### **Задача 9: Domain Adaptation**

**Условие:** Адаптируйте модель, обученную на одном домене, к другому домену.

**Требования:**
1. **Source domain:** MNIST (черно-белые цифры, четкие)
2. **Target domain:** SVHN (цветные номера домов, зашумленные)
3. Реализуйте 3 подхода:
   
   **A. Naive transfer:** Обучаем на MNIST, тестируем на SVHN
   
   **B. Fine-tuning:** Обучаем на MNIST, дообучаем на SVHN
   
   **C. Domain adversarial:** Обучаем feature extractor, который не может различить домены

4. Для варианта C реализуйте adversarial training:
   - Feature extractor F
   - Task classifier C (предсказывает цифру)
   - Domain classifier D (предсказывает домен)
   - Обучаем F, чтобы обмануть D (gradient reversal)

5. Сравните accuracy всех трех подходов на SVHN

**Ожидаемый результат:** Domain adversarial подход покажет лучшее качество на target domain.

```python
class GradientReversalLayer(torch.autograd.Function):
    """Инвертирует градиент при backprop"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

# TODO: реализуйте domain adversarial network
# TODO: обучите с gradient reversal
# TODO: сравните с naive transfer и fine-tuning
```

---

### **Задача 10: Визуализация Transfer Learning**

**Условие:** Визуализируйте, как меняются признаки в процессе fine-tuning.

**Требования:**
1. Возьмите ResNet18, предобученную на ImageNet
2. Извлеките признаки (output слоя avgpool) для 1000 изображений CIFAR-10
3. Используйте t-SNE для визуализации признаков в 2D
4. Постройте 4 графика:
   - До fine-tuning (pretrained ImageNet weights)
   - После 1 эпохи fine-tuning на CIFAR-10
   - После 5 эпох fine-tuning
   - После 20 эпох fine-tuning
5. Раскрасьте точки по классам CIFAR-10

**Ожидаемый результат:** После fine-tuning классы CIFAR-10 станут более разделимыми в пространстве признаков.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def extract_features(model, dataloader):
    """Извлекает признаки из слоя avgpool"""
    # TODO: реализуйте extraction
    pass

def visualize_tsne(features, labels, title):
    """Визуализирует признаки через t-SNE"""
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    for i in range(10):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   label=f'Class {i}', alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.show()

# TODO: визуализируйте признаки до и после fine-tuning
```

**Вопросы для анализа:**
1. Как изменяется structure пространства признаков после fine-tuning?
2. Какие классы становятся более separable?
3. Есть ли классы, которые остаются смешанными даже после fine-tuning?

---

## 📝 Дополнительные вопросы для размышления

1. **Почему transfer learning работает?**
   - Что такое иерархическое представление признаков?
   - Почему нижние слои универсальны, а верхние специфичны?

2. **Когда transfer learning НЕ работает?**
   - Приведите примеры доменов, где ImageNet pretrained модели бесполезны
   - Что делать в таких случаях?

3. **Какой learning rate выбрать?**
   - Почему для fine-tuning нужен меньший lr, чем для training с нуля?
   - Как выбрать оптимальный lr?

4. **Сколько данных нужно для fine-tuning?**
   - Минимальное количество для feature extraction?
   - Минимальное количество для full fine-tuning?

5. **Что лучше: одна большая модель или ансамбль маленьких?**
   - С точки зрения качества?
   - С точки зрения скорости inference?
   - С точки зрения памяти?

---

## 🎯 Критерии успешного выполнения

- ✅ Вы понимаете разницу между feature extraction и fine-tuning
- ✅ Вы умеете правильно замораживать/размораживать слои
- ✅ Вы знаете, как выбрать предобученную модель под задачу
- ✅ Вы умеете настраивать learning rate для разных слоев
- ✅ Вы понимаете важность data augmentation при fine-tuning
- ✅ Вы можете визуализировать, что изменилось в модели после fine-tuning
- ✅ Вы знаете продвинутые техники: distillation, domain adaptation, multi-task learning

---

## 📚 Полезные ресурсы

- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [CS231n: Transfer Learning](http://cs231n.github.io/transfer-learning/)
- [Paper: How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)
- [timm library](https://github.com/rwightman/pytorch-image-models) — огромная коллекция pretrained моделей
- [Paper: Knowledge Distillation](https://arxiv.org/abs/1503.02531)
- [Paper: Domain-Adversarial Training](https://arxiv.org/abs/1505.07818)
