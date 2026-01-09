# Meta-Learning и Few-Shot Learning

## 🟢 Основы

### Проблема Few-Shot Learning

**Задача**: обучить модель распознавать новые классы по нескольким примерам (1-5 примеров на класс)

**Мотивация**:
- Сбор данных дорог и требует времени
- Люди учатся по нескольким примерам
- Адаптация к новым задачам без переобучения с нуля

**Терминология**:
- **N-way K-shot**: N классов, K примеров на класс
- **Support set**: примеры для обучения
- **Query set**: примеры для тестирования
- **Episode/Task**: одна итерация обучения с support + query

### Transfer Learning vs Meta-Learning

```python
# Transfer Learning: обучаем на большом датасете, fine-tune на новой задаче
# Требует много данных для fine-tuning

# Meta-Learning: "учимся учиться" - оптимизируем способность быстро адаптироваться
# Работает с малым количеством данных
```

**Подходы к Meta-Learning**:
1. **Metric-based**: обучаем хорошее embedding пространство (Siamese, Prototypical Networks)
2. **Model-based**: модель с внешней памятью (Memory-Augmented NN)
3. **Optimization-based**: обучаем хорошую инициализацию (MAML)

### Siamese Networks

Обучаем сеть определять, принадлежат ли два изображения одному классу

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 4),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.Sigmoid()
        )
        
    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # label=1 для одного класса, label=0 для разных
        loss = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss

# Пример использования
model = SiameseNetwork()
criterion = ContrastiveLoss(margin=2.0)

# Два изображения
x1 = torch.randn(16, 1, 105, 105)
x2 = torch.randn(16, 1, 105, 105)
labels = torch.randint(0, 2, (16,)).float()  # 1 - same, 0 - different

out1, out2 = model(x1, x2)
loss = criterion(out1, out2, labels)
print(f"Contrastive loss: {loss.item():.4f}")
```

## 🟡 Prototypical Networks

Обучаем embedding, где классы формируют кластеры. Классифицируем по расстоянию до прототипа класса.

```python
class PrototypicalNetwork(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            self._conv_block(input_channels, 64),
            self._conv_block(64, 64),
            self._conv_block(64, 64),
            self._conv_block(64, 64)
        )
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

def compute_prototypes(embeddings, labels, n_way):
    """Вычисляем прототип (центроид) для каждого класса"""
    prototypes = []
    for c in range(n_way):
        class_embeddings = embeddings[labels == c]
        prototype = class_embeddings.mean(dim=0)
        prototypes.append(prototype)
    return torch.stack(prototypes)

def prototypical_loss(query_embeddings, query_labels, prototypes):
    """Классифицируем по расстоянию до ближайшего прототипа"""
    # Евклидово расстояние до всех прототипов
    distances = torch.cdist(query_embeddings, prototypes)
    
    # Преобразуем в log probabilities
    log_p_y = F.log_softmax(-distances, dim=1)
    
    # Cross-entropy loss
    loss = F.nll_loss(log_p_y, query_labels)
    return loss

# Пример 5-way 5-shot задачи
model = PrototypicalNetwork()
n_way, n_support, n_query = 5, 5, 15

# Support set
support_x = torch.randn(n_way * n_support, 1, 28, 28)
support_y = torch.arange(n_way).repeat_interleave(n_support)

# Query set
query_x = torch.randn(n_way * n_query, 1, 28, 28)
query_y = torch.arange(n_way).repeat_interleave(n_query)

# Forward pass
support_embeddings = model(support_x)
query_embeddings = model(query_x)

# Вычисляем прототипы и loss
prototypes = compute_prototypes(support_embeddings, support_y, n_way)
loss = prototypical_loss(query_embeddings, query_y, prototypes)

print(f"Prototypical loss: {loss.item():.4f}")
```

### Matching Networks

Используют attention mechanism для weighted k-NN в embedding пространстве

```python
class AttentionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        
    def forward(self, x):
        output, (h, c) = self.lstm(x)
        return output

class MatchingNetwork(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=64):
        super().__init__()
        # Encoder для изображений
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    def forward(self, support_x, support_y, query_x, n_way, n_shot):
        # Encode support и query
        support_features = self.encoder(support_x).view(support_x.size(0), -1)
        query_features = self.encoder(query_x).view(query_x.size(0), -1)
        
        # Attention: насколько query похож на каждый support пример
        attention = F.softmax(
            torch.matmul(query_features, support_features.t()) / (query_features.size(1) ** 0.5),
            dim=1
        )
        
        # Weighted voting
        support_labels_one_hot = F.one_hot(support_y, n_way).float()
        predictions = torch.matmul(attention, support_labels_one_hot)
        
        return predictions

# Пример
matching_net = MatchingNetwork()
predictions = matching_net(support_x, support_y, query_x, n_way, n_support)
print(f"Predictions shape: {predictions.shape}")  # (n_query, n_way)
```

## 🔴 Model-Agnostic Meta-Learning (MAML)

**Идея**: найти такую инициализацию параметров, от которой можно быстро адаптироваться к новой задаче за несколько шагов градиентного спуска

```python
import torch.optim as optim

class MAML:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=outer_lr)
        
    def inner_loop(self, support_x, support_y):
        """Адаптируемся к задаче на support set"""
        # Копируем параметры
        adapted_params = [p.clone() for p in self.model.parameters()]
        
        for step in range(self.inner_steps):
            # Forward pass с текущими параметрами
            predictions = self.forward_with_params(support_x, adapted_params)
            loss = F.cross_entropy(predictions, support_y)
            
            # Вычисляем градиенты
            grads = torch.autograd.grad(loss, adapted_params, create_graph=True)
            
            # Обновляем параметры
            adapted_params = [p - self.inner_lr * g for p, g in zip(adapted_params, grads)]
        
        return adapted_params
    
    def forward_with_params(self, x, params):
        """Forward pass с заданными параметрами"""
        # Простая реализация для демонстрации
        # В реальности нужно правильно применять params к слоям
        return self.model(x)
    
    def meta_train_step(self, tasks):
        """Один шаг мета-обучения на батче задач"""
        meta_loss = 0
        
        for task in tasks:
            support_x, support_y, query_x, query_y = task
            
            # Inner loop: адаптация к задаче
            adapted_params = self.inner_loop(support_x, support_y)
            
            # Оцениваем на query set с адаптированными параметрами
            predictions = self.forward_with_params(query_x, adapted_params)
            loss = F.cross_entropy(predictions, query_y)
            meta_loss += loss
        
        # Outer loop: обновляем мета-параметры
        meta_loss = meta_loss / len(tasks)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()

# Пример использования
base_model = PrototypicalNetwork()
maml = MAML(base_model, inner_lr=0.01, outer_lr=0.001, inner_steps=5)

# Генерируем батч задач (meta-batch)
tasks = []
for _ in range(4):  # 4 задачи в мета-батче
    support_x = torch.randn(n_way * n_support, 1, 28, 28)
    support_y = torch.arange(n_way).repeat_interleave(n_support)
    query_x = torch.randn(n_way * n_query, 1, 28, 28)
    query_y = torch.arange(n_way).repeat_interleave(n_query)
    tasks.append((support_x, support_y, query_x, query_y))

meta_loss = maml.meta_train_step(tasks)
print(f"Meta-training loss: {meta_loss:.4f}")
```

### Reptile - упрощенный MAML

```python
class Reptile:
    """Более простая альтернатива MAML без second-order gradients"""
    def __init__(self, model, inner_lr=0.01, outer_lr=0.1, inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        
    def train_on_task(self, support_x, support_y):
        """Обучаемся на одной задаче"""
        # Сохраняем начальные параметры
        initial_params = [p.clone().detach() for p in self.model.parameters()]
        
        # Обычное обучение на задаче
        optimizer = optim.SGD(self.model.parameters(), lr=self.inner_lr)
        
        for step in range(self.inner_steps):
            predictions = self.model(support_x)
            loss = F.cross_entropy(predictions, support_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Возвращаем финальные параметры
        return [p.clone().detach() for p in self.model.parameters()]
    
    def meta_step(self, task):
        """Один мета-шаг"""
        support_x, support_y, _, _ = task
        
        # Сохраняем начальные параметры
        initial_params = [p.clone().detach() for p in self.model.parameters()]
        
        # Обучаемся на задаче
        task_params = self.train_on_task(support_x, support_y)
        
        # Двигаем мета-параметры в направлении task_params
        with torch.no_grad():
            for p, p_init, p_task in zip(self.model.parameters(), initial_params, task_params):
                p.data = p_init + self.outer_lr * (p_task - p_init)

# Использование
reptile = Reptile(PrototypicalNetwork(), inner_lr=0.01, outer_lr=0.1, inner_steps=10)
for task in tasks:
    reptile.meta_step(task)
```

## Применения

### Drug Discovery
- Few-shot learning для предсказания свойств новых молекул
- Мало данных для редких заболеваний

### Robotics
- Быстрая адаптация к новым задачам и окружениям
- Transfer across different robots

### Computer Vision
- Few-shot object detection
- Fine-grained classification (породы собак, виды птиц)

### NLP
- Few-shot text classification
- Rapid adaptation to new languages

## Датасеты

- **Omniglot**: 1623 класса рукописных символов, 20 примеров на класс
- **miniImageNet**: 100 классов из ImageNet, 600 примеров на класс
- **tieredImageNet**: более сложная версия miniImageNet
- **Meta-Dataset**: мульти-доменный датасет для мета-обучения
- **FGVC Aircraft**: 100 типов самолетов

## Литература

- **Matching Networks for One Shot Learning** (Vinyals et al., 2016)
- **Prototypical Networks for Few-shot Learning** (Snell et al., 2017)
- **Model-Agnostic Meta-Learning (MAML)** (Finn et al., 2017)
- **On First-Order Meta-Learning Algorithms** (Reptile, Nichol et al., 2018)
- **Meta-Learning: A Survey** (Hospedales et al., 2020)
