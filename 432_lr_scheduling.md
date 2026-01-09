# Learning Rate Scheduling

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
import matplotlib.pyplot as plt
import numpy as np

# !pip install torch matplotlib
```

---

## 🟢 Базовый уровень: Зачем менять learning rate?

### 1.1 Проблема фиксированного LR

**Фиксированный learning rate** имеет недостатки:
- Слишком большой → модель не сходится
- Слишком маленький → медленное обучение
- **Решение:** Уменьшать LR по мере обучения

```python
def visualize_lr_problem():
    # Простая функция потерь
    def loss_fn(w):
        return (w - 3.5) ** 2
    
    # Оптимизация с разными LR
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (lr, title) in enumerate([(0.01, 'LR слишком мал'),
                                        (1.0, 'LR оптимален'),
                                        (2.5, 'LR слишком велик')]):
        w = 0.0
        history = [w]
        
        for _ in range(20):
            grad = 2 * (w - 3.5)
            w = w - lr * grad
            history.append(w)
        
        axes[idx].plot(history, marker='o')
        axes[idx].axhline(y=3.5, color='r', linestyle='--', label='Оптимум')
        axes[idx].set_title(title)
        axes[idx].set_xlabel('Шаг')
        axes[idx].set_ylabel('w')
        axes[idx].legend()
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.show()

visualize_lr_problem()
```

### 1.2 StepLR — простейший scheduler

**Идея:** Уменьшать LR каждые N эпох

```python
# Создаем модель и оптимизатор
model = nn.Sequential(
    nn.Linear(10, 50), nn.ReLU(),
    nn.Linear(50, 1)
)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# StepLR: уменьшаем LR в 10 раз каждые 10 эпох
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Цикл обучения
lrs = []
for epoch in range(50):
    # ... training code ...
    
    # Сохраняем текущий LR
    lrs.append(optimizer.param_groups[0]['lr'])
    
    # Обновляем LR
    scheduler.step()

# Визуализация
plt.plot(lrs)
plt.xlabel('Эпоха')
plt.ylabel('Learning Rate')
plt.title('StepLR: LR уменьшается каждые 10 эпох')
plt.grid(True)
plt.show()
```

### 1.3 ExponentialLR — плавное уменьшение

**Формула:** $LR_{epoch} = LR_{initial} \times \gamma^{epoch}$

```python
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = ExponentialLR(optimizer, gamma=0.95)

lrs = []
for epoch in range(50):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

plt.plot(lrs)
plt.xlabel('Эпоха')
plt.ylabel('Learning Rate')
plt.title('ExponentialLR: Плавное экспоненциальное уменьшение')
plt.grid(True)
plt.yscale('log')
plt.show()
```

---

## 🟡 Продвинутый уровень: Адаптивные schedulers

### 2.1 ReduceLROnPlateau — на основе метрики

**Идея:** Уменьшать LR, когда метрика перестает улучшаться

```python
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ReduceLROnPlateau следит за validation loss
scheduler = ReduceLROnPlateau(optimizer, 
                             mode='min',           # минимизируем loss
                             factor=0.5,           # уменьшаем в 2 раза
                             patience=3,           # ждем 3 эпохи
                             verbose=True)

# Пример обучения
from torchvision import datasets, transforms

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                  transform=transforms.ToTensor()),
    batch_size=64, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=1000
)

criterion = nn.CrossEntropyLoss()

lrs = []
val_losses = []

for epoch in range(20):
    # Training
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data.view(-1, 784))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data.view(-1, 784))
            val_loss += criterion(output, target).item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    lrs.append(optimizer.param_groups[0]['lr'])
    
    # Обновляем LR на основе val_loss
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch}: Val Loss={val_loss:.4f}, LR={lrs[-1]:.6f}")

# Визуализация
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.plot(val_losses)
ax1.set_xlabel('Эпоха')
ax1.set_ylabel('Validation Loss')
ax1.set_title('Validation Loss')
ax1.grid(True)

ax2.plot(lrs)
ax2.set_xlabel('Эпоха')
ax2.set_ylabel('Learning Rate')
ax2.set_title('Learning Rate (ReduceLROnPlateau)')
ax2.set_yscale('log')
ax2.grid(True)

plt.tight_layout()
plt.show()
```

### 2.2 CosineAnnealingLR — косинусное затухание

**Формула:** $LR_t = LR_{min} + \frac{1}{2}(LR_{max} - LR_{min})(1 + \cos(\frac{t\pi}{T}))$

```python
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.001)

lrs = []
for epoch in range(50):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

plt.plot(lrs)
plt.xlabel('Эпоха')
plt.ylabel('Learning Rate')
plt.title('CosineAnnealingLR: Плавное косинусное уменьшение')
plt.grid(True)
plt.show()
```

### 2.3 OneCycleLR — для супер-сходимости

**Идея:** Сначала увеличиваем LR, потом уменьшаем (1cycle policy)

```python
# OneCycleLR требует знать общее количество шагов
total_steps = 10 * len(train_loader)  # 10 эпох

optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = OneCycleLR(optimizer, 
                       max_lr=0.1,
                       total_steps=total_steps,
                       pct_start=0.3,  # 30% на разогрев
                       anneal_strategy='cos')

lrs = []
for epoch in range(10):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data.view(-1, 784))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()  # Вызываем каждый батч!

plt.plot(lrs)
plt.xlabel('Шаг обучения')
plt.ylabel('Learning Rate')
plt.title('OneCycleLR: Сначала растет, потом падает')
plt.grid(True)
plt.show()
```

---

## 🔴 Экспертный уровень: Продвинутые стратегии

### 3.1 Warmup + Cosine Decay

```python
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, 
                 min_lr=0, max_lr=0.001):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / \
                      (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                 (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# Визуализация
optimizer = optim.Adam(model.parameters())
scheduler = WarmupCosineScheduler(optimizer, 
                                 warmup_steps=1000,
                                 total_steps=10000,
                                 max_lr=0.001)

lrs = []
for step in range(10000):
    lr = scheduler.step()
    lrs.append(lr)

plt.plot(lrs)
plt.xlabel('Шаг')
plt.ylabel('Learning Rate')
plt.title('Warmup + Cosine Decay (популярно в Transformers)')
plt.grid(True)
plt.show()
```

### 3.2 Cyclic LR — циклическое изменение

```python
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = CyclicLR(optimizer, 
                    base_lr=0.001,
                    max_lr=0.1,
                    step_size_up=2000,
                    mode='triangular2')

lrs = []
for step in range(10000):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

plt.plot(lrs)
plt.xlabel('Шаг')
plt.ylabel('Learning Rate')
plt.title('CyclicLR: Циклические колебания LR')
plt.grid(True)
plt.show()
```

### 3.3 SequentialLR — комбинирование schedulers

```python
# Комбинирование: сначала Linear warmup, потом StepLR
optimizer = optim.SGD(model.parameters(), lr=0.1)

scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=5)
scheduler2 = StepLR(optimizer, step_size=10, gamma=0.5)

scheduler = SequentialLR(optimizer, 
                        schedulers=[scheduler1, scheduler2],
                        milestones=[5])

lrs = []
for epoch in range(50):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

plt.plot(lrs)
plt.xlabel('Эпоха')
plt.ylabel('Learning Rate')
plt.title('SequentialLR: Warmup → StepLR')
plt.grid(True)
plt.show()
```

### 3.4 Learning Rate Finder

```python
class LRFinder:
    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
    
    def find(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100):
        self.model.train()
        
        # Сохраняем начальное состояние
        model_state = self.model.state_dict()
        optim_state = self.optimizer.state_dict()
        
        # Экспоненциальный рост LR
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)
        lr = start_lr
        
        lrs = []
        losses = []
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        iterator = iter(train_loader)
        for iteration in range(num_iter):
            try:
                data, target = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                data, target = next(iterator)
            
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Сохраняем
            lrs.append(lr)
            losses.append(loss.item())
            
            # Увеличиваем LR
            lr *= lr_mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Останавливаемся если loss взрывается
            if loss.item() > losses[0] * 4:
                break
        
        # Восстанавливаем состояние
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optim_state)
        
        return lrs, losses

# Использование
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

finder = LRFinder(model, optimizer, criterion)
lrs, losses = finder.find(train_loader)

# Визуализация
plt.figure(figsize=(10, 5))
plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('LR Finder: Оптимальный LR там, где loss падает быстрее всего')
plt.grid(True)
plt.show()

# Оптимальный LR обычно в точке максимального наклона
```

---

## 💎 Заключение

**Сравнение schedulers:**

| Scheduler | Плюсы | Минусы | Когда использовать |
|-----------|-------|--------|-------------------|
| **StepLR** | Простой, предсказуемый | Резкие скачки LR | Простые задачи |
| **ExponentialLR** | Плавное уменьшение | Требует настройки gamma | Общее применение |
| **ReduceLROnPlateau** | Адаптивный к метрикам | Может застрять | Когда есть validation set |
| **CosineAnnealingLR** | Плавное, без параметров | Нужно знать T_max | Computer Vision |
| **OneCycleLR** | Быстрая сходимость | Сложная настройка | Когда важна скорость |
| **CyclicLR** | Помогает выйти из локальных минимумов | Нестабильная сходимость | Эксперименты |

**Рекомендации:**

1. **Начинающим:**
   - Используйте ReduceLROnPlateau
   - factor=0.5, patience=3-5
   - Просто и эффективно

2. **Computer Vision:**
   - CosineAnnealingLR или OneCycleLR
   - Часто используется в ResNet, EfficientNet

3. **NLP/Transformers:**
   - Warmup + Cosine Decay
   - Warmup критичен для стабильности

4. **Быстрое обучение:**
   - OneCycleLR (1cycle policy)
   - Может дать 2-3x ускорение

**Практические советы:**

```python
# Типичная конфигурация для разных задач

# 1. Общее применение (classification)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

# 2. Computer Vision (ResNet)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = CosineAnnealingLR(optimizer, T_max=200)

# 3. NLP (Transformer)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = WarmupCosineScheduler(optimizer, warmup_steps=4000, 
                                 total_steps=100000)

# 4. Быстрое обучение
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = OneCycleLR(optimizer, max_lr=0.1, 
                      total_steps=len(train_loader)*epochs)
```

**Лучшие практики:**
- ✅ Используйте LR Finder для начального LR
- ✅ Мониторьте и логируйте текущий LR
- ✅ Для Transformers обязателен warmup
- ✅ OneCycleLR вызываем каждый batch, остальные — каждую эпоху
- ✅ ReduceLROnPlateau следит за validation метрикой

**Частые ошибки:**
- ❌ Не вызывать scheduler.step()
- ❌ Вызывать scheduler.step() до optimizer.step()
- ❌ Использовать слишком агрессивное уменьшение LR
- ❌ Забыть про warmup в Transformers

> **"Learning rate scheduling — один из простейших способов значительно улучшить качество модели. Правильный scheduler может дать +2-5% accuracy."**

**Дальнейшее изучение:**
- [PyTorch LR Schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [Cyclical Learning Rates](https://arxiv.org/abs/1506.01186)
- [Super-Convergence](https://arxiv.org/abs/1708.07120)

---

## 📝 Задачи

**[Перейти к задачам по Learning Rate Scheduling →](432_lr_scheduling_tasks.md)**

Практические задания для закрепления материала:
- 🟢 Базовый уровень: StepLR, ExponentialLR, CosineAnnealing
- 🟡 Продвинутый уровень: ReduceLROnPlateau, OneCycleLR
- 🔴 Экспертный уровень: Warmup + Cosine Decay, LR Finder, сравнение всех методов
