### **Задачи: Learning Rate Scheduling**

**Цель:** Научиться использовать различные стратегии изменения learning rate для ускорения обучения и улучшения качества модели.

---

## 🟢 Базовый уровень

### **Задача 1: Демонстрация проблемы фиксированного LR**

**Условие:** Покажите, почему фиксированный learning rate неоптимален.

**Требования:**
1. Обучите простую MLP на MNIST с тремя фиксированными LR:
   - lr = 0.001 (слишком мал)
   - lr = 0.01 (оптимален)
   - lr = 0.5 (слишком велик)
2. Постройте графики train loss по эпохам для каждого LR
3. Измерьте:
   - Скорость сходимости (epochs to 95% accuracy)
   - Стабильность обучения
   - Финальную accuracy

**Ожидаемый результат:**
- Маленький LR: медленная сходимость
- Большой LR: нестабильное обучение или расходимость
- Оптимальный: быстрая сходимость

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

learning_rates = [0.001, 0.01, 0.5]

for lr in learning_rates:
    model = SimpleMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # TODO: обучите и постройте графики
```

**Вопрос:** Можно ли комбинировать преимущества большого и малого LR?

---

### **Задача 2: StepLR — ступенчатое уменьшение**

**Условие:** Используйте StepLR для постепенного уменьшения learning rate.

**Требования:**
1. Обучите модель на MNIST с StepLR
2. Параметры: `step_size=5, gamma=0.5` (уменьшаем вдвое каждые 5 эпох)
3. Логируйте текущий LR на каждой эпохе
4. Постройте два графика:
   - Learning rate vs эпоха
   - Loss vs эпоха
5. Сравните с фиксированным LR

**Ожидаемый результат:** StepLR достигает лучшей финальной accuracy.

```python
from torch.optim.lr_scheduler import StepLR

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

lrs = []
losses = []

for epoch in range(20):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    losses.append(train_loss)
    lrs.append(optimizer.param_groups[0]['lr'])
    
    scheduler.step()  # Обновляем LR после эпохи
```

---

### **Задача 3: ExponentialLR — экспоненциальное затухание**

**Условие:** Сравните StepLR и ExponentialLR.

**Требования:**
1. Обучите две модели:
   - StepLR: `step_size=10, gamma=0.1`
   - ExponentialLR: `gamma=0.95`
2. Обучайте 50 эпох
3. Визуализируйте LR schedules на одном графике
4. Сравните:
   - Плавность изменения LR
   - Финальную accuracy
   - Стабильность на поздних эпохах

**Вопрос:** В каких случаях предпочесть ExponentialLR?

---

## 🟡 Продвинутый уровень

### **Задача 4: ReduceLROnPlateau — адаптивное уменьшение**

**Условие:** Реализуйте автоматическое уменьшение LR при застое метрики.

**Требования:**
1. Используйте ReduceLROnPlateau:
   ```python
   scheduler = ReduceLROnPlateau(optimizer, 
                                mode='min',
                                factor=0.5,
                                patience=3,
                                verbose=True)
   ```
2. Обучите на MNIST, передавая validation loss в scheduler
3. Логируйте:
   - Validation loss
   - Текущий LR
   - Эпохи, когда LR изменялся
4. Визуализируйте оба графика вместе

**Ожидаемый результат:** LR уменьшается, когда val loss перестает улучшаться.

```python
for epoch in range(50):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)
    
    # Передаем метрику в scheduler
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch}: Val Loss={val_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
```

---

### **Задача 5: CosineAnnealingLR — косинусное затухание**

**Условие:** Примените CosineAnnealingLR для плавного уменьшения LR.

**Требования:**
1. Используйте CosineAnnealingLR с `T_max=50, eta_min=0.0001`
2. Обучите модель на CIFAR-10
3. Сравните с ReduceLROnPlateau:
   - Предсказуемость изменения LR
   - Финальное качество
   - Необходимость настройки гиперпараметров
4. Постройте график LR schedule (должна быть плавная косинусоида)

**Практическое применение:** CosineAnnealing популярен в Computer Vision (ResNet, EfficientNet).

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)

# LR будет плавно уменьшаться от lr_max до eta_min
```

---

### **Задача 6: OneCycleLR — суперсходимость**

**Условие:** Используйте 1cycle policy для быстрого обучения.

**Требования:**
1. Реализуйте OneCycleLR:
   ```python
   total_steps = len(train_loader) * epochs
   scheduler = OneCycleLR(optimizer,
                         max_lr=0.1,
                         total_steps=total_steps,
                         pct_start=0.3,
                         anneal_strategy='cos')
   ```
2. **Важно:** Вызывайте `scheduler.step()` после каждого батча!
3. Сравните с обычным обучением:
   - Скорость достижения 90% accuracy
   - Финальное качество
   - Стабильность обучения
4. Визуализируйте LR на каждом шаге (должен сначала расти, потом падать)

**Ожидаемый результат:** OneCycleLR может дать 2-3x ускорение.

```python
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        scheduler.step()  # После каждого батча!
        
        lrs.append(optimizer.param_groups[0]['lr'])
```

---

## 🔴 Экспертный уровень

### **Задача 7: Warmup + Cosine Decay (Transformer style)**

**Условие:** Реализуйте комбинированный scheduler, используемый в Transformers.

**Требования:**
1. Создайте кастомный scheduler:
   ```python
   class WarmupCosineScheduler:
       def __init__(self, optimizer, warmup_steps, total_steps, 
                    min_lr=0, max_lr=0.001):
           # TODO: реализуйте
       
       def step(self):
           # Linear warmup, затем cosine decay
           pass
   ```
2. Параметры: warmup_steps=1000, total_steps=10000
3. Обучите модель на сложной задаче
4. Визуализируйте LR schedule (должна быть линейный рост, потом косинус)
5. Сравните с обычным Cosine Annealing

**Вопрос:** Зачем нужен warmup? (Подумайте о состоянии модели в начале обучения)

<details>
<summary>Подсказка (нажмите, чтобы раскрыть)</summary>
Warmup стабилизирует обучение в начале, когда веса случайные. С высоким LR модель может сразу уйти в плохую область пространства параметров.
</details>

---

### **Задача 8: Learning Rate Finder**

**Условие:** Реализуйте LR Finder для автоматического подбора оптимального learning rate.

**Требования:**
1. Реализуйте алгоритм:
   - Экспоненциально увеличивайте LR от 1e-7 до 10
   - На каждом шаге записывайте loss
   - Остановитесь, если loss взрывается
2. Постройте график: loss vs learning rate (log scale)
3. Найдите оптимальный LR (там, где loss падает быстрее всего)
4. Обучите модель с найденным LR и проверьте результат

```python
class LRFinder:
    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
    
    def find(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100):
        """Ищет оптимальный learning rate"""
        # TODO: реализуйте
        lrs = []
        losses = []
        
        # Сохраняем начальное состояние
        model_state = self.model.state_dict()
        optim_state = self.optimizer.state_dict()
        
        # TODO: экспоненциально увеличиваем LR и записываем loss
        
        # Восстанавливаем состояние
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optim_state)
        
        return lrs, losses

# Использование
finder = LRFinder(model, optimizer, criterion)
lrs, losses = finder.find(train_loader)

plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('LR Finder')
plt.show()
```

---

### **Задача 9: CyclicLR — циклическое изменение**

**Условие:** Исследуйте, помогает ли циклическое изменение LR выходить из локальных минимумов.

**Требования:**
1. Используйте CyclicLR с разными режимами:
   - 'triangular' (линейный)
   - 'triangular2' (с уменьшающейся амплитудой)
   - 'exp_range' (экспоненциальный)
2. Параметры: `base_lr=0.001, max_lr=0.1, step_size_up=2000`
3. Обучите модель на задаче с множеством локальных минимумов
4. Сравните с обычным обучением
5. Визуализируйте циклы LR

**Ожидаемый результат:** Циклы помогают исследовать пространство параметров.

```python
from torch.optim.lr_scheduler import CyclicLR

scheduler = CyclicLR(optimizer,
                    base_lr=0.001,
                    max_lr=0.1,
                    step_size_up=2000,
                    mode='triangular2')
```

---

### **Задача 10: SequentialLR — комбинирование schedulers**

**Условие:** Создайте сложный schedule, комбинируя несколько schedulers.

**Требования:**
1. Создайте pipeline:
   - Эпохи 0-5: LinearLR (warmup от 0.1*lr до lr)
   - Эпохи 6-20: StepLR (уменьшение каждые 5 эпох)
   - Эпохи 21+: ExponentialLR (плавное затухание)
2. Используйте SequentialLR:
   ```python
   scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=5)
   scheduler2 = StepLR(optimizer, step_size=5, gamma=0.5)
   scheduler3 = ExponentialLR(optimizer, gamma=0.95)
   
   scheduler = SequentialLR(optimizer,
                           schedulers=[scheduler1, scheduler2, scheduler3],
                           milestones=[5, 20])
   ```
3. Обучите модель 50 эпох
4. Визуализируйте весь LR schedule
5. Сравните с простым CosineAnnealing

---

### **Задача 11: Сравнение всех schedulers**

**Условие:** Проведите комплексное сравнение всех schedulers на одной задаче.

**Требования:**
1. Обучите модель с каждым scheduler:
   - StepLR
   - ExponentialLR
   - ReduceLROnPlateau
   - CosineAnnealingLR
   - OneCycleLR
   - CyclicLR
2. Для каждого:
   - Подберите оптимальные гиперпараметры
   - Запустите 5 раз с разными seeds
   - Усредните результаты
3. Создайте сравнительную таблицу:
   - Финальная accuracy (mean ± std)
   - Epochs to 95% accuracy
   - Сложность настройки (субъективно)
4. Визуализируйте все LR schedules на одном графике

**Датасет:** CIFAR-10

**Ожидаемый результат:** 
- OneCycleLR: самый быстрый
- ReduceLROnPlateau: самый адаптивный
- CosineAnnealing: хороший баланс

---

## 💎 Заключение

### **Рекомендации по выбору scheduler:**

| Ситуация | Рекомендуемый Scheduler | Параметры |
|----------|------------------------|-----------|
| **Начинающим / прототипирование** | ReduceLROnPlateau | factor=0.5, patience=5 |
| **Computer Vision (ResNet, CNN)** | CosineAnnealingLR | T_max=epochs |
| **NLP / Transformers** | Warmup + Cosine Decay | warmup=10% от total_steps |
| **Быстрое обучение** | OneCycleLR | max_lr из LR Finder |
| **Простые задачи** | StepLR | step_size=epochs//3 |
| **Нестандартные задачи** | ReduceLROnPlateau | следит за custom метрикой |

### **Чек-лист использования schedulers:**

✅ **Обязательно:**
- [ ] Используйте какой-нибудь scheduler (лучше, чем константа)
- [ ] Логируйте текущий LR на каждой эпохе
- [ ] Визуализируйте LR schedule перед обучением
- [ ] Для OneCycleLR вызывайте .step() после каждого батча
- [ ] Для остальных schedulers — после каждой эпохи

✅ **Рекомендуется:**
- [ ] Используйте LR Finder для начального LR
- [ ] Добавьте warmup для Transformers и больших моделей
- [ ] Сохраняйте LR в логи (TensorBoard, W&B)
- [ ] Экспериментируйте с гиперпараметрами scheduler

✅ **Продвинутое:**
- [ ] Комбинируйте schedulers (SequentialLR)
- [ ] Используйте разные LR для разных частей модели
- [ ] Следите за градиентами вместе с LR
- [ ] Адаптируйте scheduler под специфику задачи

### **Типичные ошибки:**

❌ **Не делайте так:**
- Забыть вызвать `scheduler.step()`
- Вызывать `scheduler.step()` до `optimizer.step()`
- Использовать слишком агрессивное уменьшение (gamma=0.1)
- OneCycleLR с .step() после эпохи (должен быть после батча!)
- Не логировать текущий LR

✅ **Делайте так:**
- Всегда используйте scheduler
- Визуализируйте LR schedule
- Мониторьте loss вместе с LR
- Экспериментируйте с параметрами
- Используйте early stopping вместе со scheduler

### **Практический пример:**

```python
# Универсальная конфигурация для classification
model = YourModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Вариант 1: Простой и эффективный
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Вариант 2: Для быстрого обучения
scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=len(train_loader)*epochs)

# Вариант 3: Для Transformers
def get_warmup_cosine_lr(step, warmup_steps, total_steps, lr_max):
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * lr_max * (1 + np.cos(np.pi * progress))

# В training loop
for step, (data, target) in enumerate(train_loader):
    # Вариант 3: manual LR schedule
    lr = get_warmup_cosine_lr(step, warmup_steps=1000, 
                              total_steps=10000, lr_max=0.001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

### **Дополнительные ресурсы:**

1. **Статьи:**
   - [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
   - [Super-Convergence: Very Fast Training](https://arxiv.org/abs/1708.07120)
   - [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)

2. **Инструменты:**
   - `torch.optim.lr_scheduler` — все schedulers
   - TensorBoard для визуализации LR
   - Weights & Biases для tracking LR experiments

3. **Практика:**
   - Всегда начинайте с LR Finder
   - Экспериментируйте на маленьком датасете
   - Мониторьте не только loss, но и LR

> **"Правильный learning rate schedule может дать +5-10% accuracy и 2-3x ускорение обучения. Это one of the easiest wins в deep learning!"**
