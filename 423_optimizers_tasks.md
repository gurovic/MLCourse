### **Задачи: Оптимизаторы в глубоком обучении**

**Цель:** Понять работу различных оптимизаторов, научиться выбирать подходящий для конкретной задачи и настраивать гиперпараметры.

---

## 🟢 Базовый уровень

### **Задача 1: Сравнение базовых оптимизаторов на MNIST**

**Условие:** Обучите одну и ту же сеть с разными оптимизаторами.

**Требования:**
1. Используйте простую MLP: 784 → 256 → 128 → 10
2. Обучите с оптимизаторами: SGD, SGD+Momentum, Adam
3. Используйте фиксированный learning rate = 0.01 для всех
4. Постройте на одном графике:
   - Train loss по эпохам
   - Test accuracy по эпохам
5. Обучайте 10 эпох

**Ожидаемый результат:** Adam сходится быстрее всех, SGD+Momentum лучше vanilla SGD

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Шаблон
optimizers = {
    'SGD': torch.optim.SGD(model.parameters(), lr=0.01),
    'SGD+Momentum': torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'Adam': torch.optim.Adam(model.parameters(), lr=0.01)
}

# TODO: обучите модели и постройте графики
```

**Вопросы:**
- Почему Adam сходится быстрее?
- Какой оптимизатор достигает лучшей финальной accuracy?

---

### **Задача 2: Подбор learning rate**

**Условие:** Найдите оптимальный learning rate для каждого оптимизатора.

**Требования:**
1. Для SGD попробуйте lr ∈ {0.001, 0.01, 0.1, 1.0}
2. Для Adam попробуйте lr ∈ {0.0001, 0.001, 0.01, 0.1}
3. Обучите каждую комбинацию 5 эпох
4. Постройте heatmap: оптимизатор vs learning rate vs финальная accuracy
5. Определите лучшие комбинации

**Ожидаемый результат:**
- SGD работает хорошо с lr ~ 0.1
- Adam работает хорошо с lr ~ 0.001

```python
import seaborn as sns
import matplotlib.pyplot as plt

results = {}  # {(optimizer, lr): accuracy}

# TODO: заполните results и постройте heatmap
```

---

### **Задача 3: Weight decay (L2 регуляризация)**

**Условие:** Изучите влияние weight decay на обучение.

**Требования:**
1. Используйте Adam с lr=0.001
2. Варьируйте weight_decay ∈ {0, 1e-5, 1e-4, 1e-3, 1e-2}
3. Обучите модель на маленьком датасете (5000 примеров MNIST)
4. Постройте графики train/val accuracy
5. Найдите оптимальное значение weight_decay

**Вопросы:**
- Как weight decay влияет на переобучение?
- Почему слишком большой weight_decay вреден?

---

## 🟡 Продвинутый уровень

### **Задача 4: Momentum и Nesterov acceleration**

**Условие:** Сравните vanilla momentum и Nesterov momentum.

**Требования:**
1. Обучите модель с SGD + momentum (β=0.9)
2. Обучите с SGD + Nesterov momentum (β=0.9)
3. Обучите с vanilla SGD для сравнения
4. Используйте lr=0.1 для всех
5. Визуализируйте:
   - Траекторию loss в пространстве первых двух весов (PCA)
   - Скорость сходимости
6. Покажите "предвидение" Nesterov momentum

**Ожидаемый результат:** Nesterov converges faster and more smoothly

```python
optimizer_vanilla = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer_momentum = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer_nesterov = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
```

---

### **Задача 5: Adaptive learning rates: RMSprop vs Adam**

**Условие:** Глубоко разберитесь в разнице между RMSprop и Adam.

**Требования:**
1. Реализуйте RMSprop вручную (без torch.optim)
2. Реализуйте Adam вручную
3. Обучите модель обоими оптимизаторами
4. Во время обучения логируйте:
   - Среднее значение moving average of squared gradients
   - Effective learning rate для каждого параметра
5. Визуализируйте эти метрики для первых 2-3 слоев

**Вопросы:**
- В чем ключевое отличие Adam от RMSprop?
- Почему Adam часто предпочитают?

```python
class ManualAdam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]  # first moment
        self.v = [torch.zeros_like(p) for p in self.params]  # second moment
        self.t = 0
    
    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            # TODO: реализуйте Adam update rule
            pass
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
```

---

### **Задача 6: AdamW vs Adam (decoupled weight decay)**

**Условие:** Исследуйте разницу между Adam и AdamW.

**Требования:**
1. Обучите модель с Adam + weight_decay
2. Обучите с AdamW + weight_decay (используйте те же параметры)
3. Используйте weight_decay = 0.01
4. Сравните:
   - Финальную accuracy
   - Норму весов ||W|| во время обучения
   - Величину регуляризации
5. Объясните, почему AdamW лучше

**Ожидаемый результат:** AdamW дает лучшую генерализацию

```python
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
optimizer_adamw = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

---

## 🔴 Экспертный уровень

### **Задача 7: Gradient clipping для стабилизации обучения**

**Условие:** Продемонстрируйте важность gradient clipping в нестабильных задачах.

**Требования:**
1. Создайте "сложную" оптимизационную задачу (глубокая RNN или high learning rate)
2. Обучите БЕЗ gradient clipping
3. Обучите С gradient clipping (max_norm=1.0)
4. Логируйте:
   - Норму градиентов на каждом шаге
   - Loss (включая NaN случаи)
5. Визуализируйте, как clipping предотвращает взрывающиеся градиенты

```python
# Вариант 1: по значению
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

# Вариант 2: по норме (лучше)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Вопросы:**
- В чем разница между clip_grad_value и clip_grad_norm?
- Когда каждый метод предпочтительнее?

---

### **Задача 8: Learning rate warmup**

**Условие:** Реализуйте и изучите эффект от learning rate warmup.

**Требования:**
1. Реализуйте linear warmup:
   ```
   lr_t = lr_base * min(1, t / warmup_steps)
   ```
2. Обучите модель БЕЗ warmup (constant lr)
3. Обучите С warmup (1000 шагов)
4. Используйте высокий базовый lr (например, 0.1 для Adam)
5. Сравните:
   - Стабильность в начале обучения
   - Скорость сходимости
   - Финальное качество

**Ожидаемый результат:** Warmup стабилизирует начало обучения с высоким lr

```python
def get_lr_with_warmup(step, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    return base_lr

# В training loop:
for step, (x, y) in enumerate(dataloader):
    lr = get_lr_with_warmup(step, warmup_steps=1000, base_lr=0.1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

---

### **Задача 9: Сравнение на разных датасетах**

**Условие:** Протестируйте оптимизаторы на различных типах данных.

**Датасеты:**
1. Computer Vision: CIFAR-10
2. NLP: IMDB sentiment (используйте простую LSTM)
3. Tabular: любой табличный датасет с Kaggle

**Требования:**
1. Для каждого датасета обучите модель с 4 оптимизаторами:
   - SGD + Momentum
   - Adam
   - AdamW
   - RMSprop
2. Подберите лучший lr для каждого (grid search)
3. Создайте сводную таблицу результатов
4. Определите, есть ли универсальный "лучший" оптимизатор

**Ожидаемый результат:**
- Adam/AdamW часто лучше на большинстве задач
- SGD+Momentum может быть лучше на CV с правильной настройкой

---

### **Задача 10: Кастомный оптимизатор с адаптивным momentum**

**Условие:** Реализуйте свой оптимизатор, комбинирующий идеи из разных методов.

**Требования:**
1. Базируйтесь на Adam
2. Добавьте адаптивный momentum:
   ```
   β1_t = β1_base * (1 - progress)  # progress ∈ [0, 1]
   ```
3. Momentum уменьшается по мере обучения
4. Реализуйте как класс, наследующийся от `torch.optim.Optimizer`
5. Протестируйте на MNIST и сравните с обычным Adam

```python
class AdaptiveMomentumAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.step_count = 0
        self.max_steps = None  # установите в начале обучения
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        self.step_count += 1
        progress = self.step_count / self.max_steps if self.max_steps else 0
        
        for group in self.param_groups:
            # TODO: реализуйте custom update rule
            pass
        
        return loss
```

---

### **Задача 11: Second-order методы: L-BFGS**

**Условие:** Используйте квази-ньютоновский метод L-BFGS для оптимизации.

**Требования:**
1. Реализуйте обучение с `torch.optim.LBFGS`
2. LBFGS требует closure - реализуйте его правильно
3. Сравните с Adam на небольшой модели (MLP на MNIST)
4. Измерьте:
   - Количество итераций до сходимости
   - Время обучения
   - Потребление памяти
5. Объясните, почему L-BFGS редко используется в deep learning

```python
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0)

def closure():
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    return loss

optimizer.step(closure)
```

**Вопросы:**
- Почему L-BFGS не масштабируется на большие модели?
- В каких случаях L-BFGS может быть полезен?

---

## 💎 Заключение

### **Чек-лист выбора оптимизатора:**

✅ **Когда использовать каждый оптимизатор:**

**SGD + Momentum:**
- ✅ Computer Vision (особенно с правильным LR schedule)
- ✅ Когда нужна лучшая генерализация
- ✅ Когда есть время подбирать гиперпараметры
- ❌ NLP задачи (обычно)
- ❌ Когда нужен быстрый прототип

**Adam/AdamW:**
- ✅ NLP задачи (почти всегда)
- ✅ Быстрое прототипирование
- ✅ Когда данных мало
- ✅ Sparse gradients
- ❌ Иногда хуже генерализует, чем SGD

**RMSprop:**
- ✅ RNN задачи (исторически)
- ✅ Non-stationary задачи
- ❌ Обычно Adam лучше

### **Типичные настройки:**

```python
# Computer Vision (ResNet, VGG)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

# NLP (Transformers)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# Универсальный вариант (прототипирование)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

### **Отладка проблем с оптимизацией:**

**Если loss не уменьшается:**
1. Проверьте, вызывается ли `optimizer.step()`
2. Убедитесь, что `zero_grad()` вызывается перед backward
3. Проверьте градиенты: `for p in model.parameters(): print(p.grad)`
4. Попробуйте уменьшить learning rate на порядок

**Если loss становится NaN:**
1. Уменьшите learning rate
2. Добавьте gradient clipping
3. Проверьте данные на NaN/Inf
4. Используйте более стабильные функции активации (ReLU вместо Sigmoid)

**Если модель не генерализует:**
1. Добавьте weight decay
2. Попробуйте AdamW вместо Adam
3. Используйте learning rate schedule (reduce on plateau)
4. Добавьте dropout / data augmentation

### **Дополнительные ресурсы:**

1. **Статьи:**
   - "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
   - "On the Variance of the Adaptive Learning Rate and Beyond" (RAdam paper)
   - "Decoupled Weight Decay Regularization" (AdamW paper)

2. **Инструменты:**
   - `torch.optim.lr_scheduler` для learning rate schedules
   - TensorBoard для визуализации lr и gradients
   - Weights & Biases для tracking experiments

3. **Практика:**
   - Попробуйте все оптимизаторы на вашей задаче
   - Не бойтесь подбирать гиперпараметры
   - Следите за train/val curves внимательно
