# Оптимизаторы (Optimizers)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# !pip install torch matplotlib
```

---

## 🟢 Базовый уровень: Градиентный спуск

### 1.1 Что такое оптимизатор?

**Оптимизатор** — алгоритм, который обновляет веса модели для минимизации функции потерь.

**Общая формула:**
$$w_{t+1} = w_t - \eta \cdot \nabla L(w_t)$$

где $\eta$ — learning rate, $\nabla L$ — градиент функции потерь

```python
# Простейший пример оптимизации
def optimize_simple():
    # Функция: f(x) = (x - 3)^2, минимум в x=3
    x = 0.0  # начальное значение
    learning_rate = 0.1
    
    for step in range(20):
        # Градиент: df/dx = 2(x - 3)
        gradient = 2 * (x - 3)
        
        # Обновление
        x = x - learning_rate * gradient
        
        loss = (x - 3) ** 2
        if step % 5 == 0:
            print(f"Step {step}: x={x:.4f}, loss={loss:.4f}")
    
    return x

result = optimize_simple()
print(f"Результат: {result:.4f}")
```

### 1.2 Stochastic Gradient Descent (SGD)

**Базовый оптимизатор** — обновляет веса на каждом батче.

```python
# Пример модели
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# Создание оптимизатора SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Цикл обучения
for epoch in range(100):
    # Forward pass
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    y_pred = model(x)
    loss = nn.MSELoss()(y_pred, y)
    
    # Backward pass
    optimizer.zero_grad()  # Обнуляем градиенты
    loss.backward()        # Вычисляем градиенты
    optimizer.step()       # Обновляем веса
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### 1.3 Learning Rate — важнейший гиперпараметр

```python
def visualize_learning_rates():
    # Функция: f(x) = (x - 3)^2
    def optimize(lr, steps=50):
        x = 0.0
        history = [x]
        for _ in range(steps):
            gradient = 2 * (x - 3)
            x = x - lr * gradient
            history.append(x)
        return history
    
    plt.figure(figsize=(12, 4))
    
    # Слишком маленький LR
    plt.subplot(131)
    history = optimize(lr=0.01)
    plt.plot(history)
    plt.title('LR=0.01 (слишком мал)')
    plt.xlabel('Шаг')
    plt.ylabel('x')
    
    # Оптимальный LR
    plt.subplot(132)
    history = optimize(lr=0.5)
    plt.plot(history)
    plt.title('LR=0.5 (оптимально)')
    plt.xlabel('Шаг')
    
    # Слишком большой LR
    plt.subplot(133)
    history = optimize(lr=1.5)
    plt.plot(history[:20])  # Ограничиваем для читаемости
    plt.title('LR=1.5 (слишком велик)')
    plt.xlabel('Шаг')
    
    plt.tight_layout()
    plt.show()

visualize_learning_rates()
```

---

## 🟡 Продвинутый уровень: Современные оптимизаторы

### 2.1 SGD с Momentum

**Идея:** Учитываем направление предыдущих градиентов (инерция)

$$v_t = \beta v_{t-1} + \nabla L(w_t)$$
$$w_{t+1} = w_t - \eta v_t$$

```python
# Реализация с нуля
class SGDMomentum:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p) for p in self.parameters]
    
    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            # Обновление velocity
            self.velocities[i] = self.momentum * self.velocities[i] + param.grad
            
            # Обновление параметров
            param.data -= self.lr * self.velocities[i]
    
    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()

# PyTorch версия
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

**Визуализация momentum:**
```python
def compare_sgd_momentum():
    # Функция с "оврагом"
    def f(x, y):
        return x**2 + 10*y**2
    
    # SGD без momentum
    x_sgd, y_sgd = 1.0, 1.0
    history_sgd = [(x_sgd, y_sgd)]
    
    for _ in range(50):
        grad_x = 2*x_sgd
        grad_y = 20*y_sgd
        x_sgd -= 0.01 * grad_x
        y_sgd -= 0.01 * grad_y
        history_sgd.append((x_sgd, y_sgd))
    
    # SGD с momentum
    x_mom, y_mom = 1.0, 1.0
    v_x, v_y = 0.0, 0.0
    history_mom = [(x_mom, y_mom)]
    
    for _ in range(50):
        grad_x = 2*x_mom
        grad_y = 20*y_mom
        v_x = 0.9*v_x + grad_x
        v_y = 0.9*v_y + grad_y
        x_mom -= 0.01 * v_x
        y_mom -= 0.01 * v_y
        history_mom.append((x_mom, y_mom))
    
    # Визуализация
    history_sgd = np.array(history_sgd)
    history_mom = np.array(history_mom)
    
    plt.figure(figsize=(10, 5))
    plt.plot(history_sgd[:, 0], history_sgd[:, 1], 'o-', label='SGD', alpha=0.7)
    plt.plot(history_mom[:, 0], history_mom[:, 1], 's-', label='SGD+Momentum', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('SGD vs SGD+Momentum')
    plt.grid(True)
    plt.show()

compare_sgd_momentum()
```

### 2.2 RMSprop

**Идея:** Адаптивный learning rate для каждого параметра

$$s_t = \beta s_{t-1} + (1-\beta)(\nabla L(w_t))^2$$
$$w_{t+1} = w_t - \frac{\eta}{\sqrt{s_t + \epsilon}} \nabla L(w_t)$$

```python
# PyTorch версия
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)

# Пример использования
for epoch in range(100):
    # ... forward pass, loss ...
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 2.3 Adam — самый популярный оптимизатор

**Идея:** Комбинирует Momentum + RMSprop

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L(w_t)$$ (Momentum)
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L(w_t))^2$$ (RMSprop)

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$ (Bias correction)

$$w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

```python
# Базовые параметры Adam
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# Пример: обучение на MNIST
import torchvision

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True,
                               transform=torchvision.transforms.ToTensor()),
    batch_size=64, shuffle=True
)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
```

### 2.4 AdamW — улучшенный Adam

**Отличие:** Правильная реализация weight decay

```python
# AdamW с weight decay
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

---

## 🔴 Экспертный уровень: Продвинутые техники

### 3.1 Сравнение оптимизаторов

```python
def compare_optimizers():
    # Простая задача регрессии
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    def train_model(optimizer_fn, name, epochs=50):
        model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))
        optimizer = optimizer_fn(model.parameters())
        criterion = nn.MSELoss()
        
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        return losses
    
    # Сравнение
    optimizers = [
        (lambda p: optim.SGD(p, lr=0.01), 'SGD'),
        (lambda p: optim.SGD(p, lr=0.01, momentum=0.9), 'SGD+Momentum'),
        (lambda p: optim.RMSprop(p, lr=0.001), 'RMSprop'),
        (lambda p: optim.Adam(p, lr=0.001), 'Adam'),
    ]
    
    plt.figure(figsize=(10, 6))
    for opt_fn, name in optimizers:
        losses = train_model(opt_fn, name)
        plt.plot(losses, label=name)
    
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.title('Сравнение оптимизаторов')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.show()

compare_optimizers()
```

### 3.2 Gradient Clipping

**Проблема:** Взрывающиеся градиенты в RNN

```python
# Gradient clipping по норме
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    
    # Обрезаем градиенты
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()

# Gradient clipping по значению
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### 3.3 Learning Rate Warmup

```python
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            lr = self.base_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# Использование
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = WarmupScheduler(optimizer, warmup_steps=1000, base_lr=0.001)

for batch in train_loader:
    # ... training ...
    scheduler.step()
```

### 3.4 Lookahead Optimizer

```python
class Lookahead(optim.Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = {}
        
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['slow_buffer'] = torch.zeros_like(p.data)
                param_state['slow_buffer'].copy_(p.data)
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                
                # Обновление "медленных" весов каждые k шагов
                if self.optimizer.state[p]['step'] % self.k == 0:
                    param_state['slow_buffer'].add_(
                        p.data - param_state['slow_buffer'], alpha=self.alpha
                    )
                    p.data.copy_(param_state['slow_buffer'])
        
        return loss

# Использование
base_optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
```

---

## 💎 Заключение

**Сравнение оптимизаторов:**

| Оптимизатор | Плюсы | Минусы | Когда использовать |
|------------|-------|--------|-------------------|
| **SGD** | Простой, надежный | Медленная сходимость | Небольшие модели |
| **SGD+Momentum** | Быстрее SGD | Требует настройки momentum | Computer Vision |
| **RMSprop** | Адаптивный LR | Может застрять | RNN |
| **Adam** | Быстрая сходимость, мало настройки | Может переобучаться | Общее применение |
| **AdamW** | Лучше регуляризация | - | Transformers, NLP |

**Рекомендации:**
1. **По умолчанию:** Adam (lr=0.001) или AdamW
2. **Computer Vision:** SGD с momentum (lr=0.01-0.1)
3. **NLP/Transformers:** AdamW с warmup
4. **Reinforcement Learning:** Adam или RMSprop

**Лучшие практики:**
- Начинайте с Adam/AdamW с lr=0.001
- Используйте learning rate scheduling
- Мониторьте норму градиентов
- Применяйте gradient clipping для RNN
- Экспериментируйте с разными оптимизаторами

> **"Выбор оптимизатора может критически влиять на скорость и качество обучения. Adam — хороший выбор по умолчанию."**

**Дальнейшее изучение:**
- [PyTorch Optimizers](https://pytorch.org/docs/stable/optim.html)
- [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)
- [Adam paper](https://arxiv.org/abs/1412.6980)
