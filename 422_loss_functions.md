# Функции потерь (Loss Functions)

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# !pip install torch scikit-learn matplotlib
```

---

## 🟢 Базовый уровень: Что такое функция потерь?

### 1.1 Определение

**Функция потерь (Loss Function)** — метрика, которая измеряет разницу между предсказаниями модели и истинными значениями.

**Цель обучения:** Минимизировать функцию потерь
$$\theta^* = \arg\min_{\theta} L(y_{true}, y_{pred})$$

```python
# Пример: простая loss для одного предсказания
y_true = 5.0
y_pred = 3.0
loss = (y_true - y_pred) ** 2  # = 4.0
print(f"Loss: {loss}")
```

### 1.2 Mean Squared Error (MSE) — для регрессии

**Формула:**
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Пример
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
loss = mse_loss(y_true, y_pred)
print(f"MSE Loss: {loss:.4f}")

# В PyTorch
y_true_t = torch.tensor([1., 2., 3., 4., 5.])
y_pred_t = torch.tensor([1.1, 1.9, 3.2, 3.8, 5.1])
criterion = nn.MSELoss()
loss_t = criterion(y_pred_t, y_true_t)
print(f"PyTorch MSE: {loss_t.item():.4f}")
```

**Свойства MSE:**
- ✅ Гладкая, дифференцируемая
- ✅ Сильно штрафует большие ошибки (квадрат!)
- ❌ Чувствительна к выбросам

### 1.3 Mean Absolute Error (MAE)

**Формула:**
$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

```python
def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

loss = mae_loss(y_true, y_pred)
print(f"MAE Loss: {loss:.4f}")

# PyTorch
criterion = nn.L1Loss()
loss_t = criterion(y_pred_t, y_true_t)
print(f"PyTorch MAE: {loss_t.item():.4f}")
```

**Сравнение MSE vs MAE:**
```python
# При наличии выброса
y_true = np.array([1, 2, 3, 4, 100])  # 100 — выброс
y_pred = np.array([1, 2, 3, 4, 5])

print(f"MSE: {mse_loss(y_true, y_pred):.2f}")  # ~1805 (огромное!)
print(f"MAE: {mae_loss(y_true, y_pred):.2f}")  # ~19 (меньше)
```

---

## 🟡 Продвинутый уровень: Классификация

### 2.1 Binary Cross-Entropy — бинарная классификация

**Формула:**
$$BCE = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

где $\hat{y}_i$ — вероятность класса 1 (от 0 до 1)

```python
def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    # epsilon для численной стабильности
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Пример
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])  # вероятности
loss = binary_cross_entropy(y_true, y_pred)
print(f"BCE Loss: {loss:.4f}")

# PyTorch
y_true_t = torch.tensor([1., 0., 1., 1., 0.])
y_pred_t = torch.tensor([0.9, 0.1, 0.8, 0.7, 0.2])
criterion = nn.BCELoss()
loss_t = criterion(y_pred_t, y_true_t)
print(f"PyTorch BCE: {loss_t.item():.4f}")
```

**Почему логарифм?**
```python
# Визуализация BCE для одного примера
p = np.linspace(0.01, 0.99, 100)

# Если истинный класс = 1
loss_y1 = -np.log(p)
# Если истинный класс = 0
loss_y0 = -np.log(1 - p)

plt.figure(figsize=(10, 4))
plt.plot(p, loss_y1, label='y_true=1')
plt.plot(p, loss_y0, label='y_true=0')
plt.xlabel('Предсказанная вероятность')
plt.ylabel('Loss')
plt.title('Binary Cross-Entropy')
plt.legend()
plt.grid(True)
plt.show()
```

### 2.2 Cross-Entropy — многоклассовая классификация

**Формула:**
$$CE = -\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C}y_{ic}\log(\hat{y}_{ic})$$

где $C$ — количество классов

```python
# Пример: 3 класса
y_true = np.array([0, 2, 1])  # классы
y_pred = np.array([
    [0.7, 0.2, 0.1],  # предсказание для примера 1
    [0.1, 0.2, 0.7],  # предсказание для примера 2
    [0.2, 0.6, 0.2]   # предсказание для примера 3
])

def cross_entropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    n = y_true.shape[0]
    ce = -np.sum(np.log(y_pred[np.arange(n), y_true])) / n
    return ce

loss = cross_entropy(y_true, y_pred)
print(f"CE Loss: {loss:.4f}")

# PyTorch (автоматически применяет softmax!)
y_true_t = torch.tensor([0, 2, 1])
y_logits = torch.tensor([
    [2.0, 1.0, 0.1],
    [0.5, 1.0, 2.0],
    [1.0, 2.0, 1.0]
])
criterion = nn.CrossEntropyLoss()
loss_t = criterion(y_logits, y_true_t)
print(f"PyTorch CE: {loss_t.item():.4f}")
```

### 2.3 Softmax и Cross-Entropy

**Softmax** преобразует логиты в вероятности:
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j}e^{z_j}}$$

```python
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # для стабильности
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Пример
logits = np.array([[2.0, 1.0, 0.1]])
probs = softmax(logits)
print(f"Логиты: {logits}")
print(f"Вероятности: {probs}")
print(f"Сумма: {np.sum(probs)}")  # = 1.0

# PyTorch
logits_t = torch.tensor([[2.0, 1.0, 0.1]])
probs_t = torch.softmax(logits_t, dim=1)
print(f"PyTorch Softmax: {probs_t}")
```

---

## 🔴 Экспертный уровень: Специализированные loss

### 3.1 Focal Loss — для дисбаланса классов

**Формула:**
$$FL = -\alpha(1-p_t)^\gamma \log(p_t)$$

где $\gamma$ — параметр фокусировки (обычно 2)

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()

# Пример
criterion = FocalLoss(alpha=1, gamma=2)
logits = torch.randn(10, 5)
targets = torch.randint(0, 5, (10,))
loss = criterion(logits, targets)
print(f"Focal Loss: {loss.item():.4f}")
```

### 3.2 Huber Loss — робастная к выбросам

**Формула:**
$$L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}$$

```python
criterion = nn.HuberLoss(delta=1.0)
y_true = torch.tensor([1., 2., 3., 4., 100.])
y_pred = torch.tensor([1., 2., 3., 4., 5.])
loss = criterion(y_pred, y_true)
print(f"Huber Loss: {loss.item():.4f}")

# Сравнение с MSE
mse = nn.MSELoss()(y_pred, y_true)
print(f"MSE Loss: {mse.item():.4f}")  # Намного больше!
```

### 3.3 Contrastive Loss — для metric learning

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        # label = 1 если пара похожа, 0 если разная
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss

# Пример
criterion = ContrastiveLoss(margin=2.0)
embed1 = torch.randn(32, 128)
embed2 = torch.randn(32, 128)
labels = torch.randint(0, 2, (32,)).float()
loss = criterion(embed1, embed2, labels)
print(f"Contrastive Loss: {loss.item():.4f}")
```

### 3.4 Пользовательские loss функции

```python
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        # Комбинация MSE и MAE
        mse = torch.mean((y_pred - y_true) ** 2)
        mae = torch.mean(torch.abs(y_pred - y_true))
        return 0.5 * mse + 0.5 * mae

# Использование
criterion = CustomLoss()
y_pred = torch.randn(100)
y_true = torch.randn(100)
loss = criterion(y_pred, y_true)
print(f"Custom Loss: {loss.item():.4f}")
```

---

## 💎 Заключение

**Выбор функции потерь:**

| Задача | Loss Function | Когда использовать |
|--------|--------------|-------------------|
| **Регрессия** | MSE | Стандартный выбор, нет выбросов |
| | MAE | Есть выбросы |
| | Huber Loss | Компромисс MSE/MAE |
| **Бинарная классификация** | Binary Cross-Entropy | Стандартный выбор |
| | Focal Loss | Дисбаланс классов |
| **Многоклассовая** | Cross-Entropy | Стандартный выбор |
| | Focal Loss | Дисбаланс классов |
| **Metric Learning** | Contrastive/Triplet | Обучение эмбеддингов |

**Ключевые принципы:**
1. **Дифференцируемость** — loss должна иметь градиент
2. **Выбор loss ≠ метрика** — loss оптимизируется, метрика оценивается
3. **Масштаб** — разные loss имеют разные масштабы значений

**Лучшие практики:**
- Начинайте со стандартных loss (MSE, Cross-Entropy)
- Используйте Focal Loss при дисбалансе классов
- Мониторьте значение loss во время обучения
- Комбинируйте loss функции для сложных задач

**Частые ошибки:**
- ❌ Использование accuracy как loss (не дифференцируема!)
- ❌ Забыть применить sigmoid/softmax перед BCE/CE
- ❌ Не учитывать масштаб loss при комбинировании

> **"Правильный выбор функции потерь критичен для успеха модели. Loss направляет обучение, метрики оценивают результат."**

**Дальнейшее изучение:**
- [PyTorch Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [Loss Functions Explained](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)
