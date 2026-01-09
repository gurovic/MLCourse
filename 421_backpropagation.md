# Обратное распространение ошибки (Backpropagation)

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# !pip install torch matplotlib
```

---

## 🟢 Базовый уровень: Интуиция

### 1.1 Что такое обратное распространение?

**Backpropagation** — это алгоритм для эффективного вычисления градиентов функции потерь по весам нейронной сети.

**Почему это важно?**
- Градиенты показывают, как нужно изменить веса, чтобы уменьшить ошибку
- Без backpropagation обучение больших сетей было бы невозможным

**Интуиция:**
1. Делаем предсказание (forward pass)
2. Вычисляем ошибку
3. Распространяем ошибку назад по сети (backward pass)
4. Обновляем веса в направлении уменьшения ошибки

```python
# Простой пример: одна связь
x = 2.0  # вход
w = 0.5  # вес
y_true = 3.0  # целевое значение

# Forward pass
y_pred = w * x  # предсказание = 1.0

# Loss
loss = (y_pred - y_true) ** 2  # = 4.0

# Backward pass: вычисляем градиент
# dL/dw = dL/dy_pred * dy_pred/dw
dL_dy_pred = 2 * (y_pred - y_true)  # = -4.0
dy_pred_dw = x  # = 2.0
dL_dw = dL_dy_pred * dy_pred_dw  # = -8.0

# Обновление веса
learning_rate = 0.01
w_new = w - learning_rate * dL_dw  # = 0.5 - 0.01*(-8) = 0.58

print(f"Старый вес: {w}, Новый вес: {w_new}")
```

### 1.2 Цепное правило

**Цепное правило** — математическая основа backpropagation.

Если `y = f(g(x))`, то:
$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

```python
# Пример: y = (x^2 + 1)^3
x = 2.0

# Forward pass
g = x ** 2 + 1  # g = 5
y = g ** 3  # y = 125

# Backward pass с цепным правилом
dy_dg = 3 * g ** 2  # = 75
dg_dx = 2 * x  # = 4
dy_dx = dy_dg * dg_dx  # = 300

print(f"dy/dx = {dy_dx}")
```

### 1.3 Вычислительный граф

```python
# PyTorch автоматически строит граф вычислений
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(0.5, requires_grad=True)

# Forward pass
y_pred = w * x
loss = (y_pred - 3.0) ** 2

# Backward pass (PyTorch делает это автоматически!)
loss.backward()

print(f"Градиент по w: {w.grad}")  # -8.0
print(f"Градиент по x: {x.grad}")  # -4.0
```

---

## 🟡 Продвинутый уровень: Математика

### 2.1 Backpropagation для одного нейрона

Рассмотрим нейрон с активацией sigmoid:

$$z = w_1 x_1 + w_2 x_2 + b$$
$$a = \sigma(z) = \frac{1}{1 + e^{-z}}$$
$$L = (a - y_{true})^2$$

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Данные
x1, x2 = 1.0, 2.0
w1, w2 = 0.5, -0.3
b = 0.1
y_true = 1.0

# Forward pass
z = w1 * x1 + w2 * x2 + b  # = 0.5*1 + (-0.3)*2 + 0.1 = -0.4
a = sigmoid(z)  # ≈ 0.401
loss = (a - y_true) ** 2  # ≈ 0.359

# Backward pass
dL_da = 2 * (a - y_true)  # ≈ -1.198
da_dz = sigmoid_derivative(z)  # ≈ 0.240
dL_dz = dL_da * da_dz  # ≈ -0.287

# Градиенты по весам
dL_dw1 = dL_dz * x1  # ≈ -0.287
dL_dw2 = dL_dz * x2  # ≈ -0.574
dL_db = dL_dz  # ≈ -0.287

print(f"Градиенты: dw1={dL_dw1:.3f}, dw2={dL_dw2:.3f}, db={dL_db:.3f}")
```

### 2.2 Backpropagation через слои

Для сети с двумя слоями:

$$z^{[1]} = W^{[1]}x + b^{[1]}$$
$$a^{[1]} = \text{ReLU}(z^{[1]})$$
$$z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$$
$$\hat{y} = \sigma(z^{[2]})$$

```python
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        # Инициализация весов
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))
    
    def forward(self, X):
        # Сохраняем промежуточные значения для backward pass
        self.X = X
        self.z1 = np.dot(self.W1, X) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, y_true, learning_rate=0.01):
        m = y_true.shape[1]  # количество примеров
        
        # Градиенты выходного слоя
        dL_da2 = 2 * (self.a2 - y_true) / m
        da2_dz2 = self.a2 * (1 - self.a2)  # производная sigmoid
        dL_dz2 = dL_da2 * da2_dz2
        
        # Градиенты по W2 и b2
        dL_dW2 = np.dot(dL_dz2, self.a1.T)
        dL_db2 = np.sum(dL_dz2, axis=1, keepdims=True)
        
        # Градиенты скрытого слоя
        dL_da1 = np.dot(self.W2.T, dL_dz2)
        da1_dz1 = (self.z1 > 0).astype(float)  # производная ReLU
        dL_dz1 = dL_da1 * da1_dz1
        
        # Градиенты по W1 и b1
        dL_dW1 = np.dot(dL_dz1, self.X.T)
        dL_db1 = np.sum(dL_dz1, axis=1, keepdims=True)
        
        # Обновление весов
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2

# Тестирование
net = TwoLayerNet(2, 4, 1)
X = np.array([[1, 2], [3, 4]]).T
y = np.array([[1, 0]]).T

for epoch in range(1000):
    y_pred = net.forward(X)
    net.backward(y, learning_rate=0.1)
    if epoch % 100 == 0:
        loss = np.mean((y_pred - y) ** 2)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

### 2.3 Проблемы градиентов

**Затухающие градиенты (Vanishing Gradients):**
```python
# Sigmoid в глубоких сетях
def deep_sigmoid_gradients():
    layers = 10
    x = np.array([1.0])
    
    for i in range(layers):
        x = sigmoid(x)
        grad = sigmoid_derivative(x)
        print(f"Слой {i+1}: активация={x[0]:.6f}, градиент={grad[0]:.6f}")

deep_sigmoid_gradients()
# Градиент уменьшается экспоненциально!
```

**Взрывающиеся градиенты (Exploding Gradients):**
```python
# Плохая инициализация весов
W = np.random.randn(5, 5) * 10  # Большие веса!
x = np.random.randn(5, 1)

for i in range(10):
    x = np.dot(W, x)
    print(f"Итерация {i+1}: норма={np.linalg.norm(x):.2e}")
# Норма растет экспоненциально!
```

---

## 🔴 Экспертный уровень: Продвинутые техники

### 3.1 Gradient Checking

Проверка правильности реализации backpropagation:

```python
def numerical_gradient(f, x, epsilon=1e-5):
    """Численная аппроксимация градиента"""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        
        x[idx] = old_value + epsilon
        f_plus = f(x)
        
        x[idx] = old_value - epsilon
        f_minus = f(x)
        
        grad[idx] = (f_plus - f_minus) / (2 * epsilon)
        x[idx] = old_value
        it.iternext()
    
    return grad

# Пример
def f(w):
    x = np.array([1.0, 2.0])
    return np.sum((np.dot(w, x) - 3.0) ** 2)

w = np.array([0.5, 0.3])

# Аналитический градиент
x = np.array([1.0, 2.0])
pred = np.dot(w, x)
analytical_grad = 2 * (pred - 3.0) * x

# Численный градиент
numerical_grad = numerical_gradient(f, w)

print(f"Аналитический: {analytical_grad}")
print(f"Численный: {numerical_grad}")
print(f"Разница: {np.linalg.norm(analytical_grad - numerical_grad)}")
```

### 3.2 Autograd в PyTorch

```python
# PyTorch автоматически вычисляет градиенты
class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y + x ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_output * (y + 2 * x)
        grad_y = grad_output * x
        return grad_x, grad_y

# Использование
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = CustomFunction.apply(x, y)
z.backward()

print(f"dz/dx = {x.grad}")  # y + 2*x = 7
print(f"dz/dy = {y.grad}")  # x = 2
```

### 3.3 Визуализация потока градиентов

```python
def plot_gradient_flow(named_parameters):
    """Визуализация потока градиентов по слоям"""
    ave_grads = []
    layers = []
    
    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
    
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(ave_grads)), ave_grads, alpha=0.8)
    plt.xticks(range(len(layers)), layers, rotation=45, ha='right')
    plt.ylabel('Средний градиент')
    plt.title('Поток градиентов')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Пример использования
model = nn.Sequential(
    nn.Linear(10, 50), nn.ReLU(),
    nn.Linear(50, 50), nn.ReLU(),
    nn.Linear(50, 1)
)

x = torch.randn(32, 10)
y = torch.randn(32, 1)
loss = nn.MSELoss()(model(x), y)
loss.backward()

plot_gradient_flow(model.named_parameters())
```

---

## 💎 Заключение

**Ключевые концепции:**
1. **Цепное правило** — математическая основа backpropagation
2. **Forward + Backward** — два прохода для вычисления градиентов
3. **Эффективность** — backpropagation вычисляет все градиенты за один проход

**Проблемы и решения:**

| Проблема | Причина | Решение |
|----------|---------|---------|
| Затухающие градиенты | Sigmoid/Tanh в глубоких сетях | ReLU, Batch Normalization |
| Взрывающиеся градиенты | Плохая инициализация | Xavier/He initialization, Gradient Clipping |
| Медленное обучение | Маленький learning rate | Адаптивные оптимизаторы (Adam) |

**Лучшие практики:**
- Используйте автоматическое дифференцирование (PyTorch/TensorFlow)
- При написании своих функций проверяйте градиенты численно
- Мониторьте норму градиентов во время обучения
- Используйте gradient clipping для RNN

> **"Backpropagation — это не магия, а элегантное применение цепного правила. Понимание этого алгоритма критически важно для глубокого обучения."**

**Дальнейшее изучение:**
- [CS231n: Backpropagation](https://cs231n.github.io/optimization-2/)
- [Neural Networks and Deep Learning (Chapter 2)](http://neuralnetworksanddeeplearning.com/chap2.html)
- [Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/)
