# Инициализация весов (Weight Initialization)

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# !pip install torch matplotlib
```

---

## 🟢 Базовый уровень: Почему инициализация важна?

### 1.1 Проблема плохой инициализации

**Плохая инициализация** может привести к:
- Затухающим градиентам (градиенты → 0)
- Взрывающимся градиентам (градиенты → ∞)
- Медленной сходимости

```python
# Эксперимент: разные инициализации
def test_initialization(init_method, name):
    torch.manual_seed(42)
    
    # Глубокая сеть
    layers = []
    for _ in range(10):
        layer = nn.Linear(100, 100)
        
        if init_method == 'zeros':
            nn.init.zeros_(layer.weight)
        elif init_method == 'large':
            nn.init.normal_(layer.weight, mean=0, std=10.0)
        elif init_method == 'small':
            nn.init.normal_(layer.weight, mean=0, std=0.01)
        elif init_method == 'xavier':
            nn.init.xavier_uniform_(layer.weight)
        
        layers.append(layer)
        layers.append(nn.ReLU())
    
    model = nn.Sequential(*layers)
    
    # Прогон данных
    x = torch.randn(64, 100)
    
    activations = []
    with torch.no_grad():
        for layer in model:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                activations.append(x.std().item())
    
    return activations

# Сравнение
fig, ax = plt.subplots(figsize=(10, 6))

for method, label in [('zeros', 'Нули'), ('large', 'Большие'), 
                       ('small', 'Маленькие'), ('xavier', 'Xavier')]:
    acts = test_initialization(method, label)
    ax.plot(acts, marker='o', label=label)

ax.set_xlabel('Номер слоя')
ax.set_ylabel('Стандартное отклонение активаций')
ax.set_title('Влияние инициализации на распространение сигнала')
ax.legend()
ax.grid(True)
plt.show()
```

### 1.2 Xavier (Glorot) Initialization

**Идея:** Сохранить дисперсию активаций и градиентов на всех слоях.

**Формула (uniform):**
$$W \sim U\left[-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right]$$

**Формула (normal):**
$$W \sim N\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$$

```python
# Xavier инициализация
layer = nn.Linear(256, 128)

# Uniform
nn.init.xavier_uniform_(layer.weight)
print(f"Xavier Uniform: mean={layer.weight.mean():.4f}, std={layer.weight.std():.4f}")

# Normal
nn.init.xavier_normal_(layer.weight)
print(f"Xavier Normal: mean={layer.weight.mean():.4f}, std={layer.weight.std():.4f}")
```

### 1.3 He Initialization — для ReLU

**Проблема Xavier:** Не учитывает, что ReLU "убивает" половину нейронов.

**Формула He (normal):**
$$W \sim N\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

```python
# He инициализация
layer = nn.Linear(256, 128)

nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
print(f"He Uniform: std={layer.weight.std():.4f}")

nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
print(f"He Normal: std={layer.weight.std():.4f}")
```

---

## 🟡 Продвинутый уровень: Выбор инициализации

### 2.1 Инициализация для разных активаций

```python
def initialize_layer(layer, activation='relu'):
    """Правильная инициализация в зависимости от активации"""
    if activation in ['relu', 'leaky_relu']:
        # He initialization для ReLU
        nn.init.kaiming_normal_(layer.weight, nonlinearity=activation)
    elif activation in ['sigmoid', 'tanh']:
        # Xavier initialization для sigmoid/tanh
        nn.init.xavier_normal_(layer.weight)
    else:
        # По умолчанию Xavier
        nn.init.xavier_normal_(layer.weight)
    
    # Bias обычно инициализируем нулями
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

# Пример использования
class ProperlyInitializedModel(nn.Module):
    def __init__(self):
        super(ProperlyInitializedModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        # Инициализация
        initialize_layer(self.fc1, 'relu')
        initialize_layer(self.fc2, 'relu')
        initialize_layer(self.fc3, 'relu')
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### 2.2 Сравнение сходимости

```python
def compare_initializations():
    from torchvision import datasets, transforms
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                      transform=transforms.ToTensor()),
        batch_size=64, shuffle=True
    )
    
    def train_with_init(init_method, epochs=5):
        # Создаем модель
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        # Инициализация
        for layer in model:
            if isinstance(layer, nn.Linear):
                if init_method == 'default':
                    pass  # PyTorch default
                elif init_method == 'xavier':
                    nn.init.xavier_normal_(layer.weight)
                elif init_method == 'he':
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                elif init_method == 'small':
                    nn.init.normal_(layer.weight, std=0.01)
                
                nn.init.zeros_(layer.bias)
        
        # Обучение
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
                if batch_idx >= 100:  # Ограничиваем для скорости
                    break
            
            losses.append(epoch_loss / 100)
        
        return losses
    
    # Сравнение
    plt.figure(figsize=(10, 6))
    for method in ['default', 'xavier', 'he', 'small']:
        losses = train_with_init(method)
        plt.plot(losses, marker='o', label=method.capitalize())
    
    plt.xlabel('Эпоха')
    plt.ylabel('Training Loss')
    plt.title('Влияние инициализации на скорость обучения')
    plt.legend()
    plt.grid(True)
    plt.show()

compare_initializations()
```

### 2.3 Инициализация Batch Normalization

```python
# Для слоев с Batch Normalization
class ModelWithBN(nn.Module):
    def __init__(self):
        super(ModelWithBN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        
        # BatchNorm имеет свои параметры γ и β
        # По умолчанию: γ=1, β=0 (правильно!)
        # Но можно настроить:
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.bn1(torch.relu(self.fc1(x)))
        x = self.bn2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
```

---

## 🔴 Экспертный уровень: Продвинутые техники

### 3.1 LSUV Initialization

**Layer-Sequential Unit-Variance** — итеративная нормализация активаций.

```python
@torch.no_grad()
def lsuv_init(model, data_loader, target_std=1.0, tol=0.1, max_iter=10):
    """LSUV инициализация"""
    model.eval()
    
    # Получаем батч данных
    data, _ = next(iter(data_loader))
    
    # Проходим по слоям
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            # Инициализируем ортогональной матрицей
            nn.init.orthogonal_(module.weight)
            
            # Итеративно нормализуем
            for i in range(max_iter):
                # Forward pass до этого слоя
                output = model(data)
                
                # Вычисляем std активаций
                activation_std = output.std().item()
                
                if abs(activation_std - target_std) < tol:
                    break
                
                # Корректируем веса
                module.weight.data /= (activation_std + 1e-8)
            
            print(f"Layer {name}: std={activation_std:.4f} after {i+1} iterations")
    
    model.train()
    return model

# Использование
model = ProperlyInitializedModel()
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                  transform=transforms.ToTensor()),
    batch_size=64
)
model = lsuv_init(model, train_loader)
```

### 3.2 Fixup Initialization — для ResNet без BN

```python
class FixupBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FixupBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        
        # Fixup parameters
        self.scale = nn.Parameter(torch.ones(1))
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.bias3 = nn.Parameter(torch.zeros(1))
        self.bias4 = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x + self.bias1)
        out = torch.relu(out + self.bias2)
        
        out = self.conv2(out + self.bias3)
        out = out * self.scale + self.bias4
        
        out += identity
        out = torch.relu(out)
        return out

def fixup_init(model, num_layers):
    """Fixup инициализация"""
    for m in model.modules():
        if isinstance(m, FixupBasicBlock):
            nn.init.normal_(m.conv1.weight, 
                          mean=0, 
                          std=np.sqrt(2 / (m.conv1.weight.shape[0] * 
                                          np.prod(m.conv1.weight.shape[2:]))) * 
                          num_layers ** (-0.5))
            nn.init.zeros_(m.conv2.weight)
```

### 3.3 Data-Dependent Initialization

```python
@torch.no_grad()
def data_dependent_init(model, data_loader, num_batches=100):
    """Инициализация на основе данных"""
    model.eval()
    
    # Собираем статистику активаций
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(output.detach())
        return hook
    
    # Регистрируем hooks
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            handle = module.register_forward_hook(hook_fn(name))
            handles.append(handle)
    
    # Прогоняем данные
    for i, (data, _) in enumerate(data_loader):
        if i >= num_batches:
            break
        model(data)
    
    # Удаляем hooks
    for handle in handles:
        handle.remove()
    
    # Нормализуем веса на основе статистики
    for name, module in model.named_modules():
        if name in activations and isinstance(module, (nn.Linear, nn.Conv2d)):
            all_acts = torch.cat(activations[name])
            std = all_acts.std()
            mean = all_acts.mean()
            
            # Корректируем веса для нормализации активаций
            module.weight.data /= (std + 1e-8)
            if module.bias is not None:
                module.bias.data -= mean
    
    model.train()
    return model
```

### 3.4 Transfer Learning инициализация

```python
def initialize_for_transfer_learning(model, pretrained_model, freeze_early_layers=True):
    """Инициализация для transfer learning"""
    
    # Копируем веса из pretrained модели
    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()
    
    # Фильтруем только совпадающие слои
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # Замораживаем ранние слои
    if freeze_early_layers:
        for name, param in model.named_parameters():
            if name in pretrained_dict:
                param.requires_grad = False
    
    # Новые слои инициализируем He
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name not in pretrained_dict:
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    return model
```

---

## 💎 Заключение

**Рекомендации по инициализации:**

| Архитектура | Активация | Рекомендуемая инициализация |
|-------------|-----------|----------------------------|
| **MLP** | ReLU/LeakyReLU | He (Kaiming) Normal |
| **MLP** | Sigmoid/Tanh | Xavier Normal |
| **CNN** | ReLU | He Uniform для Conv2d |
| **ResNet** | ReLU | He Normal |
| **ResNet без BN** | ReLU | Fixup Initialization |
| **Transformer** | Any | Xavier Uniform |
| **RNN/LSTM** | Tanh | Xavier или Orthogonal |

**Правила выбора:**

1. **С Batch Normalization:**
   - Инициализация менее критична
   - Xavier или He — оба работают хорошо

2. **Без Batch Normalization:**
   - Инициализация ОЧЕНЬ важна
   - He для ReLU, Xavier для sigmoid/tanh

3. **Глубокие сети (>20 слоев):**
   - Рассмотрите Fixup или LSUV
   - Обязательно используйте residual connections

4. **Transfer Learning:**
   - Используйте предобученные веса
   - Новые слои: He initialization

**PyTorch defaults:**
```python
# По умолчанию PyTorch использует:
# Linear: U(-√k, √k) где k = 1/in_features
# Conv2d: U(-√k, √k) где k = 1/(in_channels * kernel_size^2)
# Это близко к Xavier, но не совсем
```

**Лучшие практики:**
- ✅ ReLU → He initialization
- ✅ Sigmoid/Tanh → Xavier initialization
- ✅ С BatchNorm → инициализация менее критична
- ✅ Bias обычно инициализируем нулями
- ✅ Проверяйте распределение активаций после инициализации

**Частые ошибки:**
- ❌ Инициализация всех весов нулями (симметрия!)
- ❌ Слишком большая инициализация (взрывающиеся градиенты)
- ❌ Слишком маленькая инициализация (затухающие градиенты)
- ❌ Использование Xavier для ReLU сетей

> **"Правильная инициализация — это первый шаг к успешному обучению. Плохая инициализация может сделать обучение невозможным."**

**Дальнейшее изучение:**
- [Understanding the difficulty of training deep networks](https://proceedings.mlr.press/v9/glorot10a.html)
- [Delving Deep into Rectifiers (He initialization)](https://arxiv.org/abs/1502.01852)
- [Fixup Initialization](https://arxiv.org/abs/1901.09321)
