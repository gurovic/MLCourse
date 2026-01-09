# Регуляризация в нейронных сетях

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# !pip install torch torchvision matplotlib
```

---

## 🟢 Базовый уровень: Что такое переобучение?

### 1.1 Проблема переобучения (Overfitting)

**Переобучение** — модель слишком хорошо запоминает тренировочные данные, но плохо обобщается на новых.

```python
# Демонстрация переобучения
def demonstrate_overfitting():
    # Простые данные
    torch.manual_seed(42)
    X_train = torch.linspace(0, 1, 20).reshape(-1, 1)
    y_train = torch.sin(2 * np.pi * X_train) + 0.3 * torch.randn_like(X_train)
    
    # Слишком сложная модель
    overfit_model = nn.Sequential(
        nn.Linear(1, 100), nn.ReLU(),
        nn.Linear(100, 100), nn.ReLU(),
        nn.Linear(100, 1)
    )
    
    # Обучение
    criterion = nn.MSELoss()
    optimizer = optim.Adam(overfit_model.parameters(), lr=0.01)
    
    for epoch in range(1000):
        optimizer.zero_grad()
        y_pred = overfit_model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    
    # Визуализация
    X_test = torch.linspace(0, 1, 200).reshape(-1, 1)
    with torch.no_grad():
        y_pred_train = overfit_model(X_train)
        y_pred_test = overfit_model(X_test)
    
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train, y_train, label='Train data', alpha=0.7)
    plt.plot(X_test, y_pred_test, 'r-', label='Prediction', linewidth=2)
    plt.plot(X_test, torch.sin(2 * np.pi * X_test), 'g--', label='True function')
    plt.legend()
    plt.title('Переобучение: модель запомнила шум')
    plt.show()

demonstrate_overfitting()
```

### 1.2 Dropout — случайное отключение нейронов

**Dropout** — во время обучения случайно "выключает" нейроны с вероятностью `p`.

```python
# Простой пример Dropout
class SimpleNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # Применяем dropout
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = SimpleNet(dropout_rate=0.5)
print(model)

# Важно: model.train() и model.eval()
model.train()  # Dropout активен
output_train = model(torch.randn(1, 784))

model.eval()  # Dropout отключен
output_eval = model(torch.randn(1, 784))
```

### 1.3 Сравнение с Dropout и без

```python
def compare_dropout():
    # Загрузка MNIST
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                      transform=transforms.ToTensor()),
        batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
        batch_size=1000
    )
    
    def train_model(use_dropout, epochs=5):
        model = SimpleNet(dropout_rate=0.5 if use_dropout else 0.0)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_losses, test_accs = [], []
        
        for epoch in range(epochs):
            # Training
            model.train()
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            # Evaluation
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
            
            test_acc = 100. * correct / len(test_loader.dataset)
            test_accs.append(test_acc)
            print(f"Epoch {epoch+1}: Test Accuracy = {test_acc:.2f}%")
        
        return test_accs
    
    # Сравнение
    print("Без Dropout:")
    acc_no_dropout = train_model(use_dropout=False)
    
    print("\nС Dropout:")
    acc_with_dropout = train_model(use_dropout=True)
    
    plt.plot(acc_no_dropout, label='Без Dropout')
    plt.plot(acc_with_dropout, label='С Dropout')
    plt.xlabel('Эпоха')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.title('Dropout уменьшает переобучение')
    plt.show()
```

---

## 🟡 Продвинутый уровень: Batch Normalization

### 2.1 Что такое Batch Normalization?

**Batch Normalization** нормализует активации каждого слоя, ускоряя обучение и стабилизируя его.

**Формула:**
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

где $\mu_B$, $\sigma_B$ — среднее и дисперсия батча, $\gamma$, $\beta$ — обучаемые параметры.

```python
class ModelWithBatchNorm(nn.Module):
    def __init__(self):
        super(ModelWithBatchNorm, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Batch Normalization
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.bn1(x)  # Нормализация перед активацией
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        x = self.fc3(x)
        return x

model = ModelWithBatchNorm()
print(model)
```

### 2.2 Эффект Batch Normalization

```python
def visualize_batchnorm_effect():
    # Данные
    x = torch.randn(100, 256)
    
    # Без Batch Norm
    fc = nn.Linear(256, 256)
    output_no_bn = torch.relu(fc(x))
    
    # С Batch Norm
    fc_bn = nn.Linear(256, 256)
    bn = nn.BatchNorm1d(256)
    output_with_bn = torch.relu(bn(fc_bn(x)))
    
    # Визуализация распределений
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(output_no_bn.detach().numpy().flatten(), bins=50, alpha=0.7)
    axes[0].set_title('Без Batch Normalization')
    axes[0].set_xlabel('Значение активации')
    axes[0].set_ylabel('Частота')
    
    axes[1].hist(output_with_bn.detach().numpy().flatten(), bins=50, alpha=0.7)
    axes[1].set_title('С Batch Normalization')
    axes[1].set_xlabel('Значение активации')
    
    plt.tight_layout()
    plt.show()

visualize_batchnorm_effect()
```

### 2.3 Layer Normalization

**Layer Normalization** — альтернатива для последовательностей (RNN, Transformers).

```python
class ModelWithLayerNorm(nn.Module):
    def __init__(self):
        super(ModelWithLayerNorm, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.ln1 = nn.LayerNorm(256)  # Layer Normalization
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.ln1(torch.relu(self.fc1(x)))
        x = self.ln2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Разница: BatchNorm нормализует по батчу, LayerNorm — по признакам
```

---

## 🔴 Экспертный уровень: Продвинутая регуляризация

### 3.1 Weight Decay (L2 Regularization)

**Weight Decay** добавляет штраф за большие веса: $L = L_{data} + \lambda \sum w^2$

```python
# Weight Decay в оптимизаторе
optimizer_no_wd = optim.Adam(model.parameters(), lr=0.001)
optimizer_with_wd = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Эквивалентно добавлению L2-регуляризации к loss
def l2_regularization(model, lambda_reg=0.01):
    l2_loss = 0
    for param in model.parameters():
        l2_loss += torch.norm(param, p=2)
    return lambda_reg * l2_loss

# В цикле обучения
loss = criterion(output, target) + l2_regularization(model)
```

### 3.2 DropConnect — Dropout для весов

```python
class DropConnect(nn.Module):
    def __init__(self, input_size, output_size, drop_prob=0.5):
        super(DropConnect, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.bias = nn.Parameter(torch.zeros(output_size))
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.training:
            # Dropout для весов, а не активаций
            mask = torch.bernoulli(torch.full_like(self.weight, 1 - self.drop_prob))
            w = self.weight * mask / (1 - self.drop_prob)
        else:
            w = self.weight
        
        return torch.matmul(x, w.t()) + self.bias
```

### 3.3 Spatial Dropout для CNN

```python
class CNNWithSpatialDropout(nn.Module):
    def __init__(self):
        super(CNNWithSpatialDropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.dropout1 = nn.Dropout2d(0.25)  # Spatial Dropout
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout3 = nn.Dropout(0.5)  # Обычный Dropout
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)  # Отключает целые feature maps
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x
```

### 3.4 Mixup Data Augmentation

```python
def mixup_data(x, y, alpha=1.0):
    """Mixup: смешивание примеров для регуляризации"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Использование в обучении
for data, target in train_loader:
    mixed_data, targets_a, targets_b, lam = mixup_data(data, target, alpha=1.0)
    
    optimizer.zero_grad()
    output = model(mixed_data)
    loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
    loss.backward()
    optimizer.step()
```

### 3.5 Комбинирование техник регуляризации

```python
class RegularizedModel(nn.Module):
    def __init__(self):
        super(RegularizedModel, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)     # Batch Normalization
        self.dropout1 = nn.Dropout(0.3)    # Dropout
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

# Оптимизатор с Weight Decay
model = RegularizedModel()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
```

---

## 💎 Заключение

**Сравнение техник регуляризации:**

| Техника | Как работает | Плюсы | Минусы | Когда использовать |
|---------|-------------|-------|--------|-------------------|
| **Dropout** | Случайно отключает нейроны | Прост, эффективен | Замедляет обучение | MLP, универсально |
| **Batch Normalization** | Нормализует активации | Ускоряет обучение | Проблемы с малыми батчами | CNN, большие батчи |
| **Layer Normalization** | Нормализует по признакам | Работает с любым размером батча | - | RNN, Transformers |
| **Weight Decay** | Штраф за большие веса | Простая реализация | Требует настройки | Универсально |
| **Mixup** | Смешивает примеры | Сильная регуляризация | Увеличивает время обучения | Computer Vision |

**Рекомендации по выбору:**

1. **Базовый набор:**
   - Dropout (0.3-0.5) в полносвязных слоях
   - Batch Normalization после сверточных/линейных слоев
   - Weight Decay (0.0001-0.01) в оптимизаторе

2. **CNN:**
   - BatchNorm после каждого сверточного слоя
   - Spatial Dropout (0.2-0.3)
   - Data Augmentation

3. **RNN/Transformers:**
   - Layer Normalization
   - Dropout на выходе RNN
   - Gradient Clipping

4. **Маленькие датасеты:**
   - Сильный Dropout (0.5-0.7)
   - Большой Weight Decay
   - Data Augmentation

**Лучшие практики:**
- Начинайте с Dropout + Batch Normalization + Weight Decay
- Используйте Dropout 0.5 для полносвязных слоев
- BatchNorm ставьте ПЕРЕД активацией (хотя есть споры)
- Мониторьте train/validation loss для детекции переобучения
- Не используйте слишком много регуляризации (недообучение!)

**Частые ошибки:**
- ❌ Забыть переключить model.eval() при инференсе
- ❌ Применять Dropout в выходном слое
- ❌ Слишком большой Weight Decay (модель не обучается)
- ❌ BatchNorm с очень маленькими батчами (<4)

> **"Регуляризация — это искусство баланса между запоминанием и обобщением. Правильная регуляризация критична для успеха модели."**

**Дальнейшее изучение:**
- [Dropout Paper](https://jmlr.org/papers/v15/srivastava14a.html)
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)
- [Understanding Deep Learning Regularization](https://arxiv.org/abs/1710.10686)
