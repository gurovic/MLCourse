### **Задачи: Инициализация весов (Weight Initialization)**

**Цель:** Понять влияние инициализации на обучение нейросетей и научиться правильно выбирать методы инициализации.

---

## 🟢 Базовый уровень

### **Задача 1: Демонстрация проблемы плохой инициализации**

**Условие:** Покажите, как плохая инициализация препятствует обучению.

**Требования:**
1. Создайте глубокую сеть (10 слоев): 100 → 100 → ... → 100 → 10
2. Обучите 3 варианта с разной инициализацией:
   - Все веса = 0 (nn.init.zeros_)
   - Очень большие веса (std=10.0)
   - Xavier initialization
3. Постройте графики loss по эпохам для каждого варианта
4. Измерьте, сколько эпох нужно для достижения 90% accuracy

**Ожидаемый результат:** 
- Нули: модель не обучается (симметрия)
- Большие веса: взрывающиеся градиенты или NaN
- Xavier: нормальное обучение

```python
import torch
import torch.nn as nn

class DeepNet(nn.Module):
    def __init__(self, init_type='xavier'):
        super().__init__()
        layers = []
        for _ in range(10):
            layer = nn.Linear(100, 100)
            # TODO: применить инициализацию
            layers.append(layer)
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers, nn.Linear(100, 10))
```

**Вопросы:**
- Почему инициализация нулями не работает?
- Что происходит с градиентами при больших весах?

---

### **Задача 2: Сравнение Xavier и He для разных активаций**

**Условие:** Исследуйте, какая инициализация лучше для ReLU vs Sigmoid.

**Требования:**
1. Постройте 2 сети с одинаковой архитектурой
2. Одна с ReLU активациями, другая с Sigmoid
3. Для каждой обучите с Xavier и He инициализацией
4. Используйте MNIST
5. Сравните:
   - Скорость сходимости (epochs to 95% accuracy)
   - Стабильность обучения
   - Финальную accuracy

**Ожидаемый результат:**
- ReLU + He > ReLU + Xavier
- Sigmoid + Xavier > Sigmoid + He

```python
def compare_init_and_activation():
    # TODO: реализуйте сравнение
    configs = [
        ('ReLU', 'xavier', nn.ReLU),
        ('ReLU', 'he', nn.ReLU),
        ('Sigmoid', 'xavier', nn.Sigmoid),
        ('Sigmoid', 'he', nn.Sigmoid)
    ]
```

---

### **Задача 3: Визуализация распространения сигнала**

**Условие:** Визуализируйте, как инициализация влияет на активации в глубокой сети.

**Требования:**
1. Создайте сеть из 20 слоев
2. Используйте 3 типа инициализации: zeros, uniform[-1,1], Xavier
3. Сделайте forward pass с random input
4. Для каждого слоя запишите std активаций
5. Постройте график: std vs номер слоя

**Ожидаемый результат:**
- Zeros: std быстро → 0
- Uniform: std растет или падает
- Xavier: std примерно постоянная (~1.0)

```python
def visualize_signal_propagation(model, x):
    """Визуализирует std активаций по слоям"""
    stds = []
    with torch.no_grad():
        for layer in model:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                stds.append(x.std().item())
    
    plt.plot(stds, marker='o')
    plt.xlabel('Номер слоя')
    plt.ylabel('Стандартное отклонение активаций')
    plt.show()
```

---

## 🟡 Продвинутый уровень

### **Задача 4: Влияние инициализации на затухающие градиенты**

**Условие:** Продемонстрируйте проблему vanishing gradients с разными инициализациями.

**Требования:**
1. Постройте глубокую сеть (15 слоев) с Sigmoid активациями
2. Обучите с инициализациями:
   - Normal(0, 0.01) — слишком маленькая
   - Normal(0, 1.0) — слишком большая
   - Xavier — правильная
3. Во время обучения логируйте норму градиентов первого слоя
4. Постройте график: ||∇W|| vs эпоха для каждой инициализации
5. Измерьте, обучается ли модель

**Ожидаемый результат:** 
- Маленькая: градиенты → 0, модель не учится
- Большая: градиенты → ∞, NaN loss
- Xavier: стабильные градиенты

```python
def track_gradients(model):
    """Отслеживает норму градиентов первого слоя"""
    first_layer = list(model.parameters())[0]
    grad_norm = first_layer.grad.norm().item()
    return grad_norm
```

---

### **Задача 5: Инициализация для ResNet архитектуры**

**Условие:** Реализуйте residual block с правильной инициализацией.

**Требования:**
1. Создайте ResidualBlock:
   ```python
   class ResidualBlock(nn.Module):
       def forward(self, x):
           residual = x
           out = self.conv1(x)
           out = F.relu(out)
           out = self.conv2(out)
           out += residual  # Skip connection
           out = F.relu(out)
           return out
   ```
2. Инициализируйте веса с He initialization
3. Постройте ResNet из 10 блоков
4. Обучите на CIFAR-10
5. Сравните с обычной сетью без skip connections

**Вопрос:** Почему skip connections помогают с инициализацией?

---

### **Задача 6: Batch Normalization vs инициализация**

**Условие:** Исследуйте, делает ли Batch Normalization инициализацию менее важной.

**Требования:**
1. Создайте две сети:
   - Без BatchNorm
   - С BatchNorm после каждого слоя
2. Для каждой обучите с плохой (std=0.01) и хорошей (Xavier) инициализацией
3. Сравните 4 комбинации:
   - No BN + Bad init
   - No BN + Good init
   - BN + Bad init
   - BN + Good init
4. Измерьте скорость сходимости

**Ожидаемый результат:** BN делает модель робастной к плохой инициализации.

---

## 🔴 Экспертный уровень

### **Задача 7: Реализация LSUV инициализации**

**Условие:** Реализуйте Layer-Sequential Unit-Variance initialization.

**Требования:**
1. Реализуйте LSUV алгоритм:
   - Ортогональная инициализация весов
   - Итеративная нормализация std активаций к 1.0
2. Примените к глубокой сети (20+ слоев)
3. Сравните с обычной Xavier инициализацией:
   - Распределение активаций по слоям
   - Скорость обучения
   - Финальное качество

```python
@torch.no_grad()
def lsuv_init(model, data_loader, target_std=1.0, tol=0.1, max_iter=10):
    """LSUV инициализация"""
    model.eval()
    data, _ = next(iter(data_loader))
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # TODO: реализуйте LSUV
            pass
```

---

### **Задача 8: Инициализация для разных типов слоев**

**Условие:** Создайте универсальную функцию инициализации для всех типов слоев.

**Требования:**
1. Реализуйте функцию, которая правильно инициализирует:
   - nn.Linear
   - nn.Conv2d
   - nn.LSTM/GRU
   - nn.Embedding
   - nn.BatchNorm
2. Учтите тип активации (ReLU, Tanh, Sigmoid)
3. Протестируйте на модели с разными типами слоев

```python
def initialize_model(model, activation='relu'):
    """Универсальная инициализация"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # TODO: He или Xavier в зависимости от activation
            pass
        elif isinstance(module, nn.Conv2d):
            # TODO: He uniform для conv
            pass
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            # TODO: Orthogonal для рекуррентных весов
            pass
        elif isinstance(module, nn.Embedding):
            # TODO: Normal(0, 1)
            pass
```

---

### **Задача 9: Fixup инициализация для ResNet без BN**

**Условие:** Реализуйте Fixup initialization для обучения глубокого ResNet без BatchNorm.

**Требования:**
1. Создайте ResNet архитектуру с Fixup parameters (bias1-4, scale)
2. Реализуйте Fixup инициализацию:
   ```python
   def fixup_init(model, num_layers):
       # Веса первого conv в каждом блоке: std = sqrt(2/fan_in) * L^(-1/2)
       # Веса второго conv: zeros
       pass
   ```
3. Обучите ResNet-50 на CIFAR-10
4. Сравните с обычным ResNet + BatchNorm

**Вопрос:** Как Fixup позволяет обучать глубокие сети без BN?

---

### **Задача 10: Анализ спектра весов**

**Условие:** Исследуйте спектр (собственные значения) матриц весов при разных инициализациях.

**Требования:**
1. Создайте глубокую сеть
2. Для каждого слоя вычислите собственные значения весовой матрицы
3. Сравните распределение собственных значений для:
   - Random normal
   - Xavier
   - He
   - Orthogonal initialization
4. Визуализируйте как histogram
5. Объясните связь со стабильностью обучения

```python
def analyze_weight_spectrum(model):
    """Анализирует собственные значения весов"""
    eigenvalues = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            eigvals = torch.linalg.eigvalsh(param @ param.T)
            eigenvalues.append(eigvals.detach().numpy())
    
    # TODO: визуализируйте
```

---

### **Задача 11: Data-Dependent Initialization**

**Условие:** Реализуйте инициализацию на основе статистики реальных данных.

**Требования:**
1. Реализуйте алгоритм:
   - Сделайте forward pass на нескольких батчах
   - Соберите статистику активаций (mean, std)
   - Скорректируйте веса для нормализации активаций
2. Сравните с обычной инициализацией на сложном датасете
3. Измерьте влияние на:
   - Начальный loss
   - Скорость сходимости первых 5 эпох
   - Финальное качество

```python
@torch.no_grad()
def data_dependent_init(model, data_loader, num_batches=100):
    """Инициализация на основе данных"""
    # TODO: собрать статистику активаций
    # TODO: нормализовать веса
    pass
```

---

## 💎 Заключение

### **Чек-лист правильной инициализации:**

✅ **Базовые правила:**
- [ ] ReLU активации → He (Kaiming) initialization
- [ ] Sigmoid/Tanh → Xavier (Glorot) initialization
- [ ] Bias обычно инициализируем нулями
- [ ] Никогда не инициализируем все веса одинаково (симметрия!)

✅ **С Batch Normalization:**
- [ ] Инициализация менее критична
- [ ] Xavier или He — оба работают
- [ ] BN параметры: γ=1, β=0 (по умолчанию правильно)

✅ **Без Batch Normalization:**
- [ ] Инициализация ОЧЕНЬ важна
- [ ] Обязательно проверить распределение активаций
- [ ] Рассмотреть LSUV или Fixup для глубоких сетей

✅ **Transfer Learning:**
- [ ] Копируем предобученные веса
- [ ] Новые слои: He initialization
- [ ] Fine-tuning: можем заморозить ранние слои

### **Типичные ошибки:**

❌ **Не делайте так:**
- Инициализация всех весов нулями
- Использование одинаковых весов (нарушает симметрию)
- Xavier для ReLU сетей (недостаточно)
- Забыть про инициализацию в кастомных слоях

✅ **Делайте так:**
- Проверяйте распределение активаций после инициализации
- Используйте правильную инициализацию для активации
- Логируйте нормы градиентов в начале обучения
- Используйте готовые функции PyTorch (они правильные!)

### **Практические советы:**

```python
# Быстрая проверка инициализации
@torch.no_grad()
def check_initialization(model, sample_input):
    """Проверяет качество инициализации"""
    activations = []
    x = sample_input
    
    for layer in model:
        x = layer(x)
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            activations.append(x)
    
    # Проверяем std активаций
    for i, act in enumerate(activations):
        std = act.std().item()
        print(f"Layer {i}: std={std:.4f}")
        
        if std < 0.1 or std > 10:
            print(f"⚠️ WARNING: Layer {i} has unusual std!")
```

### **Дополнительные ресурсы:**

1. **Статьи:**
   - [Understanding the difficulty of training deep networks](https://proceedings.mlr.press/v9/glorot10a.html) (Xavier)
   - [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) (He initialization)
   - [Fixup Initialization](https://arxiv.org/abs/1901.09321)
   - [LSUV Initialization](https://arxiv.org/abs/1511.06422)

2. **Инструменты:**
   - `torch.nn.init` — все методы инициализации
   - TensorBoard для визуализации распределения весов
   - Weights & Biases для tracking experiments

3. **Практика:**
   - Всегда проверяйте первые несколько итераций обучения
   - Если loss не падает — проверьте инициализацию
   - Используйте gradient checking для отладки

> **"Хорошая инициализация — это половина успеха. Плохая инициализация может сделать обучение невозможным, даже с идеальной архитектурой."**
