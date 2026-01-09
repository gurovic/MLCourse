### **Задачи: Обратное распространение ошибки**

**Цель:** Понять механику backpropagation, научиться вычислять градиенты вручную и проверять их программно.

---

## 🟢 Базовый уровень

### **Задача 1: Ручное вычисление градиентов для простой сети**

**Условие:** Вычислите градиенты вручную для двухслойной сети.

**Дано:**
- Сеть: x → w1 → ReLU → w2 → y
- x = 2.0, w1 = 0.5, w2 = 1.5
- Целевое значение: target = 5.0
- Loss: MSE = (y - target)²

**Требования:**
1. Вычислите forward pass вручную
2. Вычислите градиенты ∂L/∂w2 и ∂L/∂w1 по chain rule
3. Реализуйте это в PyTorch с `requires_grad=True`
4. Сравните ваши вычисления с PyTorch градиентами

**Ожидаемый результат:** Совпадение ручных вычислений с PyTorch (точность до 4 знаков)

```python
import torch

# Шаблон
x = torch.tensor([2.0], requires_grad=False)
w1 = torch.tensor([0.5], requires_grad=True)
w2 = torch.tensor([1.5], requires_grad=True)
target = torch.tensor([5.0])

# TODO: реализуйте forward pass
# TODO: вычислите loss
# TODO: вызовите backward()
# TODO: сравните с ручными вычислениями
```

---

### **Задача 2: Визуализация computational graph**

**Условие:** Создайте и визуализируйте вычислительный граф для простой функции.

**Требования:**
1. Реализуйте функцию: z = (x² + y²) * sin(x*y)
2. Используйте `torchviz` для визуализации графа
3. Вычислите градиенты ∂z/∂x и ∂z/∂y при x=2, y=3
4. Объясните структуру графа

```python
from torchviz import make_dot

# TODO: создайте граф и визуализируйте его
```

**Вопросы:**
- Сколько промежуточных узлов в графе?
- Какие операции выполняются первыми?

---

### **Задача 3: Gradient checking**

**Условие:** Реализуйте численную проверку градиентов для простой функции.

**Требования:**
1. Реализуйте функцию численного градиента:
   ```
   grad_numerical = (f(x + h) - f(x - h)) / (2 * h)
   ```
2. Сравните с аналитическим градиентом из PyTorch
3. Используйте h = 1e-5
4. Функция: f(x) = x³ - 2x² + 5x - 3

**Ожидаемый результат:** Разница < 1e-6

```python
def numerical_gradient(f, x, h=1e-5):
    """Вычисляет численный градиент функции f в точке x"""
    # TODO: реализуйте
    pass

def analytical_gradient(f, x):
    """Вычисляет градиент через PyTorch autograd"""
    # TODO: реализуйте
    pass
```

---

## 🟡 Продвинутый уровень

### **Задача 4: Backpropagation через различные функции активации**

**Условие:** Проанализируйте поведение градиентов в сетях с разными активациями.

**Требования:**
1. Создайте 3-слойную сеть с 3 вариантами активаций: Sigmoid, Tanh, ReLU
2. Для входа x ∈ [-5, 5] постройте графики:
   - Значения активаций на каждом слое
   - Величины градиентов на каждом слое
3. Обучите сеть на простой задаче (XOR) и отслеживайте градиенты
4. Определите, в какой сети градиенты "затухают" быстрее

**Вопросы:**
- Почему в Sigmoid/Tanh возникает vanishing gradient?
- Как ReLU решает эту проблему?
- Что происходит с dead neurons в ReLU?

---

### **Задача 5: Анализ vanishing/exploding gradients**

**Условие:** Продемонстрируйте проблему затухающих/взрывающихся градиентов.

**Требования:**
1. Создайте глубокую сеть (10 слоев) с Sigmoid активацией
2. Инициализируйте веса нормально с σ = 1.0
3. Сделайте forward pass и backward pass
4. Постройте график: норма градиента vs номер слоя
5. Повторите эксперимент с правильной инициализацией (Xavier)

**Ожидаемый результат:**
- С плохой инициализацией: градиенты → 0 или → ∞
- С Xavier: градиенты стабильны

```python
class DeepNet(nn.Module):
    def __init__(self, num_layers=10, init_type='normal'):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layer = nn.Linear(64, 64)
            if init_type == 'normal':
                # TODO: плохая инициализация
            elif init_type == 'xavier':
                # TODO: Xavier инициализация
            layers.append(layer)
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
```

---

### **Задача 6: Backpropagation через RNN слой**

**Условие:** Вычислите градиенты для простой рекуррентной ячейки.

**Требования:**
1. Реализуйте простую RNN ячейку вручную:
   ```
   h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
   ```
2. Сделайте forward pass для последовательности длины 3
3. Вычислите loss = ||h_3 - target||²
4. Реализуйте BPTT (backpropagation through time) вручную
5. Сравните с PyTorch autograd

**Дополнительно:**
- Покажите, как градиенты распространяются во времени
- Объясните, почему появляется vanishing gradient в длинных последовательностях

---

## 🔴 Экспертный уровень

### **Задача 7: Кастомная autograd функция**

**Условие:** Реализуйте собственную дифференцируемую функцию с custom backward pass.

**Требования:**
1. Создайте класс, наследующийся от `torch.autograd.Function`
2. Реализуйте функцию: SoftThreshold(x, λ) = sign(x) * max(|x| - λ, 0)
3. Реализуйте forward и backward методы
4. Проверьте корректность через `torch.autograd.gradcheck`
5. Используйте в простой сети для L1 регуляризации

```python
class SoftThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lambd):
        # TODO: реализуйте forward
        pass
    
    @staticmethod
    def backward(ctx, grad_output):
        # TODO: реализуйте backward
        pass
```

**Проверка:**
```python
from torch.autograd import gradcheck

test = gradcheck(SoftThreshold.apply, (torch.randn(20, 20, dtype=torch.double, requires_grad=True), 0.5))
print(f"Gradient check passed: {test}")
```

---

### **Задача 8: Визуализация backward pass**

**Условие:** Создайте интерактивную визуализацию распространения градиентов в сети.

**Требования:**
1. Постройте 4-слойную сеть
2. После backward pass извлеките градиенты всех параметров
3. Создайте heatmap визуализацию:
   - Размер каждого прямоугольника = размер слоя
   - Цвет = величина градиента (log scale)
4. Покажите, как градиенты меняются при разных:
   - Learning rates
   - Batch sizes
   - Инициализациях

**Используйте:**
- `matplotlib` или `seaborn` для heatmaps
- `register_hook` для мониторинга градиентов

```python
def visualize_gradients(model):
    """Визуализирует градиенты всех слоев"""
    gradients = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            # TODO: собрать градиенты
            pass
    # TODO: создать heatmap
```

---

### **Задача 9: Второй порядок - Hessian**

**Условие:** Вычислите матрицу Гессе (вторые производные) для простой функции.

**Требования:**
1. Функция: f(x, y) = x²y + xy² + sin(x*y)
2. Вычислите все вторые производные:
   - ∂²f/∂x²
   - ∂²f/∂y²
   - ∂²f/∂x∂y
3. Реализуйте через двойное применение autograd
4. Визуализируйте Hessian как heatmap
5. Найдите седловые точки

```python
def compute_hessian(f, x, y):
    """Вычисляет матрицу Гессе для функции f(x, y)"""
    # TODO: используйте torch.autograd.grad дважды
    pass
```

**Вопросы:**
- Что говорит Hessian о кривизне функции?
- Как можно использовать Hessian для оптимизации?

---

### **Задача 10: Memory-efficient backprop (Gradient Checkpointing)**

**Условие:** Реализуйте gradient checkpointing для экономии памяти.

**Требования:**
1. Создайте очень глубокую сеть (50+ слоев)
2. Измерьте потребление памяти при обычном backprop
3. Реализуйте checkpointing: сохраняйте активации только для части слоев
4. Измерьте:
   - Экономию памяти
   - Увеличение времени обучения
5. Найдите оптимальный баланс (checkpointing каждые N слоев)

**Используйте:**
```python
from torch.utils.checkpoint import checkpoint

class CheckpointedLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x):
        return checkpoint(self.layer, x)
```

---

### **Задача 11: Automatic Mixed Precision (AMP) и градиенты**

**Условие:** Исследуйте влияние mixed precision на backpropagation.

**Требования:**
1. Обучите модель в FP32 (обычная точность)
2. Обучите ту же модель с AMP (FP16/FP32 mixed)
3. Сравните:
   - Величины градиентов (есть ли underflow?)
   - Скорость обучения
   - Потребление памяти
   - Финальную accuracy
4. Объясните, зачем нужен loss scaling

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# TODO: реализуйте training loop с AMP
```

---

## 💎 Заключение

### **Чек-лист понимания backpropagation:**

✅ **Базовые концепции:**
- [ ] Понимаю chain rule и его применение
- [ ] Могу вычислить градиенты вручную для простых сетей
- [ ] Умею использовать autograd в PyTorch

✅ **Продвинутое:**
- [ ] Понимаю проблемы vanishing/exploding gradients
- [ ] Знаю, как правильная инициализация помогает
- [ ] Умею анализировать поведение градиентов в сети

✅ **Экспертное:**
- [ ] Могу реализовать custom autograd функции
- [ ] Понимаю BPTT и градиенты в рекуррентных сетях
- [ ] Знаю техники оптимизации памяти (checkpointing)
- [ ] Разбираюсь в mixed precision training

### **Дополнительные ресурсы:**

1. **Статьи:**
   - "Backpropagation Algorithm" by Geoffrey Hinton
   - "Understanding the difficulty of training deep feedforward neural networks" (Xavier init)
   - "Gradient-Based Learning Applied to Document Recognition" (LeNet paper)

2. **Инструменты:**
   - `torchviz` для визуализации computational graphs
   - `torch.autograd.gradcheck` для проверки градиентов
   - TensorBoard для мониторинга градиентов во время обучения

3. **Практика:**
   - Реализуйте simple MLP с нуля (без nn.Module)
   - Посмотрите исходный код PyTorch autograd
   - Попробуйте другие фреймворки (JAX, TensorFlow) для сравнения

### **Типичные ошибки:**

❌ **Не делайте так:**
- Не забывайте вызывать `zero_grad()` перед backward
- Не храните ссылки на промежуточные тензоры без необходимости
- Не используйте inplace операции с тензорами, требующими градиент
- Не пытайтесь вызвать backward дважды без `retain_graph=True`

✅ **Делайте так:**
- Используйте `with torch.no_grad()` для inference
- Detach тензоры, когда градиенты не нужны
- Проверяйте градиенты численно при отладке
- Мониторьте нормы градиентов во время обучения
