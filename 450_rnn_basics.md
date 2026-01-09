# Основы RNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# !pip install torch numpy matplotlib
```

---

## 🟢 Базовый уровень: От Feed-Forward к Recurrent

### 1.1 Зачем нужны рекуррентные сети?

**Проблема обычных нейросетей (Feed-Forward):**
- ❌ Фиксированный размер входа/выхода
- ❌ Нет памяти о предыдущих входах
- ❌ Не могут обрабатывать последовательности

**Примеры последовательностей:**
- **Текст:** "Я люблю машинное _____" → предсказать следующее слово
- **Временные ряды:** цены акций, температура, трафик
- **Аудио:** распознавание речи
- **Видео:** понимание действий

**RNN = Recurrent Neural Network:**
- ✅ Работает с последовательностями переменной длины
- ✅ Имеет "память" о предыдущих шагах
- ✅ Параметры переиспользуются на каждом шаге

---

### 1.2 Основная идея RNN

**Feed-Forward сеть:**
```
x → [NN] → y
```

**Рекуррентная сеть:**
```
x₁ → [RNN] → y₁
      ↓ ↑
      h₁    (hidden state)
      ↓
x₂ → [RNN] → y₂
      ↓ ↑
      h₂
      ↓
x₃ → [RNN] → y₃
```

**Ключевая идея:** Hidden state h передается от шага к шагу, накапливая информацию.

---

### 1.3 Математика RNN

**На каждом шаге t:**

```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```

Где:
- `x_t` — вход на шаге t
- `h_t` — hidden state на шаге t
- `y_t` — выход на шаге t
- `W_hh, W_xh, W_hy` — матрицы весов (общие для всех шагов!)
- `b_h, b_y` — bias

**Развернутая форма (unrolled):**

```
h₀ = 0  (инициализация)

h₁ = tanh(W_hh * h₀ + W_xh * x₁)
y₁ = W_hy * h₁

h₂ = tanh(W_hh * h₁ + W_xh * x₂)
y₂ = W_hy * h₂

h₃ = tanh(W_hh * h₂ + W_xh * x₃)
y₃ = W_hy * h₃
```

---

### 1.4 Простейшая реализация RNN

```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Веса для hidden state
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        # Веса для входа
        self.W_xh = nn.Linear(input_size, hidden_size)
        # Веса для выхода
        self.W_hy = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h_prev=None):
        """
        x: [batch_size, seq_len, input_size]
        h_prev: [batch_size, hidden_size] или None
        """
        batch_size, seq_len, _ = x.size()
        
        # Инициализация h_0
        if h_prev is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h_prev
        
        outputs = []
        
        # Проходим по последовательности
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch_size, input_size]
            
            # h_t = tanh(W_hh * h + W_xh * x_t)
            h = torch.tanh(self.W_hh(h) + self.W_xh(x_t))
            
            # y_t = W_hy * h_t
            y = self.W_hy(h)
            
            outputs.append(y)
        
        # Собираем выходы: [batch_size, seq_len, output_size]
        outputs = torch.stack(outputs, dim=1)
        
        return outputs, h

# Пример использования
input_size = 10
hidden_size = 20
output_size = 5
seq_len = 7
batch_size = 3

model = SimpleRNN(input_size, hidden_size, output_size)
x = torch.randn(batch_size, seq_len, input_size)

outputs, final_hidden = model(x)
print(f"Outputs shape: {outputs.shape}")  # [3, 7, 5]
print(f"Final hidden state shape: {final_hidden.shape}")  # [3, 20]
```

---

### 1.5 PyTorch RNN модуль

PyTorch предоставляет готовую реализацию RNN:

```python
# Создание RNN слоя
rnn = nn.RNN(
    input_size=10,
    hidden_size=20,
    num_layers=1,      # количество RNN слоев
    batch_first=True   # формат: [batch, seq, feature]
)

# Forward pass
x = torch.randn(3, 7, 10)  # [batch, seq_len, input_size]
h0 = torch.zeros(1, 3, 20)  # [num_layers, batch, hidden_size]

output, hn = rnn(x, h0)
print(f"Output: {output.shape}")  # [3, 7, 20]
print(f"Final hidden: {hn.shape}")  # [1, 3, 20]
```

---

## 🟡 Продвинутый уровень: Типы задач с RNN

### 2.1 Типы архитектур

#### **1. One-to-One (обычная нейросеть)**
```
x → [NN] → y
```
Пример: классификация изображений

#### **2. One-to-Many**
```
x → [RNN] → y₁, y₂, y₃, ...
```
Пример: генерация текста по seed, image captioning

```python
class OneToMany(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len):
        super().__init__()
        self.rnn = nn.RNN(output_size, hidden_size, batch_first=True)
        self.fc_init = nn.Linear(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.seq_len = seq_len
    
    def forward(self, x):
        # x: [batch, input_size]
        batch_size = x.size(0)
        
        # Инициализируем hidden state из входа
        h = torch.tanh(self.fc_init(x)).unsqueeze(0)  # [1, batch, hidden]
        
        # Генерируем последовательность
        outputs = []
        input_t = torch.zeros(batch_size, 1, self.fc_out.out_features, device=x.device)
        
        for t in range(self.seq_len):
            output_t, h = self.rnn(input_t, h)
            output_t = self.fc_out(output_t.squeeze(1))
            outputs.append(output_t)
            
            # Используем output как следующий вход
            input_t = output_t.unsqueeze(1)
        
        return torch.stack(outputs, dim=1)  # [batch, seq_len, output_size]
```

#### **3. Many-to-One**
```
x₁, x₂, x₃ → [RNN] → y
```
Пример: sentiment analysis, классификация последовательностей

```python
class ManyToOne(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        
        # Прогоняем всю последовательность
        output, h = self.rnn(x)  # h: [1, batch, hidden]
        
        # Используем только финальный hidden state
        h = h.squeeze(0)  # [batch, hidden]
        y = self.fc(h)    # [batch, output_size]
        
        return y

# Пример: sentiment analysis
model = ManyToOne(input_size=100, hidden_size=128, output_size=2)  # 2 класса: pos/neg
text = torch.randn(32, 50, 100)  # [batch=32, seq_len=50, embedding_dim=100]
sentiment = model(text)          # [32, 2]
```

#### **4. Many-to-Many (Synced)**
```
x₁, x₂, x₃ → [RNN] → y₁, y₂, y₃
```
Пример: POS tagging, named entity recognition

```python
class ManyToManySync(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        
        output, _ = self.rnn(x)  # [batch, seq_len, hidden]
        y = self.fc(output)      # [batch, seq_len, output_size]
        
        return y

# Пример: POS tagging
model = ManyToManySync(input_size=100, hidden_size=128, output_size=50)  # 50 POS тегов
sentence = torch.randn(32, 20, 100)  # [batch=32, words=20, embedding=100]
tags = model(sentence)               # [32, 20, 50]
```

#### **5. Many-to-Many (Encoder-Decoder)**
```
x₁, x₂, x₃ → [Encoder RNN] → context → [Decoder RNN] → y₁, y₂, y₃
```
Пример: машинный перевод, summarization

---

### 2.2 Пример: Character-Level Language Model

Генерируем текст посимвольно.

```python
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, h=None):
        # x: [batch, seq_len] индексы символов
        
        x = self.embedding(x)  # [batch, seq_len, hidden]
        
        if h is None:
            output, h = self.rnn(x)
        else:
            output, h = self.rnn(x, h)
        
        output = self.fc(output)  # [batch, seq_len, vocab_size]
        
        return output, h
    
    def generate(self, start_char_idx, length=100, temperature=1.0):
        """Генерирует текст"""
        self.eval()
        
        with torch.no_grad():
            chars = [start_char_idx]
            h = None
            
            for _ in range(length):
                x = torch.tensor([[chars[-1]]])  # [1, 1]
                output, h = self.forward(x, h)
                
                # Применяем temperature для разнообразия
                logits = output[0, -1] / temperature
                probs = F.softmax(logits, dim=0)
                
                # Сэмплируем следующий символ
                next_char = torch.multinomial(probs, 1).item()
                chars.append(next_char)
        
        return chars

# Обучение
vocab_size = 128  # ASCII символы
model = CharRNN(vocab_size, hidden_size=256, num_layers=2)

# Предположим, у нас есть текст в виде индексов символов
text_indices = torch.randint(0, vocab_size, (32, 100))  # [batch, seq_len]
target_indices = torch.randint(0, vocab_size, (32, 100))

output, _ = model(text_indices)
loss = F.cross_entropy(output.view(-1, vocab_size), target_indices.view(-1))

print(f"Loss: {loss.item():.4f}")

# Генерация
generated = model.generate(start_char_idx=ord('H'), length=200)
print(''.join(chr(c) for c in generated))
```

---

## 🟡 Продвинутый уровень: Проблемы RNN

### 3.1 Vanishing Gradient Problem

**Проблема:** Градиенты "затухают" при backpropagation через много шагов времени.

**Почему это происходит?**

При backpropagation через time (BPTT):

```
∂L/∂h₁ = ∂L/∂h_T * ∂h_T/∂h_{T-1} * ... * ∂h₂/∂h₁
```

Каждый член `∂h_t/∂h_{t-1}` включает W_hh и производную tanh:

```
∂h_t/∂h_{t-1} = W_hh * diag(1 - tanh²(·))
```

Если собственные числа W_hh < 1 → градиент → 0 (vanishing)
Если собственные числа W_hh > 1 → градиент → ∞ (exploding)

**Последствия:**
- ❌ RNN не может учиться долгосрочным зависимостям
- ❌ Информация из далеких шагов теряется
- ❌ Градиенты близки к 0, обучение застревает

**Пример:**

```python
# Демонстрация vanishing gradients
seq_len = 100
hidden_size = 50

model = SimpleRNN(input_size=10, hidden_size=hidden_size, output_size=5)
x = torch.randn(1, seq_len, 10, requires_grad=True)
target = torch.randn(1, seq_len, 5)

output, _ = model(x)
loss = F.mse_loss(output, target)
loss.backward()

# Проверяем градиенты на разных шагах
gradients = []
for t in range(seq_len):
    if x.grad is not None:
        grad_norm = x.grad[0, t].norm().item()
        gradients.append(grad_norm)

plt.plot(gradients)
plt.xlabel('Time step')
plt.ylabel('Gradient norm')
plt.title('Vanishing Gradients in RNN')
plt.show()
# Видим, что градиенты убывают для ранних шагов
```

---

### 3.2 Exploding Gradient Problem

**Проблема:** Градиенты растут экспоненциально.

**Решение: Gradient Clipping**

```python
# В процессе обучения
loss.backward()

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

optimizer.step()
```

**Как работает:**
```python
def clip_grad_norm(parameters, max_norm):
    """Масштабирует градиенты, если их norm > max_norm"""
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
```

---

### 3.3 Ограниченная память

**Проблема:** Hidden state фиксированного размера должен запоминать всю историю.

```
h_t = f(h_{t-1}, x_t)
```

h_t пытается закодировать информацию из x_1, x_2, ..., x_t, но размер h ограничен!

**Результат:**
- ❌ Информация из далеких шагов "забывается"
- ❌ Сложно моделировать длинные зависимости

**Пример:**
```
"Кошка, которая жила в большом доме с красной крышей, ___ мяукала."
```
Чтобы правильно предсказать "мяукала", нужно помнить "кошка" с начала предложения.

**Решение:** LSTM и GRU (следующая глава!)

---

## 🔴 Экспертный уровень: Bidirectional RNN

### 4.1 Зачем нужны двунаправленные RNN?

**Проблема обычного RNN:** Обрабатывает последовательность только слева направо, не видя будущего контекста.

**Пример:**
```
"Я иду в _____"
```
Обычный RNN видит только "Я иду в", не зная что после пропуска.

Но если нам доступна вся последовательность (не online обработка), можем использовать контекст с обеих сторон:
```
"Я иду в _____ с друзьями"
"с друзьями" помогает понять, что пропуск = "кино" / "парк" / etc
```

---

### 4.2 Архитектура Bidirectional RNN

**Идея:** Два RNN обрабатывают последовательность в разных направлениях:
- **Forward RNN:** x₁ → x₂ → x₃ → ... → xₜ
- **Backward RNN:** xₜ → ... → x₃ → x₂ → x₁

Затем объединяем их выходы.

```python
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        # bidirectional=True создает 2 RNN (forward и backward)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True)
        
        # Выход: [batch, seq, hidden*2] (concatenation forward и backward)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        
        output, _ = self.rnn(x)  # [batch, seq, hidden*2]
        y = self.fc(output)      # [batch, seq, output_size]
        
        return y

# Пример
model = BiRNN(input_size=100, hidden_size=128, output_size=50)
x = torch.randn(32, 20, 100)
y = model(x)
print(y.shape)  # [32, 20, 50]
```

**Преимущества:**
- ✅ Видит контекст с обеих сторон
- ✅ Лучше для задач, где доступна вся последовательность (NER, POS tagging)

**Недостатки:**
- ❌ В 2 раза больше параметров
- ❌ Не подходит для online/streaming задач (нужна вся последовательность)

---

## 📊 Сравнение архитектур

| Критерий | Feed-Forward | Simple RNN | Bidirectional RNN |
|----------|-------------|------------|-------------------|
| **Память** | Нет | Односторонняя | Двухсторонняя |
| **Последовательности** | Фиксированная длина | Переменная | Переменная |
| **Долгосрочные зависимости** | N/A | Плохо (vanishing gradients) | Плохо |
| **Скорость обучения** | Быстро | Медленно (sequential) | Очень медленно |
| **Online обработка** | Да | Да | Нет |

---

## 🎯 Ключевые выводы

1. **RNN обрабатывает последовательности** через рекуррентный hidden state

2. **Основная формула:**
   ```
   h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)
   ```

3. **5 типов архитектур:**
   - One-to-Many: генерация
   - Many-to-One: классификация последовательностей
   - Many-to-Many (synced): tagging
   - Many-to-Many (encoder-decoder): перевод

4. **Проблемы:**
   - Vanishing gradients → не учатся долгосрочным зависимостям
   - Exploding gradients → решается gradient clipping

5. **Bidirectional RNN** видит контекст с обеих сторон, но не работает online

6. **Для практики используйте LSTM/GRU** вместо vanilla RNN (следующая глава)

---

## 📚 Дополнительные материалы

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Sequence Models (Coursera)](https://www.coursera.org/learn/nlp-sequence-models)
- [PyTorch RNN Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
- [On the difficulty of training RNNs](https://arxiv.org/abs/1211.5063)
