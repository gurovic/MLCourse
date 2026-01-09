# LSTM и GRU

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# !pip install torch matplotlib numpy
```

---

## 🟢 Базовый уровень: Решение проблемы Vanishing Gradients

### 1.1 Проблема Vanilla RNN

**Вспомним проблему:**

Vanilla RNN:
```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)
```

**Проблемы:**
- ❌ Vanishing gradients → не учится долгосрочным зависимостям
- ❌ Информация из далеких шагов забывается
- ❌ Сложно моделировать зависимости на >10 шагов

**Пример задачи, где RNN fails:**
```
"Кот, который жил в большом доме у озера и любил спать на крыше, _____ мяукал."
```
Чтобы предсказать "мяукал", нужно помнить "кот" с начала предложения (>15 слов назад).

---

### 1.2 Основная идея LSTM

**Long Short-Term Memory (LSTM)** — специальная архитектура RNN, которая может **помнить** информацию на протяжении длинных последовательностей.

**Ключевые компоненты:**
1. **Cell state (C_t)** — "конвейерная лента" для передачи информации
2. **Gates (вентили)** — управляют потоком информации:
   - **Forget gate** — что забыть из C_{t-1}
   - **Input gate** — что добавить в C_t
   - **Output gate** — что выдать в h_t

**Визуализация:**
```
      ┌─────────────────────────────────┐
      │    Cell State (C_{t-1})         │
      └─────────────┬───────────────────┘
                    │
         ┌──────────┼──────────┐
         │  Forget  │  Input   │  Output
         │  Gate    │  Gate    │  Gate
         └──────────┴──────────┴─────────
                    │
                    ↓
              Cell State (C_t)
                    │
                    ↓
              Hidden State (h_t)
```

---

### 1.3 Математика LSTM

**На каждом шаге t:**

```python
# 1. Forget gate: что забыть из C_{t-1}
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

# 2. Input gate: что добавить в C_t
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  # candidate values

# 3. Update cell state
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

# 4. Output gate: что выдать в h_t
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

Где:
- σ — sigmoid функция (0 to 1)
- ⊙ — поэлементное умножение (element-wise)
- [h_{t-1}, x_t] — конкатенация векторов

**Ключевое отличие от RNN:**
- Cell state C_t передается **линейно** (без нелинейности!) → градиенты не затухают
- Gates управляют, что передавать, что забывать

---

### 1.4 Реализация LSTM в PyTorch

```python
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # PyTorch LSTM слой
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # [batch, seq, features]
        )
        
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x, hidden=None):
        """
        x: [batch, seq_len, input_size]
        hidden: tuple (h_0, c_0) или None
        """
        # LSTM forward
        output, (h_n, c_n) = self.lstm(x, hidden)
        
        # output: [batch, seq_len, hidden_size]
        # h_n: [num_layers, batch, hidden_size]
        # c_n: [num_layers, batch, hidden_size]
        
        # Предсказание
        predictions = self.fc(output)
        
        return predictions, (h_n, c_n)

# Пример использования
model = SimpleLSTM(input_size=10, hidden_size=20, num_layers=2)
x = torch.randn(32, 50, 10)  # [batch=32, seq_len=50, features=10]

output, (h_n, c_n) = model(x)
print(f"Output: {output.shape}")  # [32, 50, 10]
print(f"Hidden: {h_n.shape}, Cell: {c_n.shape}")  # [2, 32, 20]
```

---

## 🟡 Продвинутый уровень: GRU — упрощенная альтернатива

### 2.1 Gated Recurrent Unit (GRU)

**Проблема LSTM:** Много параметров (4 набора весов для 4 gates).

**GRU — упрощенная версия:**
- 2 gates вместо 3 (нет отдельного output gate)
- Нет отдельного cell state (используется только h_t)
- Меньше параметров → быстрее обучается

**Архитектура GRU:**
```
┌──────────┐
│  Reset   │  Update
│  Gate    │  Gate
└──────────┴─────────
      │
      ↓
  Hidden State (h_t)
```

---

### 2.2 Математика GRU

```python
# 1. Update gate: сколько информации из h_{t-1} оставить
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)

# 2. Reset gate: сколько забыть из h_{t-1}
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)

# 3. Candidate hidden state
h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)

# 4. Final hidden state (linear interpolation)
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

**Интуиция:**
- **z_t ≈ 1:** Обновить h_t новой информацией (h̃_t)
- **z_t ≈ 0:** Сохранить старую информацию (h_{t-1})
- **r_t ≈ 0:** Игнорировать прошлое при вычислении h̃_t (как reset)

---

### 2.3 Реализация GRU в PyTorch

```python
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x, hidden=None):
        """
        x: [batch, seq_len, input_size]
        hidden: h_0 или None (нет cell state!)
        """
        output, h_n = self.gru(x, hidden)
        
        # output: [batch, seq_len, hidden_size]
        # h_n: [num_layers, batch, hidden_size]
        
        predictions = self.fc(output)
        
        return predictions, h_n

# Использование
model = SimpleGRU(input_size=10, hidden_size=20, num_layers=2)
x = torch.randn(32, 50, 10)

output, h_n = model(x)
print(f"Output: {output.shape}")  # [32, 50, 10]
print(f"Hidden: {h_n.shape}")  # [2, 32, 20] (нет c_n!)
```

---

### 2.4 LSTM vs GRU: Сравнение

| Критерий | LSTM | GRU |
|----------|------|-----|
| **Gates** | 3 (forget, input, output) | 2 (update, reset) |
| **States** | 2 (hidden h, cell c) | 1 (hidden h) |
| **Параметры** | 4 × (hidden × (hidden + input)) | 3 × (hidden × (hidden + input)) |
| **Скорость** | Медленнее | Быстрее (~25% faster) |
| **Память** | Больше | Меньше |
| **Качество** | Немного лучше на очень длинных последовательностях | Сопоставимо на большинстве задач |
| **Когда использовать** | Длинные зависимости (>100 шагов) | Большинство практических задач |

**Практическое правило:**
- Начните с **GRU** (быстрее, проще)
- Если нужна максимальная точность на длинных последовательностях → **LSTM**

---

## 🟡 Продвинутый уровень: Практическое применение

### 3.1 Sentiment Analysis с LSTM

```python
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=2, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x: [batch, seq_len]
        
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        
        # LSTM
        output, (h_n, c_n) = self.lstm(embedded)
        # output: [batch, seq_len, hidden*2]
        
        # Используем последний hidden state от обоих направлений
        # h_n: [num_layers*2, batch, hidden]
        forward_hidden = h_n[-2, :, :]  # последний слой, forward
        backward_hidden = h_n[-1, :, :]  # последний слой, backward
        
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        hidden = self.dropout(hidden)
        
        logits = self.fc(hidden)
        return logits

# Обучение
model = SentimentLSTM(vocab_size=10000, embed_dim=100, hidden_size=128, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Пример батча
x = torch.randint(0, 10000, (32, 50))  # [batch, seq_len]
y = torch.randint(0, 2, (32,))  # [batch]

logits = model(x)
loss = criterion(logits, y)
print(f"Loss: {loss.item():.4f}")
```

---

### 3.2 Time Series Forecasting с GRU

```python
class TimeSeriesGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        
        output, h_n = self.gru(x)
        
        # Предсказываем следующий шаг на основе последнего hidden
        prediction = self.fc(h_n[-1])  # [batch, output_size]
        
        return prediction

# Пример: предсказание температуры
model = TimeSeriesGRU(input_size=1, hidden_size=64, num_layers=2, output_size=1)

# История: последние 24 часа
x = torch.randn(32, 24, 1)  # [batch, seq=24 hours, features=1]

# Предсказание: следующий час
prediction = model(x)  # [32, 1]
print(f"Prediction shape: {prediction.shape}")
```

---

### 3.3 Character-Level Text Generation с LSTM

```python
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden
    
    def generate(self, start_idx, length=500, temperature=1.0):
        """Генерирует текст"""
        self.eval()
        
        generated = [start_idx]
        hidden = None
        
        with torch.no_grad():
            for _ in range(length):
                x = torch.tensor([[generated[-1]]])
                logits, hidden = self.forward(x, hidden)
                
                # Sampling with temperature
                logits = logits[0, -1] / temperature
                probs = F.softmax(logits, dim=0)
                next_idx = torch.multinomial(probs, 1).item()
                
                generated.append(next_idx)
        
        return generated

# Использование
vocab_size = 128  # ASCII
model = CharLSTM(vocab_size, embed_dim=128, hidden_size=256, num_layers=2)

# Генерация
text_indices = model.generate(start_idx=ord('T'), length=200, temperature=0.8)
text = ''.join(chr(idx) for idx in text_indices if idx < 128)
print(text)
```

---

## 🔴 Экспертный уровень: Глубокое понимание

### 4.1 Почему LSTM решает Vanishing Gradients?

**Gradient flow в vanilla RNN:**
```
∂L/∂h_1 = ∂L/∂h_T * ∏(t=2 to T) ∂h_t/∂h_{t-1}

где ∂h_t/∂h_{t-1} включает W_hh и tanh'
```

Если собственные числа W_hh < 1 → произведение → 0.

**Gradient flow в LSTM:**
```
∂L/∂C_1 = ∂L/∂C_T * ∏(t=2 to T) ∂C_t/∂C_{t-1}

где C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

∂C_t/∂C_{t-1} = f_t  (элементы от 0 до 1, но НЕТ умножения на матрицу!)
```

**Ключевое отличие:**
- Cell state C передается через **поэлементное умножение**, а не матричное
- Градиенты могут течь без затухания, если forget gate f_t ≈ 1
- LSTM **учится** контролировать forget gate → решает, когда сохранять информацию

---

### 4.2 Визуализация работы Gates

```python
def visualize_gates(model, sequence):
    """Визуализирует активации gates в LSTM"""
    
    model.eval()
    
    # Hook для извлечения gate activations
    gate_activations = []
    
    def hook_fn(module, input, output):
        # Извлекаем gates из LSTM
        # output: (output, (h_n, c_n))
        gate_activations.append(output[1])  # (h_n, c_n)
    
    handle = model.lstm.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(sequence)
    
    handle.remove()
    
    # Визуализация
    # TODO: plot gate activations over time
    pass

# Пример использования
model = SimpleLSTM(input_size=10, hidden_size=20)
sequence = torch.randn(1, 50, 10)  # [1, seq_len=50, features=10]

visualize_gates(model, sequence)
```

---

### 4.3 Stacked LSTM

Многослойные LSTM для увеличения capacity:

```python
class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        hidden_sizes: список размеров для каждого слоя, например [128, 64, 32]
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Первый слой
        self.layers.append(nn.LSTM(input_size, hidden_sizes[0], batch_first=True))
        
        # Средние слои
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], batch_first=True))
        
        # Output layer
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
    
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        
        for lstm_layer in self.layers:
            x, _ = lstm_layer(x)
            # x: [batch, seq_len, hidden_size_i]
        
        # Используем последний timestep
        output = self.fc(x[:, -1, :])
        
        return output

# Пример: 3-layer LSTM
model = StackedLSTM(input_size=50, hidden_sizes=[128, 64, 32], output_size=10)
x = torch.randn(32, 20, 50)
output = model(x)
print(output.shape)  # [32, 10]
```

---

### 4.4 Peephole Connections

Модификация LSTM, где gates видят cell state:

```python
# Стандартный LSTM:
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

# Peephole LSTM:
f_t = σ(W_f · [h_{t-1}, x_t] + W_cf · C_{t-1} + b_f)
#                               ^^^^^^^^^^^^^^
#                               peephole connection
```

**Идея:** Gates могут принимать решения на основе текущего cell state.

**В PyTorch:** Нет встроенной поддержки, нужно реализовывать вручную.

---

### 4.5 LSTM Variants: сравнение

| Вариант | Особенность | Когда использовать |
|---------|-------------|-------------------|
| **Vanilla LSTM** | 3 gates, cell state | Стандартный выбор |
| **Peephole LSTM** | Gates видят C_t | Очень специфичные задачи |
| **GRU** | 2 gates, без C_t | Быстрее, меньше параметров |
| **Bidirectional LSTM** | 2 LSTM (forward + backward) | Когда доступна вся последовательность |
| **Stacked LSTM** | Несколько слоев | Сложные зависимости, много данных |

---

## 📊 Сравнительная таблица

| Критерий | Vanilla RNN | LSTM | GRU |
|----------|-------------|------|-----|
| **Vanishing Gradients** | Сильная проблема | Решена | Решена |
| **Долгосрочная память** | 5-10 шагов | 100+ шагов | 50-100 шагов |
| **Параметры** | Минимум | Максимум | Средне |
| **Скорость обучения** | Быстро | Медленно | Средне |
| **Память GPU** | Минимум | Максимум | Средне |
| **Интерпретируемость** | Простая | Сложная (gates) | Средняя |
| **Когда использовать** | Короткие зависимости | Длинные зависимости | Большинство задач |

---

## 🎯 Ключевые выводы

1. **LSTM решает vanishing gradients** через cell state и gates

2. **3 gate в LSTM:**
   - Forget gate — что забыть
   - Input gate — что добавить
   - Output gate — что выдать

3. **GRU — упрощенная альтернатива:**
   - 2 gates вместо 3
   - Быстрее на ~25%
   - Качество сопоставимо с LSTM

4. **Практические рекомендации:**
   - Начинайте с GRU
   - Переходите на LSTM только если нужна максимальная точность
   - Используйте bidirectional для задач, где доступна вся последовательность

5. **Типичные гиперпараметры:**
   - hidden_size: 128-512
   - num_layers: 2-3
   - dropout: 0.2-0.5 (между слоями)

6. **LSTM/GRU побеждают Vanilla RNN** на всех задачах с долгосрочными зависимостями

---

## 📚 Дополнительные материалы

- [Understanding LSTM Networks (Colah's Blog)](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [LSTM Paper (1997)](http://www.bioinf.jku.at/publications/older/2604.pdf)
- [GRU Paper (2014)](https://arxiv.org/abs/1406.1078)
- [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Empirical Evaluation of Gated RNNs](https://arxiv.org/abs/1412.3555)
- [LSTM: A Search Space Odyssey](https://arxiv.org/abs/1503.04069)
