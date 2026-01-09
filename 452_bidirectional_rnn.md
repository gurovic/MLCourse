# Bidirectional RNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# !pip install torch matplotlib
```

---

## 🟢 Базовый уровень: Двунаправленная обработка

### 1.1 Мотивация

**Проблема Unidirectional RNN:**

Обычный RNN обрабатывает последовательность только в одном направлении (слева направо):

```
"Кот сидит на _____"
      ↓   ↓   ↓
     RNN
```

RNN видит только "Кот сидит на", не зная что после пропуска.

**Но** если нам доступна вся последовательность (не online обработка), можем использовать **будущий контекст**:

```
"Кот сидит на _____ и мурлычет"
                    ^^^^^^^^^^^^^^
                    помогает понять: "диване" / "крыше"
```

---

### 1.2 Архитектура Bidirectional RNN

**Идея:** Два RNN обрабатывают последовательность в противоположных направлениях:

```
Forward RNN:  x₁ → x₂ → x₃ → x₄ → x₅
              ↓    ↓    ↓    ↓    ↓
              h₁   h₂   h₃   h₄   h₅

Backward RNN: x₅ ← x₄ ← x₃ ← x₂ ← x₁
              ↓    ↓    ↓    ↓    ↓
              h₅'  h₄'  h₃'  h₂'  h₁'

Concatenate:  [h₁, h₁'] [h₂, h₂'] [h₃, h₃'] ...
```

Для каждого timestep t получаем **двунаправленное представление**: `[h_t^→, h_t^←]`

---

### 1.3 Математика

**Forward RNN:**
```
h_t^→ = f(W^→ · h_{t-1}^→ + U^→ · x_t)
```

**Backward RNN:**
```
h_t^← = f(W^← · h_{t+1}^← + U^← · x_t)
```

**Объединенное представление:**
```
h_t = [h_t^→; h_t^←]  (concatenation)
```

**Размерность:**
- Forward hidden: [batch, hidden_size]
- Backward hidden: [batch, hidden_size]
- Concatenated: [batch, hidden_size × 2]

---

### 1.4 Реализация в PyTorch

```python
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        
        # bidirectional=True создает forward и backward RNN
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # ← ключевой параметр
        )
        
        # Output размер удваивается
        self.fc = nn.Linear(hidden_size * 2, input_size)
    
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        
        output, hidden = self.rnn(x)
        
        # output: [batch, seq_len, hidden_size*2]
        # hidden: [num_layers*2, batch, hidden_size]
        #         ^^^^^^^^^^^^^^
        #         удваивается из-за bidirectional
        
        predictions = self.fc(output)
        
        return predictions, hidden

# Пример
model = BiRNN(input_size=10, hidden_size=20, num_layers=2)
x = torch.randn(32, 50, 10)  # [batch, seq_len, features]

output, hidden = model(x)
print(f"Output: {output.shape}")   # [32, 50, 10]
print(f"Hidden: {hidden.shape}")   # [4, 32, 20] (2 layers × 2 directions)
```

---

## 🟡 Продвинутый уровень: Применения

### 2.1 Named Entity Recognition (NER) с BiLSTM

**Задача:** Определить entities в тексте (имена, места, организации).

```
"John works at Google in New York"
  PER   O    O  ORG   O  LOC  LOC
```

**Почему BiRNN?** Для слова "John" полезен контекст справа ("works at Google").

```python
class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_tags):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.bilstm = nn.LSTM(
            embed_dim, hidden_size, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        
        # Classifier для каждого токена
        self.fc = nn.Linear(hidden_size * 2, num_tags)
    
    def forward(self, x):
        # x: [batch, seq_len]
        
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        
        output, _ = self.bilstm(embedded)  # [batch, seq_len, hidden*2]
        
        logits = self.fc(output)  # [batch, seq_len, num_tags]
        
        return logits

# Использование
model = BiLSTM_NER(vocab_size=10000, embed_dim=100, hidden_size=128, num_tags=9)

# IOB2 tagging: B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC, O
sentence = torch.randint(0, 10000, (32, 20))  # [batch, seq_len]
logits = model(sentence)  # [32, 20, 9]

print(f"Predictions shape: {logits.shape}")
```

---

### 2.2 Part-of-Speech (POS) Tagging

**Задача:** Определить часть речи для каждого слова.

```
"The quick brown fox jumps"
 DET  ADJ   ADJ   NOUN VERB
```

```python
class BiGRU_POS(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_pos_tags):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bigru = nn.GRU(embed_dim, hidden_size, num_layers=2,
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_pos_tags)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.bigru(embedded)
        logits = self.fc(output)
        return logits
```

---

### 2.3 Sentiment Analysis с BiLSTM + Attention

Комбинация bidirectional processing и attention mechanism:

```python
class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, 
                             bidirectional=True)
        
        # Attention
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # x: [batch, seq_len]
        
        embedded = self.embedding(x)  # [batch, seq_len, embed]
        
        lstm_out, _ = self.bilstm(embedded)  # [batch, seq_len, hidden*2]
        
        # Attention scores
        attn_scores = self.attention(lstm_out)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden*2]
        
        logits = self.fc(context)
        
        return logits, attn_weights
```

---

## 🟡 Продвинутый уровень: Особенности

### 3.1 Преимущества Bidirectional RNN

✅ **Видит полный контекст:**
```
"The bank by the river"  → "bank" = берег реки
"The bank account"       → "bank" = банк
```
BiRNN видит слова после "bank" и правильно интерпретирует.

✅ **Лучше для sequence labeling:**
- NER, POS tagging, slot filling
- Каждый токен нуждается в контексте с обеих сторон

✅ **Улучшает качество:**
- На большинстве NLP задач BiRNN > Unidirectional RNN

---

### 3.2 Недостатки Bidirectional RNN

❌ **Нужна вся последовательность:**
- Нельзя использовать для online/streaming задач
- Нельзя для real-time generation

❌ **Вдвое больше параметров:**
- Два отдельных RNN (forward + backward)
- Больше памяти, медленнее обучение

❌ **Не подходит для autoregressive tasks:**
- Language modeling (предсказание следующего слова)
- Text generation
- Machine translation decoding

---

### 3.3 Когда использовать Bidirectional vs Unidirectional?

| Задача | Bidirectional | Unidirectional |
|--------|---------------|----------------|
| **NER** | ✅ Лучший выбор | ❌ Хуже качество |
| **POS Tagging** | ✅ Лучший выбор | ❌ Хуже качество |
| **Sentiment Analysis** | ✅ Доступна вся последовательность | ✅ Если online analysis |
| **Language Modeling** | ❌ Нарушает причинность | ✅ Единственный выбор |
| **Text Generation** | ❌ Невозможно | ✅ Единственный выбор |
| **Speech Recognition** | ✅ Offline processing | ✅ Real-time streaming |

---

## 🔴 Экспертный уровень: Продвинутые техники

### 4.1 Stacked Bidirectional LSTM

Многослойные bidirectional LSTM для увеличения capacity:

```python
class StackedBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        hidden_sizes: список размеров, например [128, 64]
        """
        super().__init__()
        
        layers = []
        
        # Первый слой
        layers.append(nn.LSTM(input_size, hidden_sizes[0], batch_first=True, 
                             bidirectional=True))
        
        # Остальные слои
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.LSTM(hidden_sizes[i-1] * 2, hidden_sizes[i], 
                                 batch_first=True, bidirectional=True))
        
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(hidden_sizes[-1] * 2, output_size)
    
    def forward(self, x):
        for lstm in self.layers:
            x, _ = lstm(x)
        
        # Последний timestep
        output = self.fc(x[:, -1, :])
        return output
```

---

### 4.2 Conditional Random Field (CRF) слой

Для sequence labeling часто добавляют CRF слой после BiLSTM:

```
Input → BiLSTM → Emission scores → CRF → Output tags
```

**Зачем CRF?**
- Учитывает **зависимости между метками**
- Например: B-PER не может идти после I-LOC

```python
# Simplified CRF (полная реализация сложнее)
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_tags):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, 
                             bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_size * 2, num_tags)
        
        # CRF transition scores
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.bilstm(embedded)
        emissions = self.hidden2tag(lstm_out)  # [batch, seq_len, num_tags]
        
        # CRF decoding (Viterbi algorithm)
        # Упрощено для примера
        return emissions
```

---

### 4.3 Практические советы

**1. Инициализация:**
```python
# Xavier initialization для BiRNN весов
def init_weights(m):
    if isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

model.apply(init_weights)
```

**2. Gradient Clipping:**
```python
# Обязательно для BiRNN (больше параметров → больше риск exploding gradients)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

**3. Dropout:**
```python
# Между слоями, НЕ между timesteps
nn.LSTM(..., dropout=0.3)  # применяется между слоями
```

---

## 📊 Сравнение архитектур

| Архитектура | Параметры | Скорость | Качество NER | Качество LM |
|-------------|-----------|----------|--------------|-------------|
| **Unidirectional RNN** | N | Быстро | Средне | Хорошо |
| **Bidirectional RNN** | 2N | Медленно | Отлично | N/A |
| **Unidirectional LSTM** | 4N | Средне | Хорошо | Отлично |
| **Bidirectional LSTM** | 8N | Медленно | Лучшее | N/A |

---

## 🎯 Ключевые выводы

1. **Bidirectional RNN** обрабатывает последовательность в обоих направлениях

2. **Преимущества:**
   - Видит полный контекст (прошлое + будущее)
   - Лучше для sequence labeling (NER, POS)
   
3. **Недостатки:**
   - Нужна вся последовательность заранее
   - Вдвое больше параметров
   - Не подходит для generation tasks

4. **Когда использовать:**
   - ✅ NER, POS tagging, slot filling
   - ✅ Sentiment analysis (если доступна вся последовательность)
   - ❌ Language modeling, text generation

5. **Практические советы:**
   - Используйте BiLSTM вместо BiRNN
   - Добавляйте dropout между слоями
   - Применяйте gradient clipping
   - Для NER добавьте CRF слой

---

## 📚 Дополнительные материалы

- [Bidirectional LSTM Paper](https://www.researchgate.net/publication/2329878_Bidirectional_Recurrent_Neural_Networks)
- [Named Entity Recognition with Bidirectional LSTM-CNNs](https://arxiv.org/abs/1511.08308)
- [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/abs/1603.01354)
- [PyTorch Bidirectional LSTM Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
