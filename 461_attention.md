# Attention механизм

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# !pip install torch matplotlib numpy
```

---

## 🟢 Базовый уровень: Проблема и решение

### 1.1 Проблема Seq2Seq без Attention

**Information Bottleneck:**

В классическом seq2seq весь входной текст сжимается в **один** context vector фиксированного размера:

```
Input:  "The cat sat on the mat"
        ↓  ↓  ↓  ↓  ↓  ↓
Encoder [RNN RNN RNN RNN RNN RNN] → context [h_final]
                                      ↓
Decoder                        [RNN RNN RNN]
                                ↓  ↓  ↓
Output:                        "Le chat"
```

**Проблемы:**
- ❌ Длинные последовательности → информация теряется
- ❌ Decoder не может "вернуться" к нужной части input
- ❌ Все части input одинаково важны (неправильно!)

---

### 1.2 Основная идея Attention

**Attention** позволяет decoder "обращать внимание" к **разным частям** входной последовательности на каждом шаге.

```
Input:  h₁  h₂  h₃  h₄  h₅  (encoder outputs)
         ↓   ↓   ↓   ↓   ↓
Decoder step 1: attention weights [0.1, 0.6, 0.2, 0.05, 0.05]
                         ↓
                    context₁ = weighted sum

Decoder step 2: attention weights [0.05, 0.1, 0.15, 0.5, 0.2]
                         ↓
                    context₂ = weighted sum
```

**Ключевая идея:** На каждом шаге decoder **динамически выбирает**, какие части input важны.

---

### 1.3 Математика Attention

**Входы:**
- `h_dec` — текущий hidden state decoder
- `H_enc` — все hidden states encoder [h₁, h₂, ..., h_T]

**Шаг 1: Score (энергия)**
```
e_t = score(h_dec, h_t)  для каждого h_t в H_enc
```

**Популярные score functions:**
- **Dot-product:** `score(h_dec, h_t) = h_dec · h_t`
- **General:** `score(h_dec, h_t) = h_dec^T W h_t`
- **Concat:** `score(h_dec, h_t) = v^T tanh(W[h_dec; h_t])`

**Шаг 2: Attention weights (нормализация)**
```
α = softmax(e) = [α₁, α₂, ..., α_T]
```

**Шаг 3: Context vector**
```
context = Σ α_t * h_t
```

**Шаг 4: Использование в decoder**
```
output = f(context, h_dec, previous_output)
```

---

### 1.4 Реализация Attention

```python
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Score function: concat + tanh
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        """
        hidden: [batch, hidden_size] — decoder hidden state
        encoder_outputs: [batch, seq_len, hidden_size]
        """
        seq_len = encoder_outputs.size(1)
        
        # Repeat hidden для всех encoder outputs
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # hidden: [batch, seq_len, hidden_size]
        
        # Concatenate
        energy = torch.cat([hidden, encoder_outputs], dim=2)
        # energy: [batch, seq_len, hidden_size*2]
        
        # Score
        energy = torch.tanh(self.attn(energy))
        # energy: [batch, seq_len, hidden_size]
        
        attention = self.v(energy).squeeze(2)
        # attention: [batch, seq_len]
        
        # Softmax
        return F.softmax(attention, dim=1)

# Использование
attention = AttentionLayer(hidden_size=256)

decoder_hidden = torch.randn(32, 256)  # [batch, hidden]
encoder_outputs = torch.randn(32, 20, 256)  # [batch, seq_len, hidden]

attn_weights = attention(decoder_hidden, encoder_outputs)
print(attn_weights.shape)  # [32, 20]

# Context vector
context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
print(context.shape)  # [32, 1, 256]
```

---

## 🟡 Продвинутый уровень: Self-Attention

### 2.1 От Encoder-Decoder к Self-Attention

**Encoder-Decoder Attention:** Decoder смотрит на encoder outputs.

**Self-Attention:** Последовательность смотрит **сама на себя**!

```
Input:  "The cat sat on the mat"

Self-Attention для "sat":
- Насколько "sat" связан с "The"? → 0.05
- Насколько "sat" связан с "cat"? → 0.7  (субъект действия!)
- Насколько "sat" связан с "sat"? → 0.1
- Насколько "sat" связан с "on"? → 0.05
- Насколько "sat" связан с "the"? → 0.05
- Насколько "sat" связан с "mat"? → 0.05
```

**Результат:** Каждое слово получает контекстное представление с учетом **всех** других слов.

---

### 2.2 Математика Self-Attention

**Три компонента: Query, Key, Value**

Для каждого input x_i создаем:
```
Q_i = W_Q * x_i  (query — "что я ищу?")
K_i = W_K * x_i  (key — "что я предлагаю?")
V_i = W_V * x_i  (value — "что я даю?")
```

**Attention formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

где d_k — размерность key
```

**Почему √d_k?** Стабилизирует gradient (масштабирующий фактор).

---

### 2.3 Реализация Self-Attention

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Linear projections для Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([embed_dim]))
    
    def forward(self, x):
        """
        x: [batch, seq_len, embed_dim]
        """
        batch_size = x.size(0)
        
        # Generate Q, K, V
        Q = self.query(x)  # [batch, seq_len, embed_dim]
        K = self.key(x)    # [batch, seq_len, embed_dim]
        V = self.value(x)  # [batch, seq_len, embed_dim]
        
        # Attention scores: QK^T
        scores = torch.bmm(Q, K.transpose(1, 2))
        # scores: [batch, seq_len, seq_len]
        
        # Scale
        scores = scores / self.scale
        
        # Softmax
        attention = F.softmax(scores, dim=-1)
        # attention: [batch, seq_len, seq_len]
        
        # Apply attention to values
        output = torch.bmm(attention, V)
        # output: [batch, seq_len, embed_dim]
        
        return output, attention

# Использование
self_attn = SelfAttention(embed_dim=128)
x = torch.randn(32, 20, 128)  # [batch, seq_len, embed]

output, attn_weights = self_attn(x)
print(output.shape)        # [32, 20, 128]
print(attn_weights.shape)  # [32, 20, 20] — attention между всеми парами слов
```

---

## 🟡 Продвинутый уровень: Multi-Head Attention

### 3.1 Зачем несколько "голов"?

**Проблема single-head attention:** Может фокусироваться только на **одном** типе отношений.

**Multi-Head Attention:** Несколько параллельных attention механизмов → разные головы учат разные паттерны!

```
Head 1: синтаксические отношения (субъект-глагол)
Head 2: семантические отношения (cat-animal)
Head 3: позиционные отношения (соседние слова)
...
Head h: другие паттерны
```

---

### 3.2 Архитектура Multi-Head Attention

```
Input → [Head 1] → output₁
      → [Head 2] → output₂
      → ...
      → [Head h] → outputₕ

Concatenate [output₁, output₂, ..., outputₕ]
      ↓
  Linear projection
      ↓
  Final output
```

**Формула:**
```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) W^O

где head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

---

### 3.3 Реализация Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    
    def forward(self, query, key, value, mask=None):
        """
        query, key, value: [batch, seq_len, embed_dim]
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Reshape для multi-head: [batch, seq_len, num_heads, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [batch, num_heads, seq_len, head_dim]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: [batch, num_heads, seq_len, seq_len]
        
        # Mask (optional)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
        
        # Softmax
        attention = F.softmax(scores, dim=-1)
        
        # Apply to values
        x = torch.matmul(attention, V)
        # x: [batch, num_heads, seq_len, head_dim]
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.embed_dim)
        # x: [batch, seq_len, embed_dim]
        
        # Final linear
        output = self.fc_out(x)
        
        return output, attention

# Использование
mha = MultiHeadAttention(embed_dim=512, num_heads=8)
x = torch.randn(32, 20, 512)

output, attn = mha(x, x, x)  # Self-attention
print(output.shape)  # [32, 20, 512]
print(attn.shape)    # [32, 8, 20, 20] — attention для каждой головы
```

---

## 🔴 Экспертный уровень: Визуализация Attention

### 4.1 Attention Heatmap

```python
def visualize_attention(attention_weights, source_words, target_words):
    """
    Визуализирует attention как heatmap
    attention_weights: [target_len, source_len]
    """
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, xticklabels=source_words, 
                yticklabels=target_words, cmap='viridis')
    plt.xlabel('Source')
    plt.ylabel('Target')
    plt.title('Attention Weights')
    plt.show()

# Пример
source = ["The", "cat", "sat", "on", "mat"]
target = ["Le", "chat", "assis"]
attn = torch.rand(3, 5)  # [target_len, source_len]

visualize_attention(attn.numpy(), source, target)
```

---

## 🎯 Ключевые выводы

1. **Attention** решает information bottleneck в seq2seq
2. **Self-Attention** позволяет моделировать зависимости между всеми элементами последовательности
3. **Multi-Head Attention** учит разные типы отношений параллельно
4. **Scaled Dot-Product** — эффективная score function
5. **Attention weights** можно визуализировать для интерпретируемости

---

## 📚 Материалы

- [Attention Paper (Bahdanau et al.)](https://arxiv.org/abs/1409.0473)
- [Attention is All You Need (Transformer)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
