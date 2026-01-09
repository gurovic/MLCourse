# Sequence-to-Sequence

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# !pip install torch
```

---

## 🟢 Базовый уровень: Encoder-Decoder архитектура

### 1.1 Проблема: Variable-Length Input/Output

**Задачи, где input и output разной длины:**
- **Машинный перевод:** "Hello" (1 слово) → "Привет" (1 слово) или "Good morning" (2 слова) → "Доброе утро" (2 слова)
- **Summarization:** длинный текст → короткое резюме
- **Question Answering:** вопрос → ответ
- **Dialogue:** реплика → ответ

**Обычный RNN не подходит:**
```
[RNN] принимает fixed input → fixed output
```

Нужна архитектура, которая:
- Обрабатывает последовательности **любой длины** на входе
- Генерирует последовательности **любой длины** на выходе

---

### 1.2 Архитектура Seq2Seq

**Два компонента:**

1. **Encoder:** Кодирует входную последовательность в фиксированный **context vector**
2. **Decoder:** Генерирует выходную последовательность из context vector

```
Input:  x₁ x₂ x₃ x₄
         ↓  ↓  ↓  ↓
Encoder [RNN RNN RNN RNN] → context vector (c)
                              ↓
Decoder [RNN RNN RNN] → y₁ y₂ y₃
         ↓  ↓  ↓
Output:  y₁ y₂ y₃
```

**Context vector (c)** — это финальный hidden state encoder'а, содержащий информацию о всей входной последовательности.

---

### 1.3 Математика

**Encoder:**
```
h_t^enc = RNN_enc(x_t, h_{t-1}^enc)
c = h_T^enc  (последний hidden state)
```

**Decoder:**
```
h_0^dec = c  (инициализация из context)
h_t^dec = RNN_dec(y_{t-1}, h_{t-1}^dec)
y_t = softmax(W · h_t^dec)
```

---

### 1.4 Реализация в PyTorch

```python
class Encoder(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size, batch_first=True)
    
    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return hidden  # context vector

class Decoder(nn.Module):
    def __init__(self, output_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        # x: [batch, seq_len]
        # hidden: context от encoder
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        predictions = self.fc(output)
        return predictions, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: [batch, src_len]
        # tgt: [batch, tgt_len]
        
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.fc.out_features
        
        # Encode
        context = self.encoder(src)
        
        # Decode
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size)
        
        # Первый вход decoder: <SOS> token
        decoder_input = tgt[:, 0].unsqueeze(1)
        hidden = context
        
        for t in range(1, tgt_len):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t] = output.squeeze(1)
            
            # Teacher forcing: используем ground truth или prediction?
            use_teacher = random.random() < teacher_forcing_ratio
            decoder_input = tgt[:, t].unsqueeze(1) if use_teacher else output.argmax(2)
        
        return outputs
```

---

## 🟡 Продвинутый уровень: Teacher Forcing

### 2.1 Проблема Exposure Bias

**Training time:**
```
Decoder input: <SOS> "Hello" "world"  (ground truth)
Decoder output: "Привет" "мир" <EOS>
```

**Inference time:**
```
Decoder input: <SOS> <predicted> <predicted>
Decoder output: может генерировать ошибки
```

Модель **не видела своих ошибок** во время обучения!

---

### 2.2 Teacher Forcing

**Teacher Forcing:** На обучении используем **ground truth** как вход decoder, а не predictions.

```python
for t in range(1, tgt_len):
    output, hidden = decoder(decoder_input, hidden)
    
    # Teacher forcing
    teacher_force = random.random() < teacher_forcing_ratio
    
    if teacher_force:
        decoder_input = tgt[:, t].unsqueeze(1)  # ground truth
    else:
        decoder_input = output.argmax(2)  # prediction
```

**Параметр teacher_forcing_ratio:**
- 1.0: всегда используем ground truth (быстрое обучение, но exposure bias)
- 0.0: всегда используем predictions (медленное обучение, но нет bias)
- 0.5: компромисс (рекомендуется)

---

## 🟡 Продвинутый уровень: Attention Mechanism

### 3.1 Проблема: Information Bottleneck

**Проблема:** Весь input сжимается в **один** context vector фиксированного размера!

```
Long input sequence → [single vector] → output
```

Для длинных последовательностей информация теряется.

**Решение: Attention**

Decoder **обращает внимание** к разным частям входной последовательности на каждом шаге.

```
Encoder outputs: h₁ h₂ h₃ h₄
                  ↓  ↓  ↓  ↓
Decoder step t:  [attention weights: 0.1, 0.6, 0.2, 0.1]
                          ↓
                    context_t = weighted sum
```

---

### 3.2 Реализация Attention

```python
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hidden_size]
        # encoder_outputs: [batch, src_len, hidden_size]
        
        src_len = encoder_outputs.size(1)
        
        # Repeat hidden для всех encoder outputs
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Concatenate
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        
        # Attention scores
        attention = self.v(energy).squeeze(2)  # [batch, src_len]
        
        return F.softmax(attention, dim=1)

class AttentionDecoder(nn.Module):
    def __init__(self, output_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embed_dim)
        self.attention = Attention(hidden_size)
        self.rnn = nn.GRU(embed_dim + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden, encoder_outputs):
        # input: [batch, 1]
        # hidden: [1, batch, hidden]
        # encoder_outputs: [batch, src_len, hidden]
        
        embedded = self.embedding(input)  # [batch, 1, embed_dim]
        
        # Attention weights
        attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)
        
        # Context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        
        # Concatenate embedded + context
        rnn_input = torch.cat([embedded, context], dim=2)
        
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc(output.squeeze(1))
        
        return prediction, hidden, attn_weights
```

---

## 🔴 Экспертный уровень: Практические советы

### 4.1 Inference (Greedy vs Beam Search)

**Greedy Decoding:**
```python
def greedy_decode(model, src, max_len=50):
    context = model.encoder(src)
    
    decoder_input = torch.tensor([[SOS_token]])
    hidden = context
    
    decoded = []
    
    for _ in range(max_len):
        output, hidden = model.decoder(decoder_input, hidden)
        predicted = output.argmax(1)
        
        if predicted.item() == EOS_token:
            break
        
        decoded.append(predicted.item())
        decoder_input = predicted.unsqueeze(0)
    
    return decoded
```

**Beam Search:** Держим top-K гипотез, выбираем лучшую.

---

### 4.2 Практические рекомендации

1. **Используйте LSTM/GRU вместо RNN**
2. **Bidirectional encoder** для лучшего encoding
3. **Attention обязателен** для длинных последовательностей
4. **Teacher forcing ratio:** начинайте с 1.0, постепенно снижайте
5. **Gradient clipping:** обязательно (max_norm=1.0)
6. **Word embeddings:** используйте pre-trained (Word2Vec, GloVe)

---

## 🎯 Ключевые выводы

1. **Seq2Seq = Encoder + Decoder** для variable-length sequences
2. **Context vector** хранит информацию о входной последовательности
3. **Teacher forcing** ускоряет обучение, но создает exposure bias
4. **Attention** решает information bottleneck для длинных последовательностей
5. **Beam search** лучше greedy для inference

---

## 📚 Материалы

- [Seq2Seq Paper](https://arxiv.org/abs/1409.3215)
- [Attention Mechanism Paper](https://arxiv.org/abs/1409.0473)
- [PyTorch Seq2Seq Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
