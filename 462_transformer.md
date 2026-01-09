# Transformer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# !pip install torch
```

---

## 🟢 Базовый уровень: Архитектура Transformer

### 1.1 Революция в NLP

**До Transformer:** RNN/LSTM доминировали, но медленные (sequential processing).

**Transformer (2017):** "Attention is All You Need"
- ✅ Полностью на attention (нет RNN!)
- ✅ Параллельная обработка (быстро!)
- ✅ Лучше моделирует long-range dependencies
- ✅ Стал основой для BERT, GPT, T5, и др.

---

### 1.2 Основные компоненты

**Transformer = Encoder + Decoder**

```
Input → Positional Encoding
  ↓
Encoder (N layers):
  - Multi-Head Self-Attention
  - Feed-Forward Network
  ↓
Decoder (N layers):
  - Masked Multi-Head Self-Attention
  - Multi-Head Cross-Attention
  - Feed-Forward Network
  ↓
Output
```

---

### 1.3 Positional Encoding

**Проблема:** Attention не учитывает порядок слов!

**Решение:** Добавляем positional encoding к embeddings.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
```

---

### 1.4 Encoder Layer

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        
        # Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-Attention with residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # Feed-Forward with residual
        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        
        return x
```

---

## 🟡 Продвинутый уровень: Полная реализация

### 2.1 Decoder Layer

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Masked Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        
        # Cross-Attention (attend to encoder)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        
        # Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked Self-Attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # Cross-Attention
        attn_out, _ = self.cross_attn(x, encoder_output, encoder_output, attn_mask=src_mask)
        x = x + self.dropout2(attn_out)
        x = self.norm2(x)
        
        # Feed-Forward
        ff_out = self.ff(x)
        x = x + self.dropout3(ff_out)
        x = self.norm3(x)
        
        return x
```

---

### 2.2 Полный Transformer

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                 d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embedding + Positional Encoding
        src = self.dropout(self.pos_encoding(
            self.src_embedding(src) * math.sqrt(self.d_model)))
        tgt = self.dropout(self.pos_encoding(
            self.tgt_embedding(tgt) * math.sqrt(self.d_model)))
        
        # Encoder
        encoder_output = src
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
        
        # Decoder
        decoder_output = tgt
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)
        
        # Output projection
        output = self.fc_out(decoder_output)
        
        return output
```

---

## 🎯 Ключевые выводы

1. **Transformer** полностью на attention (нет RNN!)
2. **Positional Encoding** добавляет информацию о позициях
3. **Multi-Head Attention** — ключевой компонент
4. **Residual connections** + **Layer Normalization** стабилизируют обучение
5. **Параллелизация** → быстрое обучение

---

## 📚 Материалы

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
