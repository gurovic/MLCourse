### **Задачи: Attention механизм**

**Цель:** Понять и реализовать attention mechanism, self-attention и multi-head attention.

---

## 🟢 Базовый уровень

### **Задача 1: Реализация базового Attention**

Реализуйте простой attention механизм для seq2seq модели.

```python
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # TODO: реализуйте score function
        pass
    
    def forward(self, decoder_hidden, encoder_outputs):
        # TODO: вычислите attention weights
        # TODO: создайте context vector
        pass
```

**Требования:** Обучите seq2seq с attention на задаче перевода, сравните с baseline без attention.

---

### **Задача 2: Визуализация Attention Weights**

Визуализируйте attention weights как heatmap для анализа alignment между source и target.

**Требования:** Создайте heatmap для 5 примеров перевода, проанализируйте паттерны.

---

### **Задача 3: Self-Attention с нуля**

Реализуйте self-attention layer без использования готовых библиотек.

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        # TODO: реализуйте Q, K, V computation
        # TODO: scaled dot-product attention
        pass
```

---

## 🟡 Продвинутый уровень

### **Задача 4: Multi-Head Attention**

Реализуйте multi-head attention с несколькими параллельными attention heads.

**Требования:** Сравните 1, 4, 8 heads по качеству на sentiment analysis.

---

### **Задача 5: Attention в BiLSTM**

Добавьте attention к BiLSTM для улучшения sequence classification.

**Измерьте:** accuracy, attention interpretability.

---

## 🔴 Экспертный уровень

### **Задача 6: Анализ Attention Patterns**

Проанализируйте, какие паттерны учат разные attention heads в multi-head attention.

**Требования:** Визуализируйте attention для каждой головы, определите их специализацию.

---

### **Задача 7: Sparse Attention**

Реализуйте sparse attention для обработки длинных последовательностей.

---

## 📚 Ресурсы

- [Attention Paper](https://arxiv.org/abs/1409.0473)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
