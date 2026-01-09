### **Задачи: Transformer**

**Цель:** Реализовать и обучить Transformer модель, понять ключевые компоненты.

---

## 🟢 Базовый уровень

### **Задача 1: Positional Encoding**

Реализуйте и визуализируйте positional encoding.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # TODO: создайте positional encoding matrix
        pass
```

**Требования:** Визуализируйте encoding для разных позиций как heatmap.

---

### **Задача 2: Encoder Layer**

Реализуйте один Transformer encoder layer с self-attention и feed-forward.

**Требования:** Обучите на задаче классификации, сравните с LSTM.

---

### **Задача 3: Полный Transformer**

Реализуйте полный Transformer для machine translation.

**Требования:** Обучите на простом датасете (< 10K pairs), достигните BLEU > 20.

---

## 🟡 Продвинутый уровень

### **Задача 4: Сравнение с Seq2Seq**

Сравните Transformer и LSTM Seq2Seq на одинаковой задаче перевода.

**Измерьте:** BLEU score, training speed, inference speed.

---

### **Задача 5: Визуализация Multi-Head Attention**

Визуализируйте attention patterns для разных heads в Transformer.

**Требования:** Проанализируйте, какие linguistic patterns учит каждая голова.

---

## 🔴 Экспертный уровень

### **Задача 6: Transformer для Text Classification**

Используйте только encoder часть Transformer для классификации текста.

**Архитектура:** Encoder + [CLS] token + classifier.

---

### **Задача 7: Learning Rate Scheduling**

Реализуйте warmup + decay learning rate schedule как в оригинальной статье.

---

## 📚 Ресурсы

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
