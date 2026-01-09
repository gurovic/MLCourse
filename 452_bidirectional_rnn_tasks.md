### **Задачи: Bidirectional RNN**

**Цель:** Научиться применять bidirectional RNN для задач sequence labeling, понять преимущества и ограничения двунаправленной обработки.

---

## 🟢 Базовый уровень

### **Задача 1: NER с BiLSTM**

Реализуйте Named Entity Recognition с BiLSTM для распознавания имен, мест и организаций.

```python
class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_size, num_layers=2,
                             batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_tags)
    
    def forward(self, x):
        # TODO: реализуйте forward pass
        pass
```

**Требования:** Обучите на CoNLL-2003 или синтетических данных, достигните F1 > 0.75.

---

### **Задача 2: POS Tagging с BiGRU**

Определите части речи для каждого слова в предложении используя BiGRU.

```python
class BiGRU_POS(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_pos):
        super().__init__()
        # TODO: создайте BiGRU архитектуру
        pass
```

**Требования:** Сравните BiGRU с Unidirectional GRU по качеству и скорости.

---

### **Задача 3: Сравнение Uni vs Bi на Sentiment**

Сравните unidirectional и bidirectional LSTM на задаче sentiment analysis.

**Измерьте:** accuracy, inference time, параметры. **Визуализируйте:** learning curves обеих моделей.

---

## 🟡 Продвинутый уровень

### **Задача 4: BiLSTM + Attention**

Добавьте attention mechanism к BiLSTM для улучшения качества на sentiment analysis.

```python
class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.bilstm(embedded)
        
        # Attention
        attn_scores = self.attention(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        return self.fc(context), attn_weights
```

**Требования:** Визуализируйте attention weights для тестовых примеров.

---

### **Задача 5: Slot Filling для Dialogue Systems**

Реализуйте slot filling (извлечение информации из запросов пользователей) с BiLSTM.

**Пример:**
```
Input:  "Book a flight from Moscow to Paris on Friday"
Slots:  O    O  O     O    B-from I-from O  B-to  O  B-date
```

**Требования:** Создайте датасет запросов, обучите BiLSTM, оцените точность extraction.

---

### **Задача 6: Stacked Bidirectional LSTM**

Реализуйте многослойную bidirectional LSTM и сравните с single-layer.

```python
class StackedBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        # TODO: создайте несколько BiLSTM слоев
        pass
```

**Эксперимент:** Сравните 1, 2, 3 слоя по качеству и overfitting.

---

## 🔴 Экспертный уровень

### **Задача 7: BiLSTM-CRF для NER**

Добавьте CRF слой после BiLSTM для улучшения consistency predictions.

**Требования:** Реализуйте CRF с Viterbi decoding, сравните BiLSTM vs BiLSTM-CRF.

---

### **Задача 8: Multi-task BiLSTM**

Обучите одну BiLSTM на нескольких задачах: NER + POS tagging одновременно.

```python
class MultiTaskBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_ner, num_pos):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.ner_head = nn.Linear(hidden_size * 2, num_ner)
        self.pos_head = nn.Linear(hidden_size * 2, num_pos)
    
    def forward(self, x):
        # TODO: реализуйте multi-task output
        pass
```

---

### **Задача 9: Attention Visualization**

Визуализируйте, на какие части последовательности обращает внимание BiLSTM с attention.

**Требования:** Создайте heatmap attention weights, проанализируйте паттерны для разных классов.

---

### **Задача 10: Real-World NER System**

Создайте production-ready NER систему с BiLSTM-CRF.

**Требования:**
- Pre-trained word embeddings (GloVe/FastText)
- Character-level CNN для OOV words
- API endpoint для inference
- Latency < 50ms per sentence

---

## 🎯 Критерии успешного выполнения

- ✅ Понимаете принцип работы bidirectional RNN
- ✅ Знаете преимущества и ограничения
- ✅ Умеете применять BiRNN для sequence labeling
- ✅ Можете добавить attention к BiRNN
- ✅ Понимаете, когда использовать uni vs bi

---

## 📚 Ресурсы

- [Bidirectional LSTM Paper](https://www.researchgate.net/publication/2329878_Bidirectional_Recurrent_Neural_Networks)
- [Named Entity Recognition with BiLSTM](https://arxiv.org/abs/1511.08308)
- [BiLSTM-CRF для Sequence Labeling](https://arxiv.org/abs/1603.01354)
