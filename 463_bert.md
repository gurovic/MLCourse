# BERT и его варианты

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# !pip install transformers torch
```

---

## 🟢 Базовый уровень: BERT

### 1.1 Что такое BERT?

**BERT = Bidirectional Encoder Representations from Transformers**

**Ключевые отличия от GPT:**
- **Bidirectional:** Видит контекст слева И справа
- **Encoder-only:** Использует только encoder часть Transformer
- **Pre-training:** Обучается на огромных текстах, затем fine-tuning

---

### 1.2 Архитектура

```
Input: "The cat sat on [MASK] mat"
   ↓
Token Embeddings + Positional + Segment
   ↓
12-24 Transformer Encoder Layers
   ↓
Contextualized Representations
   ↓
Task-specific heads (classification, NER, QA, etc.)
```

**Размеры:**
- **BERT-Base:** 12 layers, 768 hidden, 12 heads, 110M parameters
- **BERT-Large:** 24 layers, 1024 hidden, 16 heads, 340M parameters

---

### 1.3 Pre-training задачи

**1. Masked Language Model (MLM):**
```
Input:  "The cat [MASK] on the mat"
Target: предсказать "sat"
```

**2. Next Sentence Prediction (NSP):**
```
Sentence A: "The cat sat on the mat"
Sentence B: "It was sleeping"
Target: Is B next sentence after A? (Yes/No)
```

---

### 1.4 Использование BERT

```python
from transformers import BertTokenizer, BertModel

# Загрузка pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenization
text = "Hello, my name is BERT"
inputs = tokenizer(text, return_tensors='pt')

# Forward pass
outputs = model(**inputs)

# Outputs
last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, 768]
pooler_output = outputs.pooler_output  # [batch, 768] — [CLS] token representation
```

---

## 🟡 Продвинутый уровень: Fine-tuning BERT

### 2.1 Sentiment Analysis

```python
from transformers import BertForSequenceClassification

class SentimentClassifier:
    def __init__(self, num_classes=2):
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=num_classes)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def train(self, texts, labels):
        # Tokenize
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                               return_tensors='pt')
        
        # Forward
        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        
        # Backward
        loss.backward()
        
        return loss.item()
```

---

### 2.2 Named Entity Recognition

```python
from transformers import BertForTokenClassification

model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=9)

# Input
tokens = ["John", "lives", "in", "New", "York"]
# Labels: B-PER, O, O, B-LOC, I-LOC
```

---

## 🟡 Продвинутый уровень: Варианты BERT

### 3.1 RoBERTa

**Robustly Optimized BERT Approach**

Улучшения:
- ✅ Убрали NSP задачу
- ✅ Динамический masking
- ✅ Больше данных, больше батчей
- ✅ Результат: лучше качество

```python
from transformers import RobertaModel, RobertaTokenizer

model = RobertaModel.from_pretrained('roberta-base')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
```

---

### 3.2 DistilBERT

**Distilled version of BERT**

- ✅ 40% меньше параметров
- ✅ 60% быстрее
- ✅ 97% качества BERT

```python
from transformers import DistilBertModel

model = DistilBertModel.from_pretrained('distilbert-base-uncased')
# 66M parameters vs 110M в BERT-Base
```

---

## 🎯 Ключевые выводы

1. **BERT** — bidirectional pre-trained Transformer encoder
2. **Pre-training + Fine-tuning** — эффективная стратегия
3. **Masked Language Model** — ключевая задача pre-training
4. **RoBERTa** улучшает BERT через лучший training
5. **DistilBERT** — компактная версия для production

---

## 📚 Материалы

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Hugging Face Documentation](https://huggingface.co/docs/transformers/)
