### **Задачи: BERT**

**Цель:** Научиться использовать и fine-tune BERT для различных NLP задач.

---

## 🟢 Базовый уровень

### **Задача 1: Sentiment Analysis с BERT**

Fine-tune BERT для классификации sentiment (positive/negative).

```python
from transformers import BertForSequenceClassification, Trainer

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# TODO: подготовьте данные
# TODO: обучите модель
# TODO: оцените accuracy
```

**Требования:** Используйте IMDB dataset, достигните accuracy > 90%.

---

### **Задача 2: Named Entity Recognition**

Fine-tune BERT для NER (распознавание имен, мест, организаций).

**Требования:** Используйте CoNLL-2003, измерьте F1-score для каждого entity type.

---

### **Задача 3: Сравнение BERT vs DistilBERT**

Сравните BERT-Base и DistilBERT на задаче text classification.

**Измерьте:** accuracy, inference time, model size.

---

## 🟡 Продвинутый уровень

### **Задача 4: Question Answering**

Fine-tune BERT для extractive question answering.

```python
from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
```

**Датасет:** SQuAD или свой.

---

### **Задача 5: Multi-task Fine-tuning**

Обучите одну BERT модель на нескольких задачах одновременно (sentiment + NER).

---

## 🔴 Экспертный уровень

### **Задача 6: Domain Adaptation**

Fine-tune BERT на domain-specific данных (медицинские, юридические тексты).

**Требования:** Сравните с general BERT на domain tasks.

---

### **Задача 7: BERT для Non-English**

Используйте multilingual BERT или обучите на русском языке.

---

## 📚 Ресурсы

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Hugging Face Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
