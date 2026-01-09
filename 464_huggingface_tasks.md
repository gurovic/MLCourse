### **Задачи: Hugging Face Transformers**

**Цель:** Научиться использовать Hugging Face библиотеку для различных NLP задач.

---

## 🟢 Базовый уровень

### **Задача 1: Использование Pipelines**

Попробуйте все основные pipelines: sentiment-analysis, NER, QA, summarization, translation.

**Требования:** Протестируйте на 5 примерах каждый pipeline, проанализируйте результаты.

---

### **Задача 2: Fine-tuning с Trainer API**

Fine-tune DistilBERT для text classification используя Trainer API.

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()
```

---

### **Задача 3: Поиск и использование моделей с Hub**

Найдите pre-trained модель на Model Hub для своей задачи и используйте её.

**Требования:** Сравните минимум 3 разные модели для одной задачи.

---

## 🟡 Продвинутый уровень

### **Задача 4: Multi-task Fine-tuning**

Обучите одну модель на нескольких задачах с помощью Trainer API.

---

### **Задача 5: Custom Pipeline**

Создайте свой custom pipeline для специфической задачи.

---

## 🔴 Экспертный уровень

### **Задача 6: Model Deployment**

Разверните fine-tuned модель как REST API используя FastAPI.

---

### **Задача 7: Model Sharing**

Fine-tune модель и загрузите её на Hugging Face Hub.

---

## 📚 Ресурсы

- [Hugging Face Course](https://huggingface.co/course)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
