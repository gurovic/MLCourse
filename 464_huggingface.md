# Hugging Face Transformers

```python
from transformers import pipeline, AutoModel, AutoTokenizer
import torch

# !pip install transformers torch
```

---

## 🟢 Базовый уровень: Pipelines

### 1.1 Что такое Hugging Face?

**Hugging Face** — библиотека для работы с Transformer моделями.

**Преимущества:**
- ✅ Тысячи pre-trained моделей (BERT, GPT, T5, etc.)
- ✅ Простой API через `pipeline()`
- ✅ Поддержка PyTorch и TensorFlow
- ✅ Model Hub для sharing моделей

---

### 1.2 Использование Pipelines

**Pipelines** — самый простой способ использовать модели.

```python
# Sentiment Analysis
classifier = pipeline('sentiment-analysis')
result = classifier('I love Hugging Face!')
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# Named Entity Recognition
ner = pipeline('ner', grouped_entities=True)
result = ner('John lives in New York')
print(result)
# [{'entity_group': 'PER', 'word': 'John'}, 
#  {'entity_group': 'LOC', 'word': 'New York'}]

# Question Answering
qa = pipeline('question-answering')
result = qa(question='Where does John live?',
           context='John lives in New York')
print(result)  # {'answer': 'New York', 'score': 0.98}

# Text Generation
generator = pipeline('text-generation', model='gpt2')
result = generator('Once upon a time', max_length=50)
print(result[0]['generated_text'])

# Translation
translator = pipeline('translation_en_to_fr')
result = translator('Hello, how are you?')
print(result)  # [{'translation_text': 'Bonjour, comment allez-vous?'}]

# Summarization
summarizer = pipeline('summarization')
text = "Long article text here..."
result = summarizer(text, max_length=100, min_length=30)
print(result)
```

---

## 🟡 Продвинутый уровень: Model & Tokenizer API

### 2.1 AutoModel и AutoTokenizer

```python
from transformers import AutoModel, AutoTokenizer

# Загрузка любой модели по имени
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Использование
text = "Hello, Hugging Face!"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

print(outputs.last_hidden_state.shape)  # [1, seq_len, 768]
```

---

### 2.2 Task-Specific Models

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM
)

# Sentiment Analysis
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased-finetuned-sst-2-english')

# NER
model = AutoModelForTokenClassification.from_pretrained(
    'dslim/bert-base-NER')

# Question Answering
model = AutoModelForQuestionAnswering.from_pretrained(
    'distilbert-base-uncased-distilled-squad')

# Text Generation
model = AutoModelForCausalLM.from_pretrained('gpt2')
```

---

### 2.3 Fine-tuning с Trainer API

```python
from transformers import Trainer, TrainingArguments

# Подготовка данных
train_dataset = ...  # ваш dataset
eval_dataset = ...

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy='epoch'
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Training
trainer.train()

# Evaluation
trainer.evaluate()
```

---

## 🟡 Продвинутый уровень: Model Hub

### 3.1 Поиск моделей

```python
from huggingface_hub import list_models

# Поиск моделей
models = list_models(filter='text-classification', sort='downloads', limit=10)
for model in models:
    print(model.modelId)
```

**Model Hub:** https://huggingface.co/models

Тысячи моделей для:
- Text Classification
- NER
- Question Answering
- Translation
- Summarization
- Text Generation
- И многое другое!

---

### 3.2 Загрузка и использование

```python
# Загрузка любой модели с Hub
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
classifier = pipeline('sentiment-analysis', model=model_name)

result = classifier('I love this library!')
print(result)
```

---

### 3.3 Sharing своих моделей

```python
# После fine-tuning
model.save_pretrained('./my_awesome_model')
tokenizer.save_pretrained('./my_awesome_model')

# Загрузка на Hub
model.push_to_hub('my-username/my-awesome-model')
tokenizer.push_to_hub('my-username/my-awesome-model')
```

---

## 🔴 Экспертный уровень: Продвинутые возможности

### 4.1 Custom Pipeline

```python
from transformers import Pipeline

class MyCustomPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}
    
    def preprocess(self, inputs):
        return self.tokenizer(inputs, return_tensors='pt')
    
    def _forward(self, model_inputs):
        return self.model(**model_inputs)
    
    def postprocess(self, model_outputs):
        # Custom logic
        return model_outputs
```

---

### 4.2 Mixed Precision Training

```python
training_args = TrainingArguments(
    ...,
    fp16=True,  # Используем mixed precision (быстрее!)
)
```

---

### 4.3 Multi-GPU Training

```python
# Автоматически использует все доступные GPU
training_args = TrainingArguments(
    ...,
    per_device_train_batch_size=8,
    # С 4 GPU: effective batch size = 8 * 4 = 32
)
```

---

## 🎯 Ключевые выводы

1. **Hugging Face** — стандарт для Transformer моделей
2. **Pipelines** — простейший способ использования
3. **AutoModel/AutoTokenizer** — универсальный API
4. **Trainer** — удобный training loop
5. **Model Hub** — тысячи pre-trained моделей

---

## 📚 Материалы

- [Hugging Face Documentation](https://huggingface.co/docs/transformers/)
- [Model Hub](https://huggingface.co/models)
- [Course](https://huggingface.co/course)
