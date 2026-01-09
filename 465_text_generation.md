# Генеративные модели для текста

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration

# !pip install transformers torch
```

---

## 🟢 Базовый уровень: GPT

### 1.1 GPT архитектура

**GPT = Generative Pre-trained Transformer**

**Ключевые особенности:**
- **Decoder-only** (использует только decoder Transformer)
- **Autoregressive** (предсказывает следующее слово)
- **Unidirectional** (видит только левый контекст)

```
Input:  "The cat sat on"
Output: "the" (предсказание следующего слова)

Input:  "The cat sat on the"
Output: "mat"
```

---

### 1.2 GPT-2 использование

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузка
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Генерация текста
input_text = "Once upon a time"
inputs = tokenizer.encode(input_text, return_tensors='pt')

# Generate
outputs = model.generate(
    inputs,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

---

### 1.3 Параметры генерации

**Temperature:**
- `< 1.0` → более conservative (предсказуемо)
- `= 1.0` → нормальное распределение
- `> 1.0` → более creative (разнообразно)

**Top-k sampling:**
- Выбираем из k наиболее вероятных слов
- k=50 — хороший баланс

**Top-p (nucleus) sampling:**
- Выбираем минимальный набор слов с суммарной вероятностью p
- p=0.95 — обычный выбор

---

## 🟡 Продвинутый уровень: T5

### 2.1 Text-to-Text Transfer Transformer

**T5** — универсальная модель, которая решает **все** NLP задачи как text-to-text.

```
Translation:     "translate English to French: Hello" → "Bonjour"
Summarization:   "summarize: <article>" → "<summary>"
Question Answer: "question: <Q> context: <C>" → "<answer>"
```

---

### 2.2 T5 использование

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Summarization
text = "summarize: The tower is 324 metres tall..."
inputs = tokenizer(text, return_tensors='pt')
outputs = model.generate(**inputs, max_length=50)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Translation
text = "translate English to German: Hello, how are you?"
inputs = tokenizer(text, return_tensors='pt')
outputs = model.generate(**inputs)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## 🟡 Продвинутый уровень: Fine-tuning

### 3.1 Fine-tuning GPT-2

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Подготовка данных
texts = ["Your text corpus here..."]
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize
inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

# Model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Training
training_args = TrainingArguments(
    output_dir='./gpt2-finetuned',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

---

## 🎯 Ключевые выводы

1. **GPT** — autoregressive language model (decoder-only)
2. **Temperature, top-k, top-p** контролируют генерацию
3. **T5** — text-to-text универсальная модель
4. **Fine-tuning** адаптирует модель к вашему домену

---

## 📚 Материалы

- [GPT-2 Paper](https://openai.com/research/better-language-models)
- [T5 Paper](https://arxiv.org/abs/1910.10683)
- [Hugging Face Generation Guide](https://huggingface.co/docs/transformers/generation_strategies)
