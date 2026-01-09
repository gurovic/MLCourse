### **Задачи: Sequence-to-Sequence**

**Цель:** Научиться реализовывать и обучать seq2seq модели для задач машинного перевода, summarization и других variable-length задач.

---

## 🟢 Базовый уровень

### **Задача 1: Simple Seq2Seq для сложения чисел**

Обучите seq2seq модель складывать числа в текстовом виде.

**Пример:**
```
Input: "12+34"
Output: "46"
```

```python
class SimpleSeq2Seq(nn.Module):
    def __init__(self, input_vocab, output_vocab, embed_dim, hidden_size):
        super().__init__()
        self.encoder = Encoder(input_vocab, embed_dim, hidden_size)
        self.decoder = Decoder(output_vocab, embed_dim, hidden_size)
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # TODO: реализуйте encoder-decoder
        pass
```

**Требования:** Создайте датасет (1000 примеров), обучите 50 эпох, достигните accuracy > 95%.

---

### **Задача 2: Teacher Forcing эксперимент**

Исследуйте влияние teacher forcing ratio на качество обучения.

**Эксперименты:** Обучите модели с teacher_forcing_ratio = [0.0, 0.3, 0.5, 0.7, 1.0].

**Измерьте:** convergence speed, final accuracy, inference quality.

**Визуализируйте:** learning curves для всех вариантов.

---

### **Задача 3: Reverse Sequence Task**

Обучите seq2seq переворачивать последовательности.

**Пример:**
```
Input:  "hello world"
Output: "dlrow olleh"
```

**Требования:** Реализуйте encoder-decoder, обучите на синтетических данных.

---

## 🟡 Продвинутый уровень

### **Задача 4: Seq2Seq с Attention**

Добавьте attention mechanism к базовой seq2seq модели.

```python
class AttentionSeq2Seq(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.encoder = Encoder(...)
        self.attention_decoder = AttentionDecoder(...)
    
    def forward(self, src, tgt):
        # TODO: реализуйте с attention
        pass
```

**Требования:** Сравните с baseline (без attention) на задаче перевода, визуализируйте attention weights.

---

### **Задача 5: Machine Translation (En→Ru)**

Реализуйте простой переводчик английский → русский.

**Архитектура:**
- Bidirectional encoder
- Attention decoder
- Pre-trained embeddings

**Датасет:** Используйте готовый параллельный корпус или создайте синтетический.

**Метрики:** BLEU score.

---

### **Задача 6: Text Summarization**

Обучите seq2seq модель генерировать короткие резюме текстов.

**Подход:**
- Encoder: обрабатывает длинный текст
- Decoder: генерирует summary

**Требования:** Обучите на датасете новостей, оцените качество (ROUGE score).

---

## 🔴 Экспертный уровень

### **Задача 7: Beam Search Implementation**

Реализуйте beam search decoding для улучшения inference quality.

```python
def beam_search(model, src, beam_width=5, max_len=50):
    """
    Реализует beam search decoding
    beam_width: сколько гипотез держим
    """
    # TODO: реализуйте beam search
    pass
```

**Требования:** Сравните greedy vs beam search (beam_width=[1,3,5,10]) по качеству и скорости.

---

### **Задача 8: Scheduled Sampling**

Реализуйте scheduled sampling — постепенное снижение teacher forcing ratio.

```python
def get_teacher_forcing_ratio(epoch, total_epochs):
    """Линейное убывание от 1.0 до 0.0"""
    return 1.0 - (epoch / total_epochs)

# Training loop
for epoch in range(epochs):
    tf_ratio = get_teacher_forcing_ratio(epoch, epochs)
    train(model, dataloader, teacher_forcing_ratio=tf_ratio)
```

**Сравните:** constant ratio vs scheduled sampling.

---

### **Задача 9: Multi-layer Seq2Seq**

Реализуйте multi-layer encoder и decoder (2-3 слоя).

**Требования:** Сравните 1, 2, 3 слоя по качеству, скорости обучения и overfitting.

---

### **Задача 10: Dialogue System**

Создайте простой диалоговый бот используя seq2seq.

**Данные:** Используйте датасет диалогов (Cornell Movie Dialogs или свой).

**Архитектура:**
- Bidirectional LSTM encoder
- Attention LSTM decoder
- Beam search inference

**Требования:** Обучите модель, создайте интерактивный интерфейс для общения с ботом.

---

## 🎯 Критерии успешного выполнения

- ✅ Понимаете архитектуру seq2seq (encoder-decoder)
- ✅ Знаете, как работает teacher forcing
- ✅ Умеете добавлять attention mechanism
- ✅ Понимаете разницу между greedy и beam search
- ✅ Можете применить seq2seq для разных задач

---

## 📚 Ресурсы

- [Seq2Seq Paper](https://arxiv.org/abs/1409.3215)
- [Attention Paper](https://arxiv.org/abs/1409.0473)
- [PyTorch Seq2Seq Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
