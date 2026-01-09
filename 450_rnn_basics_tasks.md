### **Задачи: Основы RNN**

**Цель:** Понять принципы работы рекуррентных нейронных сетей, научиться обрабатывать последовательности, решить проблемы vanishing/exploding градиентов, применить RNN на практике.

---

## 🟢 Базовый уровень

### **Задача 1: Реализация Simple RNN с нуля**

**Условие:** Реализуйте простейшую рекуррентную нейросеть без использования nn.RNN.

**Требования:**
1. Реализуйте класс SimpleRNN с методами:
   - `__init__(input_size, hidden_size, output_size)`
   - `forward(x, h_prev=None)` — обработка последовательности
   - `step(x_t, h_prev)` — один шаг RNN
   
2. Используйте формулы:
   ```
   h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
   y_t = W_hy * h_t + b_y
   ```

3. Обучите на простой задаче: предсказание следующего числа в последовательности
   ```
   [1, 2, 3, 4, 5] → предсказать 6
   [2, 4, 6, 8, 10] → предсказать 12
   ```

4. Визуализируйте:
   - Loss по эпохам
   - Предсказания vs ground truth

**Ожидаемый результат:** RNN научится предсказывать простые паттерны.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # TODO: определите веса W_hh, W_xh, W_hy, b_h, b_y
        pass
    
    def step(self, x_t, h_prev):
        """Один шаг RNN"""
        # TODO: реализуйте h_t = tanh(W_hh * h_prev + W_xh * x_t + b_h)
        pass
    
    def forward(self, x, h_prev=None):
        """Обработка всей последовательности"""
        # TODO: реализуйте цикл по времени
        pass

# Генерация данных
def generate_sequences(num_sequences=1000):
    sequences = []
    targets = []
    
    for _ in range(num_sequences):
        # Арифметическая прогрессия
        start = torch.randint(1, 10, (1,)).item()
        step = torch.randint(1, 5, (1,)).item()
        
        seq = torch.tensor([start + i*step for i in range(5)], dtype=torch.float32)
        target = start + 5*step
        
        sequences.append(seq.unsqueeze(-1))  # [5, 1]
        targets.append(target)
    
    return torch.stack(sequences), torch.tensor(targets, dtype=torch.float32)

# TODO: создайте модель
# TODO: обучите на сгенерированных данных
# TODO: протестируйте на новых последовательностях
```

**Вопросы для анализа:**
1. Почему используется tanh, а не ReLU?
2. Как размер hidden_size влияет на способность модели учиться?
3. Сколько параметров в вашей RNN?

---

### **Задача 2: Sentiment Analysis с PyTorch RNN**

**Условие:** Классифицируйте отзывы на положительные/отрицательные с помощью RNN.

**Требования:**
1. Используйте датасет IMDB или простой синтетический датасет
2. Архитектура:
   - Embedding layer (vocab_size → embed_dim)
   - RNN layer (embed_dim → hidden_size)
   - Linear layer (hidden_size → 2 classes)
   
3. Реализуйте обработку текста:
   - Tokenization
   - Vocabulary building
   - Padding sequences
   
4. Обучите модель 10 эпох
5. Оцените:
   - Accuracy на test set
   - Confusion matrix
   - Примеры правильных/неправильных классификаций

**Ожидаемый результат:** Accuracy > 80% на test set.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: [batch, seq_len] — индексы слов
        
        # Embedding
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        
        # RNN
        output, hidden = self.rnn(embedded)  # hidden: [1, batch, hidden_size]
        
        # Используем финальный hidden state
        hidden = hidden.squeeze(0)  # [batch, hidden_size]
        
        # Classification
        logits = self.fc(hidden)  # [batch, num_classes]
        
        return logits

# Простой датасет для тестирования
reviews = [
    ("great movie love it", 1),
    ("terrible waste of time", 0),
    ("best film ever", 1),
    ("boring and slow", 0),
    # TODO: добавьте больше примеров
]

# TODO: реализуйте tokenization и vocabulary
# TODO: создайте DataLoader
# TODO: обучите модель
# TODO: оцените на test set
```

---

### **Задача 3: Визуализация Vanishing Gradients**

**Условие:** Продемонстрируйте проблему затухающих градиентов в RNN.

**Требования:**
1. Создайте RNN с длинной последовательностью (100-200 шагов)
2. Обучите на задаче, требующей долгосрочной памяти:
   ```
   Последовательность: [1, 0, 0, ..., 0, 0]  (99 нулей)
   Задача: предсказать первый элемент (1) в конце
   ```

3. Во время backpropagation измерьте:
   - Gradient norm на каждом временном шаге
   - Gradient norm для весов W_hh
   
4. Визуализируйте:
   - График gradient norm по временным шагам
   - Сравнение с короткой последовательностью (10 шагов)

**Ожидаемый результат:** Градиенты экспоненциально убывают для ранних шагов.

```python
import matplotlib.pyplot as plt

def measure_gradients(model, x, target, seq_len):
    """Измеряет градиенты на разных временных шагах"""
    
    # Forward + backward
    output, _ = model(x)
    loss = F.mse_loss(output[:, -1], target)
    loss.backward()
    
    # Извлекаем градиенты
    gradients = []
    
    # TODO: измерьте градиенты для каждого временного шага
    # Hint: используйте retain_grad() для промежуточных тензоров
    
    return gradients

# Создание задачи с долгосрочной зависимостью
def create_memory_task(batch_size, seq_len):
    """Создает последовательность, где нужно помнить первый элемент"""
    x = torch.zeros(batch_size, seq_len, 1)
    x[:, 0, 0] = torch.randint(0, 10, (batch_size,)).float()
    
    target = x[:, 0, 0]  # Предсказываем первый элемент
    
    return x, target

# TODO: обучите RNN на разных длинах последовательности
# TODO: визуализируйте затухание градиентов
```

**Вопросы:**
1. На каком временном шаге градиенты становятся близки к нулю?
2. Как размер hidden_size влияет на vanishing gradients?
3. Помогает ли увеличение learning rate?

---

## 🟡 Продвинутый уровень

### **Задача 4: Character-Level Text Generation**

**Условие:** Реализуйте генерацию текста на уровне символов (char-level language model).

**Требования:**
1. Используйте текстовый датасет (Shakespeare, код на Python, etc.)
2. Архитектура:
   - Embedding для символов
   - Multi-layer RNN (2-3 слоя)
   - Linear для предсказания следующего символа
   
3. Обучите модель (20+ эпох)
4. Реализуйте генерацию с:
   - Greedy sampling (argmax)
   - Temperature sampling
   - Top-k sampling
   
5. Сравните качество генерации для разных температур: [0.5, 1.0, 1.5, 2.0]

**Ожидаемый результат:** Модель генерирует осмысленный текст в стиле обучающих данных.

```python
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(self, x, h=None):
        embedded = self.embedding(x)
        output, h = self.rnn(embedded, h)
        logits = self.fc(output)
        return logits, h
    
    def generate(self, start_char_idx, length=500, temperature=1.0, top_k=None):
        """Генерирует текст"""
        self.eval()
        
        generated = [start_char_idx]
        h = None
        
        with torch.no_grad():
            for _ in range(length):
                x = torch.tensor([[generated[-1]]])
                logits, h = self.forward(x, h)
                
                # Применяем temperature
                logits = logits[0, -1] / temperature
                
                # Top-k sampling
                if top_k:
                    top_logits, top_indices = torch.topk(logits, top_k)
                    probs = F.softmax(top_logits, dim=0)
                    next_char_idx = top_indices[torch.multinomial(probs, 1)].item()
                else:
                    probs = F.softmax(logits, dim=0)
                    next_char_idx = torch.multinomial(probs, 1).item()
                
                generated.append(next_char_idx)
        
        return generated

# TODO: загрузите текстовый датасет
# TODO: подготовьте vocabulary и data loader
# TODO: обучите модель
# TODO: сгенерируйте тексты с разными температурами
# TODO: сравните результаты
```

---

### **Задача 5: Gradient Clipping — борьба с Exploding Gradients**

**Условие:** Продемонстрируйте проблему взрывающихся градиентов и эффект gradient clipping.

**Требования:**
1. Создайте RNN с большими весами (инициализация xavier_normal с gain=2)
2. Обучите на задаче sequence prediction без clipping
3. Обучите ту же модель с gradient clipping (max_norm=5.0)
4. Сравните:
   - Loss curves
   - Gradient norms по эпохам
   - Stability обучения
   
5. Экспериментируйте с разными max_norm: [1.0, 5.0, 10.0, None]

**Ожидаемый результат:** Без clipping обучение нестабильно, с clipping — сходится плавно.

```python
def train_with_clipping(model, train_loader, epochs, max_norm=None):
    """Обучает модель с или без gradient clipping"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    grad_norms = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for x, y in train_loader:
            optimizer.zero_grad()
            
            output, _ = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            
            loss.backward()
            
            # Измеряем gradient norm до clipping
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)
            
            # Gradient clipping
            if max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch+1}, Loss: {losses[-1]:.4f}")
    
    return losses, grad_norms

# TODO: обучите модели с разными max_norm
# TODO: визуализируйте сравнение
```

---

### **Задача 6: Bidirectional RNN для Named Entity Recognition**

**Условие:** Реализуйте NER систему с использованием Bidirectional RNN.

**Требования:**
1. Используйте датасет CoNLL-2003 или создайте синтетический
2. Архитектура:
   - Word embeddings
   - Bidirectional RNN (2 слоя)
   - Linear layer для классификации каждого токена
   
3. Метки: O (outside), B-PER (begin person), I-PER (inside person), B-LOC, I-LOC, etc.
4. Обучите модель
5. Оцените:
   - F1-score для каждого entity type
   - Примеры правильно/неправильно определенных entities

**Ожидаемый результат:** F1-score > 0.7 на test set.

```python
class BiRNNForNER(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_tags):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers=2, 
                         batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_tags)  # *2 для bidirectional
    
    def forward(self, x):
        # x: [batch, seq_len]
        
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        output, _ = self.rnn(embedded)  # [batch, seq_len, hidden*2]
        logits = self.fc(output)  # [batch, seq_len, num_tags]
        
        return logits

# Пример данных
sentences = [
    (["John", "lives", "in", "New", "York"], 
     ["B-PER", "O", "O", "B-LOC", "I-LOC"]),
    # TODO: добавьте больше примеров
]

# TODO: подготовьте данные
# TODO: обучите модель
# TODO: оцените F1-score
```

---

## 🔴 Экспертный уровень

### **Задача 7: Sequence-to-Sequence без Attention**

**Условие:** Реализуйте простой seq2seq для задачи перевода или суммаризации.

**Требования:**
1. Архитектура Encoder-Decoder:
   - Encoder RNN: обрабатывает входную последовательность → context vector
   - Decoder RNN: генерирует выходную последовательность из context
   
2. Реализуйте:
   - Teacher forcing (используем ground truth на обучении)
   - Greedy decoding на inference
   
3. Задача: сложение чисел в текстовом виде
   ```
   Input: "12+34"
   Output: "46"
   ```

4. Обучите и оцените точность

**Ожидаемый результат:** Модель научится складывать числа.

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return hidden  # context vector

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        src: [batch, src_len]
        tgt: [batch, tgt_len]
        """
        # TODO: реализуйте encoder-decoder с teacher forcing
        pass
    
    def generate(self, src, max_len=50):
        """Генерация без teacher forcing"""
        # TODO: реализуйте greedy decoding
        pass

# TODO: создайте датасет сложения
# TODO: обучите seq2seq
# TODO: протестируйте на новых примерах
```

---

### **Задача 8: Анализ Hidden States**

**Условие:** Визуализируйте и проанализируйте, что кодируют hidden states в RNN.

**Требования:**
1. Обучите RNN на задаче sentiment analysis
2. Для тестовых примеров извлеките hidden states на каждом шаге
3. Визуализируйте:
   - t-SNE проекцию hidden states (раскрасьте по классу)
   - Activation patterns для конкретных слов ("good", "bad", "amazing")
   - Cosine similarity между hidden states разных предложений
   
4. Проанализируйте:
   - Как меняются hidden states по ходу предложения?
   - Есть ли кластеры для разных sentiment?
   - Какие слова вызывают наибольшее изменение hidden state?

**Ожидаемый результат:** Визуализация покажет, что hidden states кодируют sentiment информацию.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def extract_hidden_states(model, sentences):
    """Извлекает hidden states для предложений"""
    model.eval()
    
    all_hiddens = []
    all_labels = []
    
    with torch.no_grad():
        for sent, label in sentences:
            # TODO: получите hidden states для каждого временного шага
            pass
    
    return all_hiddens, all_labels

def visualize_hidden_states(hiddens, labels):
    """Визуализирует hidden states через t-SNE"""
    # TODO: примените t-SNE
    # TODO: постройте scatter plot с раскраской по labels
    pass

# TODO: обучите модель
# TODO: извлеките hidden states
# TODO: визуализируйте
```

---

### **Задача 9: Сравнение RNN, GRU, LSTM**

**Условие:** Сравните vanilla RNN с GRU и LSTM на одинаковых задачах.

**Требования:**
1. Реализуйте 3 модели с одинаковой архитектурой (кроме рекуррентного слоя):
   - SimpleRNN
   - GRU
   - LSTM
   
2. Задачи для сравнения:
   - Sentiment analysis (короткие зависимости)
   - Copy task (средние зависимости)
   - Memory task (длинные зависимости)
   
3. Для каждой задачи измерьте:
   - Training time
   - Final accuracy
   - Convergence speed (epochs to reach 90% accuracy)
   - Number of parameters
   
4. Визуализируйте:
   - Learning curves для всех трех моделей
   - Bar chart сравнения метрик

**Ожидаемый результат:** LSTM/GRU лучше на задачах с долгосрочными зависимостями.

```python
def compare_architectures(task_name, train_loader, test_loader):
    """Сравнивает RNN, GRU, LSTM на одной задаче"""
    
    models = {
        'RNN': SimpleRNN(...),
        'GRU': nn.GRU(...),
        'LSTM': nn.LSTM(...),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nОбучение {name}...")
        
        # TODO: обучите модель
        # TODO: измерьте метрики
        
        results[name] = {
            'accuracy': ...,
            'train_time': ...,
            'convergence_epoch': ...,
            'num_params': ...
        }
    
    return results

# TODO: запустите сравнение на всех трех задачах
# TODO: визуализируйте результаты
```

---

### **Задача 10: Production-Ready RNN для Text Classification**

**Условие:** Создайте полный пайплайн для развертывания RNN модели в production.

**Требования:**
1. **Обучение:**
   - Используйте реальный датасет (IMDB, AG News, etc.)
   - Bidirectional RNN с pre-trained embeddings (GloVe)
   - Early stopping, learning rate scheduling
   - Сохранение лучшей модели
   
2. **Оптимизация:**
   - Dynamic batching (группируем по длине)
   - Padding optimization
   - ONNX export для faster inference
   
3. **API:**
   - Создайте Flask/FastAPI endpoint
   - Input: текст, Output: класс + confidence
   - Обработка ошибок
   
4. **Тестирование:**
   - Unit tests для preprocessing
   - Integration tests для API
   - Latency benchmarks

**Ожидаемый результат:** Работающий API с latency < 100ms.

```python
# model.py
class ProductionRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        # TODO: реализуйте архитектуру
        pass
    
    def predict_proba(self, text):
        """Возвращает вероятности классов"""
        # TODO: preprocessing + inference
        pass

# api.py
from flask import Flask, request, jsonify

app = Flask(__name__)
model = load_model('best_model.pth')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    
    # TODO: валидация
    # TODO: inference
    # TODO: возврат результата
    
    return jsonify({
        'class': ...,
        'confidence': ...,
        'latency_ms': ...
    })

# TODO: реализуйте полный пайплайн
# TODO: добавьте tests
# TODO: измерьте latency
```

---

## 📝 Дополнительные вопросы для размышления

1. **Почему RNN медленнее обучается, чем CNN?**
   - Что такое sequential processing?
   - Можно ли распараллелить RNN?

2. **Когда использовать RNN вместо Transformer?**
   - Преимущества RNN
   - Недостатки RNN

3. **Как выбрать размер hidden state?**
   - Что кодирует hidden state?
   - Как размер влияет на capacity и overfitting?

4. **Bidirectional RNN:**
   - Когда он полезен?
   - Когда его нельзя использовать?

---

## 🎯 Критерии успешного выполнения

- ✅ Вы понимаете, как работает RNN (рекуррентная формула)
- ✅ Вы умеете реализовать RNN с нуля
- ✅ Вы понимаете проблемы vanishing/exploding gradients
- ✅ Вы знаете, как применять gradient clipping
- ✅ Вы умеете использовать RNN для разных типов задач (one-to-many, many-to-one, etc.)
- ✅ Вы понимаете разницу между unidirectional и bidirectional RNN
- ✅ Вы можете визуализировать и интерпретировать hidden states

---

## 📚 Полезные ресурсы

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of RNN](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [PyTorch RNN Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
- [Sequence Models Course (Coursera)](https://www.coursera.org/learn/nlp-sequence-models)
- [On the difficulty of training RNNs](https://arxiv.org/abs/1211.5063)
- [Visualizing RNN](https://distill.pub/2019/memorization-in-rnns/)
