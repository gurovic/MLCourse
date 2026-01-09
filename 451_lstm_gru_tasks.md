### **Задачи: LSTM и GRU**

**Цель:** Понять принципы работы LSTM и GRU, научиться применять их для задач с долгосрочными зависимостями, сравнить с vanilla RNN.

---

## 🟢 Базовый уровень

### **Задача 1: Сравнение RNN, LSTM, GRU на Memory Task**

**Условие:** Сравните три архитектуры на задаче, требующей долгосрочной памяти.

**Требования:**
1. Создайте датасет "Copy Task":
   ```
   Input:  [1, 2, 3, 0, 0, 0, 0, 0]  (3 числа + 5 нулей)
   Output: [0, 0, 0, 0, 0, 1, 2, 3]  (копируем числа в конец)
   ```

2. Обучите 3 модели:
   - Vanilla RNN
   - LSTM
   - GRU

3. Для каждой модели измерьте:
   - Final loss
   - Accuracy (точность копирования)
   - Epochs to converge
   
4. Визуализируйте learning curves

**Ожидаемый результат:** LSTM/GRU обучаются, RNN fails.

```python
import torch
import torch.nn as nn

def generate_copy_task(batch_size, seq_len=10, delay=5):
    """
    Генерирует задачу копирования
    seq_len: длина входной последовательности
    delay: сколько нулей между входом и выходом
    """
    # TODO: сгенерируйте последовательности
    pass

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, cell_type='rnn'):
        super().__init__()
        
        if cell_type == 'rnn':
            self.cell = nn.RNN(input_size, hidden_size, batch_first=True)
        elif cell_type == 'lstm':
            self.cell = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif cell_type == 'gru':
            self.cell = nn.GRU(input_size, hidden_size, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        output, _ = self.cell(x)
        return self.fc(output)

# TODO: обучите три модели
# TODO: сравните результаты
```

---

### **Задача 2: Sentiment Analysis с LSTM**

**Условие:** Реализуйте sentiment analysis с использованием LSTM.

**Требования:**
1. Используйте датасет IMDB или синтетический
2. Архитектура:
   - Embedding layer
   - 2-layer LSTM (hidden_size=128)
   - Linear classifier
   
3. Добавьте:
   - Dropout между LSTM слоями
   - Bidirectional LSTM
   
4. Обучите 10 эпох
5. Сравните:
   - Unidirectional vs Bidirectional
   - 1 layer vs 2 layers
   
**Ожидаемый результат:** Bidirectional LSTM дает лучшее качество.

```python
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, bidirectional=False):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=2,
                           batch_first=True, dropout=0.3, 
                           bidirectional=bidirectional)
        
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, num_classes)
    
    def forward(self, x):
        # TODO: реализуйте forward pass
        pass

# TODO: обучите модели
# TODO: сравните uni vs bi
```

---

### **Задача 3: Time Series Forecasting с GRU**

**Условие:** Предскажите будущие значения временного ряда с помощью GRU.

**Требования:**
1. Используйте синтетический временной ряд или реальные данные (погода, акции)
2. Архитектура:
   - Input: последние N значений
   - GRU (2 слоя, hidden_size=64)
   - Output: следующее значение
   
3. Экспериментируйте с N (окно истории): [10, 20, 50, 100]
4. Визуализируйте:
   - Ground truth vs Predictions
   - MAE для разных размеров окна

**Ожидаемый результат:** Большее окно → лучшие предсказания (до определенного предела).

```python
class TimeSeriesGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: [batch, seq_len, 1]
        output, h_n = self.gru(x)
        # Предсказываем на основе последнего hidden state
        pred = self.fc(h_n[-1])  # [batch, 1]
        return pred

# TODO: создайте датасет временного ряда
# TODO: обучите модель
# TODO: визуализируйте предсказания
```

---

## 🟡 Продвинутый уровень

### **Задача 4: Character-Level Language Model с LSTM**

**Условие:** Обучите LSTM генерировать текст на уровне символов.

**Требования:**
1. Датасет: Shakespeare, код на Python, или любой текст
2. Архитектура:
   - Embedding для символов
   - 3-layer LSTM (hidden_size=256)
   - Temperature sampling
   
3. Обучите 20+ эпох
4. Генерируйте тексты с разными температурами: [0.3, 0.7, 1.0, 1.5]
5. Проанализируйте:
   - Как температура влияет на разнообразие?
   - Какие паттерны выучила модель?

**Ожидаемый результат:** Модель генерирует осмысленный текст в стиле обучающих данных.

```python
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256, num_layers=3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden
    
    def generate(self, start_char, length=500, temperature=1.0):
        """Генерирует текст"""
        # TODO: реализуйте generation с temperature sampling
        pass

# TODO: подготовьте текстовый датасет
# TODO: обучите модель
# TODO: сгенерируйте тексты с разными температурами
```

---

### **Задача 5: Сравнение LSTM и GRU: Speed vs Accuracy**

**Условие:** Детально сравните LSTM и GRU на нескольких задачах.

**Требования:**
1. Задачи для сравнения:
   - Sentiment analysis (IMDB)
   - Sequence classification (синтетика)
   - Time series forecasting
   
2. Для каждой задачи измерьте:
   - Training time per epoch
   - Inference time (100 samples)
   - Final accuracy
   - Number of parameters
   - Memory usage
   
3. Постройте:
   - Bar charts для сравнения
   - Scatter plot: speed vs accuracy
   
4. Сделайте выводы: когда GRU, когда LSTM?

**Ожидаемый результат:** GRU быстрее на ~25%, качество сопоставимо.

```python
import time

def benchmark_model(model_class, task_data, epochs=10):
    """Бенчмарк модели"""
    
    model = model_class()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training time
    start = time.time()
    for epoch in range(epochs):
        # TODO: training loop
        pass
    train_time = time.time() - start
    
    # Inference time
    model.eval()
    test_data = task_data['test']
    start = time.time()
    with torch.no_grad():
        for batch in test_data:
            _ = model(batch)
    inference_time = (time.time() - start) / len(test_data)
    
    # Accuracy
    accuracy = evaluate(model, test_data)
    
    # Parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    return {
        'train_time': train_time,
        'inference_time': inference_time,
        'accuracy': accuracy,
        'num_params': num_params
    }

# TODO: запустите benchmark для LSTM и GRU на всех задачах
# TODO: визуализируйте результаты
```

---

### **Задача 6: Визуализация Gate Activations**

**Условие:** Визуализируйте, как работают gates в LSTM во время inference.

**Требования:**
1. Обучите LSTM на задаче sentiment analysis
2. Для тестовых предложений извлеките:
   - Forget gate activations
   - Input gate activations
   - Output gate activations
   
3. Визуализируйте:
   - Heatmap: gates × timesteps
   - Какие слова вызывают высокие activations?
   - Сравните positive vs negative reviews
   
4. Проанализируйте:
   - Когда forget gate закрывается/открывается?
   - Какие паттерны видны?

**Ожидаемый результат:** Визуализация покажет, как LSTM управляет памятью.

```python
def extract_gate_activations(model, sentence):
    """Извлекает activations gates из LSTM"""
    
    # Нужно модифицировать LSTM для доступа к gates
    # Или использовать hooks
    
    activations = {
        'forget': [],
        'input': [],
        'output': []
    }
    
    # TODO: реализуйте extraction
    
    return activations

def visualize_gates(sentence, activations, words):
    """Визуализирует gates как heatmap"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    for idx, (gate_name, ax) in enumerate(zip(['forget', 'input', 'output'], axes)):
        sns.heatmap(activations[gate_name], ax=ax, cmap='viridis',
                   xticklabels=words, yticklabels=False)
        ax.set_title(f'{gate_name.capitalize()} Gate Activations')
    
    plt.tight_layout()
    plt.show()

# TODO: обучите LSTM
# TODO: извлеките и визуализируйте gates
```

---

## 🔴 Экспертный уровень

### **Задача 7: Реализация LSTM Cell с нуля**

**Условие:** Реализуйте LSTM cell с нуля без использования nn.LSTM.

**Требования:**
1. Реализуйте класс LSTMCell с:
   - Forget gate
   - Input gate
   - Output gate
   - Cell state update
   
2. Реализуйте класс CustomLSTM:
   - Использует ваш LSTMCell
   - Поддерживает multi-layer
   - Поддерживает dropout
   
3. Обучите на задаче и сравните с nn.LSTM:
   - Accuracy
   - Speed
   
**Ожидаемый результат:** Ваша реализация работает, но медленнее PyTorch.

```python
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        # Gates: forget, input, output, cell
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, x_t, h_prev, c_prev):
        """
        x_t: [batch, input_size]
        h_prev: [batch, hidden_size]
        c_prev: [batch, hidden_size]
        """
        # Concatenate input and hidden
        combined = torch.cat([x_t, h_prev], dim=1)
        
        # Forget gate
        f_t = torch.sigmoid(self.W_f(combined))
        
        # Input gate
        i_t = torch.sigmoid(self.W_i(combined))
        
        # Candidate cell state
        c_tilde = torch.tanh(self.W_c(combined))
        
        # Update cell state
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Output gate
        o_t = torch.sigmoid(self.W_o(combined))
        
        # Hidden state
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(LSTMCell(input_size, hidden_size))
        
        for _ in range(num_layers - 1):
            self.layers.append(LSTMCell(hidden_size, hidden_size))
    
    def forward(self, x, hidden=None):
        """
        x: [batch, seq_len, input_size]
        """
        # TODO: реализуйте forward pass через все timesteps и layers
        pass

# TODO: обучите и сравните с nn.LSTM
```

---

### **Задача 8: Attention-Enhanced LSTM**

**Условие:** Добавьте attention mechanism к LSTM для улучшения quality.

**Требования:**
1. Реализуйте LSTM с attention:
   - LSTM encoder: обрабатывает всю последовательность
   - Attention: взвешивает hidden states
   - Classifier: использует attended representation
   
2. Обучите на sentiment analysis
3. Сравните с baseline LSTM:
   - Accuracy
   - Attention weights visualization
   
4. Проанализируйте:
   - На какие слова attention обращает больше внимания?
   - Помогает ли attention интерпретируемости?

**Ожидаемый результат:** Attention LSTM показывает лучшее качество и интерпретируемость.

```python
class AttentionLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        
        # Attention weights
        self.attention = nn.Linear(hidden_size, 1)
        
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: [batch, seq_len]
        
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, hidden]
        
        # Attention scores
        attn_scores = self.attention(lstm_out)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch, seq_len, 1]
        
        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden]
        
        # Classification
        logits = self.fc(context)
        
        return logits, attn_weights.squeeze(-1)

# TODO: обучите модель
# TODO: визуализируйте attention weights для тестовых примеров
```

---

### **Задача 9: Multi-Task Learning с Shared LSTM**

**Условие:** Обучите одну LSTM на нескольких связанных задачах одновременно.

**Требования:**
1. Задачи (на текстовых данных):
   - Sentiment classification (positive/negative)
   - Topic classification (sports/politics/tech)
   - Length prediction (short/medium/long)
   
2. Архитектура:
   - Shared LSTM encoder
   - 3 отдельных classification heads
   
3. Экспериментируйте с:
   - Weights для losses разных задач
   - Gradient normalization
   
4. Сравните с 3 отдельными моделями

**Ожидаемый результат:** Multi-task модель эффективнее и может улучшить качество.

```python
class MultiTaskLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        
        # Shared encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        
        # Task-specific heads
        self.sentiment_head = nn.Linear(hidden_size, 2)
        self.topic_head = nn.Linear(hidden_size, 3)
        self.length_head = nn.Linear(hidden_size, 3)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (h_n, _) = self.lstm(embedded)
        
        # Use final hidden state
        h = h_n[-1]
        
        # Task predictions
        sentiment = self.sentiment_head(h)
        topic = self.topic_head(h)
        length = self.length_head(h)
        
        return sentiment, topic, length

# Training loop
def train_multitask(model, dataloader, epochs):
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        for x, y_sent, y_topic, y_len in dataloader:
            optimizer.zero_grad()
            
            pred_sent, pred_topic, pred_len = model(x)
            
            # Multi-task loss
            loss_sent = F.cross_entropy(pred_sent, y_sent)
            loss_topic = F.cross_entropy(pred_topic, y_topic)
            loss_len = F.cross_entropy(pred_len, y_len)
            
            # Weighted sum
            loss = loss_sent + 0.5 * loss_topic + 0.3 * loss_len
            
            loss.backward()
            optimizer.step()

# TODO: создайте multi-task датасет
# TODO: обучите и сравните с single-task моделями
```

---

### **Задача 10: LSTM for Anomaly Detection**

**Условие:** Используйте LSTM для детекции аномалий во временных рядах.

**Требования:**
1. Подход: Autoencoder LSTM
   - Encoder LSTM: кодирует нормальную последовательность
   - Decoder LSTM: восстанавливает последовательность
   - Аномалия = высокая reconstruction error
   
2. Обучите на нормальных данных (только здоровые паттерны)
3. Тестируйте на данных с аномалиями
4. Определите threshold для детекции аномалий
5. Визуализируйте:
   - Original vs Reconstructed
   - Reconstruction error over time
   - Detected anomalies

**Ожидаемый результат:** LSTM Autoencoder успешно детектирует аномалии.

```python
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len):
        super().__init__()
        
        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Decoder
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)
        
        self.seq_len = seq_len
    
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        
        # Encode
        _, (h_n, c_n) = self.encoder(x)
        
        # Decode: repeat hidden state for all timesteps
        decoder_input = h_n.repeat(1, self.seq_len, 1).transpose(0, 1)
        decoder_out, _ = self.decoder(decoder_input, (h_n, c_n))
        
        # Reconstruct
        reconstruction = self.output_layer(decoder_out)
        
        return reconstruction

def detect_anomalies(model, data, threshold):
    """Детектирует аномалии на основе reconstruction error"""
    model.eval()
    
    with torch.no_grad():
        reconstruction = model(data)
        error = torch.mean((data - reconstruction) ** 2, dim=(1, 2))
    
    anomalies = error > threshold
    return anomalies, error

# TODO: обучите autoencoder на нормальных данных
# TODO: подберите threshold
# TODO: детектируйте аномалии на тестовых данных
```

---

## 📝 Дополнительные вопросы

1. **Почему LSTM лучше RNN на длинных последовательностях?**
2. **Когда GRU предпочтительнее LSTM?**
3. **Как выбрать размер hidden_size?**
4. **Влияет ли порядок gates на качество?**
5. **Можно ли visualize что LSTM "запомнила"?**

---

## 🎯 Критерии успешного выполнения

- ✅ Понимаете архитектуру LSTM (3 gates, cell state)
- ✅ Понимаете архитектуру GRU (2 gates, упрощенная)
- ✅ Знаете, почему LSTM решает vanishing gradients
- ✅ Умеете применять LSTM/GRU для разных задач
- ✅ Можете сравнить LSTM, GRU, RNN по качеству и скорости
- ✅ Умеете визуализировать gate activations

---

## 📚 Полезные ресурсы

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [LSTM Paper](http://www.bioinf.jku.at/publications/older/2604.pdf)
- [GRU Paper](https://arxiv.org/abs/1406.1078)
- [Empirical Evaluation of Gated RNNs](https://arxiv.org/abs/1412.3555)
- [PyTorch LSTM Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
