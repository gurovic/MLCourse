# Обзор и сравнение библиотек Deep Learning

```python
# Установка библиотек для сравнения
# !pip install torch torchvision tensorflow keras jax jaxlib
```

---

## Введение: Ландшафт фреймворков глубокого обучения

**Deep Learning frameworks** — это программные библиотеки, предоставляющие инструменты для построения, обучения и развертывания нейронных сетей. Выбор фреймворка влияет на скорость разработки, производительность и возможности масштабирования.

**Основные игроки (по состоянию на 2026):**
- **PyTorch** — гибкость и простота для исследований
- **TensorFlow/Keras** — промышленное развертывание и мобильные устройства
- **JAX** — высокопроизводительные вычисления и автоматическое дифференцирование
- **ONNX** — формат обмена моделями между фреймворками
- **Исторические:** Caffe, Theano, MXNet (Apache)

**Критерии выбора фреймворка:**
- Простота API и кривая обучения
- Производительность (скорость обучения/инференса)
- Экосистема (предобученные модели, инструменты)
- Поддержка развертывания (mobile, edge, cloud)
- Сообщество и документация

---

## PyTorch: Фреймворк для исследователей

### Основные характеристики

**Разработчик:** Meta AI (Facebook AI Research)  
**Год создания:** 2016  
**Философия:** Pythonic API, динамический граф вычислений, простота отладки

**Преимущества:**
- ✅ Интуитивный Python-подобный синтаксис
- ✅ Динамический граф (Define-by-Run) — удобная отладка
- ✅ Доминирование в исследовательском сообществе (85-90% статей на конференциях)
- ✅ Отличная интеграция с NumPy
- ✅ TorchScript для оптимизации и развертывания
- ✅ Активное сообщество и быстрое внедрение новых идей

**Недостатки:**
- ❌ Меньше готовых решений для промышленного развертывания
- ❌ Менее зрелая мобильная поддержка по сравнению с TF Lite
- ❌ Требует больше кода для production-ready решений

### Пример кода: Простая нейросеть

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Генерация синтетических данных для классификации
X = torch.randn(1000, 20)
y = (X.sum(dim=1) > 0).long()

# Определение модели
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(20, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Обучение
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    total_loss = 0
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(loader):.4f}")
```

**Ключевые особенности кода:**
- `nn.Module` — базовый класс для всех моделей
- `forward()` — определяет прямой проход (forward pass)
- `optimizer.zero_grad()` — обязательное обнуление градиентов
- `loss.backward()` — автоматическое вычисление градиентов
- `optimizer.step()` — обновление весов

### Экосистема PyTorch

| Библиотека | Назначение |
|-----------|-----------|
| **torchvision** | Computer Vision (датасеты, трансформации, модели) |
| **torchtext** | NLP (обработка текста, датасеты) |
| **torchaudio** | Обработка аудио |
| **PyTorch Lightning** | Упрощение кода обучения |
| **Hugging Face Transformers** | Предобученные NLP/CV модели |
| **timm** | Современные архитектуры компьютерного зрения |
| **TorchServe** | Развертывание моделей в production |

---

## TensorFlow/Keras: Промышленный стандарт

### Основные характеристики

**Разработчик:** Google Brain  
**Год создания:** 2015 (TensorFlow), 2017 (Keras интегрирован)  
**Философия:** Масштабирование, production-ready, кроссплатформенность

**Преимущества:**
- ✅ TensorBoard — мощная визуализация обучения
- ✅ TensorFlow Lite — оптимизация для мобильных устройств
- ✅ TensorFlow.js — запуск моделей в браузере
- ✅ TensorFlow Serving — готовая система для production
- ✅ Keras API — высокоуровневый простой интерфейс

**Недостатки:**
- ❌ Более сложный низкоуровневый API (TensorFlow 2.x исправил это)
- ❌ Статический граф (хотя Eager Execution решает проблему)
- ❌ Медленнее адаптируется к новым исследовательским идеям

### Пример кода: Аналогичная модель на Keras

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Генерация синтетических данных
X = np.random.randn(1000, 20).astype(np.float32)
y = (X.sum(axis=1) > 0).astype(np.int32)

# Определение модели (Sequential API)
model = keras.Sequential([
    layers.Dense(50, activation='relu', input_shape=(20,)),
    layers.Dense(2, activation='softmax')
])

# Компиляция
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Обучение
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
print(f"Финальная точность: {history.history['accuracy'][-1]:.4f}")
```

**Отличия от PyTorch:**
- Более высокоуровневый API — меньше кода
- `compile()` — явное указание оптимизатора и функции потерь
- `fit()` — единая функция для обучения (в PyTorch нужен цикл)
- Автоматическая валидация через `validation_split`

### Functional API: Более сложные архитектуры

```python
# Пример с несколькими входами
input_a = keras.Input(shape=(10,), name='input_a')
input_b = keras.Input(shape=(5,), name='input_b')

# Обработка первого входа
x = layers.Dense(30, activation='relu')(input_a)
x = layers.Dense(20, activation='relu')(x)

# Обработка второго входа
y = layers.Dense(15, activation='relu')(input_b)

# Объединение
combined = layers.concatenate([x, y])
output = layers.Dense(1, activation='sigmoid')(combined)

model = keras.Model(inputs=[input_a, input_b], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy')
```

### Экосистема TensorFlow

| Инструмент | Назначение |
|-----------|-----------|
| **TensorFlow Hub** | Репозиторий предобученных моделей |
| **TensorFlow Lite** | Оптимизация для мобильных/embedded |
| **TensorFlow.js** | Запуск моделей в браузере |
| **TensorFlow Extended (TFX)** | Production ML pipeline |
| **TensorFlow Serving** | Развертывание моделей через REST/gRPC |
| **Keras Tuner** | Автоматический подбор гиперпараметров |

---

## JAX: Следующее поколение вычислений

### Основные характеристики

**Разработчик:** Google Research  
**Год создания:** 2018  
**Философия:** Композируемые трансформации, функциональное программирование

**Преимущества:**
- ✅ Автоматическое дифференцирование (grad, jacobian, hessian)
- ✅ JIT-компиляция через XLA (очень быстро)
- ✅ Автоматическая векторизация (vmap)
- ✅ Простое распараллеливание (pmap)
- ✅ Чистый функциональный стиль

**Недостатки:**
- ❌ Меньше высокоуровневых абстракций (нужны библиотеки типа Flax, Haiku)
- ❌ Крутая кривая обучения (функциональное программирование)
- ❌ Меньшее сообщество и экосистема

### Пример кода: JAX + Flax

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Определение модели
class SimpleNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=50)(x)
        x = nn.relu(x)
        x = nn.Dense(features=2)(x)
        return x

# Инициализация
model = SimpleNet()
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones((1, 20)))

# Функция потерь
def loss_fn(params, X, y):
    logits = model.apply(params, X)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss

# Градиенты
grad_fn = jax.grad(loss_fn)

# Оптимизатор
optimizer = optax.adam(0.001)
opt_state = optimizer.init(params)

# Шаг обучения (JIT-компилируется для скорости)
@jax.jit
def train_step(params, opt_state, X, y):
    grads = grad_fn(params, X, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Обучение
X = jax.random.normal(key, (1000, 20))
y = (X.sum(axis=1) > 0).astype(jnp.int32)

for epoch in range(10):
    params, opt_state = train_step(params, opt_state, X, y)
    loss = loss_fn(params, X, y)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
```

**Особенности JAX:**
- Функциональный стиль — параметры передаются явно
- `@jax.jit` — JIT-компиляция для максимальной скорости
- `jax.grad()` — автоматическое дифференцирование любых функций
- Требует понимания функционального программирования
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state


### Экосистема JAX

| Библиотека | Назначение |
|-----------|-----------|
| **Flax** | Нейросети (высокоуровневый API) |
| **Haiku** | Нейросети от DeepMind |
| **Optax** | Оптимизаторы и градиентные трансформации |
| **Equinox** | Современный ООП-подход к нейросетям |
| **Diffrax** | Дифференциальные уравнения |

---

## Сравнительная таблица фреймворков

### Общее сравнение

| Критерий | PyTorch | TensorFlow/Keras | JAX |
|----------|---------|------------------|-----|
| **Кривая обучения** | Легкая | Средняя | Сложная |
| **Скорость разработки** | Высокая | Высокая (Keras) | Средняя |
| **Производительность** | Высокая | Высокая | Очень высокая |
| **Гибкость** | Очень высокая | Средняя | Очень высокая |
| **Production-ready** | Средний | Отличный | Средний |
| **Мобильная поддержка** | Средняя | Отличная | Слабая |
| **Исследовательское сообщество** | Доминирует | Уменьшается | Растет |
| **Документация** | Отличная | Отличная | Хорошая |
| **Отладка** | Простая | Средняя | Сложная |

### Производительность (бенчмарки)

**Время обучения ResNet-50 на ImageNet (1 эпоха, V100 GPU):**
- PyTorch: ~2.5 часа
- TensorFlow: ~2.4 часа
- JAX: ~2.3 часа

*Примечание: бенчмарки приблизительные и зависят от версий библиотек, конфигурации, и оптимизации кода*

**Скорость инференса (batch=1, CPU):**
- PyTorch: 10 мс
- TensorFlow: 9 мс
- ONNX Runtime: 7 мс

### Экосистема и сообщество (2024)

| Метрика | PyTorch | TensorFlow | JAX |
|---------|---------|------------|-----|
| GitHub Stars | 80K+ | 185K+ | 30K+ |
| Статьи на конференциях ML | ~85-90% | ~5-10% | ~5% |
| Вакансии (LinkedIn) | 15K+ | 12K+ | 500+ |
| StackOverflow вопросов | 90K+ | 180K+ | 3K+ |

*Примечание: данные являются приблизительными и отражают общие тенденции*

---

## Специализированные фреймворки

### PyTorch Lightning

**Назначение:** Упрощение кода обучения PyTorch, устранение boilerplate

**Преимущества:**
- Автоматическое логирование метрик
- Простое распределенное обучение на нескольких GPU
- Встроенные callbacks для early stopping, checkpointing
- Меньше кода — фокус на логике модели, а не на деталях обучения
- Совместимость с обычным PyTorch кодом

**Пример структуры:**
```python
import pytorch_lightning as pl

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Определение слоев
    
    def forward(self, x):
        # Прямой проход
        pass
    
    def training_step(self, batch, batch_idx):
        # Логика одного шага обучения
        return loss
    
    def configure_optimizers(self):
        # Настройка оптимизатора
        return optimizer
```

Обучение упрощается до одной строки: `trainer.fit(model, train_loader)`

### Hugging Face Transformers

**Назначение:** Предобученные трансформеры для NLP и CV

**Концепция:** Hub с тысячами готовых моделей, которые можно использовать для:
- Классификации текста
- Генерации текста
- Перевода
- Ответов на вопросы
- Классификации изображений

**Основной workflow:**
1. Выбрать модель на [huggingface.co/models](https://huggingface.co/models)
2. Загрузить модель и токенайзер через `AutoModel` и `AutoTokenizer`
3. Fine-tune на своих данных (опционально)
4. Использовать для предсказаний

**Поддержка:**
- 200+ архитектур (BERT, GPT, T5, CLIP, ...)
- 50K+ предобученных моделей
- Работает с PyTorch и TensorFlow
- Pipelines для быстрого прототипирования

**Пример использования pipelines:**
```python
from transformers import pipeline

# Классификация тональности (sentiment analysis)
classifier = pipeline("sentiment-analysis")
result = classifier("I love machine learning!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
```

### Fast.ai

**Назначение:** Высокоуровневый API для быстрого прототипирования

```python
from fastai.vision.all import *

# Создание DataLoader
dls = ImageDataLoaders.from_folder('path/to/data', valid_pct=0.2)

# Обучение за 2 строки
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(5)
```

**Преимущества:**
- Очень быстрое прототипирование
- Отличные курсы для обучения
- Встроенные best practices

---

## ONNX: Совместимость фреймворков

### Что такое ONNX?

**Open Neural Network Exchange** — открытый формат для представления моделей машинного обучения.

**Зачем нужен:**
- Обучение в одном фреймворке, инференс в другом
- Оптимизация моделей для production
- Кроссплатформенное развертывание

### Конвертация модели PyTorch → ONNX

```python
import torch
import torch.onnx

# Обученная модель (например, наша SimpleNet)
model = SimpleNet()
model.eval()  # Переводим в режим инференса

# Создаем пример входных данных
dummy_input = torch.randn(1, 20)

# Экспорт в ONNX
torch.onnx.export(
    model,                    # модель
    dummy_input,              # пример входа
    "model.onnx",            # имя файла
    export_params=True,       # сохранить веса
    input_names=['input'],    # имена входов
    output_names=['output'],  # имена выходов
    dynamic_axes={            # динамические размерности
        'input': {0: 'batch_size'}, 
        'output': {0: 'batch_size'}
    }
)
print("Модель сохранена в model.onnx")
```
```

### Инференс через ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Загрузка модели
session = ort.InferenceSession("model.onnx")

# Инференс
input_data = np.random.randn(5, 20).astype(np.float32)
outputs = session.run(None, {'input': input_data})
print(outputs[0])
```

**Преимущества ONNX Runtime:**
- Быстрее на 2-10x чем нативный инференс
- Кроссплатформенность (CPU, GPU, mobile)
- Оптимизации (квантизация, pruning)

---

## Выбор фреймворка: Рекомендации

### Для исследований и прототипирования
**Выбор: PyTorch**
- Быстрое итерирование идей
- Простая отладка
- Поддержка сообщества

### Для production и масштабирования
**Выбор: TensorFlow/Keras**
- Готовые инструменты развертывания
- Мобильная оптимизация
- Зрелая экосистема

### Для высокопроизводительных вычислений
**Выбор: JAX**
- Максимальная скорость
- Сложные математические операции
- Исследования в reinforcement learning

### Для NLP задач
**Выбор: PyTorch + Hugging Face**
- Лучшие предобученные модели
- Активное развитие
- Большое сообщество

### Для компьютерного зрения
**Выбор: PyTorch + torchvision/timm**
- Современные архитектуры
- Простота fine-tuning
- Отличная документация

### Для мобильных приложений
**Выбор: TensorFlow Lite**
- Лучшая оптимизация для мобильных
- Большой выбор предобученных моделей
- Отличная документация

### Для edge-устройств (IoT)
**Выбор: TensorFlow Lite или ONNX Runtime**
- Минимальный размер моделей
- Оптимизация для слабого железа
- Кроссплатформенность

---

## Тренды и будущее

### Современные тенденции (2024-2026)

1. **Унификация API**
   - Keras 3.0 поддерживает PyTorch, JAX, TensorFlow как бэкенды
   - ONNX становится стандартом обмена моделями

2. **Оптимизация для inference**
   - Квантизация (INT8, FP16)
   - Pruning и дистилляция
   - Специализированные процессоры (TPU, Apple Neural Engine)

3. **Федеративное обучение**
   - TensorFlow Federated
   - PySyft для приватного ML

4. **MLOps интеграция**
   - Weights & Biases
   - MLflow
   - TensorBoard

### Новые игроки

**Mojo** (2023-2025) — новый язык программирования для AI:
- Python-совместимость с C++ производительностью
- Оптимизирован для современных процессоры (CPU/GPU/TPU)
- Находится в активной разработке, экосистема формируется

**Apple MLX** (2023) — фреймворк от Apple для Apple Silicon (M1/M2/M3/M4):

```python
# Пример MLX (на Apple Silicon)
import mlx.core as mx
import mlx.nn as nn

# Похож на PyTorch, но оптимизирован для Apple Silicon
model = nn.Sequential(
    nn.Linear(20, 50),
    nn.ReLU(),
    nn.Linear(50, 2)
)
```

---

## Практические примеры миграции

### Миграция PyTorch → TensorFlow

```python
# PyTorch
class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.fc = nn.Linear(32*26*26, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 32*26*26)
        x = self.fc(x)
        return x

# Аналог в TensorFlow/Keras
def keras_model():
    return keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(10)
    ])
```

### Миграция TensorFlow → PyTorch

```python
# TensorFlow/Keras
keras_model = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(None, 10)),
    layers.LSTM(32),
    layers.Dense(1)
])

# Аналог в PyTorch
class PyTorchLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(10, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x[:, -1, :])  # Берем последний временной шаг
        return x
```

---

## Заключение

**Выбор фреймворка зависит от ваших целей:**

### Для школьников и начинающих
1. **Первый фреймворк:** Начните с **Keras** (простой API) или **PyTorch** (популярен в обучающих материалах)
2. **Для экспериментов:** PyTorch — легко отлаживать и быстро итерировать
3. **Для проектов:** TensorFlow/Keras — много готовых решений и туториалов

### Универсальные рекомендации по областям
1. **Исследователям:** PyTorch (гибкость, динамический граф, активное сообщество)
2. **ML-инженерам:** TensorFlow (production-ready инструменты, зрелая экосистема)
3. **Исследователям в области оптимизации:** JAX (максимальная производительность)
4. **Для NLP:** PyTorch + Hugging Face (лучшие предобученные модели)
5. **Для мобильных приложений:** TensorFlow Lite (лучшая оптимизация)

### Популярные комбинации
- **Исследование:** PyTorch + Hugging Face + Weights & Biases
- **Production:** TensorFlow + TensorFlow Serving + Kubernetes
- **Мобильная разработка:** TensorFlow Lite + Core ML (iOS)
- **Высокая производительность:** JAX + TPU + XLA

### Ключевые советы для изучения
- **Не привязывайтесь к одному фреймворку** — понимание основ позволит легко переключаться
- **Используйте ONNX** для переноса моделей между фреймворками
- **Следите за трендами**, но выбирайте проверенные решения для серьезных проектов
- **Экосистема важнее** самого фреймворка (библиотеки, сообщество, документация)
- **Начните с практики** — теория важна, но практический опыт критичен

### Траектория обучения
1. **Месяц 1:** Изучите основы одного фреймворка (PyTorch или Keras)
2. **Месяц 2:** Реализуйте 3-5 проектов в выбранном фреймворке
3. **Месяц 3:** Познакомьтесь со вторым фреймворком, сравните подходы
4. **Месяц 4+:** Изучайте специализированные инструменты (Hugging Face, Lightning, etc.)

> **"Лучший фреймворк — тот, который позволяет вам решать задачи быстро и эффективно. В 2026 году это означает знание хотя бы PyTorch или TensorFlow, плюс понимание экосистемы вокруг них."**

### Что важно понимать
- Все фреймворки решают одни и те же задачи, но по-разному
- Концепции (градиенты, backpropagation, оптимизаторы) универсальны
- Изучив один фреймворк хорошо, вы легко освоите другие
- Главное — понимать, **что** делает код, а не только **как** он написан

---

## Полезные ресурсы

### Официальная документация
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [TensorFlow Docs](https://www.tensorflow.org/api_docs)
- [JAX Docs](https://jax.readthedocs.io/)

### Сравнения и бенчмарки
- [Papers With Code - Framework Comparison](https://paperswithcode.com/)
- [MLPerf - ML Benchmarks](https://mlcommons.org/en/training-normal-20/)

### Обучающие материалы
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [JAX Tutorial](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
