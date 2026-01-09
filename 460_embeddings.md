# Эмбеддинги слов

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# !pip install torch numpy matplotlib scikit-learn gensim
```

---

## 🟢 Базовый уровень: От One-Hot к Dense Embeddings

### 1.1 Проблема: One-Hot Encoding

**Обычное представление слов:**
```python
vocab = ["cat", "dog", "bird"]
cat  = [1, 0, 0]
dog  = [0, 1, 0]
bird = [0, 0, 1]
```

**Проблемы:**
- ❌ **Размерность = vocab_size** (10K-100K слов → очень большие векторы)
- ❌ **Нет семантики:** расстояние между любыми двумя словами одинаковое
- ❌ **Разреженность:** только один элемент = 1, остальные = 0

---

### 1.2 Word Embeddings: Плотные представления

**Идея:** Представлять слова как **плотные** векторы в низкоразмерном пространстве (50-300 измерений).

```python
cat  = [0.2, 0.8, -0.3, 0.1, ...]  # 100 измерений
dog  = [0.3, 0.7, -0.2, 0.2, ...]
bird = [-0.1, 0.3, 0.8, -0.5, ...]
```

**Свойство:** Семантически похожие слова → близкие векторы!

```
distance(cat, dog) < distance(cat, airplane)
```

---

### 1.3 Embedding Layer в PyTorch

```python
# Создание embedding слоя
vocab_size = 10000
embed_dim = 100

embedding = nn.Embedding(vocab_size, embed_dim)

# Индексы слов
word_indices = torch.LongTensor([45, 123, 678])  # [cat, dog, bird]

# Получение embeddings
word_vectors = embedding(word_indices)
print(word_vectors.shape)  # [3, 100]
```

**Обучение embeddings:** Веса embedding слоя обучаются вместе с моделью через backpropagation.

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        
        # Embeddings обучаются!
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        output, (h_n, _) = self.lstm(embedded)
        logits = self.fc(h_n[-1])
        return logits
```

---

## 🟡 Продвинутый уровень: Word2Vec

### 2.1 Архитектуры Word2Vec

**Две архитектуры:**

**1. CBOW (Continuous Bag of Words):** Предсказывает центральное слово по контексту.
```
Context: "the quick [?] jumps over"
Target:  "fox"
```

**2. Skip-gram:** Предсказывает контекст по центральному слову.
```
Center:  "fox"
Targets: "the", "quick", "jumps", "over"
```

**Skip-gram обычно лучше** для малых датасетов и редких слов.

---

### 2.2 Обучение Word2Vec

```python
from gensim.models import Word2Vec

# Корпус текстов (list of tokenized sentences)
sentences = [
    ["the", "quick", "brown", "fox"],
    ["jumps", "over", "the", "lazy", "dog"],
    # ... миллионы предложений
]

# Обучение Word2Vec
model = Word2Vec(
    sentences,
    vector_size=100,  # размерность embeddings
    window=5,         # размер окна контекста
    min_count=2,      # минимальная частота слова
    sg=1,             # 1=skip-gram, 0=CBOW
    workers=4
)

# Использование
vector_fox = model.wv['fox']  # вектор слова "fox"
print(vector_fox.shape)  # (100,)

# Похожие слова
similar = model.wv.most_similar('fox', topn=5)
print(similar)
# [('dog', 0.85), ('cat', 0.82), ('wolf', 0.79), ...]
```

---

### 2.3 Семантические свойства

**Удивительное свойство:** Векторная арифметика отражает семантику!

```python
# King - Man + Woman ≈ Queen
result = model.wv.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=1
)
print(result)  # [('queen', 0.73)]

# Paris - France + Germany ≈ Berlin
result = model.wv.most_similar(
    positive=['paris', 'germany'],
    negative=['france']
)
print(result[0])  # ('berlin', ...)
```

---

## 🟡 Продвинутый уровень: GloVe

### 3.1 Отличия от Word2Vec

**Global Vectors (GloVe):** Использует статистику совместной встречаемости слов.

**Идея:** Если слова часто появляются вместе → их векторы должны быть близки.

```python
# Загрузка pre-trained GloVe
import gensim.downloader as api

glove = api.load('glove-wiki-gigaword-100')  # 100-мерные векторы

# Использование
vector = glove['computer']
similar = glove.most_similar('computer', topn=5)
print(similar)
```

---

## 🟡 Продвинутый уровень: FastText

### 4.1 Embeddings для подслов

**Проблема Word2Vec/GloVe:** Нет представлений для OOV (out-of-vocabulary) слов.

**FastText:** Использует **character n-grams**.

```
Word: "apple"
N-grams: <ap, app, ppl, ple, le>
Embedding(apple) = sum(embeddings of n-grams)
```

**Преимущество:** Может генерировать embeddings для **неизвестных слов**!

```python
from gensim.models import FastText

model = FastText(sentences, vector_size=100, window=5, min_count=1)

# Работает даже для слов вне словаря
vector_unknown = model.wv['unknownword123']  # генерирует вектор!
```

---

## 🔴 Экспертный уровень: Использование Pre-trained Embeddings

### 5.1 Загрузка в PyTorch модель

```python
import gensim.downloader as api

# Загружаем pre-trained embeddings
glove = api.load('glove-wiki-gigaword-100')

# Создаем vocabulary
word2idx = {word: idx for idx, word in enumerate(glove.index_to_key)}
vocab_size = len(word2idx)

# Создаем weight matrix для Embedding
embedding_matrix = np.zeros((vocab_size, 100))
for word, idx in word2idx.items():
    embedding_matrix[idx] = glove[word]

# Создаем Embedding с pre-trained weights
embedding = nn.Embedding(vocab_size, 100)
embedding.weight = nn.Parameter(torch.FloatTensor(embedding_matrix))

# Опция 1: Заморозить embeddings (не обучать)
embedding.weight.requires_grad = False

# Опция 2: Fine-tune embeddings
embedding.weight.requires_grad = True
```

---

### 5.2 Визуализация Embeddings

```python
def visualize_embeddings(model, words, method='tsne'):
    """Визуализирует word embeddings в 2D"""
    
    # Получаем векторы
    vectors = np.array([model.wv[word] for word in words])
    
    # Снижаем размерность до 2D
    if method == 'tsne':
        tsne = TSNE(n_components=2, random_state=42)
        vectors_2d = tsne.fit_transform(vectors)
    
    # Визуализация
    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
    
    for i, word in enumerate(words):
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]))
    
    plt.title('Word Embeddings Visualization')
    plt.show()

# Пример
words = ['king', 'queen', 'man', 'woman', 'prince', 'princess',
         'car', 'truck', 'bike', 'vehicle']
visualize_embeddings(model, words)
```

---

## 📊 Сравнение подходов

| Метод | Размер | OOV | Семантика | Когда использовать |
|-------|--------|-----|-----------|-------------------|
| **One-Hot** | vocab_size | ❌ | ❌ | Не используется |
| **Word2Vec** | 50-300 | ❌ | ✅ | Стандартный выбор |
| **GloVe** | 50-300 | ❌ | ✅ | Альтернатива Word2Vec |
| **FastText** | 50-300 | ✅ | ✅ | Много морфологии, OOV |
| **Learned** | 50-300 | ❌ | ✅ | Task-specific |

---

## 🎯 Ключевые выводы

1. **Embeddings** представляют слова как плотные векторы
2. **Word2Vec** (CBOW/Skip-gram) обучается предсказывать контекст
3. **GloVe** использует статистику co-occurrence
4. **FastText** работает с character n-grams (handles OOV)
5. **Pre-trained embeddings** ускоряют обучение на малых данных
6. **Векторная арифметика** отражает семантические отношения

---

## 📚 Материалы

- [Word2Vec Paper](https://arxiv.org/abs/1301.3781)
- [GloVe Paper](https://nlp.stanford.edu/pubs/glove.pdf)
- [FastText Paper](https://arxiv.org/abs/1607.04606)
- [Gensim Documentation](https://radimrehurek.com/gensim/)
