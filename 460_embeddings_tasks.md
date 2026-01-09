### **Задачи: Эмбеддинги слов**

**Цель:** Научиться работать с word embeddings, обучать Word2Vec/FastText, использовать pre-trained embeddings, визуализировать семантические отношения.

---

## 🟢 Базовый уровень

### **Задача 1: Обучение Word2Vec с нуля**

Обучите Word2Vec модель на корпусе текстов.

```python
from gensim.models import Word2Vec

# Подготовка данных
sentences = [
    ["machine", "learning", "is", "awesome"],
    ["deep", "learning", "is", "powerful"],
    # ... добавьте больше предложений
]

# Обучение
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)

# Тестирование
print(model.wv.most_similar('learning'))
```

**Требования:** Используйте датасет новостей или Wikipedia, обучите модель, найдите похожие слова для 10 примеров.

---

### **Задача 2: Визуализация Embeddings с t-SNE**

Визуализируйте word embeddings в 2D пространстве.

```python
from sklearn.manifold import TSNE

def visualize_embeddings(model, words):
    vectors = [model.wv[word] for word in words]
    tsne = TSNE(n_components=2)
    vectors_2d = tsne.fit_transform(vectors)
    
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]))
    plt.show()
```

**Требования:** Выберите 50 слов из разных категорий (животные, города, профессии), визуализируйте, проанализируйте кластеры.

---

### **Задача 3: Семантическая арифметика**

Проверьте семантические аналогии с word embeddings.

**Примеры:**
```python
# King - Man + Woman = Queen?
result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'])

# Paris - France + Germany = Berlin?
result = model.wv.most_similar(positive=['paris', 'germany'], negative=['france'])
```

**Требования:** Создайте 20 аналогий, проверьте сколько правильных, проанализируйте ошибки.

---

## 🟡 Продвинутый уровень

### **Задача 4: Сравнение Word2Vec и GloVe**

Сравните Word2Vec и GloVe на одинаковых задачах.

```python
import gensim.downloader as api

w2v = api.load('word2vec-google-news-300')
glove = api.load('glove-wiki-gigaword-100')

# Сравните на similarity tasks
```

**Требования:** Измерьте correlation на similarity datasets (SimLex-999, WordSim-353).

---

### **Задача 5: FastText для OOV слов**

Обучите FastText и продемонстрируйте работу с OOV словами.

```python
from gensim.models import FastText

model = FastText(sentences, vector_size=100, window=5, min_count=1)

# Тестируйте на неизвестных словах
print(model.wv['unknownword123'])  # работает!
```

**Требования:** Создайте тестовый набор с опечатками и редкими словами, сравните FastText с Word2Vec.

---

### **Задача 6: Fine-tuning Pre-trained Embeddings**

Используйте pre-trained GloVe для sentiment analysis, сравните frozen vs fine-tuned.

```python
class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_embeddings, freeze=True):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze)
        self.lstm = nn.LSTM(100, 128, batch_first=True)
        self.fc = nn.Linear(128, 2)
```

**Требования:** Обучите две модели (frozen и fine-tuned), сравните accuracy и overfitting.

---

## 🔴 Экспертный уровень

### **Задача 7: Обучение Domain-Specific Embeddings**

Обучите специализированные embeddings на техническом корпусе (медицинские тексты, код, научные статьи).

**Требования:** Сравните domain-specific vs general embeddings на domain tasks.

---

### **Задача 8: Multilingual Embeddings**

Обучите или используйте multilingual embeddings (fastText multilingual).

```python
# Cross-lingual similarity
en_vector = model.wv['computer']
ru_vector = model.wv['компьютер']
similarity = cosine_similarity(en_vector, ru_vector)
```

**Требования:** Протестируйте на cross-lingual tasks (translation pairs, similarity).

---

### **Задача 9: Contextualized Embeddings (ELMo preview)**

Сравните static embeddings (Word2Vec) с contextualized (используя простую BiLSTM).

**Идея:** Word2Vec дает один вектор для "bank", но значение зависит от контекста ("river bank" vs "financial bank").

---

### **Задача 10: Embeddings для Recommendation**

Используйте item embeddings для рекомендательной системы.

**Подход:**
- Обучите embeddings для товаров/фильмов на основе user interactions
- Используйте похожесть embeddings для рекомендаций

---

## 🎯 Критерии успешного выполнения

- ✅ Понимаете difference between one-hot и dense embeddings
- ✅ Умеете обучать Word2Vec/FastText
- ✅ Знаете как использовать pre-trained embeddings
- ✅ Можете визуализировать embeddings
- ✅ Понимаете семантические свойства (векторная арифметика)

---

## 📚 Ресурсы

- [Word2Vec Paper](https://arxiv.org/abs/1301.3781)
- [GloVe Paper](https://nlp.stanford.edu/pubs/glove.pdf)
- [FastText Paper](https://arxiv.org/abs/1607.04606)
- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [Pre-trained Embeddings](https://nlp.stanford.edu/projects/glove/)
