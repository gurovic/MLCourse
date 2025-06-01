# **Feature Engineering для текстовых данных**  

## **Введение в Feature Engineering для текста**  
Преобразование текста в числовые признаки — ключевой этап в NLP. Рассмотрим:  
- 📊 **Классические подходы** (статистические признаки)  
- 🔍 **Лингвистические особенности** (морфология, синтаксис)  
- 🧩 **Специализированные техники** для русского языка  

**Основные задачи:**  
- Улучшение качества моделей ML  
- Снижение размерности  
- Выделение смысловых компонентов  

---

## **🟢 Базовый уровень (Статистические признаки)**  

### **1.1 Частотные характеристики**  
```python
def text_statistics(text):
    return {
        'num_chars': len(text),
        'num_words': len(text.split()),
        'num_unique_words': len(set(text.split())),
        'avg_word_length': sum(len(word) for word in text.split()) / len(text.split()),
        'num_sentences': text.count('.') + text.count('!') + text.count('?')
    }

text = "Машинное обучение. Это интересно!"
stats = text_statistics(text)
# {'num_chars': 27, 'num_words': 4, ...}
```

### **1.2 Признаки на основе N-грамм**  
```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["машинное обучение работает"]
vectorizer = CountVectorizer(ngram_range=(2, 2))  # биграммы
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())  # ['машинное обучение', 'обучение работает']
```

### **1.3 Признаки стиля текста**  
```python
def style_features(text):
    return {
        'punct_density': sum(1 for char in text if char in '.,!?;:') / len(text),
        'capitals_ratio': sum(1 for char in text if char.isupper()) / len(text),
        'exclamations': text.count('!')
    }
```

---

## **🟡 Продвинутый уровень (Лингвистические признаки)**  

### **2.1 Морфологические признаки (для русского языка)**  
```python
from pymorphy2 import MorphAnalyzer

morph = MorphAnalyzer()

def morph_features(word):
    parse = morph.parse(word)[0]
    return {
        'POS': parse.tag.POS,  # часть речи
        'case': parse.tag.case,
        'tense': parse.tag.tense if parse.tag.tense else None
    }

morph_features("бегу")  # {'POS': 'VERB', 'case': None, 'tense': 'pres'}
```

### **2.2 Синтаксическая сложность**  
```python
import spacy

nlp = spacy.load("ru_core_news_sm")

def syntax_complexity(text):
    doc = nlp(text)
    return {
        'avg_deps': sum(len(token.dep_) for token in doc) / len(doc),
        'num_clauses': sum(1 for sent in doc.sents for token in sent if token.dep_ == 'conj')
    }
```

### **2.3 Эмбеддинги FastText**  
```python
import fasttext
import fasttext.util

# Загрузка русской модели
ft = fasttext.load_model('cc.ru.300.bin')

def get_text_vector(text):
    words = text.split()
    return sum(ft.get_word_vector(word) for word in words) / len(words)  # усредненный вектор
```

---

## **🔴 Экспертный уровень (Композитные признаки)**  

### **3.1 Тематические признаки (LDA)**  
```python
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=5)
lda.fit(tfidf_matrix)  # матрица TF-IDF

def get_topic_features(text):
    tfidf = vectorizer.transform([text])
    return lda.transform(tfidf)[0]  # распределение по темам
```

### **3.2 Персонализированные признаки**  
```python
def domain_specific_features(text):
    return {
        'has_legal_terms': int(any(word in text for word in ['договор', 'сторона'])),
        'has_tech_terms': int(any(word in text for word in ['алгоритм', 'интерфейс']))
    }
```

### **3.3 Комбинирование признаков**  
```python
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

class TextStatsTransformer(BaseEstimator, TransformerMixin):
    def transform(self, texts):
        return [list(text_statistics(text).values()) for text in texts]

pipeline = FeatureUnion([
    ('tfidf', TfidfVectorizer()),
    ('stats', TextStatsTransformer())
])
```

---

## **📌 Тренировочные задания**  

### **🟢 Базовый уровень**  
1. Для текста "НЛП - это искусственный интеллект" вычислите:  
   - Количество символов и слов  
   - Плотность знаков препинания  

### **🟡 Продвинутый уровень**  
1. Постройте матрицу биграмм для корпуса русских пословиц  
2. Извлеките морфологические признаки для 10 русских глаголов  

### **🔴 Экспертный уровень**  
1. Создайте композитный признак "научность текста" на основе:  
   - Длины предложений  
   - Частоты терминов  
   - Части речи  

---

## **💡 Заключение**  
**Ключевые принципы Feature Engineering для текста:**  
1. **Комбинируйте подходы** (статистика + лингвистика + тематика)  
2. **Учитывайте специфику языка** (особенно для русского)  
3. **Экспериментируйте с композицией признаков**  
4. **Контролируйте размерность** (отбирайте значимые признаки)  

**Практические советы:**  
- Для коротких текстов: упор на статистические и стилевые признаки  
- Для тематического анализа: LDA + ключевые слова  
- Для классификации: TF-IDF + эмбеддинги  

> **"Хорошие признаки часто важнее сложных моделей!"**  

**Инструменты для русского языка:**  
- `pymorphy2` — морфологический анализ  
- `natasha` — извлечение именованных сущностей  
- `ru_core_news_sm` (spacy) — синтаксический разбор
