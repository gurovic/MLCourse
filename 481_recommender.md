# Рекомендательные системы (Recommender Systems)

## 🟢 Основы

### Задача рекомендаций

**Цель**: предсказать релевантность товаров/контента для пользователя

**Типы данных**:
- Explicit feedback: рейтинги (1-5 звезд)
- Implicit feedback: клики, просмотры, покупки

**Проблема разреженности**: большинство пользователей оценили малую часть товаров

### Collaborative Filtering

**User-based**: находим похожих пользователей, рекомендуем то, что понравилось им

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Матрица user-item (строки - пользователи, столбцы - товары)
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 0, 5, 4]
])

# Находим похожесть между пользователями
user_similarity = cosine_similarity(ratings)
print("User similarity:\n", user_similarity)

# Предсказываем рейтинг для user 0, item 2
user_id = 0
item_id = 2

# Взвешенная сумма рейтингов похожих пользователей
numerator = 0
denominator = 0

for other_user in range(len(ratings)):
    if other_user != user_id and ratings[other_user, item_id] > 0:
        sim = user_similarity[user_id, other_user]
        numerator += sim * ratings[other_user, item_id]
        denominator += abs(sim)

predicted_rating = numerator / denominator if denominator > 0 else 0
print(f"Predicted rating for user {user_id}, item {item_id}: {predicted_rating:.2f}")
```

**Item-based**: находим похожие товары на основе паттернов оценок

```python
# Находим похожесть между товарами
item_similarity = cosine_similarity(ratings.T)
print("Item similarity:\n", item_similarity)

# Предсказываем рейтинг user 0, item 2 на основе похожих товаров
user_ratings = ratings[user_id]
numerator = 0
denominator = 0

for item in range(len(user_ratings)):
    if user_ratings[item] > 0:
        sim = item_similarity[item_id, item]
        numerator += sim * user_ratings[item]
        denominator += abs(sim)

predicted_rating = numerator / denominator if denominator > 0 else 0
print(f"Predicted rating: {predicted_rating:.2f}")
```

### Content-Based Filtering

Рекомендуем на основе характеристик товаров и профиля пользователя

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Описания фильмов
movies = [
    "action adventure superhero",
    "romantic comedy love",
    "action thriller spy",
    "comedy family kids"
]

# TF-IDF векторизация
vectorizer = TfidfVectorizer()
movie_features = vectorizer.fit_transform(movies)

# Пользователь любит фильм 0
user_profile = movie_features[0]

# Находим похожие фильмы
similarities = cosine_similarity(user_profile, movie_features)
print("Recommendations:", similarities[0])
```

## 🟡 Продвинутые методы

### Matrix Factorization

Разложение матрицы рейтингов на латентные факторы пользователей и товаров

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=20):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Инициализация
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()
        
        # Скалярное произведение + biases
        dot_product = (user_emb * item_emb).sum(dim=1)
        prediction = dot_product + user_b + item_b + self.global_bias
        
        return prediction

# Пример обучения
num_users, num_items = 1000, 500
model = MatrixFactorization(num_users, num_items, embedding_dim=50)

# Синтетические данные
user_ids = torch.randint(0, num_users, (100,))
item_ids = torch.randint(0, num_items, (100,))
ratings = torch.randn(100) * 2 + 3  # Рейтинги 1-5

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Один шаг обучения
predictions = model(user_ids, item_ids)
loss = criterion(predictions, ratings)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.4f}")
```

### Neural Collaborative Filtering (NCF)

Заменяем скалярное произведение на нейронную сеть

```python
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=20, hidden_dims=[64, 32, 16]):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        
        # Конкатенация embeddings
        x = torch.cat([user_emb, item_emb], dim=1)
        output = self.mlp(x).squeeze()
        
        return output

# Использование
ncf_model = NCF(num_users, num_items, embedding_dim=32)
predictions = ncf_model(user_ids, item_ids)
print(f"NCF predictions shape: {predictions.shape}")
```

### Wide & Deep Learning

Комбинация memorization (линейная модель) и generalization (глубокая сеть)

```python
class WideAndDeep(nn.Module):
    def __init__(self, num_users, num_items, num_features, embedding_dim=20):
        super().__init__()
        # Wide part: линейная модель
        self.wide = nn.Linear(num_features, 1)
        
        # Deep part: embeddings + MLP
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        self.deep = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
    def forward(self, user_ids, item_ids, wide_features):
        # Wide part
        wide_output = self.wide(wide_features)
        
        # Deep part
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        deep_input = torch.cat([user_emb, item_emb], dim=1)
        deep_output = self.deep(deep_input)
        
        # Комбинируем
        output = torch.sigmoid(wide_output + deep_output)
        return output.squeeze()

# Пример
num_features = 10
wide_deep_model = WideAndDeep(num_users, num_items, num_features)
wide_features = torch.randn(100, num_features)
predictions = wide_deep_model(user_ids, item_ids, wide_features)
print(f"Wide & Deep predictions: {predictions.shape}")
```

## 🔴 Продвинутые техники

### Factorization Machines (FM)

Моделируют взаимодействия между всеми парами признаков

```python
class FactorizationMachine(nn.Module):
    def __init__(self, num_features, embedding_dim=10):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)
        self.embeddings = nn.Embedding(num_features, embedding_dim)
        
    def forward(self, x):
        # x: (batch, num_features) - sparse one-hot encoded
        # Линейная часть
        linear_part = self.linear(x)
        
        # Взаимодействия второго порядка
        # (sum of squares - square of sum) / 2
        emb = self.embeddings.weight  # (num_features, embedding_dim)
        square_of_sum = torch.matmul(x, emb) ** 2  # (batch, embedding_dim)
        sum_of_square = torch.matmul(x ** 2, emb ** 2)  # (batch, embedding_dim)
        
        interaction = 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)
        
        output = linear_part + interaction
        return output.squeeze()

# FM для рекомендаций
fm_model = FactorizationMachine(num_features=100, embedding_dim=20)
sparse_features = torch.randn(32, 100)
predictions = fm_model(sparse_features)
print(f"FM predictions: {predictions.shape}")
```

### Session-Based Recommendations с RNN

Учитываем последовательность действий пользователя

```python
class SessionRNN(nn.Module):
    def __init__(self, num_items, embedding_dim=50, hidden_dim=100):
        super().__init__()
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_items)
        
    def forward(self, item_sequence):
        # item_sequence: (batch, seq_len)
        emb = self.item_embeddings(item_sequence)  # (batch, seq_len, embedding_dim)
        output, hidden = self.gru(emb)  # output: (batch, seq_len, hidden_dim)
        
        # Предсказываем следующий item
        logits = self.fc(output[:, -1, :])  # Используем последний шаг
        return logits

# Пример
session_model = SessionRNN(num_items=500, embedding_dim=50, hidden_dim=100)
# Последовательность просмотров
item_sequence = torch.randint(0, 500, (16, 10))  # batch=16, seq_len=10
logits = session_model(item_sequence)
print(f"Next item predictions: {logits.shape}")  # (16, 500)
```

### Метрики оценки

```python
import numpy as np
from sklearn.metrics import ndcg_score

def precision_at_k(y_true, y_pred, k=10):
    """Precision@K"""
    top_k = y_pred.argsort()[-k:][::-1]
    return np.sum(y_true[top_k]) / k

def recall_at_k(y_true, y_pred, k=10):
    """Recall@K"""
    top_k = y_pred.argsort()[-k:][::-1]
    return np.sum(y_true[top_k]) / np.sum(y_true)

def hit_rate_at_k(y_true, y_pred, k=10):
    """Hit Rate@K - хотя бы один релевантный в топ-K"""
    top_k = y_pred.argsort()[-k:][::-1]
    return int(np.sum(y_true[top_k]) > 0)

# Пример
y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([0.1, 0.9, 0.2, 0.7, 0.3, 0.1, 0.8, 0.4])

print(f"Precision@3: {precision_at_k(y_true, y_pred, k=3):.3f}")
print(f"Recall@3: {recall_at_k(y_true, y_pred, k=3):.3f}")
print(f"Hit Rate@3: {hit_rate_at_k(y_true, y_pred, k=3)}")
print(f"NDCG@3: {ndcg_score([y_true], [y_pred], k=3):.3f}")
```

### Cold Start Problem

**Проблема**: как рекомендовать новым пользователям/товарам без истории?

**Решения**:
1. **Hybrid approaches**: комбинировать content-based и collaborative filtering
2. **Side information**: использовать демографические данные, теги, описания
3. **Transfer learning**: переносить знания из других доменов
4. **Active learning**: задавать вопросы новым пользователям

```python
class HybridRecommender(nn.Module):
    """Гибридный рекомендатель для холодного старта"""
    def __init__(self, num_users, num_items, content_dim, embedding_dim=20):
        super().__init__()
        # Collaborative filtering part
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # Content-based part
        self.content_encoder = nn.Sequential(
            nn.Linear(content_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        
        # Mixing weight
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, user_ids, item_ids, item_content, is_cold_start):
        # Collaborative filtering prediction
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        cf_pred = (user_emb * item_emb).sum(dim=1)
        
        # Content-based prediction
        content_emb = self.content_encoder(item_content)
        cb_pred = (user_emb * content_emb).sum(dim=1)
        
        # Для холодного старта используем больше content-based
        if is_cold_start:
            return cb_pred
        else:
            # Взвешенная комбинация
            return torch.sigmoid(self.alpha) * cf_pred + (1 - torch.sigmoid(self.alpha)) * cb_pred
```

## Датасеты

- **MovieLens**: рейтинги фильмов (100K, 1M, 20M версии)
- **Amazon Product Reviews**: покупки и отзывы
- **Last.fm**: прослушивания музыки
- **Netflix Prize**: исторический датасет фильмов
- **Yelp**: рейтинги ресторанов и бизнесов

## Литература

- **Neural Collaborative Filtering** (He et al., 2017)
- **Wide & Deep Learning** (Cheng et al., 2016)
- **DeepFM** (Guo et al., 2017)
- **Session-based Recommendations with RNNs** (Hidasi et al., 2016)
