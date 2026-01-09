# Задачи: Рекомендательные системы

## Задача 1: User-Based Collaborative Filtering (🟢)

Реализуйте user-based collaborative filtering на датасете MovieLens 100K.

**Требования**:
- Вычислите cosine similarity между пользователями
- Предскажите рейтинги для всех пар (user, movie), где рейтинг отсутствует
- Используйте k=10 ближайших соседей
- Оцените качество с помощью RMSE на test set

**Метрика**: RMSE < 1.0

## Задача 2: Item-Based Collaborative Filtering (🟢)

Реализуйте item-based collaborative filtering и сравните с user-based.

**Требования**:
- Вычислите similarity между фильмами
- Предскажите рейтинги
- Сравните RMSE с user-based подходом
- Проанализируйте, когда какой метод работает лучше

## Задача 3: Matrix Factorization с PyTorch (🟡)

Реализуйте Matrix Factorization модель и обучите на MovieLens.

**Требования**:
- Embedding dimension = 50
- Добавьте user и item biases
- Используйте L2 регуляризацию (weight_decay=0.01)
- Обучите 20 эпох
- Постройте график loss

**Метрика**: Test RMSE < 0.9

## Задача 4: Neural Collaborative Filtering (🟡)

Реализуйте NCF модель с MLP слоями.

**Требования**:
- Embedding dimension = 32
- MLP архитектура: [128, 64, 32, 16]
- Dropout = 0.2 между слоями
- Обучите с Adam optimizer, lr=0.001
- Сравните с Matrix Factorization

## Задача 5: Content-Based Filtering (🟢)

Постройте content-based рекомендатель для фильмов на основе жанров и тегов.

**Требования**:
- Используйте TF-IDF для текстовых описаний
- Постройте user profile как среднее фильмов, которые он оценил высоко
- Рекомендуйте top-10 фильмов для каждого пользователя
- Оцените Precision@10 и Recall@10

## Задача 6: Hybrid Recommender (🟡)

Создайте гибридную систему, комбинирующую collaborative и content-based подходы.

**Требования**:
- Обучите отдельно CF и content-based модели
- Комбинируйте их предсказания с весами α и (1-α)
- Подберите оптимальное α на валидации
- Покажите, что гибрид лучше каждого метода по отдельности

## Задача 7: Implicit Feedback (🟡)

Работа с неявным feedback (клики, просмотры) вместо явных рейтингов.

**Требования**:
- Преобразуйте рейтинги в бинарные метки (понравилось/не понравилось)
- Используйте Binary Cross-Entropy loss
- Примените negative sampling (1 positive : 4 negative)
- Оцените с помощью Hit Rate@10 и NDCG@10

## Задача 8: Session-Based Recommendations (🔴)

Реализуйте RNN для предсказания следующего item в сессии пользователя.

**Требования**:
- Используйте GRU с hidden_dim=100
- Входная последовательность - последние 10 просмотров
- Предскажите вероятности для всех items
- Используйте CrossEntropyLoss
- Оцените Recall@10 и MRR (Mean Reciprocal Rank)

**Метрика**: Recall@10 > 0.15

## Задача 9: Cold Start с Side Information (🔴)

Решите проблему холодного старта, используя дополнительную информацию.

**Требования**:
- Для новых users используйте демографические данные (возраст, пол)
- Для новых items используйте content features (жанры, теги)
- Постройте separate encoders для side information
- Протестируйте на подмножестве users/items с минимальным количеством взаимодействий

**Метрика**: RMSE на cold start users < 1.1

## Задача 10: Production Recommender System (🔴)

Постройте полноценную рекомендательную систему для production.

**Требования**:
1. **Offline training**: обучите NCF модель на полном датасете
2. **Online serving**: 
   - Сохраните embeddings в эффективном формате
   - Реализуйте быстрый поиск k ближайших items (FAISS)
3. **A/B testing framework**:
   - Разделите users на control/treatment группы
   - Логируйте CTR для каждой группы
4. **Monitoring**:
   - Отслеживайте distribution shift в user behavior
   - Определите, когда нужен retrain
5. **API**:
   - Создайте REST API с FastAPI
   - Endpoint: GET /recommendations/{user_id}?k=10
   - Latency < 50ms для 99th percentile

**Бонус**: реализуйте online learning для адаптации к новым данным в реальном времени
