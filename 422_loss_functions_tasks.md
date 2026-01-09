### **Задачи: Функции потерь (Loss Functions)**

**Цель:** Понять различия между функциями потерь и научиться выбирать правильную для задачи.

---

## 🟢 Базовый уровень

### **Задача 1: MSE vs MAE для регрессии**

**Условие:** Сравните MSE и MAE на данных с выбросами.

**Данные:**
```python
# Генерация данных с выбросами
np.random.seed(42)
X = np.linspace(0, 10, 100)
y_true = 2*X + 1 + np.random.randn(100)*0.5
# Добавляем выбросы
y_true[[10, 20, 30]] = [50, 60, 55]
```

**Требования:**
1. Обучите две линейные модели: одну с MSE, другую с MAE (используйте `nn.L1Loss()`)
2. Визуализируйте данные и предсказания обеих моделей
3. Сравните, какая модель лучше игнорирует выбросы

**Вопрос:** Почему MAE более робастна к выбросам?

---

### **Задача 2: Binary Cross-Entropy для классификации**

**Условие:** Реализуйте бинарную классификацию с BCE loss.

**Датасет:** Используйте `make_moons` из sklearn

**Требования:**
1. Создайте датасет: `X, y = make_moons(n_samples=200, noise=0.2)`
2. Постройте простую MLP: 2 → 10 → 1 (с sigmoid на выходе)
3. Используйте `nn.BCELoss()`
4. Визуализируйте границу решений

**Дополнительно:** 
- Что происходит, если забыть sigmoid на выходе?
- Попробуйте `nn.BCEWithLogitsLoss()` (более стабильная версия)

---

### **Задача 3: Cross-Entropy для многоклассовой классификации**

**Условие:** Примените CE loss к классификации MNIST.

**Требования:**
1. Постройте MLP для MNIST (10 классов)
2. Используйте `nn.CrossEntropyLoss()` (НЕ применяйте softmax в модели!)
3. Логируйте loss на каждой эпохе
4. Постройте график обучения

**Важно:** CrossEntropyLoss в PyTorch уже включает softmax!

---

## 🟡 Продвинутый уровень

### **Задача 4: Focal Loss для дисбаланса классов**

**Условие:** Реализуйте Focal Loss и сравните с обычной CE.

**Датасет:** MNIST, но с дисбалансом (оставьте только 10% примеров класса "0")

**Требования:**
1. Реализуйте Focal Loss:
   ```python
   class FocalLoss(nn.Module):
       def __init__(self, gamma=2):
           super().__init__()
           self.gamma = gamma
       
       def forward(self, inputs, targets):
           ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
           p_t = torch.exp(-ce_loss)
           focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()
           return focal_loss
   ```

2. Обучите две модели: одну с CE, другую с Focal Loss
3. Сравните accuracy на редком классе "0"

**Вопрос:** Как γ влияет на фокусировку на сложных примерах?

---

### **Задача 5: Взвешенный Cross-Entropy**

**Условие:** Используйте веса классов для борьбы с дисбалансом.

**Требования:**
1. Создайте дисбалансированный датасет (соотношение 1:10:100 между классами)
2. Вычислите веса классов: `weights = 1.0 / class_counts`
3. Используйте `nn.CrossEntropyLoss(weight=weights)`
4. Сравните с обычной CE без весов

**Метрика:** Используйте F1-score для каждого класса (не только accuracy!)

---

### **Задача 6: Huber Loss для регрессии**

**Условие:** Примените Huber Loss как компромисс между MSE и MAE.

**Требования:**
1. Сгенерируйте данные с выбросами (как в Задаче 1)
2. Обучите 3 модели: с MSE, MAE, Huber Loss
3. Постройте сравнительный график потерь по эпохам
4. Визуализируйте предсказания всех трех моделей

**Параметр:** Экспериментируйте с δ (delta) в Huber Loss

---

## 🔴 Экспертный уровень

### **Задача 7: Кастомная Loss функция**

**Условие:** Создайте свою loss для специфической задачи.

**Сценарий:** Прогнозирование цен квартир, где:
- Переоценка (predicted > actual) — хуже, чем недооценка
- Штраф за переоценку должен быть в 2 раза больше

**Требования:**
1. Реализуйте asymmetric loss:
   ```python
   class AsymmetricLoss(nn.Module):
       def forward(self, pred, target):
           error = pred - target
           loss = torch.where(error > 0,
                            2.0 * error**2,  # переоценка
                            error**2)        # недооценка
           return loss.mean()
   ```

2. Обучите модель с этой loss
3. Проверьте, что модель действительно делает меньше переоценок

---

### **Задача 8: Contrastive Loss для Similarity Learning**

**Условие:** Обучите сеть различать похожие/непохожие пары изображений.

**Датасет:** MNIST (создайте пары)

**Требования:**
1. Создайте пары:
   - Positive pairs: две цифры одного класса
   - Negative pairs: цифры разных классов

2. Реализуйте Contrastive Loss:
   ```python
   def contrastive_loss(output1, output2, label, margin=1.0):
       distance = F.pairwise_distance(output1, output2)
       loss = label * distance**2 + \
              (1 - label) * torch.clamp(margin - distance, min=0)**2
       return loss.mean()
   ```

3. Обучите Siamese network (две одинаковые CNN с shared weights)
4. Визуализируйте embeddings с t-SNE

**Цель:** Похожие цифры должны быть близко в embedding space.

---

### **Задача 9: Triplet Loss**

**Условие:** Реализуйте Triplet Loss для Face Recognition задачи.

**Концепция:** 
- Anchor: исходное изображение
- Positive: изображение того же класса
- Negative: изображение другого класса

**Требования:**
1. Реализуйте Triplet Loss:
   ```python
   def triplet_loss(anchor, positive, negative, margin=1.0):
       pos_dist = F.pairwise_distance(anchor, positive)
       neg_dist = F.pairwise_distance(anchor, negative)
       loss = torch.clamp(pos_dist - neg_dist + margin, min=0)
       return loss.mean()
   ```

2. Создайте triplets из MNIST
3. Обучите embedding network
4. Проверьте: distance(anchor, positive) < distance(anchor, negative)

---

### **Задача 10: Multi-task Loss**

**Условие:** Обучите модель на двух задачах одновременно.

**Задачи:**
1. Классификация цифры (CE loss)
2. Регрессия "толщины" линий (MSE loss)

**Требования:**
1. Создайте multi-head модель:
   ```python
   class MultiTaskModel(nn.Module):
       def __init__(self):
           self.shared = nn.Sequential(...)  # общие слои
           self.classifier = nn.Linear(128, 10)  # для классификации
           self.regressor = nn.Linear(128, 1)    # для регрессии
   ```

2. Комбинируйте losses:
   ```python
   total_loss = ce_loss + lambda_reg * mse_loss
   ```

3. Экспериментируйте с λ (вес регрессионной loss)
4. Сравните с моделями, обученными по отдельности

---

## 💎 Бонусная задача

### **Задача 11: Loss Landscape Visualization**

**Условие:** Визуализируйте пространство loss функции.

**Требования:**
1. Обучите простую модель на MNIST
2. Зафиксируйте финальные веса
3. Исследуйте loss вокруг этой точки:
   ```python
   # Добавляем возмущения к весам
   for dx in range(-10, 10):
       for dy in range(-10, 10):
           perturbed_weights = original_weights + dx*direction1 + dy*direction2
           loss = evaluate_loss(perturbed_weights)
   ```

4. Постройте 3D поверхность (или contour plot) loss landscape

**Анализ:**
- Насколько "гладкий" loss landscape?
- Есть ли локальные минимумы?
- Как выглядит область вокруг найденного минимума?

---

## Полезные ресурсы

- [PyTorch Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [Focal Loss Paper](https://arxiv.org/abs/1708.02002)
- [Metric Learning Survey](https://arxiv.org/abs/2003.08505)
