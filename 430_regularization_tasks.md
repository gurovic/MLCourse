### **Задачи: Регуляризация в нейронных сетях**

**Цель:** Научиться применять различные техники регуляризации для борьбы с переобучением.

---

## 🟢 Базовый уровень

### **Задача 1: Демонстрация переобучения**

**Условие:** Создайте модель, которая переобучается, и визуализируйте это.

**Требования:**
1. Используйте маленький subset MNIST (только 1000 примеров для обучения)
2. Постройте "слишком большую" модель: 784→512→512→10
3. Обучите без регуляризации 50 эпох
4. Постройте график train loss vs validation loss

**Ожидаемое наблюдение:** Train loss продолжает падать, val loss растет после ~10 эпох.

**Вопрос:** На какой эпохе начинается переобучение?

---

### **Задача 2: Борьба с переобучением через Dropout**

**Условие:** Примените Dropout к модели из Задачи 1.

**Требования:**
1. Добавьте `nn.Dropout(0.5)` после каждого hidden слоя
2. Обучите с теми же условиями (1000 примеров, 50 эпох)
3. Сравните графики train/val loss с/без Dropout
4. Сравните финальную accuracy на test set

**Эксперимент:** Попробуйте разные значения p: [0.2, 0.3, 0.5, 0.7]

---

### **Задача 3: Batch Normalization**

**Условие:** Добавьте Batch Normalization и сравните с Dropout.

**Требования:**
1. Постройте модель с BatchNorm после каждого Linear слоя:
   ```python
   self.fc1 = nn.Linear(784, 512)
   self.bn1 = nn.BatchNorm1d(512)
   ```
2. Обучите на полном MNIST
3. Сравните с моделью без BN:
   - Скорость сходимости (epochs to 95% accuracy)
   - Финальную accuracy
   - Стабильность обучения

**Важно:** model.train() vs model.eval() для BatchNorm!

---

## 🟡 Продвинутый уровень

### **Задача 4: Weight Decay vs L2 Regularization**

**Условие:** Реализуйте L2 регуляризацию вручную и сравните с weight_decay.

**Требования:**
1. **Вариант 1:** Используйте `optimizer = Adam(params, weight_decay=0.01)`
2. **Вариант 2:** Добавьте L2 penalty в loss:
   ```python
   l2_lambda = 0.01
   l2_penalty = sum(p.pow(2.0).sum() for p in model.parameters())
   loss = ce_loss + l2_lambda * l2_penalty
   ```
3. Сравните результаты (должны быть близки)
4. Визуализируйте распределение весов (гистограмма)

**Вопрос:** Почему регуляризация делает веса меньше?

---

### **Задача 5: Комбинация техник**

**Условие:** Найдите оптимальную комбинацию Dropout + BatchNorm + Weight Decay.

**Сетка экспериментов:**
- Dropout: [0, 0.3, 0.5]
- BatchNorm: [True, False]
- Weight Decay: [0, 0.0001, 0.001]

**Требования:**
1. Обучите модели для всех 18 комбинаций (3×2×3)
2. Для каждой запишите:
   - Val accuracy
   - Время обучения
   - Gap между train и val accuracy (мера переобучения)
3. Найдите best configuration
4. Создайте таблицу результатов

---

### **Задача 6: Layer Normalization для RNN-like задачи**

**Условие:** Сравните BatchNorm и LayerNorm на последовательных данных.

**Датасет:** Создайте искусственную задачу с последовательностями переменной длины

**Требования:**
1. Постройте модель, обрабатывающую последовательности
2. Реализуйте две версии:
   - С `nn.BatchNorm1d` (проблемы с малыми батчами)
   - С `nn.LayerNorm` (работает с любым размером батча)
3. Обучите с batch_size=[4, 16, 64]
4. Сравните стабильность обучения

**Вопрос:** Почему LayerNorm лучше для маленьких батчей?

---

## 🔴 Экспертный уровень

### **Задача 7: Dropout в Test Time**

**Условие:** Используйте Dropout для оценки неопределенности (uncertainty estimation).

**Концепция:** MC Dropout — оставляем Dropout активным во время инференса.

**Требования:**
1. Обучите модель с Dropout
2. Во время теста сделайте N=100 forward passes с активным Dropout:
   ```python
   model.train()  # Держим Dropout активным!
   predictions = [model(x) for _ in range(100)]
   ```
3. Вычислите:
   - Среднее предсказание
   - Стандартное отклонение (мера неопределенности)
4. Визуализируйте примеры с высокой/низкой неопределенностью

**Анализ:** Высокая неопределенность часто коррелирует с ошибками модели.

---

### **Задача 8: Spectral Normalization**

**Условие:** Реализуйте Spectral Normalization для стабилизации обучения GAN.

**Требования:**
1. Используйте `torch.nn.utils.spectral_norm()`:
   ```python
   self.fc1 = nn.utils.spectral_norm(nn.Linear(784, 256))
   ```
2. Обучите дискриминатор GAN с/без spectral norm
3. Сравните:
   - Стабильность gradients
   - Качество генерации (визуально)
   - Loss curves

**Вопрос:** Как spectral normalization ограничивает Lipschitz константу?

---

### **Задача 9: Mixup Augmentation как регуляризация**

**Условие:** Реализуйте Mixup для классификации изображений.

**Требования:**
1. Реализуйте mixup:
   ```python
   def mixup_data(x, y, alpha=1.0):
       lam = np.random.beta(alpha, alpha)
       index = torch.randperm(x.size(0))
       mixed_x = lam * x + (1 - lam) * x[index]
       return mixed_x, y, y[index], lam
   ```
2. Обучите две модели: с/без Mixup
3. Сравните:
   - Test accuracy
   - Робастность к adversarial examples (попробуйте FGSM attack)
   - Калибровку предсказаний (confidence vs accuracy)

---

### **Задача 10: Adaptive Dropout**

**Условие:** Реализуйте Dropout с адаптивной вероятностью.

**Идея:** Dropout rate меняется в зависимости от уровня переобучения.

**Требования:**
1. Реализуйте адаптивный Dropout:
   ```python
   class AdaptiveDropout(nn.Module):
       def __init__(self, initial_p=0.5):
           super().__init__()
           self.p = initial_p
       
       def update_p(self, train_loss, val_loss):
           if val_loss > train_loss * 1.1:  # Переобучение
               self.p = min(0.8, self.p + 0.05)
           else:
               self.p = max(0.1, self.p - 0.05)
   ```
2. Обучите модель с адаптивным Dropout
3. Визуализируйте изменение p по эпохам
4. Сравните с фиксированным Dropout

---

## 💎 Бонусная задача

### **Задача 11: Автоматический подбор регуляризации**

**Условие:** Используйте Optuna для подбора гиперпараметров регуляризации.

**Требования:**
1. Определите пространство поиска:
   ```python
   dropout_rate = trial.suggest_float('dropout', 0.1, 0.7)
   weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
   use_batchnorm = trial.suggest_categorical('bn', [True, False])
   ```
2. Оптимизируйте validation accuracy
3. Запустите 50 trials
4. Постройте importance plot параметров

**Анализ:** Какой параметр важнее для предотвращения переобучения?

---

## Полезные ресурсы

- [Dropout Paper](https://jmlr.org/papers/v15/srivastava14a.html)
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)
- [Mixup Paper](https://arxiv.org/abs/1710.09412)
- [Understanding Regularization](https://towardsdatascience.com/regularization-in-deep-learning-l1-l2-and-dropout-377e75acc036)
