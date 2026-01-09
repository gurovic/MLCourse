# Задачи: Отладка нейронных сетей

## Задача 1: Debug Checklist Implementation (🟢)

Реализуйте функцию debug_model, которая проверяет модель перед обучением.

**Требования**:
- Проверка размерностей входа/выхода
- Проверка initial loss (должен быть близок к -log(1/num_classes))
- Проверка наличия градиентов во всех слоях
- Проверка на NaN/Inf
- Красиво форматированный вывод

## Задача 2: Gradient Flow Visualization (🟢)

Создайте функцию для визуализации потока градиентов.

**Требования**:
- После backward pass постройте bar chart градиентов по слоям
- Покажите mean и max gradient для каждого слоя
- Детектируйте vanishing gradients (mean < 1e-5)
- Детектируйте exploding gradients (mean > 100)

## Задача 3: Learning Rate Finder (🟡)

Реализуйте LR Finder для поиска оптимального learning rate.

**Требования**:
- Экспоненциально увеличивайте LR от 1e-7 до 10
- Постройте график loss vs LR
- Остановитесь если loss взрывается (> 4 * min_loss)
- Предложите оптимальный LR

**Метрика**: найденный LR должен давать хорошую сходимость

## Задача 4: Dead ReLU Detection (🟡)

Реализуйте детектор "мертвых" ReLU нейронов.

**Требования**:
- Используйте forward hooks для сбора активаций
- Для каждого ReLU слоя посчитайте % нейронов, всегда выдающих 0
- Предупреждайте если > 50% нейронов мертвы
- Предложите решения (LeakyReLU, BatchNorm, меньший LR)

## Задача 5: Overfitting Test (🟢)

Проверьте, может ли модель переобучиться на маленьком датасете.

**Требования**:
- Возьмите 10 примеров из train set
- Обучайте модель до 100% accuracy
- Если не достигается за 100 эпох - есть проблема
- Выведите diagnostic info

## Задача 6: Activation Statistics Tracker (🟡)

Соберите статистику активаций по всем слоям.

**Требования**:
- Mean, std, min, max для каждого слоя
- Sparsity (% нулевых активаций)
- Отслеживайте изменение статистик во время обучения
- Визуализируйте как boxplots

## Задача 7: Loss Landscape 2D (🔴)

Визуализируйте loss landscape вокруг текущей точки.

**Требования**:
- Генерируйте 2 случайных direction vectors
- Создайте 2D сетку в пространстве параметров
- Вычислите loss в каждой точке сетки
- Постройте contour plot
- Отметьте текущую позицию

## Задача 8: BatchNorm Statistics Monitor (🔴)

Отслеживайте статистики BatchNorm слоев.

**Требования**:
- running_mean и running_var для каждого BN слоя
- Gamma (scale) и beta (shift) параметры
- Детектируйте внутренний covariate shift
- Постройте t-SNE активаций до и после BN

## Задача 9: Gradient Noise Analysis (🔴)

Проанализируйте шум в градиентах для разных batch sizes.

**Требования**:
- Вычислите gradient variance для batch_size = [8, 16, 32, 64, 128]
- Постройте зависимость variance от batch_size
- Оцените оптимальный batch_size (balance между скоростью и качеством)
- Используйте накопление градиентов для симуляции больших батчей

## Задача 10: Full Diagnostic Suite (🔴)

Создайте комплексный инструмент для автоматической диагностики.

**Требования**:
1. **Pre-training checks**:
   - Размерности
   - Initial loss
   - Gradient flow
   - Overfitting test
   
2. **During training monitoring**:
   - Learning curves (train/val loss & accuracy)
   - Gradient norms
   - Weight histograms
   - Activation statistics
   - Dead neurons tracking
   
3. **Post-training analysis**:
   - Final metrics
   - Confusion matrix
   - Per-class performance
   - Error analysis
   
4. **Auto-suggestions**:
   - Если не обучается: предложить увеличить LR или проверить данные
   - Если переобучается: предложить regularization
   - Если vanishing gradients: предложить BatchNorm или ResNet
   
5. **Report generation**:
   - HTML отчет со всеми графиками и метриками
   - Markdown summary для quick overview

**Бонус**: интеграция с TensorBoard или Weights & Biases
