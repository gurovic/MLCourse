# Задачи: Интерпретируемость нейронных сетей

## 🟢 Базовый уровень

### Задача 1: Saliency Maps
Обучите простую CNN на MNIST. Вычислите saliency maps для 10 примеров (по одному на класс). Визуализируйте результаты.

### Задача 2: Feature Importance
Для табличных данных (например, Breast Cancer Wisconsin) обучите MLP и вычислите важность признаков через анализ весов первого слоя.

### Задача 3: Анализ предсказаний
Найдите примеры из test set, где модель:
- Уверенно правильно предсказывает
- Уверенно неправильно предсказывает
- Не уверена в предсказании
Визуализируйте saliency maps для каждого случая.

## 🟡 Средний уровень

### Задача 4: Grad-CAM
Реализуйте Grad-CAM для ResNet-18 на CIFAR-10. Визуализируйте важные регионы для:
- Правильных предсказаний
- Неправильных предсказаний
Сравните, где смотрит модель в каждом случае.

### Задача 5: SHAP для табличных данных
Используйте SHAP для объяснения предсказаний MLP на датасете с категориальными и числовыми признаками. Создайте:
- Force plots для отдельных примеров
- Summary plots для всего test set
- Dependence plots для топ-3 признаков

### Задача 6: LIME для изображений
Примените LIME к предобученному ResNet-18. Сравните LIME explanations с Grad-CAM для одних и тех же изображений.

### Задача 7: Integrated Gradients
Реализуйте Integrated Gradients и сравните с обычными градиентами (saliency maps). Покажите, что IG более стабильны.

## 🔴 Продвинутый уровень

### Задача 8: Attention Visualization
Используйте предобученный BERT для текстовой классификации. Визуализируйте attention weights:
- Для разных слоев
- Для разных attention heads
- Найдите паттерны (какие слои фокусируются на синтаксисе, какие на семантике)

### Задача 9: Counterfactual Explanations
Создайте систему для генерации counterfactual examples:
- Для изображений (MNIST или CIFAR-10)
- Минимизируйте изменения (L2 distance)
- Визуализируйте, что изменилось
- Проанализируйте, осмысленны ли изменения

### Задача 10: Comprehensive Analysis
Выберите задачу классификации (изображения или текст). Примените все методы интерпретируемости:
- Saliency Maps / Integrated Gradients
- Grad-CAM (для изображений) / Attention (для текста)
- SHAP
- LIME
Создайте dashboard с визуализациями. Сравните, какие методы дают наиболее полезные insights.

**Datasets:**
- MNIST: http://yann.lecun.com/exdb/mnist/
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- Breast Cancer Wisconsin: sklearn.datasets.load_breast_cancer
- IMDB Reviews: https://ai.stanford.edu/~amaas/data/sentiment/

**Tools:**
- Captum: https://captum.ai/
- SHAP: https://github.com/slundberg/shap
- LIME: https://github.com/marcotcr/lime
