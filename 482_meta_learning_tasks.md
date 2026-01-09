# Задачи: Meta-Learning и Few-Shot Learning

## Задача 1: Siamese Network на Omniglot (🟢)

Реализуйте Siamese Network для one-shot learning на датасете Omniglot.

**Требования**:
- Архитектура: 4 convolutional блока
- Contrastive Loss с margin=2.0
- Обучите на 20 эпох
- Оцените точность на 20-way one-shot задаче

**Метрика**: Accuracy > 90%

## Задача 2: Prototypical Networks (🟡)

Реализуйте Prototypical Networks для few-shot classification.

**Требования**:
- Encoder: 4 convolutional блока с BatchNorm
- Embedding dimension = 64
- Тестируйте на 5-way 1-shot и 5-way 5-shot
- Постройте график accuracy vs число shots (1, 5, 10, 20)

**Метрика**: 5-way 5-shot accuracy > 75%

## Задача 3: Matching Networks (🟡)

Реализуйте Matching Networks с attention механизмом.

**Требования**:
- Bidirectional LSTM для encoding support set
- Attention-based classification
- Сравните с Prototypical Networks на miniImageNet
- Проанализируйте attention weights

## Задача 4: MAML Implementation (🔴)

Реализуйте полную версию MAML алгоритма.

**Требования**:
- Inner loop: 5 gradient steps, lr=0.01
- Outer loop: Adam optimizer, lr=0.001
- Second-order gradients (create_graph=True)
- Meta-batch size = 4 tasks
- Обучите на Omniglot

**Метрика**: 5-way 1-shot accuracy > 95%

## Задача 5: Reptile vs MAML (🔴)

Сравните Reptile и MAML на одной задаче.

**Требования**:
- Реализуйте обе версии с одинаковой базовой архитектурой
- Сравните:
  - Скорость обучения (время на эпоху)
  - Финальная accuracy
  - Memory usage
- Постройте learning curves для обоих методов

## Задача 6: Cross-Domain Meta-Learning (🔴)

Обучите мета-модель на одном датасете, протестируйте на другом.

**Требования**:
- Meta-train на miniImageNet
- Meta-test на CUB-200 (птицы) и Cars-196 (автомобили)
- Сравните с обычным transfer learning
- Проанализируйте, насколько хорошо переносится способность к few-shot learning

## Задача 7: Few-Shot Object Detection (🔴)

Адаптируйте мета-обучение для задачи object detection.

**Требования**:
- Используйте Faster R-CNN как базовую модель
- Реализуйте few-shot fine-tuning для новых классов
- Датасет: COCO или Pascal VOC
- Тестируйте с 1, 5, 10 примерами нового класса

**Метрика**: mAP@0.5 > 0.3 для 5-shot

## Задача 8: Meta-Learning для NLP (🟡)

Примените мета-обучение к text classification.

**Требования**:
- Используйте BERT в качестве encoder
- Few-shot sentiment analysis на разных доменах
- Support set: 5 примеров per class
- Query set: 100 примеров per class
- Датасеты: Amazon reviews, Yelp, IMDB

## Задача 9: Task-Conditional Meta-Learning (🔴)

Реализуйте meta-learner, который адаптируется к разным типам задач.

**Требования**:
- Обучите на смеси задач: classification, regression, segmentation
- Добавьте task embedding для conditioning
- Используйте FiLM layers для task-specific modulation
- Оцените на каждом типе задачи отдельно

## Задача 10: Continual Meta-Learning (🔴)

Реализуйте систему, которая продолжает мета-обучение на новых задачах без забывания старых.

**Требования**:
1. **Base meta-learner**: MAML или Prototypical Networks
2. **Continual learning component**:
   - Elastic Weight Consolidation (EWC) для защиты важных параметров
   - Experience Replay буфер с примерами из старых задач
3. **Stream of tasks**: 
   - 10 различных доменов (animals, vehicles, food, etc.)
   - Приходят последовательно
4. **Evaluation**:
   - Backward transfer: accuracy на старых задачах после обучения на новых
   - Forward transfer: быстрота адаптации к новым задачам
5. **Metrics**:
   - Average accuracy across all tasks
   - Forgetting measure

**Бонус**: визуализируйте embedding space и покажите, что задачи разных доменов образуют кластеры
