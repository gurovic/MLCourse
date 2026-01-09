# Задачи: Мультимодальное обучение

## 🟢 Базовый уровень

### Задача 1: Simple Feature Fusion
Создайте модель для классификации, которая использует:
- Изображения (CNN encoder)
- Табличные данные (MLP encoder)
Объедините признаки через конкатенацию. Обучите на синтетических данных.

### Задача 2: Image Captioning Dataset
Подготовьте датасет для image captioning:
- Загрузите Flickr8k или COCO Captions (subset)
- Создайте vocabulary
- Реализуйте DataLoader
- Визуализируйте 10 примеров (изображение + подписи)

### Задача 3: Basic Image Captioning
Реализуйте простую модель image captioning (CNN + LSTM). Обучите на малом датасете (Flickr8k) и сгенерируйте подписи для тестовых изображений.

## 🟡 Средний уровень

### Задача 4: CLIP для Zero-shot Classification
Используйте предобученный CLIP для zero-shot классификации:
- Выберите датасет (CIFAR-10, Fashion-MNIST, или кастомный)
- Создайте текстовые описания классов
- Классифицируйте изображения без дообучения
- Сравните с supervised baseline

### Задача 5: Image-Text Retrieval
Реализуйте систему поиска:
- По текстовому запросу находить похожие изображения
- По изображению находить похожие текстовые описания
Используйте CLIP или обучите свою модель на Flickr30k.

### Задача 6: Visual Question Answering
Создайте VQA модель:
- Image encoder (ResNet/ViT)
- Question encoder (BERT/LSTM)
- Fusion mechanism (attention/concatenation)
Обучите на VQA v2 (subset) и достигните разумной точности.

### Задача 7: Attention Visualization
Для VQA модели визуализируйте attention weights:
- Покажите, на какие части изображения смотрит модель
- Для разных типов вопросов (what, where, how many)
- Проанализируйте, осмысленны ли attention patterns

## 🔴 Продвинутый уровень

### Задача 8: CLIP-like Training
Обучите CLIP-подобную модель с нуля:
- Image encoder (ResNet-50 или ViT)
- Text encoder (Transformer)
- Contrastive loss
Обучите на Conceptual Captions (subset). Оцените на retrieval tasks.

### Задача 9: Multimodal Transformer
Реализуйте multimodal transformer:
- Unified encoder для текста и изображений
- Token-level fusion
- Обучите на нескольких задачах одновременно (MLM, ITM, captioning)
Сравните с separate encoders подходом.

### Задача 10: Video Understanding
Создайте модель для video classification:
- 3D CNN или I3D для visual features
- Spectrogram CNN для audio features
- Temporal modeling (LSTM/Transformer)
- Multimodal fusion
Обучите на Kinetics-400 (subset) или UCF-101.

**Datasets:**
- Flickr8k: https://www.kaggle.com/datasets/adityajn105/flickr8k
- Flickr30k: http://shannon.cs.illinois.edu/DenotationGraph/
- COCO Captions: https://cocodataset.org/#captions-2015
- VQA v2: https://visualqa.org/
- Conceptual Captions: https://ai.google.com/research/ConceptualCaptions/
- UCF-101: https://www.crcv.ucf.edu/data/UCF101.php

**Pretrained Models:**
- CLIP: https://github.com/openai/CLIP
- BLIP: https://github.com/salesforce/BLIP
- Hugging Face Multimodal: https://huggingface.co/models?pipeline_tag=image-text-to-text
