# Object Detection

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# !pip install torch torchvision matplotlib pillow pycocotools
```

---

## 🟢 Базовый уровень: От классификации к детекции

### 1.1 Задача Object Detection

**Классификация vs Detection:**

| Задача | Вход | Выход | Пример |
|--------|------|-------|--------|
| **Image Classification** | Изображение | Метка класса | "кошка" |
| **Object Localization** | Изображение | Класс + bounding box | "кошка" + [x, y, w, h] |
| **Object Detection** | Изображение | Несколько объектов (класс + bbox) | [("кошка", box1), ("собака", box2)] |
| **Instance Segmentation** | Изображение | Класс + точная маска | Попиксельная сегментация |

**Object Detection решает:**
- Что находится на изображении? (классификация)
- Где находится? (локализация)
- Сколько объектов? (может быть 0, 1, много)

---

### 1.2 Основные понятия

#### **Bounding Box**

Прямоугольник, описывающий объект:
```python
# Формат 1: (x_min, y_min, x_max, y_max) — углы
bbox = [100, 50, 300, 250]

# Формат 2: (x_center, y_center, width, height) — YOLO формат
bbox_yolo = [200, 150, 200, 200]

# Формат 3: (x, y, w, h) — COCO формат (верхний левый угол)
bbox_coco = [100, 50, 200, 200]
```

#### **Intersection over Union (IoU)**

Метрика для оценки качества предсказания bbox:

```
IoU = Area of Overlap / Area of Union
```

```python
def calculate_iou(box1, box2):
    """
    Вычисляет IoU для двух bbox в формате [x_min, y_min, x_max, y_max]
    """
    # Пересечение
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Объединение
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# Пример
gt_box = [100, 100, 200, 200]  # Ground truth
pred_box = [120, 120, 220, 220]  # Предсказание
iou = calculate_iou(gt_box, pred_box)
print(f"IoU: {iou:.3f}")  # 0.537
```

**Интерпретация IoU:**
- IoU > 0.5 — обычно считается правильным detection
- IoU > 0.7 — хорошее detection
- IoU > 0.9 — отличное detection

#### **Non-Maximum Suppression (NMS)**

Удаление дублирующихся детекций одного и того же объекта.

**Алгоритм NMS:**
1. Сортируем все предсказания по confidence score
2. Берем предсказание с максимальным score
3. Удаляем все предсказания с IoU > threshold (обычно 0.5)
4. Повторяем для оставшихся предсказаний

```python
def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression
    boxes: [N, 4] (x_min, y_min, x_max, y_max)
    scores: [N] confidence scores
    """
    # Сортируем по убыванию score
    indices = scores.argsort()[::-1]
    
    keep = []
    while len(indices) > 0:
        # Берем bbox с максимальным score
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Вычисляем IoU с остальными
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        ious = np.array([calculate_iou(current_box, box) for box in other_boxes])
        
        # Оставляем только те, у которых IoU < threshold
        indices = indices[1:][ious < iou_threshold]
    
    return keep

# Пример
boxes = np.array([
    [100, 100, 200, 200],
    [110, 110, 210, 210],  # Дубликат первого
    [300, 300, 400, 400],
])
scores = np.array([0.9, 0.85, 0.8])

keep_indices = nms(boxes, scores, iou_threshold=0.5)
print(f"Оставляем: {keep_indices}")  # [0, 2]
```

---

## 🟡 Продвинутый уровень: Семейство R-CNN

### 2.1 Эволюция R-CNN

#### **R-CNN (2014) — Region-based CNN**

**Идея:** Используем selective search для генерации region proposals, затем классифицируем каждый region.

**Алгоритм:**
1. **Selective Search:** Генерируем ~2000 region proposals
2. **Warp:** Масштабируем каждый region до фиксированного размера (227x227)
3. **CNN:** Прогоняем через CNN (AlexNet) для извлечения признаков
4. **SVM:** Классифицируем признаки с помощью SVM
5. **Bounding Box Regression:** Уточняем координаты bbox

**Проблемы:**
- ❌ Очень медленно (2000 forward pass через CNN)
- ❌ Обучение в несколько этапов
- ❌ ~47 секунд на одно изображение

---

#### **Fast R-CNN (2015)**

**Улучшения:**
- ✅ Один forward pass через CNN для всего изображения
- ✅ RoI Pooling для извлечения признаков из proposals
- ✅ Multi-task loss (классификация + bbox regression)

**Алгоритм:**
1. Прогоняем всё изображение через CNN → feature map
2. Selective search генерирует proposals
3. RoI pooling извлекает фиксированные признаки для каждого proposal
4. FC layers для классификации и bbox regression

**Результат:** ~0.3 секунды на изображение (в 150 раз быстрее R-CNN)

---

#### **Faster R-CNN (2015) — современный стандарт**

**Главное улучшение:** Заменяем selective search на **Region Proposal Network (RPN)** — нейросеть, которая предлагает regions.

**Архитектура:**

```
Изображение
    ↓
Backbone CNN (ResNet50)
    ↓
Feature Map
    ↓
    ├─→ RPN (генерирует proposals)
    │       ↓
    └─→ RoI Pooling (использует proposals)
            ↓
        Classifier + Bbox Regressor
```

**Region Proposal Network (RPN):**
- Скользящее окно по feature map
- Для каждой позиции предсказываем K anchor boxes (разных размеров и aspect ratios)
- Для каждого anchor: objectness score (есть объект или фон?) + bbox refinement

```python
# Anchors для одной позиции feature map
anchors = [
    # width, height
    (128, 128),   # квадрат
    (128, 256),   # вертикальный
    (256, 128),   # горизонтальный
    # + еще 6 вариантов разных размеров
]
```

**Обучение Faster R-CNN:**
```python
# Загружаем предобученную модель
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Адаптируем под свои классы (например, 10 классов + фон)
num_classes = 11  # 10 классов + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Обучение
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

for images, targets in train_loader:
    # targets: [{'boxes': Tensor[N, 4], 'labels': Tensor[N]}, ...]
    
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
```

**Inference:**
```python
model.eval()
with torch.no_grad():
    predictions = model(images)

# predictions: [{'boxes': Tensor[M, 4], 'labels': Tensor[M], 'scores': Tensor[M]}, ...]
```

---

## 🟡 Продвинутый уровень: YOLO — You Only Look Once

### 3.1 Главная идея YOLO

**Отличие от R-CNN:**
- R-CNN: двухэтапный (proposals → classification)
- YOLO: одноэтапный (single shot detection)

**Как работает YOLO:**
1. Делим изображение на SxS grid (например, 7x7)
2. Каждая ячейка grid предсказывает:
   - B bounding boxes (обычно B=2)
   - Для каждого bbox: 5 значений (x, y, w, h, confidence)
   - C вероятностей классов
3. Один forward pass через CNN → все предсказания сразу

```
Выход YOLO: [S, S, B*5 + C]
Для S=7, B=2, C=20: [7, 7, 30]
```

---

### 3.2 YOLOv5 — популярная реализация

**Архитектура YOLOv5:**
- **Backbone:** CSPDarknet53 (извлечение признаков)
- **Neck:** PANet (агрегация признаков с разных уровней)
- **Head:** YOLO head (предсказания bbox + классов)

**Особенности:**
- ✅ Очень быстрый (30-60 FPS на GPU)
- ✅ Несколько версий: YOLOv5n, YOLOv5s, YOLOv5m, YOLOv5l, YOLOv5x (от маленькой к большой)
- ✅ Легко обучить на своих данных
- ✅ Export в ONNX, TensorRT для production

**Использование YOLOv5:**

```python
# !pip install ultralytics

from ultralytics import YOLO

# Загрузка предобученной модели
model = YOLO('yolov5s.pt')  # small версия

# Inference
results = model('image.jpg')

# Результаты
for result in results:
    boxes = result.boxes  # bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = box.cls[0]
        print(f"Класс {class_id}, confidence {confidence:.2f}, bbox [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

# Визуализация
result.show()
```

**Обучение на своих данных:**

```python
# Формат данных: YOLO формат
# Структура:
# dataset/
#   images/
#     train/
#     val/
#   labels/
#     train/
#     val/

# Файл аннотации (one file per image, .txt):
# <class_id> <x_center> <y_center> <width> <height>
# Координаты нормализованы (0-1)

# Обучение
model = YOLO('yolov5s.pt')
results = model.train(
    data='data.yaml',  # путь к конфигурации датасета
    epochs=100,
    imgsz=640,
    batch=16,
    name='my_detector'
)
```

---

### 3.3 YOLOv8 — последняя версия (2023)

**Улучшения в YOLOv8:**
- ✅ Anchor-free (не нужны предопределенные anchors)
- ✅ Новый backbone и head
- ✅ Лучшая точность при той же скорости
- ✅ Unified API для detection, segmentation, classification, pose

```python
from ultralytics import YOLO

# Загрузка YOLOv8
model = YOLO('yolov8n.pt')  # nano (самая маленькая)
# Или: yolov8s, yolov8m, yolov8l, yolov8x

# Inference
results = model('image.jpg')

# Обучение
model.train(data='coco128.yaml', epochs=100, imgsz=640)

# Валидация
metrics = model.val()

# Export
model.export(format='onnx')
```

---

## 🟡 Продвинутый уровень: RetinaNet и Focal Loss

### 4.1 Проблема Class Imbalance

**В object detection:**
- Большинство regions — background (negative examples)
- Мало regions содержат объекты (positive examples)
- Соотношение может быть 1000:1

**Проблема:**
- Модель "забивает" на hard examples (сложные объекты)
- Фокусируется на easy negatives (очевидный фон)
- Качество падает

---

### 4.2 Focal Loss

**Cross-Entropy Loss:**
```
CE(p) = -log(p)  если true label
       = -log(1-p)  если false label
```

**Проблема CE:** Дает большой loss даже для easy examples (p=0.9 → loss=0.1)

**Focal Loss:**
```
FL(p) = -(1-p)^γ * log(p)

где γ — focusing parameter (обычно γ=2)
```

**Эффект:**
- Easy examples (p близко к 1) → (1-p)^γ ≈ 0 → почти нулевой loss
- Hard examples (p близко к 0.5) → (1-p)^γ ≈ 0.25 → нормальный loss

```python
import torch.nn.functional as F

def focal_loss(predictions, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss для борьбы с class imbalance
    predictions: [N, num_classes] логиты
    targets: [N] метки классов
    """
    ce_loss = F.cross_entropy(predictions, targets, reduction='none')
    p_t = torch.exp(-ce_loss)  # вероятность правильного класса
    
    focal_weight = (1 - p_t) ** gamma
    loss = alpha * focal_weight * ce_loss
    
    return loss.mean()
```

---

### 4.3 RetinaNet Architecture

**Компоненты:**
- **Backbone:** ResNet50/101 + FPN (Feature Pyramid Network)
- **Classification subnet:** Предсказывает класс для каждого anchor
- **Box regression subnet:** Предсказывает bbox offsets
- **Focal Loss:** Для обучения classifier

**Использование RetinaNet:**

```python
# Загрузка предобученной модели
model = retinanet_resnet50_fpn(pretrained=True)

# Адаптация под свои классы
num_classes = 10
model.head.classification_head.num_classes = num_classes

# Inference
model.eval()
with torch.no_grad():
    predictions = model(images)
```

---

## 🔴 Экспертный уровень: Современные подходы

### 5.1 Feature Pyramid Network (FPN)

**Проблема:** Объекты разного размера сложно детектировать одной сетью.
- Маленькие объекты → нужны high-resolution features (ранние слои)
- Большие объекты → нужны semantic features (поздние слои)

**Решение FPN:** Создаем pyramid из feature maps разных разрешений.

```
Backbone (bottom-up):
Input (512x512)
    ↓
C1 (256x256)
    ↓
C2 (128x128)
    ↓
C3 (64x64)
    ↓
C4 (32x32)
    ↓
C5 (16x16)

FPN (top-down + lateral connections):
P5 ← C5
P4 ← C4 + upsample(P5)
P3 ← C3 + upsample(P4)
P2 ← C2 + upsample(P3)

Итого: [P2, P3, P4, P5] — разные масштабы
```

**Использование:**
- Маленькие объекты детектируем на P2 (большое разрешение)
- Большие объекты детектируем на P5 (много семантики)

---

### 5.2 Метрики качества Object Detection

#### **Precision и Recall**

```python
def calculate_precision_recall(predictions, ground_truths, iou_threshold=0.5):
    """
    predictions: [{'boxes': [...], 'scores': [...]}]
    ground_truths: [{'boxes': [...]}]
    """
    tp = 0  # true positives
    fp = 0  # false positives
    total_gt = sum(len(gt['boxes']) for gt in ground_truths)
    
    for pred, gt in zip(predictions, ground_truths):
        matched_gt = set()
        
        for pred_box in pred['boxes']:
            # Ищем лучший match с ground truth
            best_iou = 0
            best_idx = -1
            
            for idx, gt_box in enumerate(gt['boxes']):
                if idx in matched_gt:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
    
    fn = total_gt - tp  # false negatives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall
```

#### **Average Precision (AP)**

AP — площадь под кривой precision-recall.

```python
def calculate_ap(precisions, recalls):
    """Вычисляет AP по precision-recall кривой"""
    # Сортируем по recall
    sorted_indices = np.argsort(recalls)
    recalls = np.array(recalls)[sorted_indices]
    precisions = np.array(precisions)[sorted_indices]
    
    # Интерполяция precision
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    
    # Вычисляем AP (площадь под кривой)
    ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    return ap
```

#### **mean Average Precision (mAP)**

mAP — усредненный AP по всем классам.

```
mAP@0.5 = average AP при IoU threshold = 0.5
mAP@0.75 = average AP при IoU threshold = 0.75
mAP@[0.5:0.95] = average AP для IoU от 0.5 до 0.95 с шагом 0.05 (COCO metric)
```

---

### 5.3 Современные техники

#### **1. Data Augmentation для Detection**

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Blur(blur_limit=3, p=0.1),
    A.Resize(640, 640),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# Применение
transformed = transform(image=image, bboxes=bboxes, labels=labels)
```

#### **2. Multi-Scale Training**

Обучаем на изображениях разных размеров для robustness:

```python
scales = [480, 512, 544, 576, 608, 640]

for epoch in range(num_epochs):
    for batch in dataloader:
        # Случайный выбор размера для этого батча
        scale = random.choice(scales)
        images = F.interpolate(images, size=(scale, scale))
        
        # Обучение
        loss = model(images, targets)
        loss.backward()
        optimizer.step()
```

#### **3. Test-Time Augmentation (TTA)**

Усредняем предсказания для нескольких аугментированных версий:

```python
def tta_predict(model, image):
    predictions = []
    
    # Оригинал
    pred = model(image)
    predictions.append(pred)
    
    # Horizontal flip
    pred_flip = model(flip_horizontal(image))
    predictions.append(unflip(pred_flip))
    
    # Multi-scale
    for scale in [0.8, 1.0, 1.2]:
        pred_scale = model(resize(image, scale))
        predictions.append(resize_boxes(pred_scale, 1/scale))
    
    # Объединение (Weighted Box Fusion)
    final_pred = weighted_box_fusion(predictions)
    return final_pred
```

---

## 📊 Сравнительная таблица

| Модель | Скорость (FPS) | mAP (COCO) | Архитектура | Когда использовать |
|--------|----------------|------------|-------------|-------------------|
| **Faster R-CNN** | 5-7 | 37.0% | Two-stage | Высокая точность, не критична скорость |
| **RetinaNet** | 10-15 | 39.1% | One-stage + Focal Loss | Баланс точности и скорости |
| **YOLOv5s** | 140 | 37.4% | One-stage | Real-time приложения |
| **YOLOv8m** | 80 | 50.2% | One-stage, anchor-free | Баланс скорости и точности |
| **EfficientDet** | 30-40 | 51.0% | One-stage + BiFPN | Максимальная точность при ограниченных ресурсах |

---

## 🎯 Ключевые выводы

1. **Two-stage (Faster R-CNN)** — высокая точность, но медленно
2. **One-stage (YOLO, RetinaNet)** — быстро, но может уступать в точности
3. **Focal Loss** решает проблему class imbalance
4. **FPN** позволяет детектировать объекты разных размеров
5. **mAP** — основная метрика для оценки качества detection
6. **NMS** необходим для удаления дубликатов
7. **YOLOv8** — лучший выбор для большинства практических задач

---

## 📚 Дополнительные материалы

- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
- [YOLO Paper](https://arxiv.org/abs/1506.02640)
- [YOLOv5 Documentation](https://docs.ultralytics.com/)
- [RetinaNet Paper (Focal Loss)](https://arxiv.org/abs/1708.02002)
- [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144)
- [COCO Dataset](https://cocodataset.org/#download)
- [MMDetection Library](https://github.com/open-mmlab/mmdetection)
