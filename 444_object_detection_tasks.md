### **Задачи: Object Detection**

**Цель:** Научиться работать с моделями object detection, понять различия между two-stage и one-stage детекторами, освоить метрики (IoU, mAP), применить на практике.

---

## 🟢 Базовый уровень

### **Задача 1: IoU и NMS — реализация с нуля**

**Условие:** Реализуйте функции для вычисления IoU и Non-Maximum Suppression.

**Требования:**
1. Реализуйте функцию `calculate_iou(box1, box2)`
   - Формат bbox: [x_min, y_min, x_max, y_max]
   
2. Реализуйте функцию `nms(boxes, scores, iou_threshold)`
   - Удаляет дублирующиеся детекции
   
3. Протестируйте на примерах:
   - Два полностью совпадающих bbox → IoU = 1.0
   - Два непересекающихся bbox → IoU = 0.0
   - Два частично пересекающихся → 0 < IoU < 1
   
4. Визуализируйте результаты NMS:
   - До NMS: много overlapping boxes
   - После NMS: только уникальные детекции

**Ожидаемый результат:** NMS правильно удаляет дубликаты, оставляя boxes с максимальным score.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_iou(box1, box2):
    """
    Вычисляет IoU для двух bbox
    box1, box2: [x_min, y_min, x_max, y_max]
    """
    # TODO: реализуйте вычисление intersection
    # TODO: реализуйте вычисление union
    # TODO: верните IoU = intersection / union
    pass

def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression
    boxes: numpy array [N, 4]
    scores: numpy array [N]
    """
    # TODO: реализуйте NMS алгоритм
    pass

def visualize_boxes(image, boxes, scores=None, title="Boxes"):
    """Визуализирует bounding boxes на изображении"""
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)
    
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        
        rect = patches.Rectangle((x_min, y_min), width, height,
                                linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        if scores is not None:
            ax.text(x_min, y_min - 5, f'{scores[i]:.2f}',
                   color='red', fontsize=12, weight='bold')
    
    ax.set_title(title)
    plt.show()

# Тестовые данные
boxes = np.array([
    [100, 100, 200, 200],
    [110, 110, 210, 210],  # Сильное пересечение с первым
    [105, 105, 205, 205],  # Еще одно пересечение
    [300, 300, 400, 400],  # Отдельный объект
])
scores = np.array([0.95, 0.88, 0.82, 0.90])

# TODO: примените NMS
# TODO: визуализируйте до и после
```

**Вопросы для анализа:**
1. Как выбор `iou_threshold` влияет на результаты NMS?
2. Что произойдет, если два объекта действительно перекрываются?
3. Как NMS работает с multi-class detection?

---

### **Задача 2: Faster R-CNN на COCO**

**Условие:** Используйте предобученный Faster R-CNN для детекции объектов на изображениях.

**Требования:**
1. Загрузите предобученный Faster R-CNN (ResNet50 backbone)
2. Реализуйте функцию inference для одного изображения
3. Примените на 10 тестовых изображениях
4. Визуализируйте результаты:
   - Bounding boxes с labels
   - Confidence scores
   
5. Настройте threshold для confidence score (0.5, 0.7, 0.9)
6. Сравните результаты для разных threshold

**Ожидаемый результат:** Модель корректно детектирует и локализует объекты.

```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# COCO классы
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    # ... (всего 91 класс)
]

def detect_objects(image_path, model, threshold=0.5):
    """Детектирует объекты на изображении"""
    
    # Загрузка и preprocessing
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)
    
    # Inference
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    
    # Фильтрация по threshold
    keep = predictions['scores'] > threshold
    boxes = predictions['boxes'][keep].cpu().numpy()
    labels = predictions['labels'][keep].cpu().numpy()
    scores = predictions['scores'][keep].cpu().numpy()
    
    return image, boxes, labels, scores

def visualize_detection(image, boxes, labels, scores):
    """Визуализирует результаты детекции"""
    # TODO: реализуйте визуализацию с подписями классов
    pass

# Загрузка модели
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# TODO: протестируйте на изображениях
# TODO: сравните результаты для разных threshold
```

---

### **Задача 3: YOLOv8 Inference**

**Условие:** Используйте YOLOv8 для real-time детекции на видео.

**Требования:**
1. Установите ultralytics: `pip install ultralytics`
2. Загрузите YOLOv8n (nano версия)
3. Реализуйте детекцию на:
   - Статичных изображениях
   - Видео файле
   - Webcam (если доступна)
   
4. Измерьте FPS для разных версий:
   - YOLOv8n (nano)
   - YOLOv8s (small)
   - YOLOv8m (medium)
   
5. Сравните:
   - Скорость (FPS)
   - Качество детекций (визуально)
   - Размер модели

**Ожидаемый результат:** YOLOv8n работает в real-time (>30 FPS), более крупные модели точнее, но медленнее.

```python
from ultralytics import YOLO
import cv2
import time

def benchmark_yolo(model_name, video_path, num_frames=100):
    """Измеряет FPS для YOLOv8"""
    
    model = YOLO(f'{model_name}.pt')
    cap = cv2.VideoCapture(video_path)
    
    times = []
    frame_count = 0
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.time()
        results = model(frame, verbose=False)
        end = time.time()
        
        times.append(end - start)
        frame_count += 1
    
    cap.release()
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    return fps

# TODO: измерьте FPS для yolov8n, yolov8s, yolov8m
# TODO: визуализируйте сравнение

def detect_video(model, video_path, output_path):
    """Применяет детекцию к видео"""
    # TODO: реализуйте обработку видео с сохранением результата
    pass
```

**Вопросы:**
1. Почему YOLOv8 быстрее Faster R-CNN?
2. Как выбрать версию YOLO для production?
3. Что такое inference time vs FPS?

---

## 🟡 Продвинутый уровень

### **Задача 4: Fine-tuning Faster R-CNN на своих данных**

**Условие:** Обучите Faster R-CNN детектировать объекты на пользовательском датасете.

**Требования:**
1. Используйте небольшой датасет (100-500 изображений, 2-3 класса)
   - Например: детекция лиц, автомобилей, фруктов
   
2. Подготовьте данные в формате COCO или Pascal VOC
3. Адаптируйте предобученный Faster R-CNN:
   - Замените classification head под свои классы
   - Fine-tune всю модель
   
4. Обучите 10-20 эпох
5. Оцените качество:
   - Визуально на test set
   - Вычислите mAP@0.5

**Ожидаемый результат:** Модель научится детектировать ваши объекты с хорошим качеством.

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class CustomDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transforms=None):
        # TODO: загрузите изображения и аннотации
        pass
    
    def __getitem__(self, idx):
        # Вернуть: image (tensor), target (dict с 'boxes', 'labels')
        pass
    
    def __len__(self):
        pass

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    return total_loss / len(dataloader)

# Адаптация модели
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 3 + 1  # 3 класса + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# TODO: подготовьте DataLoader
# TODO: обучите модель
# TODO: оцените на test set
```

---

### **Задача 5: YOLOv8 Training на Custom Dataset**

**Условие:** Обучите YOLOv8 на своем датасете с нуля.

**Требования:**
1. Подготовьте данные в YOLO формате:
   ```
   dataset/
     images/
       train/
       val/
     labels/
       train/
       val/
   data.yaml
   ```

2. Аннотации в формате:
   ```
   <class_id> <x_center> <y_center> <width> <height>
   ```
   (координаты нормализованы 0-1)

3. Обучите YOLOv8s 50-100 эпох
4. Используйте аугментации:
   - Horizontal flip
   - Mosaic
   - MixUp
   
5. Отслеживайте метрики:
   - mAP@0.5
   - mAP@0.5:0.95
   - Precision/Recall
   
6. Экспортируйте модель в ONNX для deployment

**Ожидаемый результат:** Модель достигает mAP > 0.7 на test set.

```python
from ultralytics import YOLO

# data.yaml:
# train: ../dataset/images/train
# val: ../dataset/images/val
# nc: 3  # number of classes
# names: ['class1', 'class2', 'class3']

# Обучение
model = YOLO('yolov8s.pt')
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='custom_detector',
    # Аугментации
    hsv_h=0.015,  # image HSV-Hue augmentation
    hsv_s=0.7,    # image HSV-Saturation augmentation
    hsv_v=0.4,    # image HSV-Value augmentation
    degrees=10,   # rotation
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1
)

# Валидация
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50}")
print(f"mAP@0.5:0.95: {metrics.box.map}")

# Export
model.export(format='onnx')
```

---

### **Задача 6: Focal Loss — борьба с Class Imbalance**

**Условие:** Реализуйте Focal Loss и сравните с обычным Cross-Entropy.

**Требования:**
1. Реализуйте Focal Loss с нуля:
   ```python
   FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
   ```
   
2. Создайте синтетический датасет с сильным дисбалансом:
   - 90% negative examples
   - 10% positive examples
   
3. Обучите простую модель классификации с:
   - Cross-Entropy Loss
   - Focal Loss (γ=2)
   
4. Сравните:
   - Loss curves
   - Precision/Recall на hard examples
   - Общее качество

**Ожидаемый результат:** Focal Loss лучше справляется с hard examples.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        inputs: [N, num_classes] logits
        targets: [N] class indices
        """
        # TODO: реализуйте Focal Loss
        pass

# Создание imbalanced датасета
def create_imbalanced_dataset(num_samples=1000, imbalance_ratio=0.1):
    # TODO: создайте датасет с дисбалансом
    pass

# TODO: обучите модели с CE и Focal Loss
# TODO: сравните результаты
```

---

## 🔴 Экспертный уровень

### **Задача 7: Реализация Simple Detector с нуля**

**Условие:** Реализуйте упрощенный one-stage детектор (аналог YOLO) с нуля.

**Требования:**
1. Архитектура:
   - Backbone: ResNet18 (pretrained)
   - Detection head: предсказания bbox + класс для каждой ячейки grid
   
2. Формат выхода:
   - Grid: 7x7
   - Для каждой ячейки: [x, y, w, h, objectness, class_probs]
   
3. Loss function:
   - Localization loss (MSE для bbox coordinates)
   - Objectness loss (BCE)
   - Classification loss (CE)
   
4. Обучите на простом датасете (100-200 изображений)
5. Реализуйте post-processing (NMS)

**Ожидаемый результат:** Детектор работает, хотя и уступает YOLO по качеству.

```python
class SimpleDetector(nn.Module):
    def __init__(self, num_classes, grid_size=7):
        super().__init__()
        
        # Backbone
        resnet = torchvision.models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Detection head
        # Output: [batch, grid, grid, 5 + num_classes]
        # где 5 = (x, y, w, h, objectness)
        self.head = nn.Sequential(
            # TODO: реализуйте detection head
        )
        
        self.grid_size = grid_size
        self.num_classes = num_classes
    
    def forward(self, x):
        # TODO: реализуйте forward pass
        pass

def compute_loss(predictions, targets, grid_size):
    """
    predictions: [batch, grid, grid, 5 + num_classes]
    targets: [batch, grid, grid, 5 + num_classes]
    """
    # TODO: реализуйте комбинированный loss
    pass

# TODO: обучите детектор
# TODO: реализуйте inference с NMS
```

---

### **Задача 8: Multi-Scale Detection с FPN**

**Условие:** Реализуйте Feature Pyramid Network для детекции объектов разных размеров.

**Требования:**
1. Реализуйте FPN:
   - Bottom-up pathway (backbone)
   - Top-down pathway с lateral connections
   - Получите feature maps на разных уровнях [P2, P3, P4, P5]
   
2. Для каждого уровня:
   - Anchor boxes разных размеров
   - Отдельные heads для предсказаний
   
3. Обучите на датасете с объектами разных размеров
4. Сравните с baseline (без FPN):
   - AP для small objects
   - AP для medium objects
   - AP для large objects

**Ожидаемый результат:** FPN значительно улучшает детекцию маленьких объектов.

```python
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        
        # Top-down convs
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) 
            for _ in in_channels_list
        ])
    
    def forward(self, features):
        """
        features: list of [C2, C3, C4, C5] от backbone
        returns: list of [P2, P3, P4, P5]
        """
        # TODO: реализуйте top-down pathway
        pass

# TODO: интегрируйте FPN в детектор
# TODO: обучите и сравните с baseline
```

---

### **Задача 9: Ensemble детекторов**

**Условие:** Создайте ensemble из нескольких детекторов для повышения качества.

**Требования:**
1. Обучите 3 разных детектора:
   - Faster R-CNN (ResNet50)
   - YOLOv8m
   - RetinaNet
   
2. Реализуйте методы объединения:
   - **NMS Ensemble:** Объединяем все predictions и применяем NMS
   - **Weighted Box Fusion (WBF):** Умное взвешивание overlapping boxes
   - **Voting:** Оставляем boxes, которые предсказали >= 2 модели
   
3. Сравните:
   - Каждая модель по отдельности
   - NMS Ensemble
   - WBF
   - Voting
   
4. Измерьте:
   - mAP
   - Inference time
   - Memory usage

**Ожидаемый результат:** Ensemble дает прирост mAP на 2-5%, но медленнее.

```python
def weighted_box_fusion(predictions_list, weights=None, iou_threshold=0.5):
    """
    Weighted Box Fusion для объединения predictions от разных моделей
    predictions_list: list of [{'boxes': [...], 'scores': [...], 'labels': [...]}]
    """
    # TODO: реализуйте WBF алгоритм
    pass

def ensemble_predict(models, image):
    """Получает predictions от всех моделей"""
    predictions = []
    for model in models:
        pred = model(image)
        predictions.append(pred)
    return predictions

# TODO: обучите 3 модели
# TODO: реализуйте все методы ensemble
# TODO: сравните результаты
```

---

### **Задача 10: Real-World Application — детекция на edge device**

**Условие:** Оптимизируйте модель для deployment на edge device (Raspberry Pi, Jetson Nano).

**Требования:**
1. Выберите легкую модель: YOLOv8n или MobileNet-based detector
2. Оптимизируйте для inference:
   - Quantization (INT8)
   - Pruning
   - ONNX export
   - TensorRT (если есть GPU)
   
3. Измерьте до и после оптимизации:
   - Inference time
   - Model size (MB)
   - mAP (потери качества)
   
4. Реализуйте video streaming детекцию:
   - Webcam input
   - Real-time detection (>15 FPS)
   - Display results

**Ожидаемый результат:** Модель работает в real-time на edge device с минимальными потерями качества.

```python
from ultralytics import YOLO

# 1. Обучаем модель
model = YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs=100)

# 2. Quantization
model.export(format='onnx', int8=True)

# 3. Benchmark
def benchmark_model(model_path, num_iterations=100):
    """Измеряет inference time"""
    # TODO: реализуйте бенчмарк
    pass

# 4. Real-time video detection
def realtime_detection(model, camera_id=0):
    """Детекция на видео с webcam"""
    import cv2
    
    cap = cv2.VideoCapture(camera_id)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detection
        results = model(frame)
        
        # Draw boxes
        annotated = results[0].plot()
        
        # Display
        cv2.imshow('Detection', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# TODO: оптимизируйте модель
# TODO: запустите real-time детекцию
```

---

## 📝 Дополнительные вопросы для размышления

1. **Two-stage vs One-stage:**
   - Когда лучше использовать каждый подход?
   - Можно ли сделать one-stage такой же точной, как two-stage?

2. **Anchor boxes:**
   - Зачем нужны anchors?
   - Как выбрать оптимальные размеры и aspect ratios?
   - Что такое anchor-free детекторы?

3. **Multi-scale detection:**
   - Почему FPN улучшает детекцию маленьких объектов?
   - Какие еще есть подходы для multi-scale?

4. **Production deployment:**
   - Как выбрать модель для production?
   - Что важнее: accuracy или latency?
   - Как оптимизировать для конкретного hardware?

---

## 🎯 Критерии успешного выполнения

- ✅ Вы понимаете метрики IoU, mAP, Precision/Recall
- ✅ Вы умеете реализовать и применять NMS
- ✅ Вы знаете разницу между two-stage и one-stage детекторами
- ✅ Вы умеете обучать детекторы на своих данных
- ✅ Вы понимаете, как работает Focal Loss и FPN
- ✅ Вы можете выбрать оптимальную модель для задачи
- ✅ Вы умеете оптимизировать модель для production

---

## 📚 Полезные ресурсы

- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
- [YOLO Paper](https://arxiv.org/abs/1506.02640)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Focal Loss Paper](https://arxiv.org/abs/1708.02002)
- [FPN Paper](https://arxiv.org/abs/1612.03144)
- [COCO Dataset](https://cocodataset.org/#download)
- [MMDetection Library](https://github.com/open-mmlab/mmdetection)
- [TorchVision Detection Reference](https://pytorch.org/vision/stable/models.html#object-detection)
