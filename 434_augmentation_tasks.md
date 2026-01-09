### **Задачи: Data Augmentation**

**Цель:** Научиться применять различные техники аугментации данных для улучшения качества моделей и борьбы с переобучением.

---

## 🟢 Базовый уровень

### **Задача 1: Демонстрация эффекта аугментации**

**Условие:** Покажите, как аугментация улучшает генерализацию модели.

**Требования:**
1. Обучите две модели на CIFAR-10:
   - Без аугментации (только ToTensor + Normalize)
   - С базовой аугментацией (RandomCrop, RandomHorizontalFlip)
2. Используйте маленький датасет (10000 примеров)
3. Сравните:
   - Train accuracy
   - Test accuracy
   - Gap между train и test (мера переобучения)
4. Постройте графики train/test accuracy для обеих моделей

**Ожидаемый результат:** С аугментацией меньше переобучение, выше test accuracy.

```python
import torchvision.transforms as transforms

# Без аугментации
transform_simple = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# С аугментацией
transform_augmented = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# TODO: обучите и сравните модели
```

**Вопрос:** Почему аугментация помогает генерализации?

---

### **Задача 2: Базовые геометрические трансформации**

**Условие:** Исследуйте влияние различных геометрических трансформаций.

**Требования:**
1. Протестируйте на MNIST каждую трансформацию отдельно:
   - RandomRotation(10)
   - RandomRotation(30)
   - RandomAffine(degrees=15, translate=(0.1, 0.1))
   - RandomPerspective(distortion_scale=0.2)
2. Для каждой:
   - Визуализируйте 10 примеров аугментированных изображений
   - Обучите модель
   - Измерьте test accuracy
3. Определите, какие трансформации полезны для MNIST

**Ожидаемый результат:** 
- Небольшие повороты помогают
- Слишком сильные искажения вредят

```python
def visualize_augmentation(dataset, transform_name):
    """Визуализирует примеры аугментации"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        img, label = dataset[0]  # Одно и то же изображение
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f'{transform_name} #{i+1}')
        ax.axis('off')
    plt.show()
```

---

### **Задача 3: ColorJitter для изображений**

**Условие:** Примените ColorJitter и изучите его влияние.

**Требования:**
1. Используйте CIFAR-10
2. Создайте трансформации с разной интенсивностью ColorJitter:
   - Слабая: brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
   - Средняя: brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
   - Сильная: brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2
3. Визуализируйте примеры каждой
4. Обучите модели и сравните результаты

**Вопрос:** Может ли слишком сильная аугментация навредить?

```python
from torchvision.transforms import ColorJitter

transform_weak_color = transforms.Compose([
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor()
])

# TODO: создайте и протестируйте остальные варианты
```

---

## 🟡 Продвинутый уровень

### **Задача 4: RandAugment — автоматическая аугментация**

**Условие:** Используйте RandAugment и сравните с ручной аугментацией.

**Требования:**
1. Создайте три варианта аугментации для CIFAR-10:
   - Ручная (RandomCrop + RandomFlip + ColorJitter)
   - RandAugment(num_ops=2, magnitude=9)
   - RandAugment(num_ops=3, magnitude=14)
2. Обучите модели для каждого варианта (3 запуска каждая)
3. Сравните:
   - Среднюю test accuracy
   - Стандартное отклонение (стабильность)
   - Время обучения
4. Визуализируйте примеры аугментаций

**Ожидаемый результат:** RandAugment дает хорошие результаты с минимальной настройкой.

```python
from torchvision.transforms import RandAugment

transform_randaug = transforms.Compose([
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# RandAugment применяет N случайных трансформаций с заданной силой
```

---

### **Задача 5: Cutout — вырезание случайных областей**

**Условие:** Реализуйте и примените Cutout аугментацию.

**Требования:**
1. Реализуйте класс `Cutout`:
   ```python
   class Cutout:
       def __init__(self, n_holes=1, length=16):
           self.n_holes = n_holes
           self.length = length
       
       def __call__(self, img):
           """Вырезает случайные квадраты из изображения"""
           # TODO: реализуйте
           pass
   ```
2. Протестируйте с разными параметрами:
   - n_holes=1, length=8
   - n_holes=1, length=16
   - n_holes=2, length=12
3. Визуализируйте примеры
4. Обучите на CIFAR-10 и сравните с базовой аугментацией

**Вопрос:** Почему Cutout помогает модели лучше обобщать?

**Подсказка:** Cutout заставляет модель использовать всё изображение, а не полагаться на одну область.

---

### **Задача 6: Mixup — смешивание изображений**

**Условие:** Реализуйте Mixup аугментацию.

**Требования:**
1. Реализуйте функцию mixup:
   ```python
   def mixup_data(x, y, alpha=1.0):
       """Смешивает изображения и метки"""
       lam = np.random.beta(alpha, alpha)
       batch_size = x.size(0)
       index = torch.randperm(batch_size)
       
       mixed_x = lam * x + (1 - lam) * x[index]
       y_a, y_b = y, y[index]
       
       return mixed_x, y_a, y_b, lam
   
   def mixup_criterion(criterion, pred, y_a, y_b, lam):
       """Loss для mixup"""
       return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
   ```
2. Интегрируйте в training loop
3. Визуализируйте примеры смешанных изображений
4. Сравните с обычным обучением:
   - Test accuracy
   - Калибровка (confidence vs correctness)

**Ожидаемый результат:** Mixup улучшает генерализацию и калибровку.

---

## 🔴 Экспертный уровень

### **Задача 7: CutMix — комбинация Cutout и Mixup**

**Условие:** Реализуйте CutMix аугментацию.

**Требования:**
1. Реализуйте CutMix:
   ```python
   def cutmix_data(x, y, alpha=1.0):
       """Вырезает и вставляет область из другого изображения"""
       lam = np.random.beta(alpha, alpha)
       batch_size = x.size(0)
       index = torch.randperm(batch_size)
       
       # Выбираем случайный box
       _, _, H, W = x.size()
       cut_rat = np.sqrt(1. - lam)
       cut_w = int(W * cut_rat)
       cut_h = int(H * cut_rat)
       
       cx = np.random.randint(W)
       cy = np.random.randint(H)
       
       bbx1 = np.clip(cx - cut_w // 2, 0, W)
       bby1 = np.clip(cy - cut_h // 2, 0, H)
       bbx2 = np.clip(cx + cut_w // 2, 0, W)
       bby2 = np.clip(cy + cut_h // 2, 0, H)
       
       # TODO: вставить region из другого изображения
       pass
   ```
2. Визуализируйте примеры CutMix
3. Сравните все методы на CIFAR-10:
   - Baseline (базовая аугментация)
   - Cutout
   - Mixup
   - CutMix
4. Создайте таблицу результатов

**Ожидаемый результат:** CutMix часто показывает лучшие результаты.

---

### **Задача 8: Test-Time Augmentation (TTA)**

**Условие:** Используйте аугментацию во время инференса для улучшения предсказаний.

**Требования:**
1. Реализуйте TTA:
   ```python
   @torch.no_grad()
   def predict_with_tta(model, image, n_augmentations=10):
       """Предсказание с TTA"""
       model.eval()
       predictions = []
       
       for _ in range(n_augmentations):
           # Применяем случайную аугментацию
           augmented = augment_image(image)
           pred = model(augmented)
           predictions.append(pred)
       
       # Усредняем предсказания
       mean_pred = torch.stack(predictions).mean(dim=0)
       return mean_pred
   ```
2. Протестируйте на CIFAR-10 test set
3. Сравните accuracy:
   - Без TTA
   - С TTA (5 аугментаций)
   - С TTA (10 аугментаций)
4. Измерьте увеличение времени инференса

**Вопрос:** Когда TTA наиболее полезна?

---

### **Задача 9: AutoAugment — оптимизация политики аугментации**

**Условие:** Используйте AutoAugment или создайте свою политику аугментации.

**Требования:**
1. Если доступно в torchvision, используйте AutoAugment для CIFAR-10:
   ```python
   from torchvision.transforms import AutoAugment, AutoAugmentPolicy
   
   transform = transforms.Compose([
       AutoAugment(AutoAugmentPolicy.CIFAR10),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)
   ])
   ```
2. Если нет, создайте свою политику:
   - Определите набор трансформаций
   - Случайно применяйте N трансформаций
   - Подберите оптимальные параметры (probability, magnitude)
3. Сравните с RandAugment
4. Визуализируйте примеры

**Ожидаемый результат:** AutoAugment/RandAugment дают state-of-the-art результаты.

---

### **Задача 10: Аугментация для специфических доменов**

**Условие:** Создайте кастомную аугментацию для специфической задачи.

**Выберите один домен:**

**A. Медицинские изображения:**
- Elastic deformation (эластичная деформация)
- Grid distortion
- Gaussian noise
- Сохранение важных структур

**B. Текстовые данные (OCR/документы):**
- Rotation небольшая (±5°)
- Изменение толщины линий
- Blur/Sharpen
- Искажение перспективы

**C. Satellite imagery:**
- Multi-spectral augmentation
- Seasonal changes simulation
- Cloud simulation
- Shadow augmentation

**Требования:**
1. Реализуйте 3-5 специфичных аугментаций
2. Создайте композицию аугментаций
3. Визуализируйте примеры
4. Обучите модель и покажите улучшение
5. Объясните, почему эти аугментации подходят для домена

```python
class DomainSpecificAugmentation:
    """Кастомная аугментация для специфического домена"""
    def __init__(self, augmentation_type='medical'):
        self.aug_type = augmentation_type
    
    def __call__(self, img):
        if self.aug_type == 'medical':
            # TODO: elastic deformation, grid distortion
            pass
        elif self.aug_type == 'ocr':
            # TODO: rotation, thickness, blur
            pass
        # TODO: добавьте другие домены
        
        return img
```

---

### **Задача 11: Сравнительный анализ всех методов**

**Условие:** Проведите комплексное сравнение всех методов аугментации.

**Требования:**
1. Реализуйте все методы:
   - Baseline (RandomCrop + RandomFlip)
   - + ColorJitter
   - + Cutout
   - + Mixup
   - + CutMix
   - + RandAugment
   - + TTA
2. Для каждого метода:
   - Обучите 5 моделей с разными seeds
   - Усредните результаты
   - Вычислите std
3. Создайте comprehensive таблицу:
   ```
   | Method      | Test Acc (%) | Std    | Training Time | Inference Time |
   |-------------|--------------|--------|---------------|----------------|
   | Baseline    | 85.2         | 0.3    | 1x            | 1x             |
   | + ColorJit  | 86.1         | 0.2    | 1.1x          | 1x             |
   | ...         |              |        |               |                |
   ```
4. Визуализируйте результаты (bar plot)
5. Напишите выводы: какие методы лучше комбинировать

**Датасет:** CIFAR-10 или CIFAR-100

---

## 💎 Заключение

### **Рекомендации по аугментации:**

| Задача | Рекомендуемые аугментации | Почему |
|--------|---------------------------|--------|
| **Natural Images (CIFAR, ImageNet)** | RandomCrop, RandomFlip, ColorJitter, RandAugment/AutoAugment | Естественные вариации |
| **Medical Images** | Elastic deformation, Rotation, Flip (осторожно!), Gaussian noise | Вариации съемки |
| **OCR/Documents** | Small rotation, Perspective, Blur/Sharpen | Условия сканирования |
| **Faces** | RandomFlip (horizontal only!), ColorJitter, RandomErasing | Естественные вариации |
| **Satellite** | Rotation, Flip (все направления), Color/Brightness | Угол съемки, погода |

### **Чек-лист по аугментации:**

✅ **Базовые правила:**
- [ ] Аугментация только для training, не для validation/test!
- [ ] Визуализируйте примеры перед обучением
- [ ] Начинайте с простых аугментаций
- [ ] Не применяйте аугментации, нарушающие семантику (horizontal flip для текста)

✅ **Для разных данных:**
- [ ] Изображения: Geometric + Color + Cutout/Mixup
- [ ] Текст: Synonym replacement, Back translation, EDA
- [ ] Аудио: Time stretch, Pitch shift, Add noise
- [ ] Табличные данные: Gaussian noise, SMOTE

✅ **Продвинутое:**
- [ ] RandAugment/AutoAugment для автоматической оптимизации
- [ ] Mixup/CutMix для лучшей калибровки
- [ ] TTA для улучшения инференса
- [ ] Domain-specific аугментации

### **Типичные ошибки:**

❌ **Не делайте так:**
- Применять аугментацию на validation/test
- Слишком сильные искажения (теряется семантика)
- Одинаковые аугментации для всех доменов
- Не визуализировать результаты аугментации

✅ **Делайте так:**
- Всегда проверяйте, что аугментация сохраняет метку
- Начинайте с консервативных параметров
- Постепенно увеличивайте интенсивность
- Используйте TTA для важных предсказаний

### **Практический pipeline:**

```python
# Стандартный pipeline для CIFAR-10
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                        (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                        (0.2023, 0.1994, 0.2010))
])

# С Mixup в training loop
for data, target in train_loader:
    if np.random.rand() < 0.5:  # 50% вероятность
        data, target_a, target_b, lam = mixup_data(data, target)
        loss = mixup_criterion(criterion, output, target_a, target_b, lam)
    else:
        output = model(data)
        loss = criterion(output, target)
```

### **Ожидаемый прирост качества:**

- **Baseline** → + ColorJitter: +0.5-1.0% accuracy
- **Baseline** → + Cutout: +1.0-2.0% accuracy
- **Baseline** → + Mixup: +1.5-2.5% accuracy
- **Baseline** → + RandAugment: +2.0-3.0% accuracy
- **Baseline** → + AutoAugment: +2.5-4.0% accuracy
- **Лучшая модель** → + TTA: +0.5-1.0% accuracy

### **Дополнительные ресурсы:**

1. **Статьи:**
   - [AutoAugment: Learning Augmentation Policies](https://arxiv.org/abs/1805.09501)
   - [RandAugment](https://arxiv.org/abs/1909.13719)
   - [Mixup](https://arxiv.org/abs/1710.09412)
   - [CutMix](https://arxiv.org/abs/1905.04899)
   - [Cutout](https://arxiv.org/abs/1708.04552)

2. **Библиотеки:**
   - `torchvision.transforms` — базовые трансформации
   - `albumentations` — продвинутые аугментации для CV
   - `imgaug` — еще одна библиотека аугментаций
   - `nlpaug` — аугментации для текста

3. **Практика:**
   - Всегда визуализируйте аугментации перед обучением
   - Экспериментируйте с комбинациями
   - Используйте аугментации, подходящие для вашего домена

> **"Data Augmentation — это самый простой способ получить 'больше данных' бесплатно. Правильная аугментация может дать +3-5% accuracy с минимальными усилиями!"**
