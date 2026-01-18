Глава: Эффективная обработка данных: батчи, Dataset и DataLoader

🎯 Зачем нам батчи? Мотивирующий пример

Представьте, что вам нужно прочитать и запомнить энциклопедию из 1000 страниц:

· Вариант A: Прочитать всю книгу за один раз, затем повторить 10 раз
· Вариант B: Читать по 20 страниц в день, повторяя пройденное

Какой метод эффективнее? Второй! В машинном обучении работает тот же принцип. Батчи позволяют:

1. Экономить память (не нужно загружать все данные сразу)
2. Ускорять обучение (параллельные вычисления)
3. Улучшать качество (шум в градиентах помогает избегать локальных минимумов)

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import time

# Создадим реалистичный набор данных
X, y = make_classification(
    n_samples=5000,      # 5000 примеров
    n_features=20,       # 20 признаков
    n_classes=3,         # 3 класса
    n_informative=8,     # 8 информативных признаков
    random_state=42
)

print("📊 Обзор нашего набора данных:")
print(f"• Примеров: {X.shape[0]:,}")
print(f"• Признаков у каждого примера: {X.shape[1]}")
print(f"• Классов: {len(np.unique(y))}")
print(f"• Размер в памяти: {X.nbytes / 1024 / 1024:.1f} МБ")
```

---

🔄 Попробуйте сами #1: Сравнение разных подходов

```python
# ИССЛЕДУЙТЕ РАЗНЫЕ СТРАТЕГИИ ОБУЧЕНИЯ:

class SimpleClassifier(nn.Module):
    """Простой классификатор для экспериментов"""
    def __init__(self, input_size=20, hidden_size=32, output_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# Преобразуем данные в тензоры PyTorch
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

def train_single_example(model, X, y, epochs=5):
    """Обучение на одном примере за раз (Stochastic Gradient Descent)"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    losses = []
    start_time = time.time()
    
    print("Стратегия: Один пример за раз")
    for epoch in range(epochs):
        epoch_loss = 0
        
        # Перемешиваем данные каждый эпоху
        indices = torch.randperm(len(X))
        
        for idx in indices:
            # Берем ОДИН пример
            x_single = X[idx:idx+1]
            y_single = y[idx:idx+1]
            
            optimizer.zero_grad()
            predictions = model(x_single)
            loss = criterion(predictions, y_single)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(X)
        losses.append(avg_loss)
        print(f"  Эпоха {epoch}: средняя ошибка = {avg_loss:.4f}")
    
    return losses, time.time() - start_time

def train_mini_batch(model, X, y, batch_size=32, epochs=5):
    """Обучение мини-батчами"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    losses = []
    start_time = time.time()
    
    print(f"Стратегия: Мини-батчи по {batch_size} примеров")
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        # Перемешиваем данные
        indices = torch.randperm(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Обрабатываем батчами
        for i in range(0, len(X), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        losses.append(avg_loss)
        print(f"  Эпоха {epoch}: средняя ошибка = {avg_loss:.4f}")
    
    return losses, time.time() - start_time

def train_full_batch(model, X, y, epochs=5):
    """Обучение на всех данных сразу (Full Batch)"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    losses = []
    start_time = time.time()
    
    print("Стратегия: Все данные сразу")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # ВСЕ данные за раз
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Эпоха {epoch}: ошибка = {loss.item():.4f}")
    
    return losses, time.time() - start_time

# Запускаем сравнение
print("=" * 60)
print("СРАВНЕНИЕ ТРЕХ СТРАТЕГИЙ ОБУЧЕНИЯ")
print("=" * 60)

results = {}
strategies = [
    ("Один пример", 1, train_single_example),
    ("Мини-батч 32", 32, lambda m, X, y, e: train_mini_batch(m, X, y, 32, e)),
    ("Полный батч", len(X), train_full_batch)
]

for name, batch_size, train_func in strategies:
    print(f"\n{name}:")
    model = SimpleClassifier()
    losses, duration = train_func(model, X_tensor[:1000], y_tensor[:1000], epochs=3)
    results[name] = {
        "losses": losses,
        "time": duration,
        "final_loss": losses[-1]
    }

# Визуализируем результаты
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# График ошибок
for name, res in results.items():
    axes[0].plot(res["losses"], marker='o', label=name, linewidth=2)

axes[0].set_xlabel('Эпоха')
axes[0].set_ylabel('Ошибка')
axes[0].set_title('Сравнение скорости сходимости')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# График времени
names = list(results.keys())
times = [results[name]["time"] for name in names]
bars = axes[1].bar(names, times, color=['blue', 'green', 'red'])

axes[1].set_xlabel('Стратегия')
axes[1].set_ylabel('Время (секунды)')
axes[1].set_title('Сравнение времени обучения')
axes[1].grid(True, alpha=0.3, axis='y')

# Добавляем значения на столбцы
for bar, time_val in zip(bars, times):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time_val:.2f}s', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("КЛЮЧЕВЫЕ ВЫВОДЫ:")
print("=" * 60)
for name, res in results.items():
    print(f"{name:15} | Время: {res['time']:6.2f}с | Финальная ошибка: {res['final_loss']:.4f}")

print("\n💡 Наблюдения:")
print("• Мини-батчи дают баланс между скоростью и стабильностью")
print("• Обучение по одному примеру очень шумное и медленное")
print("• Полный батч может быть медленным для больших данных")
```

---

🗂️ Dataset: Организуем данные правильно

Что такое Dataset и зачем он нужен?

Проблема: Когда у нас миллионы изображений или текстов, мы не можем загрузить их все в память сразу.

Решение: Dataset — это абстракция, которая:

1. Хранит информацию о данных
2. Загружает данные по требованию (ленивая загрузка)
3. Позволяет применять преобразования

```python
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    """Простейший Dataset для табличных данных"""
    
    def __init__(self, features, labels):
        """
        Args:
            features: тензор или массив с признаками
            labels: тензор или массив с метками
        """
        self.features = torch.FloatTensor(features) if not isinstance(features, torch.Tensor) else features
        self.labels = torch.LongTensor(labels) if not isinstance(labels, torch.Tensor) else labels
        
        # Проверяем согласованность размеров
        assert len(self.features) == len(self.labels), \
            f"Разное количество примеров: features={len(self.features)}, labels={len(self.labels)}"
    
    def __len__(self):
        """Возвращает общее количество примеров"""
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Возвращает один пример по индексу
        Важно: этот метод вызывается при обращении dataset[idx]
        """
        return {
            'features': self.features[idx],
            'label': self.labels[idx]
        }
    
    def get_stats(self):
        """Выводит статистику о данных"""
        print("📊 Статистика Dataset:")
        print(f"  • Примеров: {len(self):,}")
        print(f"  • Признаков: {self.features.shape[1]}")
        print(f"  • Классов: {len(torch.unique(self.labels))}")
        print(f"  • Распределение классов:")
        unique, counts = torch.unique(self.labels, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"    - Класс {cls.item()}: {count} примеров ({count/len(self)*100:.1f}%)")

# Создаем наш первый Dataset
dataset = SimpleDataset(X[:1000], y[:1000])
dataset.get_stats()

# Проверяем работу
print("\n🔍 Проверка доступа к данным:")
print(f"Длина dataset: {len(dataset)}")

# Получаем несколько примеров
for i in range(3):
    sample = dataset[i]
    print(f"Пример {i}: features.shape = {sample['features'].shape}, label = {sample['label']}")
```

---

🔄 Попробуйте сами #2: Создайте свой Dataset

```python
# СОЗДАЙТЕ И НАСТРОЙТЕ СОБСТВЕННЫЙ DATASET:

class AdvancedDataset(Dataset):
    """Расширенный Dataset с дополнительными функциями"""
    
    def __init__(self, features, labels, normalize=True, add_noise=False):
        """
        Args:
            features: матрица признаков
            labels: вектор меток
            normalize: нормализовать ли данные
            add_noise: добавлять ли случайный шум для аугментации
        """
        self.features = torch.FloatTensor(features).clone()
        self.labels = torch.LongTensor(labels).clone()
        self.normalize = normalize
        self.add_noise = add_noise
        
        # Вычисляем статистики для нормализации
        if self.normalize:
            self.feature_mean = self.features.mean(dim=0)
            self.feature_std = self.features.std(dim=0) + 1e-8  # Добавляем маленькое число для стабильности
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx].clone()
        label = self.labels[idx].clone()
        
        # Нормализация
        if self.normalize:
            features = (features - self.feature_mean) / self.feature_std
        
        # Добавление шума (только во время обучения)
        if self.add_noise and torch.rand(1) > 0.5:
            noise = torch.randn_like(features) * 0.1  # 10% шума
            features = features + noise
        
        return features, label
    
    def split(self, train_ratio=0.8):
        """Разделяет dataset на тренировочную и валидационную части"""
        indices = torch.randperm(len(self))
        split_idx = int(len(self) * train_ratio)
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Создаем новые датасеты
        train_dataset = AdvancedDataset(
            self.features[train_indices],
            self.labels[train_indices],
            normalize=False,  # Не нормализуем повторно
            add_noise=self.add_noise
        )
        
        val_dataset = AdvancedDataset(
            self.features[val_indices],
            self.labels[val_indices],
            normalize=False,
            add_noise=False  # На валидации шум не добавляем
        )
        
        return train_dataset, val_dataset

# ИЗМЕНИТЕ ЭТИ ПАРАМЕТРЫ:
NORMALIZE_DATA = True      # Нормализовать ли данные?
ADD_NOISE = True           # Добавлять ли шум для аугментации?
TRAIN_RATIO = 0.8          # Доля данных для обучения

# Создаем расширенный dataset с вашими параметрами
print("=" * 60)
print("СОЗДАНИЕ ADVANCEDDATASET")
print("=" * 60)
print(f"Параметры:")
print(f"  • Нормализация: {NORMALIZE_DATA}")
print(f"  • Добавление шума: {ADD_NOISE}")
print(f"  • Доля тренировочных данных: {TRAIN_RATIO:.0%}")

# Используем все данные
full_dataset = AdvancedDataset(
    features=X,
    labels=y,
    normalize=NORMALIZE_DATA,
    add_noise=ADD_NOISE
)

print(f"\nПолный dataset:")
print(f"  • Примеров: {len(full_dataset):,}")
print(f"  • Размер features: {full_dataset.features.shape}")

# Разделяем на train/val
train_dataset, val_dataset = full_dataset.split(train_ratio=TRAIN_RATIO)

print(f"\nПосле разделения:")
print(f"  • Тренировочных примеров: {len(train_dataset):,}")
print(f"  • Валидационных примеров: {len(val_dataset):,}")
print(f"  • Соотношение: {len(train_dataset)/len(full_dataset):.1%} / {len(val_dataset)/len(full_dataset):.1%}")

# Проверяем данные
print("\n🔍 Проверка данных:")
print("Тренировочный dataset (первые 3 примера):")
for i in range(3):
    features, label = train_dataset[i]
    print(f"  Пример {i}: label={label}, features[0:3]={features[:3].tolist()}")

print("\nВалидационный dataset (первые 3 примера):")
for i in range(3):
    features, label = val_dataset[i]
    print(f"  Пример {i}: label={label}, features[0:3]={features[:3].tolist()}")

# Проверяем нормализацию
if NORMALIZE_DATA:
    print("\n📐 Проверка нормализации:")
    train_features = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    print(f"  Среднее значение features (должно быть ~0): {train_features.mean():.6f}")
    print(f"  Стандартное отклонение (должно быть ~1): {train_features.std():.6f}")

# Вопросы для анализа:
# 1. Что происходит, когда normalize=False?
# 2. Как добавление шума влияет на обучение?
# 3. Почему важно не добавлять шум на валидационных данных?
# 4. Какое оптимальное соотношение train/val для вашей задачи?
```

---

🔄 DataLoader: Автоматизируем создание батчей

Что такое DataLoader и зачем он нужен?

DataLoader — это мощный инструмент PyTorch, который:

1. Автоматически создает батчи из Dataset
2. Перемешивает данные
3. Параллельно загружает данные (ускорение)
4. Обрабатывает неполные батчи

```python
from torch.utils.data import DataLoader

# Создаем простой DataLoader
simple_loader = DataLoader(
    dataset=train_dataset,      # Наш Dataset
    batch_size=32,              # Размер батча
    shuffle=True,               # Перемешивать ли данные каждый эпоху
    num_workers=0,              # Число процессов для загрузки (0 для отладки)
    drop_last=False            # Отбрасывать ли неполный последний батч
)

print("Создан DataLoader с параметрами:")
print(f"  • batch_size = {simple_loader.batch_size}")
print(f"  • shuffle = {simple_loader.shuffle}")
print(f"  • num_workers = {simple_loader.num_workers}")
print(f"  • drop_last = {simple_loader.drop_last}")
print(f"  • Всего батчей: {len(simple_loader)}")
print(f"  • Примеров в последнем батче: {len(train_dataset) % simple_loader.batch_size or simple_loader.batch_size}")

# Посмотрим, как работает DataLoader
print("\n🔍 Первые 3 батча из DataLoader:")
for batch_idx, (batch_features, batch_labels) in enumerate(simple_loader):
    if batch_idx >= 3:
        break
    
    print(f"\nБатч #{batch_idx + 1}:")
    print(f"  Размер features: {batch_features.shape}")
    print(f"  Размер labels: {batch_labels.shape}")
    print(f"  Диапазон меток в батче: {torch.unique(batch_labels).tolist()}")
    
    # Проверяем перемешивание
    if batch_idx == 0:
        print(f"  Первые 5 меток: {batch_labels[:5].tolist()}")
        if simple_loader.shuffle:
            print("  ✓ Данные перемешаны (метки вразнобой)")
```

---

🔄 Попробуйте сами #3: Эксперименты с DataLoader

```python
# ЭКСПЕРИМЕНТИРУЙТЕ С РАЗНЫМИ ПАРАМЕТРАМИ DATALOADER:

# НАСТРОЙТЕ ЭТИ ПАРАМЕТРЫ:
BATCH_SIZE = 64          # Размер батча (попробуйте: 16, 32, 64, 128, 256)
SHUFFLE = True           # Перемешивать данные? (True/False)
NUM_WORKERS = 0          # Число процессов (0, 2, 4) - осторожно с большими значениями!
DROP_LAST = False        # Отбрасывать неполный батч? (True/False)

print("=" * 60)
print("ЭКСПЕРИМЕНТЫ С DATALOADER")
print("=" * 60)
print(f"Параметры:")
print(f"  • batch_size = {BATCH_SIZE}")
print(f"  • shuffle = {SHUFFLE}")
print(f"  • num_workers = {NUM_WORKERS}")
print(f"  • drop_last = {DROP_LAST}")

# Создаем DataLoader с вашими параметрами
experiment_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS,
    drop_last=DROP_LAST
)

# Анализируем DataLoader
print(f"\n📊 Анализ DataLoader:")
print(f"  • Всего батчей: {len(experiment_loader)}")

# Проверяем размеры батчей
batch_sizes = []
for batch_features, _ in experiment_loader:
    batch_sizes.append(len(batch_features))

print(f"  • Размеры батчей: {set(batch_sizes)}")

if len(set(batch_sizes)) > 1:
    print(f"  ⚠  Не все батчи одинакового размера!")
    if DROP_LAST:
        print("     (Но вы выбрали drop_last=True, так что последний батч отброшен)")

# Измеряем скорость загрузки
print("\n⏱️  Измерение скорости загрузки:")

# Вариант 1: Без DataLoader (вручную)
start_time = time.time()
manual_batches = 0
indices = torch.randperm(len(train_dataset)) if SHUFFLE else torch.arange(len(train_dataset))

for i in range(0, len(train_dataset), BATCH_SIZE):
    batch_indices = indices[i:i+BATCH_SIZE]
    batch_features = torch.stack([train_dataset[idx][0] for idx in batch_indices])
    batch_labels = torch.stack([train_dataset[idx][1] for idx in batch_indices])
    manual_batches += 1
    
manual_time = time.time() - start_time

# Вариант 2: С DataLoader
start_time = time.time()
loader_batches = 0
for batch_features, batch_labels in experiment_loader:
    loader_batches += 1
    
loader_time = time.time() - start_time

print(f"  • Вручную: {manual_time:.3f} секунд ({manual_batches} батчей)")
print(f"  • DataLoader: {loader_time:.3f} секунд ({loader_batches} батчей)")
print(f"  • Ускорение: {manual_time/loader_time:.1f}x")

# Обучение с использованием DataLoader
def train_with_dataloader(model, train_loader, val_dataset, epochs=3):
    """Обучение модели с использованием DataLoader"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    
    print(f"\n🎯 Обучение с DataLoader (batch_size={BATCH_SIZE}):")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        # DataLoader автоматически создает батчи
        for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # Показываем прогресс
            if (batch_idx + 1) % max(1, len(train_loader) // 4) == 0:
                print(f"    Батч {batch_idx + 1}/{len(train_loader)}, "
                      f"ошибка: {loss.item():.4f}")
        
        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        
        # Оценка на валидации
        model.eval()
        with torch.no_grad():
            # Для валидации можно использовать весь dataset или создавать DataLoader
            val_features = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
            val_labels = torch.stack([val_dataset[i][1] for i in range(len(val_dataset))])
            
            val_predictions = model(val_features)
            val_accuracy = (val_predictions.argmax(dim=1) == val_labels).float().mean().item()
        
        print(f"  Эпоха {epoch + 1}: train_loss={avg_loss:.4f}, val_acc={val_accuracy:.2%}")
    
    return train_losses, val_accuracy

# Создаем и обучаем модель
print("\n" + "=" * 60)
print("ОБУЧЕНИЕ С ИСПОЛЬЗОВАНИЕМ DATALOADER")
print("=" * 60)

model = SimpleClassifier()
train_losses, final_val_acc = train_with_dataloader(
    model=model,
    train_loader=experiment_loader,
    val_dataset=val_dataset,
    epochs=3
)

# Визуализируем процесс обучения
plt.figure(figsize=(8, 4))
plt.plot(train_losses, 'b-', linewidth=2, marker='o')
plt.xlabel('Эпоха')
plt.ylabel('Средняя ошибка')
plt.title(f'Обучение с DataLoader (batch_size={BATCH_SIZE})')
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n✅ Финальная точность на валидации: {final_val_acc:.2%}")

# Вопросы для анализа:
# 1. Что происходит при очень маленьком batch_size (например, 4)?
# 2. Как shuffle влияет на обучение?
# 3. Что делает num_workers и когда его увеличивать?
# 4. Когда использовать drop_last=True?
# 5. Почему валидационные данные обычно не перемешивают?
```

---

🩺 Диагностика проблем с данными

Чеклист для отладки Dataset и DataLoader

```python
def diagnose_data_issues(dataset, dataloader):
    """Диагностика распространенных проблем с данными"""
    
    print("=" * 60)
    print("ДИАГНОСТИКА DATASET И DATALOADER")
    print("=" * 60)
    
    # 1. Проверка Dataset
    print("\n1. Проверка Dataset:")
    print(f"   • Размер: {len(dataset)} примеров")
    
    # Проверяем несколько примеров
    print("   • Проверка первых 3 примеров:")
    for i in range(min(3, len(dataset))):
        try:
            features, label = dataset[i]
            print(f"     Пример {i}: features.shape={features.shape}, label={label}, "
                  f"features.dtype={features.dtype}, label.dtype={label.dtype}")
            
            # Проверяем на NaN/Inf
            if torch.isnan(features).any():
                print(f"     ⚠  В примере {i} есть NaN в features!")
            if torch.isinf(features).any():
                print(f"     ⚠  В примере {i} есть Inf в features!")
                
        except Exception as e:
            print(f"     ⚠  Ошибка при загрузке примера {i}: {e}")
    
    # 2. Проверка DataLoader
    print("\n2. Проверка DataLoader:")
    print(f"   • Батчей: {len(dataloader)}")
    print(f"   • Batch size: {dataloader.batch_size}")
    
    # Проверяем несколько батчей
    print("   • Проверка первых 2 батчей:")
    for batch_idx, (features, labels) in enumerate(dataloader):
        if batch_idx >= 2:
            break
        
        print(f"     Батч {batch_idx}:")
        print(f"       • features.shape: {features.shape}")
        print(f"       • labels.shape: {labels.shape}")
        print(f"       • Уникальные метки: {torch.unique(labels).tolist()}")
        
        # Проверяем распределение меток в батче
        unique, counts = torch.unique(labels, return_counts=True)
        print(f"       • Распределение меток: ", end="")
        for cls, count in zip(unique, counts):
            print(f"класс {cls.item()}: {count} ", end="")
        print()
        
        # Проверяем значения features
        print(f"       • features: min={features.min():.3f}, max={features.max():.3f}, "
              f"mean={features.mean():.3f}, std={features.std():.3f}")
    
    # 3. Проверка скорости загрузки
    print("\n3. Проверка скорости загрузки:")
    
    import time
    start_time = time.time()
    num_batches = 0
    total_samples = 0
    
    for batch_idx, (features, labels) in enumerate(dataloader):
        num_batches += 1
        total_samples += len(features)
        
        if batch_idx >= 10:  # Проверяем только первые 10 батчей
            break
    
    load_time = time.time() - start_time
    
    print(f"   • Загружено {num_batches} батчей ({total_samples} примеров)")
    print(f"   • Время: {load_time:.3f} секунд")
    print(f"   • Скорость: {total_samples/load_time:.1f} примеров/сек")
    
    if load_time > 1.0 and dataloader.num_workers == 0:
        print("   ⚠  Загрузка медленная, рассмотрите увеличение num_workers")
    
    # 4. Проверка перемешивания
    if dataloader.shuffle:
        print("\n4. Проверка перемешивания:")
        
        # Собираем метки из первых двух батчей
        first_batch_labels = []
        second_batch_labels = []
        
        for batch_idx, (_, labels) in enumerate(dataloader):
            if batch_idx == 0:
                first_batch_labels = labels.tolist()
            elif batch_idx == 1:
                second_batch_labels = labels.tolist()
                break
        
        # Проверяем, отличаются ли батчи
        if first_batch_labels and second_batch_labels:
            if set(first_batch_labels) != set(second_batch_labels):
                print("   ✅ Батчи содержат разные метки (перемешивание работает)")
            else:
                print("   ⚠  Батчи содержат одинаковые метки (проверьте shuffle)")
    
    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ:")
    print("=" * 60)
    
    if len(dataset) < 1000:
        print("• Маленький dataset: используйте batch_size 16-64")
    elif len(dataset) < 10000:
        print("• Средний dataset: используйте batch_size 32-128")
    else:
        print("• Большой dataset: используйте batch_size 64-256")
    
    if dataloader.num_workers == 0:
        print("• num_workers=0: нормально для отладки, увеличьте для обучения")
    
    if not dataloader.shuffle:
        print("• shuffle=False: нормально для валидации/теста, но для обучения лучше True")

# Запускаем диагностику
diagnose_data_issues(train_dataset, experiment_loader)
```

---

📊 Практическое руководство: выбор параметров

Как выбрать размер батча?

```python
def find_optimal_batch_size(dataset, model_class, max_batch_size=256):
    """Поиск оптимального размера батча"""
    
    print("=" * 60)
    print("ПОИСК ОПТИМАЛЬНОГО РАЗМЕРА БАТЧА")
    print("=" * 60)
    
    batch_sizes = [16, 32, 64, 128, 256]
    results = []
    
    for batch_size in batch_sizes:
        if batch_size > len(dataset):
            print(f"Пропускаем batch_size={batch_size} (больше размера dataset)")
            continue
        
        print(f"\nТестируем batch_size={batch_size}")
        
        # Создаем DataLoader
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Создаем новую модель
        model = model_class()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Измеряем время обучения на одной эпохе
        start_time = time.time()
        model.train()
        
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
        
        epoch_time = time.time() - start_time
        
        # Измеряем использование памяти
        import psutil
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # в МБ
        
        results.append({
            'batch_size': batch_size,
            'time_per_epoch': epoch_time,
            'memory_usage': memory_usage,
            'batches_per_epoch': len(loader)
        })
        
        print(f"  • Время на эпоху: {epoch_time:.2f} сек")
        print(f"  • Использование памяти: {memory_usage:.1f} МБ")
        print(f"  • Батчей в эпохе: {len(loader)}")
    
    # Визуализируем результаты
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    batch_sizes_plot = [r['batch_size'] for r in results]
    times = [r['time_per_epoch'] for r in results]
    memory = [r['memory_usage'] for r in results]
    
    axes[0].plot(batch_sizes_plot, times, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Размер батча')
    axes[0].set_ylabel('Время на эпоху (сек)')
    axes[0].set_title('Зависимость времени от размера батча')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(batch_sizes_plot, memory, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Размер батча')
    axes[1].set_ylabel('Использование памяти (МБ)')
    axes[1].set_title('Зависимость памяти от размера батча')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Рекомендация
    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИЯ ПО ВЫБОРУ BATCH_SIZE:")
    print("=" * 60)
    
    # Находим оптимальный batch_size (баланс времени и памяти)
    normalized_times = [t/max(times) for t in times]
    normalized_memory = [m/max(memory) for m in memory]
    
    # Суммарная "стоимость" (меньше = лучше)
    costs = [t + m for t, m in zip(normalized_times, normalized_memory)]
    best_idx = costs.index(min(costs))
    best_batch = results[best_idx]['batch_size']
    
    print(f"Оптимальный batch_size для этого dataset: {best_batch}")
    print(f"\nОбъяснение:")
    print(f"• batch_size={best_batch} дает хороший баланс между:")
    print(f"  - Скоростью: {results[best_idx]['time_per_epoch']:.2f} сек/эпоху")
    print(f"  - Памятью: {results[best_idx]['memory_usage']:.1f} МБ")
    print(f"  - Количеством батчей: {results[best_idx]['batches_per_epoch']}")

# Запускаем поиск оптимального batch_size (на подмножестве данных для скорости)
small_dataset = SimpleDataset(X[:1000], y[:1000])
find_optimal_batch_size(small_dataset, SimpleClassifier)
```

---

🎓 Ключевые выводы

Про батчи:

1. Мини-батчи — золотая середина между скоростью и стабильностью
2. Размер батча влияет на:
   · Скорость обучения
   · Использование памяти
   · Качество градиентов
3. Правило: batch_size обычно выбирают степенью двойки (32, 64, 128...)

Про Dataset:

1. Dataset — это интерфейс к вашим данным
2. Ключевые методы: __len__() и __getitem__()
3. Преимущества: ленивая загрузка, преобразования данных

Про DataLoader:

1. Автоматизирует создание батчей
2. Параллелизация: num_workers ускоряет загрузку
3. Перемешивание: shuffle=True для обучения, False для валидации/теста
4. Обработка краев: drop_last для неполных батчей

Практические рекомендации:

1. Начинайте с: batch_size=32, shuffle=True, num_workers=0 (для отладки)
2. Для больших данных: увеличивайте num_workers (2-4)
3. Всегда проверяйте: корректность загрузки данных перед обучением
4. Используйте разные: стратегии для train/val/test данных

Типичные значения:

```python
# Для обучения
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,      # или 64 для больших данных
    shuffle=True,       # важно перемешивать!
    num_workers=2,      # ускоряет загрузку
    drop_last=False     # обычно False
)

# Для валидации/теста
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=32,
    shuffle=False,      # не перемешиваем!
    num_workers=0,      # для простоты
    drop_last=False
)
```

---

🚀 Что дальше?

Теперь вы умеете:

1. Эффективно организовывать данные с помощью Dataset
2. Автоматизировать создание батчей с помощью DataLoader
3. Выбирать оптимальные параметры для вашей задачи
4. Диагностировать проблемы с загрузкой данных

Следующие шаги:

1. Попробуйте применить эти знания к вашим данным
2. Экспериментируйте с разными размерами батчей
3. Изучите более сложные Dataset для изображений и текста
4. Оптимизируйте загрузку данных для ускорения обучения

Помните: Правильная организация данных — это половина успеха в машинном обучении. Хороший DataLoader может ускорить обучение в несколько раз!

---

Для дальнейшего изучения:

· Официальная документация PyTorch: Data Loading
· Примеры Dataset для разных типов данных
· Оптимизация загрузки данных для больших датасетов
