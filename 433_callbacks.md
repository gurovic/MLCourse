# Early Stopping и Callbacks

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# !pip install torch matplotlib
```

---

## 🟢 Базовый уровень: Early Stopping

### 1.1 Что такое Early Stopping?

**Early Stopping** — остановка обучения, когда validation метрика перестает улучшаться.

**Зачем?**
- Предотвращает переобучение
- Экономит время обучения
- Автоматически находит оптимальное количество эпох

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=True):
        """
        patience: сколько эпох ждать улучшения
        min_delta: минимальное улучшение метрики
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# Использование
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

for epoch in range(100):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)
    
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    # Проверяем early stopping
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping на эпохе {epoch}")
        break
```

### 1.2 Early Stopping с сохранением лучшей модели

```python
class EarlyStoppingWithCheckpoint:
    def __init__(self, patience=5, path='best_model.pt', verbose=True):
        self.patience = patience
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: {self.counter}/{self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
    
    def save_checkpoint(self, model):
        if self.verbose:
            print(f'Saving model to {self.path}')
        torch.save(model.state_dict(), self.path)

# Использование
early_stopping = EarlyStoppingWithCheckpoint(patience=5, path='best_model.pt')

for epoch in range(100):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)
    
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

# Загрузка лучшей модели
model.load_state_dict(torch.load('best_model.pt'))
```

### 1.3 Визуализация эффекта Early Stopping

```python
def demonstrate_early_stopping():
    # Симуляция обучения с переобучением
    np.random.seed(42)
    epochs = 50
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Имитация: train loss падает, val loss сначала падает, потом растет
        train_loss = 1.0 * np.exp(-epoch / 10) + 0.05 * np.random.randn()
        val_loss = 1.0 * np.exp(-epoch / 10) + 0.1 * np.random.randn()
        
        # После эпохи 20 начинается переобучение
        if epoch > 20:
            val_loss += 0.02 * (epoch - 20)
        
        train_losses.append(max(0, train_loss))
        val_losses.append(max(0, val_loss))
    
    # Early stopping остановился бы на эпохе ~25
    best_epoch = np.argmin(val_losses)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axvline(best_epoch, color='r', linestyle='--', 
                label=f'Early Stop (epoch {best_epoch})')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.title('Early Stopping предотвращает переобучение')
    plt.legend()
    plt.grid(True)
    plt.show()

demonstrate_early_stopping()
```

---

## 🟡 Продвинутый уровень: Callbacks

### 2.1 Базовая система Callbacks

```python
class Callback:
    """Базовый класс для callbacks"""
    def on_epoch_begin(self, epoch, logs=None):
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        pass
    
    def on_batch_begin(self, batch, logs=None):
        pass
    
    def on_batch_end(self, batch, logs=None):
        pass
    
    def on_train_begin(self, logs=None):
        pass
    
    def on_train_end(self, logs=None):
        pass

class CallbackList:
    """Управление списком callbacks"""
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []
    
    def on_epoch_begin(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
    
    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)
```

### 2.2 ModelCheckpoint Callback

```python
class ModelCheckpoint(Callback):
    """Сохраняет модель периодически или при улучшении метрики"""
    def __init__(self, filepath, monitor='val_loss', mode='min', 
                 save_best_only=True, verbose=1):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        if mode == 'min':
            self.best = float('inf')
            self.monitor_op = lambda x, y: x < y
        else:
            self.best = float('-inf')
            self.monitor_op = lambda x, y: x > y
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if not self.save_best_only or self.monitor_op(current, self.best):
            if self.verbose:
                print(f'\nEpoch {epoch}: {self.monitor} improved from '
                      f'{self.best:.4f} to {current:.4f}, saving model to {self.filepath}')
            
            self.best = current
            torch.save(logs['model'].state_dict(), self.filepath)

# Использование
checkpoint = ModelCheckpoint(
    filepath='best_model.pt',
    monitor='val_loss',
    mode='min',
    save_best_only=True
)
```

### 2.3 History Callback — логирование метрик

```python
class History(Callback):
    """Сохраняет историю всех метрик"""
    def __init__(self):
        self.history = {}
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            if k != 'model':  # Не логируем саму модель
                self.history.setdefault(k, []).append(v)
    
    def plot(self):
        """Визуализация истории"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        if 'train_loss' in self.history:
            axes[0].plot(self.history['train_loss'], label='Train Loss')
        if 'val_loss' in self.history:
            axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Эпоха')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        if 'train_acc' in self.history:
            axes[1].plot(self.history['train_acc'], label='Train Acc')
        if 'val_acc' in self.history:
            axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Эпоха')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()

# Использование
history = History()
```

---

## 🔴 Экспертный уровень: Продвинутые Callbacks

### 3.1 ProgressBar Callback

```python
class ProgressBar(Callback):
    """Отображает прогресс обучения"""
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.current_epoch = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        print(f'\nEpoch {epoch+1}/{self.total_epochs}')
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics_str = ' - '.join([f'{k}: {v:.4f}' for k, v in logs.items() 
                                   if k != 'model'])
        print(f'Epoch {epoch+1}/{self.total_epochs} - {metrics_str}')
```

### 3.2 TensorBoard Callback

```python
from torch.utils.tensorboard import SummaryWriter

class TensorBoardCallback(Callback):
    """Логирование в TensorBoard"""
    def __init__(self, log_dir='runs'):
        self.writer = SummaryWriter(log_dir)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        for name, value in logs.items():
            if name != 'model' and isinstance(value, (int, float)):
                self.writer.add_scalar(name, value, epoch)
    
    def on_train_end(self, logs=None):
        self.writer.close()

# Использование
# tensorboard --logdir=runs
tensorboard_callback = TensorBoardCallback(log_dir='runs/experiment_1')
```

### 3.3 LearningRateMonitor Callback

```python
class LearningRateMonitor(Callback):
    """Мониторинг и логирование learning rate"""
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lrs = []
    
    def on_epoch_end(self, epoch, logs=None):
        lr = self.optimizer.param_groups[0]['lr']
        self.lrs.append(lr)
        if logs is not None:
            logs['lr'] = lr
    
    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.lrs)
        plt.xlabel('Эпоха')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate по эпохам')
        plt.grid(True)
        plt.yscale('log')
        plt.show()
```

### 3.4 GradientMonitor — мониторинг градиентов

```python
class GradientMonitor(Callback):
    """Мониторинг нормы градиентов"""
    def __init__(self, model):
        self.model = model
        self.grad_norms = []
    
    def on_batch_end(self, batch, logs=None):
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.grad_norms.append(total_norm)
    
    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.grad_norms)
        plt.xlabel('Шаг обучения')
        plt.ylabel('Норма градиента')
        plt.title('Gradient Norm (следить за взрывающимися градиентами)')
        plt.grid(True)
        plt.show()
```

### 3.5 Полный пример с Callbacks

```python
def train_with_callbacks(model, train_loader, val_loader, optimizer, 
                        criterion, callbacks, epochs=50):
    """Обучение с поддержкой callbacks"""
    callback_list = CallbackList(callbacks)
    
    # Начало обучения
    callback_list.on_train_begin()
    
    for epoch in range(epochs):
        # Начало эпохи
        callback_list.on_epoch_begin(epoch)
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            callback_list.on_batch_begin(batch_idx)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            callback_list.on_batch_end(batch_idx)
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Конец эпохи
        logs = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'model': model
        }
        callback_list.on_epoch_end(epoch, logs)
        
        # Проверка early stopping
        if any(hasattr(cb, 'early_stop') and cb.early_stop for cb in callbacks):
            print(f"Early stopping на эпохе {epoch}")
            break
    
    # Конец обучения
    callback_list.on_train_end()

# Использование всех callbacks вместе
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10)
)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

callbacks = [
    History(),
    ModelCheckpoint('best_model.pt', monitor='val_loss', mode='min'),
    EarlyStoppingWithCheckpoint(patience=5),
    ProgressBar(total_epochs=50),
    LearningRateMonitor(optimizer),
    TensorBoardCallback('runs/experiment'),
]

train_with_callbacks(model, train_loader, val_loader, optimizer, 
                    criterion, callbacks, epochs=50)
```

---

## 💎 Заключение

**Основные Callbacks:**

| Callback | Назначение | Когда использовать |
|----------|-----------|-------------------|
| **EarlyStopping** | Останавливает при переобучении | Всегда |
| **ModelCheckpoint** | Сохраняет лучшую модель | Всегда |
| **History** | Логирует метрики | Для анализа |
| **LearningRateMonitor** | Следит за LR | С LR schedulers |
| **GradientMonitor** | Следит за градиентами | RNN, глубокие сети |
| **TensorBoard** | Визуализация в реальном времени | Длительное обучение |
| **ProgressBar** | Показывает прогресс | Удобство |

**Рекомендуемый набор callbacks:**

```python
# Минимальный набор
callbacks = [
    EarlyStoppingWithCheckpoint(patience=10, path='model.pt'),
    History()
]

# Полный набор для серьезного проекта
callbacks = [
    EarlyStoppingWithCheckpoint(patience=10, path='best_model.pt'),
    History(),
    ModelCheckpoint('checkpoint_{epoch}.pt', save_best_only=False),
    LearningRateMonitor(optimizer),
    GradientMonitor(model),
    TensorBoardCallback('runs/experiment'),
    ProgressBar(total_epochs=100)
]
```

**Лучшие практики:**
- ✅ Всегда используйте EarlyStopping (patience=5-10)
- ✅ Всегда сохраняйте лучшую модель (ModelCheckpoint)
- ✅ Логируйте все метрики (History или TensorBoard)
- ✅ Мониторьте gradients для RNN
- ✅ Делайте checkpoint'ы периодически (на случай сбоя)

**Типичная конфигурация:**

```python
# Для быстрых экспериментов
early_stopping = EarlyStoppingWithCheckpoint(patience=5)

# Для production
callbacks = [
    EarlyStopping(patience=10, min_delta=0.001),
    ModelCheckpoint('best.pt', monitor='val_acc', mode='max'),
    ModelCheckpoint('checkpoint_{epoch}.pt', save_best_only=False),
    TensorBoardCallback('runs/production'),
    History()
]
```

**Частые ошибки:**
- ❌ Не использовать early stopping (тратите время)
- ❌ Забыть сохранить лучшую модель
- ❌ Слишком маленький patience (останавливается слишком рано)
- ❌ Не логировать метрики (сложно анализировать)

> **"Callbacks — это автоматизация best practices. Правильная настройка callbacks экономит время и улучшает результаты."**

**Дальнейшее изучение:**
- [PyTorch Lightning Callbacks](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html)
- [Keras Callbacks](https://keras.io/api/callbacks/)
- [Fast.ai Callbacks](https://docs.fast.ai/callback.core.html)

---

## 📝 Задачи

**[Перейти к задачам по Callbacks →](433_callbacks_tasks.md)**

Практические задания для закрепления материала:
- 🟢 Базовый уровень: Early Stopping, сохранение лучшей модели
- 🟡 Продвинутый уровень: система callbacks, ModelCheckpoint, LRScheduler
- 🔴 Экспертный уровень: TensorBoard, GradientMonitor, полный training pipeline
