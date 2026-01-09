### **Задачи: Early Stopping и Callbacks**

**Цель:** Научиться использовать callbacks для контроля процесса обучения, автоматического сохранения моделей и предотвращения переобучения.

---

## 🟢 Базовый уровень

### **Задача 1: Простой Early Stopping**

**Условие:** Реализуйте базовый Early Stopping для предотвращения переобучения.

**Требования:**
1. Создайте класс `EarlyStopping`:
   ```python
   class EarlyStopping:
       def __init__(self, patience=5, min_delta=0):
           """
           patience: сколько эпох ждать улучшения
           min_delta: минимальное улучшение метрики
           """
           pass
       
       def __call__(self, val_loss):
           # Проверяет, нужно ли остановить обучение
           pass
   ```
2. Обучите модель на маленьком MNIST (5000 примеров) с patience=5
3. Визуализируйте train/val loss и отметьте точку останова
4. Сравните с обучением без Early Stopping (все 100 эпох)

**Ожидаемый результат:** Early Stopping останавливает обучение до переобучения.

```python
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

for epoch in range(100):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)
    
    print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
    
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping на эпохе {epoch}")
        break
```

---

### **Задача 2: Early Stopping с сохранением лучшей модели**

**Условие:** Расширьте Early Stopping для автоматического сохранения лучшей модели.

**Требования:**
1. Модифицируйте класс `EarlyStopping`:
   - Сохраняйте модель при улучшении метрики
   - Возвращайте путь к лучшей модели
2. После обучения загрузите лучшую модель
3. Сравните test accuracy:
   - Финальной модели
   - Лучшей сохраненной модели

**Ожидаемый результат:** Лучшая модель имеет выше test accuracy.

```python
class EarlyStoppingWithCheckpoint:
    def __init__(self, patience=5, path='best_model.pt', verbose=True):
        # TODO: реализуйте
        pass
    
    def __call__(self, val_loss, model):
        # Сохраняет модель при улучшении
        pass
    
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# Использование
early_stopping = EarlyStoppingWithCheckpoint(patience=5)

for epoch in range(100):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)
    
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        break

# Загрузка лучшей модели
model.load_state_dict(torch.load('best_model.pt'))
```

---

### **Задача 3: Визуализация эффекта Early Stopping**

**Условие:** Продемонстрируйте, как Early Stopping предотвращает переобучение.

**Требования:**
1. Создайте искусственный сценарий переобучения:
   - Маленький датасет (1000 примеров)
   - Большая модель (переобучается легко)
2. Обучите две модели:
   - С Early Stopping (patience=5)
   - Без Early Stopping (100 эпох)
3. Постройте графики:
   - Train/Val loss для обеих моделей
   - Train/Val accuracy для обеих моделей
4. Отметьте точку Early Stop на графиках

**Ожидаемый результат:** Без Early Stop видно переобучение (val loss растет).

---

## 🟡 Продвинутый уровень

### **Задача 4: Система Callbacks**

**Условие:** Создайте гибкую систему callbacks для управления обучением.

**Требования:**
1. Реализуйте базовый класс `Callback` с методами:
   - `on_epoch_begin(epoch, logs)`
   - `on_epoch_end(epoch, logs)`
   - `on_batch_begin(batch, logs)`
   - `on_batch_end(batch, logs)`
   - `on_train_begin(logs)`
   - `on_train_end(logs)`
2. Реализуйте `CallbackList` для управления несколькими callbacks
3. Интегрируйте в training loop

```python
class Callback:
    def on_epoch_begin(self, epoch, logs=None):
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        pass
    
    # TODO: остальные методы

class CallbackList:
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []
    
    def on_epoch_begin(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    # TODO: остальные методы

# Training loop с callbacks
callbacks = CallbackList([callback1, callback2, callback3])

callbacks.on_train_begin()
for epoch in range(epochs):
    callbacks.on_epoch_begin(epoch)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        callbacks.on_batch_begin(batch_idx)
        # ... training step ...
        callbacks.on_batch_end(batch_idx, logs={'loss': loss.item()})
    
    callbacks.on_epoch_end(epoch, logs={'train_loss': train_loss, 'val_loss': val_loss})

callbacks.on_train_end()
```

---

### **Задача 5: ModelCheckpoint Callback**

**Условие:** Создайте callback для гибкого сохранения моделей.

**Требования:**
1. Реализуйте `ModelCheckpoint` callback:
   - Параметры: filepath, monitor='val_loss', mode='min', save_best_only=True
   - Поддержка форматов имени: 'model_epoch{epoch:02d}_loss{val_loss:.4f}.pt'
   - Автоматическое удаление старых чекпоинтов
2. Опции:
   - `save_best_only=True`: сохранять только лучшую модель
   - `save_best_only=False`: сохранять каждые N эпох
3. Протестируйте на CIFAR-10

```python
class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', mode='min',
                 save_best_only=True, save_freq=1, verbose=1):
        # TODO: реализуйте
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        
        if self.save_best_only:
            if self._is_improvement(current):
                self._save_model(epoch, current)
        else:
            if epoch % self.save_freq == 0:
                self._save_model(epoch, current)

# Использование
checkpoint = ModelCheckpoint(
    filepath='checkpoints/model_epoch{epoch:02d}_loss{val_loss:.4f}.pt',
    monitor='val_loss',
    save_best_only=True
)
```

---

### **Задача 6: LearningRateScheduler Callback**

**Условие:** Интегрируйте LR scheduling в систему callbacks.

**Требования:**
1. Создайте `LRSchedulerCallback`:
   - Работает с любым PyTorch scheduler
   - Автоматически вызывает scheduler.step()
   - Логирует текущий LR
2. Поддержите schedulers, требующие метрику (ReduceLROnPlateau)
3. Комбинируйте с ModelCheckpoint и EarlyStopping

```python
class LRSchedulerCallback(Callback):
    def __init__(self, scheduler, monitor=None):
        """
        scheduler: PyTorch scheduler
        monitor: метрика для ReduceLROnPlateau
        """
        self.scheduler = scheduler
        self.monitor = monitor
    
    def on_epoch_end(self, epoch, logs=None):
        if self.monitor is not None:
            # ReduceLROnPlateau требует метрику
            metric = logs.get(self.monitor)
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
        
        # Логируем текущий LR
        current_lr = self.scheduler.optimizer.param_groups[0]['lr']
        logs['learning_rate'] = current_lr

# Использование
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)
lr_callback = LRSchedulerCallback(scheduler, monitor='val_loss')

callbacks = CallbackList([lr_callback, checkpoint, early_stopping])
```

---

## 🔴 Экспертный уровень

### **Задача 7: TensorBoard Callback**

**Условие:** Создайте callback для автоматического логирования в TensorBoard.

**Требования:**
1. Реализуйте `TensorBoardCallback`:
   - Логирует loss, accuracy, learning rate
   - Сохраняет гистограммы весов
   - Визуализирует примеры предсказаний
2. Поддержите:
   - Скалярные метрики (loss, accuracy)
   - Распределения (weights, gradients)
   - Изображения (примеры из валидации)
3. Интегрируйте с обучением на CIFAR-10

```python
from torch.utils.tensorboard import SummaryWriter

class TensorBoardCallback(Callback):
    def __init__(self, log_dir='runs', log_freq=1):
        self.writer = SummaryWriter(log_dir)
        self.log_freq = log_freq
    
    def on_epoch_end(self, epoch, logs=None):
        # Логируем скалярные метрики
        for key, value in logs.items():
            self.writer.add_scalar(key, value, epoch)
        
        # Логируем гистограммы весов
        if epoch % self.log_freq == 0:
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name, param, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'{name}.grad', param.grad, epoch)
    
    def on_train_end(self, logs=None):
        self.writer.close()

# TODO: добавьте визуализацию примеров предсказаний
```

---

### **Задача 8: GradientMonitor Callback**

**Условие:** Создайте callback для мониторинга градиентов и диагностики проблем обучения.

**Требования:**
1. Реализуйте `GradientMonitor`:
   - Вычисляет нормы градиентов для каждого слоя
   - Детектирует vanishing/exploding gradients
   - Логирует статистику градиентов
2. Предупреждает, если:
   - ||grad|| < 1e-7 (vanishing)
   - ||grad|| > 100 (exploding)
   - grad is NaN/Inf
3. Визуализирует нормы градиентов по слоям

```python
class GradientMonitor(Callback):
    def __init__(self, log_freq=10, warn_threshold=100):
        self.log_freq = log_freq
        self.warn_threshold = warn_threshold
    
    def on_batch_end(self, batch, logs=None):
        if batch % self.log_freq != 0:
            return
        
        grad_norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm
                
                # Проверяем на проблемы
                if grad_norm < 1e-7:
                    print(f"⚠️ WARNING: Vanishing gradient in {name}: {grad_norm}")
                elif grad_norm > self.warn_threshold:
                    print(f"⚠️ WARNING: Exploding gradient in {name}: {grad_norm}")
                elif not np.isfinite(grad_norm):
                    print(f"⚠️ WARNING: NaN/Inf gradient in {name}")
        
        logs['grad_norms'] = grad_norms

# Визуализация
def plot_gradient_flow(grad_norms):
    """Визуализирует нормы градиентов по слоям"""
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(grad_norms)), list(grad_norms.values()))
    plt.xticks(range(len(grad_norms)), list(grad_norms.keys()), rotation=45, ha='right')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Flow')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
```

---

### **Задача 9: ProgressBar Callback**

**Условие:** Создайте красивый прогресс-бар для мониторинга обучения.

**Требования:**
1. Используйте `tqdm` для progress bar
2. Отображайте:
   - Прогресс эпохи (батчи)
   - Текущий loss
   - ETA (estimated time of arrival)
   - Метрики (accuracy, LR)
3. Поддержите вложенные прогресс-бары (эпохи и батчи)

```python
from tqdm import tqdm

class ProgressBarCallback(Callback):
    def __init__(self):
        self.epoch_bar = None
        self.batch_bar = None
    
    def on_train_begin(self, logs=None):
        self.epoch_bar = tqdm(total=logs.get('epochs'), desc='Training')
    
    def on_epoch_begin(self, epoch, logs=None):
        total_batches = logs.get('total_batches')
        self.batch_bar = tqdm(total=total_batches, 
                             desc=f'Epoch {epoch+1}',
                             leave=False)
    
    def on_batch_end(self, batch, logs=None):
        self.batch_bar.set_postfix({
            'loss': f"{logs.get('loss', 0):.4f}",
            'lr': f"{logs.get('lr', 0):.6f}"
        })
        self.batch_bar.update(1)
    
    def on_epoch_end(self, epoch, logs=None):
        self.batch_bar.close()
        self.epoch_bar.set_postfix({
            'train_loss': f"{logs.get('train_loss', 0):.4f}",
            'val_loss': f"{logs.get('val_loss', 0):.4f}",
            'val_acc': f"{logs.get('val_acc', 0):.2f}%"
        })
        self.epoch_bar.update(1)
    
    def on_train_end(self, logs=None):
        self.epoch_bar.close()
```

---

### **Задача 10: Composite Training Pipeline**

**Условие:** Объедините все callbacks в полноценный training pipeline.

**Требования:**
1. Создайте `Trainer` класс, объединяющий:
   - Model, Optimizer, Criterion
   - CallbackList
   - Training/Validation loops
2. Используйте callbacks:
   - EarlyStopping
   - ModelCheckpoint
   - LRScheduler
   - TensorBoard
   - GradientMonitor
   - ProgressBar
3. Добавьте методы:
   - `fit(train_loader, val_loader, epochs)`
   - `evaluate(test_loader)`
   - `predict(data)`

```python
class Trainer:
    def __init__(self, model, optimizer, criterion, callbacks=None, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = CallbackList(callbacks or [])
        self.device = device
    
    def fit(self, train_loader, val_loader, epochs):
        """Обучение с callbacks"""
        self.callbacks.on_train_begin({'epochs': epochs})
        
        for epoch in range(epochs):
            logs = {'epoch': epoch, 'total_batches': len(train_loader)}
            self.callbacks.on_epoch_begin(epoch, logs)
            
            # Training
            train_loss = self._train_epoch(train_loader, logs)
            
            # Validation
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            logs.update({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            self.callbacks.on_epoch_end(epoch, logs)
        
        self.callbacks.on_train_end()
    
    def _train_epoch(self, train_loader, epoch_logs):
        """Один эпоха обучения"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            batch_logs = {'batch': batch_idx}
            self.callbacks.on_batch_begin(batch_idx, batch_logs)
            
            # Training step
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_logs['loss'] = loss.item()
            batch_logs['lr'] = self.optimizer.param_groups[0]['lr']
            
            self.callbacks.on_batch_end(batch_idx, batch_logs)
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def _validate_epoch(self, val_loader):
        """Валидация"""
        self.model.eval()
        total_loss = 0
        correct = 0
        
        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
        
        val_loss = total_loss / len(val_loader)
        val_acc = 100. * correct / len(val_loader.dataset)
        
        return val_loss, val_acc

# Использование
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = SimpleCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

callbacks = [
    EarlyStopping(patience=10, path='best_model.pt'),
    ModelCheckpoint('checkpoints/model_{epoch:02d}.pt', save_best_only=True),
    LRSchedulerCallback(scheduler, monitor='val_loss'),
    TensorBoardCallback(log_dir='runs/experiment1'),
    GradientMonitor(log_freq=50),
    ProgressBarCallback()
]

trainer = Trainer(model, optimizer, criterion, callbacks=callbacks, device='cuda')
trainer.fit(train_loader, val_loader, epochs=50)
```

---

## 💎 Заключение

### **Преимущества callbacks:**

✅ **Модульность:**
- Каждый callback отвечает за одну функцию
- Легко добавлять/удалять callbacks
- Переиспользование кода

✅ **Гибкость:**
- Контроль на любом этапе обучения
- Кастомные callbacks для специфических задач
- Комбинирование нескольких callbacks

✅ **Удобство:**
- Автоматизация рутинных задач
- Мониторинг обучения в реальном времени
- Предотвращение ошибок (Early Stopping)

### **Рекомендуемый минимум callbacks:**

```python
# Базовый набор для любого проекта
callbacks = [
    ModelCheckpoint('best_model.pt', monitor='val_loss', save_best_only=True),
    EarlyStopping(patience=10),
    LRSchedulerCallback(scheduler, monitor='val_loss'),
    ProgressBarCallback()
]

# Для экспериментов добавить
callbacks.extend([
    TensorBoardCallback(log_dir='runs/exp1'),
    GradientMonitor(log_freq=100)
])
```

### **Типичные ошибки:**

❌ **Не делайте так:**
- Забывать передавать model в callbacks
- Не обновлять logs dictionary
- Вызывать callbacks в неправильном порядке
- Создавать слишком сложные callbacks (разбивайте!)

✅ **Делайте так:**
- Используйте logs для передачи данных между callbacks
- Делайте callbacks независимыми друг от друга
- Логируйте всё важное (loss, metrics, LR)
- Тестируйте callbacks отдельно

### **Дополнительные ресурсы:**

1. **Библиотеки:**
   - PyTorch Lightning — встроенная система callbacks
   - Keras — вдохновение для callback API
   - Catalyst — продвинутые callbacks для PyTorch

2. **Практика:**
   - Начните с простых callbacks (EarlyStopping, Checkpoint)
   - Постепенно добавляйте мониторинг
   - Создавайте кастомные callbacks для своих задач

> **"Callbacks превращают training loop из монолитного кода в модульную систему. Это must-have для любого серьезного проекта!"**
