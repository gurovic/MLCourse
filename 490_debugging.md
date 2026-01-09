# Отладка нейронных сетей (Debugging Neural Networks)

## 🟢 Основы диагностики

### Типичные проблемы обучения

**1. Модель не обучается (loss не уменьшается)**
- Слишком большой learning rate
- Плохая инициализация весов
- Неправильная функция потерь
- Ошибка в forward pass

**2. Модель переобучается**
- Недостаточно данных
- Модель слишком сложная
- Нет регуляризации

**3. Медленное обучение**
- Слишком маленький learning rate
- Неудачная архитектура
- Плохая нормализация данных

### Checklist для начала отладки

```python
import torch
import torch.nn as nn

def debug_checklist(model, data, target):
    """Базовые проверки перед обучением"""
    
    # 1. Проверка размерностей
    print("=" * 50)
    print("1. Checking dimensions...")
    output = model(data)
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Target shape: {target.shape}")
    assert output.shape[0] == target.shape[0], "Batch size mismatch!"
    
    # 2. Проверка диапазона выходных значений
    print("\n2. Checking output range...")
    print(f"Output min: {output.min().item():.4f}")
    print(f"Output max: {output.max().item():.4f}")
    print(f"Output mean: {output.mean().item():.4f}")
    
    # 3. Проверка loss
    print("\n3. Checking loss...")
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    print(f"Initial loss: {loss.item():.4f}")
    print(f"Expected random loss (log(num_classes)): {np.log(output.shape[1]):.4f}")
    
    # 4. Проверка градиентов
    print("\n4. Checking gradients...")
    loss.backward()
    
    has_grad = []
    no_grad = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                has_grad.append(name)
            else:
                no_grad.append(name)
        else:
            no_grad.append(name)
    
    print(f"Layers with gradients: {len(has_grad)}")
    print(f"Layers without gradients: {len(no_grad)}")
    if no_grad:
        print(f"WARNING: No gradients for: {no_grad}")
    
    # 5. Проверка на NaN/Inf
    print("\n5. Checking for NaN/Inf...")
    has_nan = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
                has_nan = True
            if torch.isinf(param.grad).any():
                print(f"Inf gradient in {name}")
                has_nan = True
    
    if not has_nan:
        print("No NaN/Inf found ✓")
    
    print("=" * 50)

# Пример использования
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 5)
)

data = torch.randn(32, 10)
target = torch.randint(0, 5, (32,))

debug_checklist(model, data, target)
```

## 🟡 Диагностика градиентов

### Vanishing Gradients

**Симптомы**: ранние слои не обучаются, градиенты близки к нулю

```python
def check_gradient_flow(model):
    """Визуализируем поток градиентов через слои"""
    import matplotlib.pyplot as plt
    
    ave_grads = []
    max_grads = []
    layers = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and "bias" not in name and param.grad is not None:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(ave_grads)), ave_grads, alpha=0.7, label="mean")
    plt.bar(range(len(max_grads)), max_grads, alpha=0.7, label="max")
    plt.hlines(0, 0, len(ave_grads), linewidth=2, color="k")
    plt.xticks(range(len(ave_grads)), layers, rotation=45, ha='right')
    plt.xlabel("Layers")
    plt.ylabel("Gradient magnitude")
    plt.legend()
    plt.title("Gradient Flow")
    plt.tight_layout()
    plt.savefig('gradient_flow.png')
    plt.close()
    
    # Проверка на vanishing/exploding
    if max(ave_grads) < 1e-5:
        print("WARNING: Potential vanishing gradients!")
    if max(ave_grads) > 100:
        print("WARNING: Potential exploding gradients!")

# Решение vanishing gradients:
# 1. Используйте ReLU вместо sigmoid/tanh
# 2. Batch Normalization
# 3. Residual connections
# 4. Правильная инициализация (Xavier, He)
```

### Exploding Gradients

**Симптомы**: loss становится NaN, веса растут очень быстро

```python
# Решение: gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Или по значению
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### Dead ReLU Problem

```python
def check_dead_relu(model, data):
    """Проверяем, сколько ReLU нейронов всегда выдают 0"""
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Регистрируем hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    with torch.no_grad():
        model(data)
    
    # Анализируем активации
    for name, act in activations.items():
        dead_neurons = (act == 0).all(dim=0).sum().item()
        total_neurons = act.shape[1] if len(act.shape) > 1 else 1
        dead_ratio = dead_neurons / total_neurons
        
        print(f"{name}: {dead_neurons}/{total_neurons} dead ({dead_ratio*100:.1f}%)")
        
        if dead_ratio > 0.5:
            print(f"  WARNING: More than 50% dead neurons!")
    
    # Удаляем hooks
    for hook in hooks:
        hook.remove()

# Решение dead ReLU:
# 1. Используйте LeakyReLU или ELU
# 2. Уменьшите learning rate
# 3. Проверьте инициализацию
# 4. Batch Normalization
```

## 🔴 Продвинутая диагностика

### Learning Rate Finder

```python
import matplotlib.pyplot as plt

class LRFinder:
    """Находим оптимальный learning rate"""
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        
    def range_test(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100):
        lrs = []
        losses = []
        
        # Сохраняем начальное состояние
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        
        # Логарифмическая сетка для LR
        lr = start_lr
        mult = (end_lr / start_lr) ** (1 / num_iter)
        
        # Обучаем с увеличивающимся LR
        iterator = iter(train_loader)
        for iteration in range(num_iter):
            try:
                data, target = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                data, target = next(iterator)
            
            # Устанавливаем новый LR
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Training step
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # Записываем
            lrs.append(lr)
            losses.append(loss.item())
            
            # Увеличиваем LR
            lr *= mult
            
            # Останавливаемся если loss взрывается
            if loss.item() > 4 * min(losses):
                break
        
        # Восстанавливаем модель
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        
        # Визуализируем
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True)
        plt.savefig('lr_finder.png')
        plt.close()
        
        # Рекомендуем LR
        min_loss_idx = losses.index(min(losses))
        suggested_lr = lrs[max(0, min_loss_idx - 10)]
        print(f"Suggested LR: {suggested_lr:.2e}")
        
        return lrs, losses

# Использование
lr_finder = LRFinder(model, optimizer, criterion)
lrs, losses = lr_finder.range_test(train_loader, start_lr=1e-6, end_lr=1)
```

### Loss Landscape Visualization

```python
import numpy as np

def plot_loss_landscape_2d(model, criterion, data, target, device='cpu'):
    """Визуализируем loss landscape вокруг текущей точки"""
    
    # Сохраняем текущие параметры
    params = [p.clone() for p in model.parameters()]
    
    # Генерируем два случайных направления
    direction1 = [torch.randn_like(p) for p in params]
    direction2 = [torch.randn_like(p) for p in params]
    
    # Нормализуем направления
    norm1 = sum([d.norm() for d in direction1])
    norm2 = sum([d.norm() for d in direction2])
    direction1 = [d / norm1 for d in direction1]
    direction2 = [d / norm2 for d in direction2]
    
    # Создаем сетку
    alphas = np.linspace(-1, 1, 20)
    betas = np.linspace(-1, 1, 20)
    losses = np.zeros((len(alphas), len(betas)))
    
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Двигаемся в направлениях
            for p, p_orig, d1, d2 in zip(model.parameters(), params, direction1, direction2):
                p.data = p_orig + alpha * d1 + beta * d2
            
            # Вычисляем loss
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, target)
                losses[i, j] = loss.item()
    
    # Восстанавливаем параметры
    for p, p_orig in zip(model.parameters(), params):
        p.data = p_orig
    
    # Визуализируем
    plt.figure(figsize=(10, 8))
    plt.contourf(alphas, betas, losses.T, levels=20, cmap='viridis')
    plt.colorbar(label='Loss')
    plt.xlabel('Direction 1')
    plt.ylabel('Direction 2')
    plt.title('Loss Landscape')
    plt.plot(0, 0, 'r*', markersize=20, label='Current point')
    plt.legend()
    plt.savefig('loss_landscape.png')
    plt.close()

# Использование
plot_loss_landscape_2d(model, criterion, data, target)
```

### Activation Statistics

```python
class ActivationStats:
    """Собираем статистику активаций по слоям"""
    def __init__(self, model):
        self.model = model
        self.stats = {}
        self.hooks = []
        
    def register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                if name not in self.stats:
                    self.stats[name] = []
                
                with torch.no_grad():
                    self.stats[name].append({
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item(),
                        'sparsity': (output == 0).float().mean().item()
                    })
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d)):
                self.hooks.append(
                    module.register_forward_hook(hook_fn(name))
                )
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def print_stats(self):
        for name, stats_list in self.stats.items():
            # Усредняем по всем батчам
            avg_stats = {
                key: np.mean([s[key] for s in stats_list])
                for key in stats_list[0].keys()
            }
            
            print(f"\n{name}:")
            print(f"  Mean: {avg_stats['mean']:.4f}")
            print(f"  Std: {avg_stats['std']:.4f}")
            print(f"  Range: [{avg_stats['min']:.4f}, {avg_stats['max']:.4f}]")
            print(f"  Sparsity: {avg_stats['sparsity']*100:.1f}%")

# Использование
stats = ActivationStats(model)
stats.register_hooks()

# Прогоняем несколько батчей
for data, target in train_loader:
    model(data)
    if len(stats.stats[list(stats.stats.keys())[0]]) >= 10:
        break

stats.print_stats()
stats.remove_hooks()
```

### Overfitting Test

```python
def overfitting_test(model, criterion, optimizer):
    """Проверяем, может ли модель переобучиться на маленьком датасете"""
    print("Running overfitting test...")
    
    # Создаем маленький датасет (10 примеров)
    small_data = torch.randn(10, 10)
    small_target = torch.randint(0, 5, (10,))
    
    # Пытаемся переобучиться
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(small_data)
        loss = criterion(output, small_target)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            acc = (output.argmax(1) == small_target).float().mean()
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Acc = {acc.item():.4f}")
    
    # Финальная проверка
    final_acc = (output.argmax(1) == small_target).float().mean()
    
    if final_acc < 0.95:
        print("\nWARNING: Model cannot overfit small dataset!")
        print("Possible issues:")
        print("  - Bug in forward pass")
        print("  - Learning rate too low")
        print("  - Optimizer not working")
    else:
        print(f"\n✓ Model can overfit (acc = {final_acc.item():.2f})")

overfitting_test(model, criterion, optimizer)
```

## Чеклист отладки

1. ✅ **Sanity checks**
   - Размерности совпадают
   - Loss начинается с разумного значения
   - Градиенты не NaN/Inf

2. ✅ **Overfit small dataset**
   - Модель может достичь 100% accuracy на 10-100 примерах

3. ✅ **Gradient flow**
   - Все слои получают градиенты
   - Нет vanishing/exploding gradients

4. ✅ **Learning rate**
   - Используйте LR finder
   - Пробуйте разные значения

5. ✅ **Regularization**
   - Начните без регуляризации
   - Добавляйте постепенно если переобучаетесь

## Литература

- **A Recipe for Training Neural Networks** (Andrej Karpathy blog post)
- **Visualizing the Loss Landscape** (Li et al., 2018)
- **Cyclical Learning Rates** (Smith, 2017)
