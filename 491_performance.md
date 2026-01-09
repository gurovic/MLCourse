# Оптимизация производительности нейронных сетей

## 🟢 Основы оптимизации

### Mixed Precision Training

**Идея**: используем FP16 вместо FP32 для ускорения вычислений и экономии памяти

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Инициализация
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()

# Training loop
for data, target in train_loader:
    data, target = data.cuda(), target.cuda()
    
    optimizer.zero_grad()
    
    # Автоматическое mixed precision
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # Масштабируем loss перед backward
    scaler.scale(loss).backward()
    
    # Обновляем веса с unscaling
    scaler.step(optimizer)
    scaler.update()

# Преимущества:
# - 2-3x ускорение
# - 50% экономии памяти
# - Минимальная потеря точности
```

### Gradient Accumulation

**Задача**: эмулируем большой batch_size на ограниченной памяти

```python
accumulation_steps = 4  # Эффективный batch_size = batch_size * accumulation_steps

model.zero_grad()

for i, (data, target) in enumerate(train_loader):
    data, target = data.cuda(), target.cuda()
    
    # Forward pass
    output = model(data)
    loss = criterion(output, target)
    
    # Нормализуем loss
    loss = loss / accumulation_steps
    
    # Накапливаем градиенты
    loss.backward()
    
    # Обновляем веса каждые accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        model.zero_grad()
```

### DataLoader Optimization

```python
from torch.utils.data import DataLoader

# Оптимальные настройки
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Параллельная загрузка данных
    pin_memory=True,  # Быстрее CPU->GPU transfer
    prefetch_factor=2,  # Предзагрузка батчей
    persistent_workers=True  # Переиспользуем worker процессы
)

# Для very large datasets
from torch.utils.data import IterableDataset

class StreamingDataset(IterableDataset):
    """Датасет, который грузит данные по требованию"""
    def __init__(self, file_paths):
        self.file_paths = file_paths
        
    def __iter__(self):
        for path in self.file_paths:
            # Грузим и обрабатываем on-the-fly
            data = load_and_process(path)
            yield data
```

## 🟡 Distributed Training

### DataParallel (простой, но не оптимальный)

```python
# Распараллеливание по нескольким GPU
model = nn.DataParallel(model)
model = model.cuda()

# Минусы:
# - Реплицирует модель на каждой GPU каждую итерацию (медленно)
# - Несбалансированная нагрузка на GPU 0
```

### DistributedDataParallel (рекомендуется)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """Инициализация process group"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)
    
    # Создаем модель на конкретном GPU
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Distributed sampler разделяет данные между процессами
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4)
    
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Важно для правильного shuffling
        
        for data, target in train_loader:
            data, target = data.to(rank), target.to(rank)
            
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    cleanup()

# Запуск
import torch.multiprocessing as mp

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
```

### Fully Sharded Data Parallel (FSDP)

**Для очень больших моделей**: sharding параметров, градиентов и optimizer states

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# Автоматический wrap по размеру
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy, 
    min_num_params=1e6  # Wrap модули с >1M параметрами
)

model = MyLargeModel()
fsdp_model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=torch.distributed.fsdp.MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16
    )
)

# Преимущества:
# - Позволяет обучать модели, не влезающие в память одной GPU
# - Линейное масштабирование с количеством GPU
```

## 🔴 Advanced Optimizations

### Model Quantization

**Post-Training Quantization**: FP32 → INT8 после обучения

```python
import torch.quantization

# Подготовка модели
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Калибровка на небольшом датасете
with torch.no_grad():
    for data, _ in calibration_loader:
        model(data)

# Конвертация в quantized модель
torch.quantization.convert(model, inplace=True)

# Размер уменьшается в 4x, inference ускоряется в 2-4x
```

**Quantization-Aware Training**: учитываем quantization во время обучения

```python
import torch.quantization

model = MyModel()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_prepared = torch.quantization.prepare_qat(model, inplace=False)

# Обучаем как обычно
for epoch in range(num_epochs):
    train(model_prepared, train_loader)

# Конвертируем в quantized
model_quantized = torch.quantization.convert(model_prepared.eval(), inplace=False)
```

### Model Pruning

**Удаление неважных весов для сжатия модели**

```python
import torch.nn.utils.prune as prune

# Unstructured pruning (удаляем отдельные веса)
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.3)  # Удаляем 30%
        prune.remove(module, 'weight')  # Применяем маску

# Structured pruning (удаляем целые каналы/нейроны)
prune.ln_structured(
    model.conv1, 
    name='weight', 
    amount=0.5,  # Удаляем 50% каналов
    n=2,  # L2 norm
    dim=0  # По выходным каналам
)

# Iterative pruning + fine-tuning
for sparsity in [0.2, 0.4, 0.6, 0.8]:
    prune_model(model, sparsity)
    fine_tune(model, train_loader, epochs=5)
```

### Knowledge Distillation для compression

```python
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, labels):
        # Hard target loss
        hard_loss = self.criterion(student_logits, labels)
        
        # Soft target loss (distillation)
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        
        # Комбинируем
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# Teacher (большая модель)
teacher = LargeModel()
teacher.load_state_dict(torch.load('teacher.pth'))
teacher.eval()

# Student (маленькая модель)
student = SmallModel()

criterion = DistillationLoss(alpha=0.5, temperature=3.0)
optimizer = torch.optim.Adam(student.parameters(), lr=0.001)

for data, labels in train_loader:
    with torch.no_grad():
        teacher_logits = teacher(data)
    
    student_logits = student(data)
    loss = criterion(student_logits, teacher_logits, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Profiling и Bottleneck Detection

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("model_training"):
        for i, (data, target) in enumerate(train_loader):
            if i >= 10:  # Профилируем 10 итераций
                break
            
            data, target = data.cuda(), target.cuda()
            
            with record_function("forward"):
                output = model(data)
                loss = criterion(output, target)
            
            with record_function("backward"):
                optimizer.zero_grad()
                loss.backward()
            
            with record_function("optimizer_step"):
                optimizer.step()

# Выводим отчет
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Экспорт в Chrome trace viewer
prof.export_chrome_trace("trace.json")
```

### Checkpointing для экономии памяти

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(...)
        self.layer2 = nn.Sequential(...)
        self.layer3 = nn.Sequential(...)
        
    def forward(self, x):
        # Используем gradient checkpointing
        # Не сохраняем промежуточные активации, пересчитываем их при backward
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        x = self.layer3(x)
        return x

# Компромисс: экономим память, но тратим больше времени
```

### Compilation с torch.compile (PyTorch 2.0+)

```python
# Автоматическая оптимизация модели
model = MyModel()
compiled_model = torch.compile(model)

# Backends:
# - "inductor" (default): torch's own compiler
# - "aot_eager": ahead-of-time compilation
# - "cudagraphs": CUDA graph optimization

compiled_model = torch.compile(model, mode="reduce-overhead", backend="inductor")

# Ускорение: 20-50% для многих моделей
```

## Best Practices

1. **Профилируйте перед оптимизацией**: найдите реальные bottlenecks
2. **Mixed precision по умолчанию**: почти всегда win-win
3. **Batch size**: максимальный, который влезает в память
4. **DataLoader**: num_workers=4-8, pin_memory=True
5. **DDP вместо DP**: для multi-GPU
6. **Компилируйте модель**: torch.compile для PyTorch 2.0+

## Литература

- **PyTorch Performance Tuning Guide**
- **Efficient Training of Large Models** (DeepSpeed, Megatron)
- **Mixed Precision Training** (Micikevicius et al., 2018)
