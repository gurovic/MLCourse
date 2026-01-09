# Задачи: Оптимизация производительности

## Задача 1: Mixed Precision Training (🟢)

Реализуйте обучение с automatic mixed precision (AMP).

**Требования**:
- Используйте torch.cuda.amp.autocast и GradScaler
- Обучите ResNet на CIFAR-10
- Сравните время обучения FP32 vs FP16
- Сравните memory usage
- Убедитесь, что accuracy не ухудшилась

**Метрика**: Speedup > 1.5x, accuracy difference < 1%

## Задача 2: Gradient Accumulation (🟢)

Эмулируйте большой batch_size с gradient accumulation.

**Требования**:
- Реальный batch_size = 32
- Accumulation steps = 4 (эффективный batch = 128)
- Сравните с обычным training (batch=128)
- Измерьте memory usage для обоих вариантов

## Задача 3: DataLoader Optimization (🟢)

Оптимизируйте загрузку данных.

**Требования**:
- Измерьте время с num_workers = [0, 1, 2, 4, 8]
- Включите pin_memory и prefetch_factor
- Постройте график throughput vs num_workers
- Определите оптимальное значение

## Задача 4: DistributedDataParallel (🟡)

Реализуйте multi-GPU обучение с DDP.

**Требования**:
- Используйте torch.distributed и DDP
- Поддержка 2-4 GPU
- DistributedSampler для данных
- Измерьте speedup vs single GPU
- Проверьте, что результат идентичен single GPU

**Метрика**: Linear scaling (2 GPU → 2x speedup)

## Задача 5: Model Profiling (🟡)

Профилируйте модель и найдите bottlenecks.

**Требования**:
- Используйте torch.profiler
- Профилируйте CPU и CUDA time
- Выделите top-10 самых медленных операций
- Экспортируйте trace для Chrome viewer
- Предложите оптимизации на основе профиля

## Задача 6: Post-Training Quantization (🟡)

Сожмите модель с помощью quantization.

**Требования**:
- Обучите FP32 модель на MNIST/CIFAR
- Примените dynamic quantization
- Примените static quantization с калибровкой
- Сравните размер модели (FP32 vs INT8)
- Сравните inference speed и accuracy

**Метрика**: 4x compression, accuracy drop < 2%

## Задача 7: Knowledge Distillation (🔴)

Сожмите ResNet-50 в MobileNet с помощью distillation.

**Требования**:
- Teacher: ResNet-50 (предобученная на ImageNet)
- Student: MobileNetV2
- Distillation loss с temperature = 3-5
- Alpha (hard vs soft loss) = 0.3-0.7
- Сравните: student с distillation vs student без distillation

**Метрика**: Student accuracy > baseline + 3%

## Задача 8: Model Pruning (🔴)

Примените iterative magnitude pruning.

**Требования**:
- Начните с 90% dense модели
- Итеративно увеличивайте sparsity: 70%, 50%, 30%, 10%
- Fine-tune 5 эпох после каждого prune
- Постройте график accuracy vs sparsity
- Измерьте actual speedup на inference

**Метрика**: 70% sparsity, accuracy drop < 3%

## Задача 9: FSDP для Large Models (🔴)

Обучите большую модель (>1B параметров) с FSDP.

**Требования**:
- Модель: GPT-style transformer с 1-3B параметров
- Используйте FSDP для sharding
- Mixed precision (FP16/BF16)
- Gradient checkpointing для экономии памяти
- Измерьте memory usage per GPU
- Сравните с DDP (если влезает)

## Задача 10: Production Optimization Pipeline (🔴)

Создайте полный pipeline оптимизации для production.

**Требования**:
1. **Training optimization**:
   - Mixed precision
   - DDP на 4-8 GPU
   - Gradient accumulation
   - Efficient data loading
   
2. **Model compression**:
   - Knowledge distillation (если нужно)
   - Quantization (INT8)
   - Pruning (optional, 50% sparsity)
   
3. **Inference optimization**:
   - TorchScript compilation
   - ONNX export
   - Batch inference
   - GPU/CPU оптимизация
   
4. **Benchmarking**:
   - Latency (p50, p95, p99)
   - Throughput (samples/sec)
   - Memory usage
   - Model size
   
5. **Автоматизация**:
   - Скрипт, который принимает модель и датасет
   - Применяет все оптимизации
   - Выдает оптимизированную модель + отчет
   
**Метрики**:
- Inference latency < 50ms (p99)
- Throughput > 100 samples/sec
- Model size < 50MB
- Accuracy drop < 2%

**Бонус**: поддержка разных backends (ONNX Runtime, TensorRT, OpenVINO)
