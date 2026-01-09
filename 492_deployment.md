# Развертывание моделей (Model Deployment)

## 🟢 Основы

### Экспорт модели

**PyTorch → TorchScript**

```python
import torch

# Способ 1: Tracing (запускаем пример и записываем операции)
model = MyModel()
model.eval()

example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Сохранение
traced_model.save("model_traced.pt")

# Загрузка
loaded_model = torch.jit.load("model_traced.pt")

# Способ 2: Scripting (анализирует код напрямую, поддерживает control flow)
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")
```

**PyTorch → ONNX**

```python
import torch.onnx

model = MyModel()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# Проверка ONNX модели
import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
```

### Inference с ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Загружаем модель
session = ort.InferenceSession("model.onnx")

# Подготовка input
input_name = session.get_inputs()[0].name
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Inference
outputs = session.run(None, {input_name: input_data})
print(f"Output shape: {outputs[0].shape}")

# Преимущества ONNX Runtime:
# - Быстрее PyTorch на CPU (2-10x)
# - Кросс-платформенность
# - Оптимизации для inference
```

## 🟡 REST API для Inference

### FastAPI сервер

```python
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

app = FastAPI()

# Загружаем модель при старте
model = torch.jit.load("model_traced.pt")
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Читаем изображение
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Preprocessing
    input_tensor = transform(image).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    # Результат
    results = {
        "predictions": [
            {"class_id": int(idx), "probability": float(prob)}
            for idx, prob in zip(top5_idx[0], top5_prob[0])
        ]
    }
    
    return results

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Запуск: uvicorn main:app --host 0.0.0.0 --port 8000
```

### Batch Inference

```python
from typing import List
import asyncio

class BatchPredictor:
    def __init__(self, model, batch_size=32, timeout=0.1):
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = asyncio.Queue()
        self.results = {}
        
    async def start(self):
        """Background task для batch processing"""
        while True:
            batch = []
            request_ids = []
            
            # Собираем батч
            try:
                while len(batch) < self.batch_size:
                    request_id, data = await asyncio.wait_for(
                        self.queue.get(), timeout=self.timeout
                    )
                    batch.append(data)
                    request_ids.append(request_id)
            except asyncio.TimeoutError:
                if not batch:
                    continue
            
            # Обрабатываем батч
            batch_tensor = torch.stack(batch)
            with torch.no_grad():
                outputs = self.model(batch_tensor)
            
            # Распределяем результаты
            for request_id, output in zip(request_ids, outputs):
                self.results[request_id] = output
    
    async def predict(self, data):
        """Добавляем запрос в очередь"""
        request_id = id(data)
        await self.queue.put((request_id, data))
        
        # Ждем результат
        while request_id not in self.results:
            await asyncio.sleep(0.01)
        
        result = self.results.pop(request_id)
        return result

# Использование в FastAPI
predictor = BatchPredictor(model, batch_size=32)

@app.on_event("startup")
async def startup():
    asyncio.create_task(predictor.start())

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    image = preprocess_image(await file.read())
    result = await predictor.predict(image)
    return {"prediction": result.tolist()}
```

## 🔴 Production-Ready Deployment

### Docker контейнеризация

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и модель
COPY . .

# Expose порт
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/health || exit 1

# Запуск
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  model-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/model.onnx
      - BATCH_SIZE=32
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Мониторинг и логирование

```python
from prometheus_client import Counter, Histogram, start_http_server
import logging
import time

# Метрики
REQUEST_COUNT = Counter('requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency')
ERROR_COUNT = Counter('errors_total', 'Total errors')

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def add_metrics(request, call_next):
    REQUEST_COUNT.inc()
    
    start_time = time.time()
    try:
        response = await call_next(request)
        REQUEST_LATENCY.observe(time.time() - start_time)
        return response
    except Exception as e:
        ERROR_COUNT.inc()
        logger.error(f"Error processing request: {e}")
        raise

# Запускаем Prometheus metrics endpoint
start_http_server(9090)
```

### Model Versioning

```python
from pathlib import Path
import json

class ModelRegistry:
    def __init__(self, registry_path="model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        
    def register_model(self, model_path, version, metadata):
        """Регистрируем новую версию модели"""
        version_dir = self.registry_path / f"v{version}"
        version_dir.mkdir(exist_ok=True)
        
        # Копируем модель
        import shutil
        shutil.copy(model_path, version_dir / "model.onnx")
        
        # Сохраняем метаданные
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Registered model version {version}")
    
    def load_model(self, version="latest"):
        """Загружаем конкретную версию модели"""
        if version == "latest":
            versions = sorted([d.name for d in self.registry_path.iterdir() if d.is_dir()])
            version = versions[-1]
        
        model_path = self.registry_path / version / "model.onnx"
        metadata_path = self.registry_path / version / "metadata.json"
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        return model_path, metadata
    
    def list_versions(self):
        """Список всех версий"""
        versions = []
        for version_dir in self.registry_path.iterdir():
            if version_dir.is_dir():
                metadata_path = version_dir / "metadata.json"
                with open(metadata_path) as f:
                    metadata = json.load(f)
                versions.append({
                    "version": version_dir.name,
                    **metadata
                })
        return versions

# Использование
registry = ModelRegistry()

# Регистрация
registry.register_model(
    "model.onnx",
    version="1.0.0",
    metadata={
        "accuracy": 0.92,
        "training_date": "2024-01-15",
        "dataset": "ImageNet",
        "framework": "PyTorch 2.0"
    }
)

# Загрузка
model_path, metadata = registry.load_model(version="latest")
```

### A/B Testing

```python
import random

class ABTester:
    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio
        self.metrics = {"a": [], "b": []}
        
    def predict(self, user_id, data):
        """Выбираем модель на основе user_id"""
        # Consistent hashing для одного пользователя
        use_model_a = hash(user_id) % 100 < (self.split_ratio * 100)
        
        if use_model_a:
            result = self.model_a.predict(data)
            variant = "a"
        else:
            result = self.model_b.predict(data)
            variant = "b"
        
        return result, variant
    
    def log_metric(self, variant, metric_value):
        """Логируем метрику для варианта"""
        self.metrics[variant].append(metric_value)
    
    def get_statistics(self):
        """Статистика по вариантам"""
        return {
            "a": {
                "mean": np.mean(self.metrics["a"]),
                "std": np.std(self.metrics["a"]),
                "count": len(self.metrics["a"])
            },
            "b": {
                "mean": np.mean(self.metrics["b"]),
                "std": np.std(self.metrics["b"]),
                "count": len(self.metrics["b"])
            }
        }

ab_tester = ABTester(model_v1, model_v2, split_ratio=0.5)

@app.post("/predict_ab")
async def predict_ab(user_id: str, file: UploadFile = File(...)):
    image = await preprocess_image(file)
    result, variant = ab_tester.predict(user_id, image)
    
    # Логируем для аналитики
    logger.info(f"User {user_id} got variant {variant}")
    
    return {"prediction": result, "variant": variant}
```

### Serving с TorchServe

```bash
# Архивируем модель для TorchServe
torch-model-archiver \
    --model-name resnet50 \
    --version 1.0 \
    --model-file model.py \
    --serialized-file model.pth \
    --handler image_classifier \
    --extra-files index_to_name.json

# Запуск TorchServe
torchserve --start \
    --model-store model_store \
    --models resnet50=resnet50.mar \
    --ncs
```

### Edge Deployment (Mobile/IoT)

```python
# Оптимизация для mobile
import torch.quantization

# Quantization
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
model_prepared = torch.quantization.prepare(model)
model_quantized = torch.quantization.convert(model_prepared)

# Оптимизация для mobile
from torch.utils.mobile_optimizer import optimize_for_mobile

scripted_model = torch.jit.script(model_quantized)
optimized_model = optimize_for_mobile(scripted_model)
optimized_model._save_for_lite_interpreter("model_mobile.ptl")

# Размер модели уменьшится в 4x, inference ускорится
```

## Best Practices

1. **Версионирование**: храните все версии моделей с метаданными
2. **Мониторинг**: отслеживайте latency, throughput, errors
3. **A/B Testing**: тестируйте новые модели на части трафика
4. **Graceful Degradation**: fallback на простую модель при ошибках
5. **Security**: валидация входных данных, rate limiting
6. **Caching**: кешируйте результаты для популярных запросов
7. **Auto-scaling**: масштабируйтесь по нагрузке (Kubernetes HPA)

## Литература

- **TorchServe Documentation**
- **ONNX Runtime Performance Tuning**
- **FastAPI Best Practices**
- **MLOps: Continuous Delivery for ML** (Sato et al.)
