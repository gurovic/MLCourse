# 493. MLOps для нейронных сетей

## 🟢 Основы (Basic Level)

### Введение в MLOps

**MLOps (Machine Learning Operations)** - практики для автоматизации и мониторинга ML моделей в production.

**Основные компоненты:**
- Experiment tracking
- Model versioning
- Reproducibility
- Model registry
- Deployment automation
- Monitoring

### Weights & Biases (wandb)

Популярный инструмент для tracking экспериментов.

```python
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

# Инициализация проекта
wandb.init(
    project="my-neural-network",
    config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32,
        "architecture": "ResNet18",
        "dataset": "CIFAR-10"
    }
)

config = wandb.config

# Простая модель
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)

optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()

# Обучение с логированием
for epoch in range(config.epochs):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        epoch_acc += pred.eq(target.view_as(pred)).sum().item()
    
    # Логирование метрик
    wandb.log({
        "epoch": epoch,
        "train_loss": epoch_loss / len(train_loader),
        "train_acc": epoch_acc / len(train_loader.dataset)
    })
    
    # Валидация
    model.eval()
    val_loss = 0
    val_acc = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            val_acc += pred.eq(target.view_as(pred)).sum().item()
    
    wandb.log({
        "val_loss": val_loss / len(val_loader),
        "val_acc": val_acc / len(val_loader.dataset)
    })

# Сохранение модели
torch.save(model.state_dict(), "model.pth")
wandb.save("model.pth")

wandb.finish()
```

### Логирование артефактов

```python
# Логирование confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Получение предсказаний
y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1)
        y_true.extend(target.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

# Создание confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Логирование в wandb
wandb.log({"confusion_matrix": wandb.Image(plt)})
plt.close()
```

## 🟡 Средний уровень (Intermediate Level)

### MLflow

Платформа для полного ML lifecycle management.

```python
import mlflow
import mlflow.pytorch

# Настройка MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-experiment")

# Начало run
with mlflow.start_run():
    # Логирование параметров
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 10)
    mlflow.log_param("optimizer", "Adam")
    
    # Обучение модели
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Логирование метрик
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)
    
    # Сохранение модели
    mlflow.pytorch.log_model(model, "model")
    
    # Логирование артефактов
    torch.save(model.state_dict(), "model_weights.pth")
    mlflow.log_artifact("model_weights.pth")
    
    # Логирование графиков
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.legend()
    plt.savefig("loss_curves.png")
    mlflow.log_artifact("loss_curves.png")
    plt.close()
```

### Model Registry

```python
# Регистрация модели
model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
mlflow.register_model(model_uri, "MyModelName")

# Переход модели в production
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Получение последней версии
model_versions = client.get_latest_versions("MyModelName", stages=["None"])
latest_version = model_versions[0].version

# Перевод в production
client.transition_model_version_stage(
    name="MyModelName",
    version=latest_version,
    stage="Production"
)

# Загрузка production модели
model = mlflow.pytorch.load_model(
    model_uri=f"models:/MyModelName/Production"
)
```

### DVC (Data Version Control)

Версионирование данных и моделей.

```bash
# Установка
pip install dvc

# Инициализация
dvc init

# Добавление данных под контроль версий
dvc add data/train_images.zip
git add data/train_images.zip.dvc data/.gitignore
git commit -m "Add training data"

# Настройка remote storage (S3, Google Drive, etc.)
dvc remote add -d myremote s3://mybucket/path
dvc push

# Получение данных на другой машине
dvc pull
```

```python
# dvc.yaml - определение pipeline
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - data/raw
    params:
      - prepare.split_ratio
    outs:
      - data/processed
      
  train:
    cmd: python src/train.py
    deps:
      - data/processed
      - src/train.py
    params:
      - train.epochs
      - train.learning_rate
    outs:
      - models/model.pth
    metrics:
      - metrics/train_metrics.json:
          cache: false
          
  evaluate:
    cmd: python src/evaluate.py
    deps:
      - models/model.pth
      - data/processed
    metrics:
      - metrics/eval_metrics.json:
          cache: false
```

## 🔴 Продвинутый уровень (Expert Level)

### Полный MLOps Pipeline

```python
# config.py
from dataclasses import dataclass
from typing import List

@dataclass
class TrainConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    model_name: str = "resnet18"
    seed: int = 42
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

# train.py
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import wandb

@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig):
    # Setup reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Initialize tracking
    wandb.init(
        project=cfg.project_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    mlflow.set_experiment(cfg.experiment_name)
    
    with mlflow.start_run():
        # Log config
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        
        # Create model
        model = create_model(cfg.model)
        
        # Training loop with callbacks
        trainer = Trainer(
            model=model,
            config=cfg,
            callbacks=[
                EarlyStoppingCallback(patience=5),
                ModelCheckpointCallback(save_dir="checkpoints"),
                LoggingCallback(wandb=wandb, mlflow=mlflow)
            ]
        )
        
        trainer.fit(train_loader, val_loader)
        
        # Evaluate
        test_metrics = trainer.evaluate(test_loader)
        
        # Log final metrics
        mlflow.log_metrics(test_metrics)
        wandb.log(test_metrics)
        
        # Register model
        mlflow.pytorch.log_model(model, "model")
        
    wandb.finish()

if __name__ == "__main__":
    train()
```

### Model Monitoring

```python
# monitoring.py
import prometheus_client as prom
from fastapi import FastAPI
import torch

app = FastAPI()

# Prometheus metrics
REQUEST_COUNT = prom.Counter('model_request_count', 'Total requests')
REQUEST_LATENCY = prom.Histogram('model_request_latency_seconds', 'Request latency')
PREDICTION_DISTRIBUTION = prom.Histogram('model_prediction_distribution', 'Prediction distribution')

@app.post("/predict")
@REQUEST_LATENCY.time()
def predict(data: dict):
    REQUEST_COUNT.inc()
    
    # Preprocessing
    input_tensor = preprocess(data)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(dim=1).item()
    
    # Log prediction
    PREDICTION_DISTRIBUTION.observe(prediction)
    
    return {"prediction": prediction, "confidence": float(output.max())}

@app.get("/metrics")
def metrics():
    return prom.generate_latest()

# Model drift detection
class DriftDetector:
    def __init__(self, reference_data, threshold=0.05):
        self.reference_data = reference_data
        self.threshold = threshold
        
    def detect(self, new_data):
        """KS test for drift detection"""
        from scipy.stats import ks_2samp
        
        drift_detected = False
        for feature_idx in range(new_data.shape[1]):
            statistic, pvalue = ks_2samp(
                self.reference_data[:, feature_idx],
                new_data[:, feature_idx]
            )
            if pvalue < self.threshold:
                drift_detected = True
                print(f"Drift detected in feature {feature_idx}, p-value: {pvalue}")
        
        return drift_detected
```

### CI/CD для ML

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          
      - name: Run tests
        run: |
          pytest tests/
          
      - name: Check code quality
        run: |
          flake8 src/
          black --check src/
          
  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Pull data
        run: |
          dvc pull
          
      - name: Train model
        run: |
          python src/train.py
          
      - name: Evaluate model
        run: |
          python src/evaluate.py
          
      - name: Check model performance
        run: |
          python src/check_metrics.py --min-accuracy 0.85
          
  deploy:
    needs: train
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          # Deploy to cloud (AWS, GCP, Azure)
          echo "Deploying model..."
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: ml-model
        image: myregistry/ml-model:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: MODEL_PATH
          value: "/models/model.pth"
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Ссылки

- [Weights & Biases Docs](https://docs.wandb.ai/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Hydra Config](https://hydra.cc/)
- [MLOps Principles](https://ml-ops.org/)

## Tools

- Experiment Tracking: W&B, MLflow, Neptune.ai
- Model Registry: MLflow, DVC
- Pipeline Orchestration: Kubeflow, Airflow, Metaflow
- Monitoring: Prometheus, Grafana, Evidently
