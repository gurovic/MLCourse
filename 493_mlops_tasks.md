# Задачи: MLOps для нейронных сетей

## 🟢 Базовый уровень

### Задача 1: Weights & Biases Setup
Обучите простую CNN на MNIST с логированием в W&B. Залогируйте:
- Гиперпараметры
- Training/validation loss
- Training/validation accuracy
- Примеры предсказаний на каждой эпохе

### Задача 2: MLflow Tracking
Запустите MLflow tracking server локально. Проведите 5 экспериментов с разными learning rates и сравните результаты в UI.

### Задача 3: Model Checkpointing
Реализуйте автоматическое сохранение лучшей модели по validation accuracy. Используйте PyTorch Lightning или собственный callback.

## 🟡 Средний уровень

### Задача 4: Experiment Comparison
Проведите сравнение 3 архитектур (LeNet, simple CNN, ResNet-18) на CIFAR-10. Используйте W&B или MLflow для tracking. Создайте отчет с визуализацией.

### Задача 5: DVC Pipeline
Создайте DVC pipeline для:
- Загрузки данных
- Предобработки
- Обучения модели
- Evaluation
Версионируйте данные и модель.

### Задача 6: Model Registry
Создайте систему для регистрации моделей в MLflow Model Registry. Реализуйте переход моделей через стадии: None → Staging → Production.

### Задача 7: Hyperparameter Sweeps
Используйте W&B Sweeps или Optuna для автоматического поиска гиперпараметров. Оптимизируйте learning rate, batch size, dropout rate для вашей модели.

## 🔴 Продвинутый уровень

### Задача 8: Полный MLOps Pipeline
Создайте complete pipeline с:
- Hydra для конфигурации
- DVC для версионирования данных
- MLflow для tracking и registry
- CI/CD с GitHub Actions
- Автоматическими тестами

### Задача 9: Model Monitoring
Разверните модель с FastAPI и добавьте:
- Prometheus metrics (request count, latency, predictions distribution)
- Grafana dashboard для визуализации
- Drift detection на входных данных

### Задача 10: Kubernetes Deployment
Создайте Kubernetes deployment для вашей модели:
- Docker образ с моделью
- Deployment с 3 репликами
- Service для load balancing
- HorizontalPodAutoscaler для автоскейлинга
- Мониторинг с Prometheus + Grafana

**Tools:**
- Weights & Biases: https://wandb.ai/
- MLflow: https://mlflow.org/
- DVC: https://dvc.org/
- Hydra: https://hydra.cc/
- Prometheus: https://prometheus.io/
