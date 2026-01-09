# Задачи: Развертывание моделей

## Задача 1: TorchScript Export (🟢)

Экспортируйте PyTorch модель в TorchScript.

**Требования**:
- Реализуйте оба способа: tracing и scripting
- Протестируйте на модели с control flow (if/for)
- Сравните размер файлов .pth vs .pt
- Убедитесь, что выходы идентичны
- Измерьте inference speed

## Задача 2: ONNX Export и Inference (🟢)

Экспортируйте модель в ONNX и запустите с ONNX Runtime.

**Требования**:
- Экспортируйте ResNet/EfficientNet в ONNX
- Поддержка dynamic batch size
- Проверьте корректность с onnx.checker
- Запустите inference с onnxruntime
- Сравните скорость: PyTorch vs ONNX Runtime (CPU)

**Метрика**: ONNX Runtime должен быть быстрее в 2-3x на CPU

## Задача 3: REST API с FastAPI (🟡)

Создайте REST API для image classification.

**Требования**:
- Endpoint POST /predict принимает изображение
- Возвращает top-5 классов с вероятностями
- Health check endpoint GET /health
- Обработка ошибок (invalid image, too large file)
- Документация через Swagger UI

## Задача 4: Batch Inference API (🟡)

Реализуйте batch inference для эффективной обработки.

**Требования**:
- Собирайте requests в батчи (batch_size=32)
- Timeout для формирования батча (100ms)
- Async processing с asyncio
- Измерьте throughput (requests/sec)
- Сравните с sequential processing

**Метрика**: Throughput должен вырасти в 5-10x

## Задача 5: Docker Deployment (🟡)

Контейнеризируйте модель с Docker.

**Требования**:
- Dockerfile с оптимизированным размером (<1GB)
- Multi-stage build для уменьшения размера
- Health check в контейнере
- docker-compose.yml для запуска
- Volume для моделей

## Задача 6: Model Registry (🔴)

Создайте систему версионирования моделей.

**Требования**:
1. **Registration**: сохранение модели с метаданными
   - Version, accuracy, training_date, dataset
   - MD5 checksum для integrity
2. **Loading**: загрузка по версии или "latest"
3. **Comparison**: сравнение метрик разных версий
4. **Rollback**: откат на предыдущую версию
5. **API endpoints**:
   - POST /models - регистрация
   - GET /models - список версий
   - GET /models/{version} - конкретная версия
   - POST /models/{version}/promote - сделать latest

## Задача 7: Monitoring и Logging (🔴)

Добавьте полный monitoring stack.

**Требования**:
1. **Metrics** (Prometheus):
   - Request count, latency (p50, p95, p99)
   - Error rate
   - Model prediction distribution
   - Resource usage (CPU, Memory, GPU)
   
2. **Logging** (structured JSON):
   - Request ID для трacing
   - Input/output metadata
   - Errors с stack traces
   
3. **Alerting**:
   - Latency > 200ms для 95th percentile
   - Error rate > 1%
   - Drift detection (distribution shift)
   
4. **Grafana Dashboard**:
   - Real-time метрики
   - Historical trends

## Задача 8: A/B Testing Framework (🔴)

Реализуйте систему A/B тестирования моделей.

**Требования**:
- Consistent hashing для user_id (один пользователь всегда видит одну модель)
- Split ratio configurable (70/30, 50/50, etc.)
- Метрики per variant (accuracy, latency, user satisfaction)
- Statistical significance test (t-test, chi-square)
- Automatic winner selection при достижении confidence
- Gradual rollout (canary deployment)

## Задача 9: Edge Deployment (🔴)

Оптимизируйте модель для mobile/edge устройства.

**Требования**:
1. **Optimization**:
   - Quantization (INT8)
   - Pruning (50% sparsity)
   - Knowledge distillation (если нужно)
   
2. **Mobile Export**:
   - PyTorch Mobile (.ptl)
   - TFLite (если используете TF)
   - Core ML (для iOS)
   
3. **Benchmarking**:
   - Model size < 10MB
   - Inference latency < 100ms на mobile CPU
   - Accuracy drop < 3%
   
4. **Integration**:
   - Android app или iOS app (опционально)
   - On-device inference demo

## Задача 10: Production ML Pipeline (🔴)

Создайте end-to-end production pipeline.

**Требования**:
1. **Training Pipeline**:
   - Data validation (schema, distribution)
   - Automated training with hyperparameter search
   - Model evaluation и validation
   - Artifact storage (model, metrics, configs)
   
2. **Deployment Pipeline**:
   - CI/CD с GitHub Actions/GitLab CI
   - Automated testing (unit, integration, load)
   - Blue-green deployment или canary
   - Rollback mechanism
   
3. **Serving Infrastructure**:
   - Kubernetes deployment с autoscaling
   - Load balancer (nginx/Ingress)
   - Multiple model versions simultaneously
   - Feature store для preprocessing
   
4. **Monitoring & Ops**:
   - Data drift detection
   - Model performance monitoring
   - Automatic retraining trigger
   - Alerting и incident response
   
5. **Security**:
   - API authentication (JWT tokens)
   - Rate limiting
   - Input validation и sanitization
   - Audit logging

**Deliverables**:
- Infrastructure as Code (Terraform/Helm charts)
- CI/CD configuration
- Monitoring dashboards
- Documentation
- Load testing results

**Метрики**:
- Deployment time < 10 minutes
- Zero-downtime deployments
- 99.9% uptime SLA
- Latency p99 < 200ms
- Automatic scaling при load spike

**Бонус**: multi-cloud deployment (AWS + GCP) с failover
