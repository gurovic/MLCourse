# Интерпретируемость нейронных сетей

## 🟢 Основы (Basic Level)

### Зачем нужна интерпретируемость?

**Interpretability (Интерпретируемость)** - способность объяснить, почему модель приняла конкретное решение.

**Важность:**
- Доверие пользователей
- Отладка модели
- Соответствие регуляциям (GDPR, финансы, медицина)
- Обнаружение bias
- Научные открытия

### Feature Importance

Простейший способ - важность входных признаков.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Анализ весов первого слоя
model = SimpleModel(input_size=20, hidden_size=10, num_classes=2)
weights = model.fc1.weight.data.abs().mean(dim=0).numpy()

plt.figure(figsize=(10, 5))
plt.bar(range(len(weights)), weights)
plt.xlabel('Feature Index')
plt.ylabel('Average Absolute Weight')
plt.title('Feature Importance (Weight-based)')
plt.show()
```

### Saliency Maps для изображений

Градиент выхода по входу показывает важность каждого пикселя.

```python
def compute_saliency(model, image, target_class):
    """
    Вычисление saliency map для изображения.
    
    Args:
        model: обученная модель
        image: входное изображение (1, C, H, W)
        target_class: целевой класс
    
    Returns:
        saliency: карта важности (H, W)
    """
    model.eval()
    image.requires_grad = True
    
    # Forward pass
    output = model(image)
    
    # Backprop для target_class
    model.zero_grad()
    output[0, target_class].backward()
    
    # Saliency = абсолютное значение градиента
    saliency = image.grad.data.abs()
    saliency = saliency.max(dim=1)[0]  # Max across channels
    saliency = saliency.squeeze().cpu().numpy()
    
    return saliency

# Пример использования
from torchvision import models, transforms
from PIL import Image

# Загрузка предобученной модели
model = models.resnet18(pretrained=True)

# Загрузка и предобработка изображения
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# img = Image.open('cat.jpg')
# input_tensor = transform(img).unsqueeze(0)
# 
# # Получение предсказания
# with torch.no_grad():
#     output = model(input_tensor)
#     pred_class = output.argmax().item()
# 
# # Вычисление saliency map
# saliency = compute_saliency(model, input_tensor, pred_class)
# 
# # Визуализация
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# axes[0].imshow(img)
# axes[0].set_title('Original Image')
# axes[0].axis('off')
# 
# axes[1].imshow(saliency, cmap='hot')
# axes[1].set_title('Saliency Map')
# axes[1].axis('off')
# plt.show()
```

## 🟡 Средний уровень (Intermediate Level)

### Grad-CAM (Gradient-weighted Class Activation Mapping)

Визуализация важных регионов для CNN.

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def __call__(self, x, class_idx=None):
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()
        
        # Compute weights
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3])
        
        # Weight activations
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[:, i].unsqueeze(-1).unsqueeze(-1)
        
        # Create heatmap
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)  # ReLU
        heatmap /= torch.max(heatmap)  # Normalize
        
        return heatmap.cpu().numpy()

# Использование
from torchvision import models
import cv2

model = models.resnet18(pretrained=True)
model.eval()

# Target layer (последний сверточный слой)
target_layer = model.layer4[-1]

gradcam = GradCAM(model, target_layer)

# img = cv2.imread('cat.jpg')
# img_tensor = transform(img).unsqueeze(0)
# 
# heatmap = gradcam(img_tensor)
# 
# # Overlay heatmap on original image
# heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
# heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
# superimposed_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
# 
# plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
# plt.title('Grad-CAM Visualization')
# plt.axis('off')
# plt.show()
```

### SHAP (SHapley Additive exPlanations)

Unified framework для интерпретации моделей.

```python
import shap
import torch

# Простая модель для табличных данных
class TabularModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TabularModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Загрузка данных и модели
# X_train, X_test = load_data()
# model = TabularModel(input_size=X_train.shape[1], num_classes=2)
# model.load_state_dict(torch.load('model.pth'))
# model.eval()

# Wrapper для SHAP
def model_predict(x):
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x)
        output = model(x_tensor)
        return torch.softmax(output, dim=1).numpy()

# SHAP explainer
explainer = shap.KernelExplainer(model_predict, X_train[:100])

# Объяснение для одного примера
shap_values = explainer.shap_values(X_test[0:1])

# Визуализация
shap.initjs()
shap.force_plot(explainer.expected_value[1], 
                shap_values[1][0], 
                X_test[0:1],
                feature_names=feature_names)

# Summary plot для всех примеров
shap_values_all = explainer.shap_values(X_test[:100])
shap.summary_plot(shap_values_all[1], X_test[:100], feature_names=feature_names)
```

### LIME (Local Interpretable Model-agnostic Explanations)

Локальное объяснение через аппроксимацию простой моделью.

```python
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

def predict_fn(images):
    """Wrapper для модели"""
    batch = torch.stack([transform(Image.fromarray(img)) for img in images])
    with torch.no_grad():
        outputs = model(batch)
        probabilities = torch.softmax(outputs, dim=1)
    return probabilities.cpu().numpy()

# LIME explainer для изображений
explainer = lime_image.LimeImageExplainer()

# img = np.array(Image.open('cat.jpg'))
# 
# explanation = explainer.explain_instance(
#     img, 
#     predict_fn,
#     top_labels=5,
#     hide_color=0,
#     num_samples=1000
# )
# 
# # Визуализация
# from skimage.segmentation import mark_boundaries
# 
# temp, mask = explanation.get_image_and_mask(
#     explanation.top_labels[0],
#     positive_only=True,
#     num_features=5,
#     hide_rest=False
# )
# 
# plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
# plt.title('LIME Explanation')
# plt.axis('off')
# plt.show()
```

## 🔴 Продвинутый уровень (Expert Level)

### Integrated Gradients

Более стабильный метод attribution, чем простые градиенты.

```python
def integrated_gradients(model, image, target_class, baseline=None, steps=50):
    """
    Вычисление Integrated Gradients.
    
    Args:
        model: модель
        image: входное изображение
        target_class: целевой класс
        baseline: базовое изображение (по умолчанию нули)
        steps: количество шагов интерполяции
    
    Returns:
        attributions: важность каждого пикселя
    """
    if baseline is None:
        baseline = torch.zeros_like(image)
    
    # Создание интерполированных изображений
    scaled_inputs = [baseline + (float(i) / steps) * (image - baseline) 
                     for i in range(steps + 1)]
    scaled_inputs = torch.cat(scaled_inputs, dim=0)
    
    scaled_inputs.requires_grad = True
    
    # Forward pass
    outputs = model(scaled_inputs)
    
    # Gradients для target_class
    model.zero_grad()
    target_outputs = outputs[:, target_class]
    target_outputs.backward(torch.ones_like(target_outputs))
    
    # Среднее градиентов
    gradients = scaled_inputs.grad
    avg_gradients = torch.mean(gradients, dim=0, keepdim=True)
    
    # Integrated gradients
    integrated_grads = (image - baseline) * avg_gradients
    
    return integrated_grads

# # Использование
# model = models.resnet18(pretrained=True)
# model.eval()
# 
# ig_attributions = integrated_gradients(model, input_tensor, pred_class, steps=50)
# 
# # Визуализация
# attribution_map = ig_attributions.squeeze().abs().max(dim=0)[0].cpu().numpy()
# plt.imshow(attribution_map, cmap='hot')
# plt.title('Integrated Gradients')
# plt.colorbar()
# plt.show()
```

### Attention Visualization (для Transformers)

```python
def visualize_attention(model, tokenizer, text, layer_idx=-1, head_idx=0):
    """
    Визуализация attention weights в Transformer.
    
    Args:
        model: BERT-like model
        tokenizer: соответствующий tokenizer
        text: входной текст
        layer_idx: индекс слоя (-1 для последнего)
        head_idx: индекс attention head
    """
    # Токенизация
    inputs = tokenizer(text, return_tensors="pt")
    
    # Forward pass с выводом attentions
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions  # Tuple of attention weights
    
    # Получение attention weights для выбранного слоя и головы
    attention = attentions[layer_idx][0, head_idx].detach().numpy()
    
    # Токены
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attention, cmap='viridis')
    
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    
    plt.colorbar(im)
    plt.title(f'Attention Weights (Layer {layer_idx}, Head {head_idx})')
    plt.tight_layout()
    plt.show()

# # Использование с BERT
# from transformers import BertTokenizer, BertModel
# 
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# 
# text = "The cat sat on the mat."
# visualize_attention(model, tokenizer, text, layer_idx=-1, head_idx=0)
```

### Counterfactual Explanations

"Что нужно изменить, чтобы предсказание изменилось?"

```python
def generate_counterfactual(model, image, target_class, current_class, 
                           lr=0.1, max_iterations=100, lambda_reg=0.01):
    """
    Генерация counterfactual explanation.
    
    Args:
        model: модель
        image: исходное изображение
        target_class: желаемый класс
        current_class: текущий класс
        lr: learning rate
        max_iterations: максимальное количество итераций
        lambda_reg: регуляризация (минимизируем изменения)
    
    Returns:
        counterfactual: измененное изображение
    """
    model.eval()
    
    # Копия изображения
    counterfactual = image.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([counterfactual], lr=lr)
    
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        
        output = model(counterfactual)
        
        # Loss: максимизировать вероятность target_class + минимизировать изменения
        target_loss = -output[0, target_class]
        l2_loss = torch.norm(counterfactual - image)
        
        loss = target_loss + lambda_reg * l2_loss
        loss.backward()
        optimizer.step()
        
        # Проверка успеха
        pred_class = output.argmax().item()
        if pred_class == target_class:
            print(f"Counterfactual found at iteration {iteration}")
            break
    
    return counterfactual.detach()

# # Использование
# original_pred = model(input_tensor).argmax().item()
# target = 243  # Bulldog class в ImageNet
# 
# counterfactual = generate_counterfactual(
#     model, input_tensor, target_class=target, current_class=original_pred
# )
# 
# # Визуализация изменений
# diff = (counterfactual - input_tensor).abs().squeeze().max(dim=0)[0]
# 
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(input_tensor.squeeze().permute(1, 2, 0).cpu())
# axes[0].set_title(f'Original (Class {original_pred})')
# axes[1].imshow(counterfactual.squeeze().permute(1, 2, 0).cpu())
# axes[1].set_title(f'Counterfactual (Class {target})')
# axes[2].imshow(diff.cpu(), cmap='hot')
# axes[2].set_title('Difference')
# plt.show()
```

## Ссылки

- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [SHAP](https://github.com/slundberg/shap)
- [LIME](https://github.com/marcotcr/lime)
- [Integrated Gradients Paper](https://arxiv.org/abs/1703.01365)
- [Captum (PyTorch Interpretability)](https://captum.ai/)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)

## Tools

- Captum: https://captum.ai/
- SHAP: https://github.com/slundberg/shap
- LIME: https://github.com/marcotcr/lime
- Alibi: https://github.com/SeldonIO/alibi
