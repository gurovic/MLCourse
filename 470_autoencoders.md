# Autoencoders и VAE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# !pip install torch torchvision matplotlib
```

---

## 🟢 Базовый уровень: Autoencoders

### 1.1 Что такое Autoencoder?

**Autoencoder** — нейросеть для обучения сжатому представлению данных без учителя.

```
Input → [Encoder] → Latent (bottleneck) → [Decoder] → Reconstructed Output
```

**Цель:** Минимизировать разницу между входом и выходом.

---

### 1.2 Простой Autoencoder

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # для изображений [0, 1]
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def encode(self, x):
        return self.encoder(x)

# Training
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for data, _ in train_loader:
        data = data.view(data.size(0), -1)  # flatten
        
        # Forward
        reconstructed = model(data)
        loss = criterion(reconstructed, data)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

### 1.3 Применения Autoencoder

1. **Сжатие данных:** latent code = сжатое представление
2. **Denoising:** обучить восстанавливать чистое изображение из зашумленного
3. **Anomaly Detection:** высокая reconstruction loss = аномалия
4. **Feature Learning:** использовать encoder как feature extractor

---

## 🟡 Продвинутый уровень: Convolutional Autoencoder

### 2.1 Архитектура для изображений

```python
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28x28 → 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 14x14 → 7x7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)  # 7x7 → 1x1 (latent)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # 1x1 → 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 7x7 → 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),   # 14x14 → 28x28
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
```

---

## 🟡 Продвинутый уровень: Variational Autoencoder (VAE)

### 3.1 Проблема стандартного Autoencoder

- Latent space **не структурирован**
- Нельзя просто сэмплировать из latent space для генерации

**Решение:** VAE — вероятностная модель с структурированным латентным пространством.

---

### 3.2 VAE архитектура

```
Input → Encoder → [μ, log(σ²)] → Sample z ~ N(μ, σ²) → Decoder → Output
```

**Ключевая идея:** Encoder предсказывает **распределение** (μ, σ), а не одну точку.

---

### 3.3 Reparameterization Trick

Как сэмплировать z ~ N(μ, σ²) так, чтобы можно было backprop?

```python
# Reparameterization trick
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)  # ε ~ N(0, 1)
    z = mu + eps * std           # z ~ N(μ, σ)
    return z
```

---

### 3.4 VAE Implementation

```python
class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
```

---

### 3.5 VAE Loss Function

**Loss = Reconstruction Loss + KL Divergence**

```python
def vae_loss(reconstructed, original, mu, logvar):
    # Reconstruction loss (MSE or BCE)
    recon_loss = F.binary_cross_entropy(reconstructed, original, reduction='sum')
    
    # KL divergence: KL(N(μ, σ²) || N(0, 1))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss

# Training
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for data, _ in train_loader:
        data = data.view(data.size(0), -1)
        
        reconstructed, mu, logvar = model(data)
        loss = vae_loss(reconstructed, data, mu, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 🔴 Экспертный уровень: Генерация с VAE

### 4.1 Sampling новых данных

```python
# После обучения VAE
model.eval()

# Сэмплируем из prior N(0, 1)
z = torch.randn(64, latent_dim)

# Генерируем изображения
with torch.no_grad():
    generated = model.decode(z)
    generated = generated.view(-1, 1, 28, 28)

# Визуализация
plt.figure(figsize=(8, 8))
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.imshow(generated[i, 0], cmap='gray')
    plt.axis('off')
plt.show()
```

---

### 4.2 Интерполяция в латентном пространстве

```python
# Encode два изображения
z1 = model.encode(image1)[0]  # μ
z2 = model.encode(image2)[0]

# Интерполяция
alphas = torch.linspace(0, 1, 10)
for alpha in alphas:
    z = alpha * z1 + (1 - alpha) * z2
    generated = model.decode(z)
    # Показать generated
```

---

## 🎯 Ключевые выводы

1. **Autoencoder** — unsupervised learning сжатого представления
2. **VAE** — probabilistic autoencoder с структурированным latent space
3. **Reparameterization trick** позволяет обучать VAE с помощью backprop
4. **KL divergence** регуляризирует latent space к N(0, 1)
5. **VAE генерирует** новые данные через sampling из latent space

---

## 📚 Материалы

- [Autoencoding Variational Bayes](https://arxiv.org/abs/1312.6114) (VAE paper)
- [Tutorial on VAE](https://arxiv.org/abs/1606.05908)
