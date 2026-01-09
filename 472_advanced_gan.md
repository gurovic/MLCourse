# Продвинутые GAN архитектуры

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# !pip install torch torchvision
```

---

## 🟢 Базовый уровень: DCGAN

### 1.1 Deep Convolutional GAN

**DCGAN** — архитектура GAN с convolutional layers для лучшего качества изображений.

**Ключевые принципы:**
1. Заменить pooling на strided convolutions (D) и transposed convolutions (G)
2. Использовать BatchNorm в обеих сетях
3. Убрать fully connected hidden layers
4. ReLU в G (кроме output — Tanh)
5. LeakyReLU в D

---

### 1.2 DCGAN Generator

```python
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_channels=1):
        super().__init__()
        
        self.model = nn.Sequential(
            # Input: latent_dim
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),  # 1x1 → 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 4x4 → 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 8x8 → 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, num_channels, 4, 2, 1, bias=False),  # 16x16 → 32x32
            nn.Tanh()
        )
    
    def forward(self, z):
        # z shape: [batch, latent_dim, 1, 1]
        return self.model(z)
```

---

### 1.3 DCGAN Discriminator

```python
class DCGANDiscriminator(nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        
        self.model = nn.Sequential(
            # Input: 32x32
            nn.Conv2d(num_channels, 128, 4, 2, 1, bias=False),  # 32x32 → 16x16
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 16x16 → 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 8x8 → 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),  # 4x4 → 1x1
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1)
```

---

## 🟡 Продвинутый уровень: Wasserstein GAN (WGAN)

### 2.1 Проблемы стандартного GAN

- **Vanishing gradients** когда D слишком хорош
- **Mode collapse**
- Нет хорошей метрики для convergence

**Решение:** WGAN использует Wasserstein distance вместо JS divergence.

---

### 2.2 WGAN Loss

**Вместо:**
```python
d_loss = -(log D(x) + log(1 - D(G(z))))
g_loss = -log D(G(z))
```

**WGAN:**
```python
d_loss = -E[D(x)] + E[D(G(z))]  # Critic loss
g_loss = -E[D(G(z))]            # Generator loss
```

**Важно:** D теперь называется **Critic** (не выводит вероятность).

---

### 2.3 Weight Clipping

Для поддержания Lipschitz constraint в WGAN:

```python
# После каждого update Discriminator
for p in discriminator.parameters():
    p.data.clamp_(-0.01, 0.01)  # clip weights to [-0.01, 0.01]
```

---

### 2.4 WGAN Training

```python
# Train Discriminator (Critic) more often
for _ in range(5):  # 5 critic updates per generator update
    # Real
    d_real = discriminator(real_images)
    # Fake
    z = torch.randn(batch_size, latent_dim, 1, 1)
    fake_images = generator(z)
    d_fake = discriminator(fake_images.detach())
    
    # WGAN loss
    d_loss = -(d_real.mean() - d_fake.mean())
    
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()
    
    # Weight clipping
    for p in discriminator.parameters():
        p.data.clamp_(-0.01, 0.01)

# Train Generator
z = torch.randn(batch_size, latent_dim, 1, 1)
fake_images = generator(z)
d_fake = discriminator(fake_images)
g_loss = -d_fake.mean()

g_optimizer.zero_grad()
g_loss.backward()
g_optimizer.step()
```

---

## 🟡 Продвинутый уровень: Progressive GAN

### 3.1 Идея Progressive Growing

Постепенно увеличиваем разрешение изображений: 4x4 → 8x8 → 16x16 → ... → 1024x1024

**Преимущества:**
- Более стабильное обучение
- Высокое разрешение (1024x1024 и выше)
- Меньше mode collapse

---

### 3.2 Fade-in новых слоев

```
Stage 1: Train 4x4
Stage 2: Add 8x8 layers, gradually fade in (α from 0 to 1)
Stage 3: Add 16x16 layers, fade in
...
```

---

## 🔴 Экспертный уровень: StyleGAN

### 4.1 Архитектура StyleGAN

**Ключевые инновации:**
1. **Mapping network:** z → w (более disentangled latent space)
2. **Adaptive Instance Normalization (AdaIN):** inject style в каждом слое
3. **Stochastic variation:** добавляем noise для мелких деталей

```
z → MappingNetwork → w
                      ↓
        Generator with AdaIN(w) at each layer
```

---

### 4.2 Style Mixing

```python
# Генерация с двумя стилями
w1 = mapping_network(z1)
w2 = mapping_network(z2)

# Используем w1 для coarse layers (низкое разрешение)
# Используем w2 для fine layers (высокое разрешение)
```

**Результат:** Лицо одного человека со стилем волос другого.

---

### 4.3 Truncation Trick

Для улучшения качества (за счет разнообразия):

```python
w_avg = compute_average_w(many_samples)

# Truncation
w_truncated = w_avg + psi * (w - w_avg)  # psi ∈ [0, 1]
```

**psi=0:** все изображения одинаковые (среднее)  
**psi=1:** полное разнообразие  
**psi=0.7:** хороший баланс качества и разнообразия

---

## 🎯 Ключевые выводы

1. **DCGAN** — convolutional architecture для лучших изображений
2. **WGAN** — Wasserstein distance для стабильного обучения
3. **Progressive GAN** — постепенный рост разрешения
4. **StyleGAN** — control стиля через latent space manipulation

---

## 📚 Материалы

- [DCGAN Paper](https://arxiv.org/abs/1511.06434)
- [WGAN Paper](https://arxiv.org/abs/1701.07875)
- [Progressive GAN](https://arxiv.org/abs/1710.10196)
- [StyleGAN](https://arxiv.org/abs/1812.04948)
- [StyleGAN2](https://arxiv.org/abs/1912.04958)
