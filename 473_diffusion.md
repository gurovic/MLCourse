# Диффузионные модели

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# !pip install torch numpy matplotlib
```

---

## 🟢 Базовый уровень: Основы диффузии

### 1.1 Что такое диффузионные модели?

**Идея:** Постепенно добавляем шум к изображению, затем учимся **обращать** этот процесс.

```
x₀ (real) → x₁ → x₂ → ... → xₜ (pure noise)
         ↑ Reverse process (denoising)
```

**Forward process (diffusion):** Добавляем Gaussian noise  
**Reverse process:** Нейросеть учится удалять шум

---

### 1.2 Forward Diffusion Process

Постепенно добавляем шум по расписанию:

```python
def forward_diffusion(x0, t, alphas):
    """
    x0: оригинальное изображение
    t: timestep (0 to T)
    alphas: noise schedule
    """
    alpha_t = alphas[t]
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
    
    # Добавляем шум
    noise = torch.randn_like(x0)
    xt = sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * noise
    
    return xt, noise
```

**Математика:**
```
q(xₜ | x₀) = N(xₜ; √(ᾱₜ)x₀, (1-ᾱₜ)I)
```

---

### 1.3 Noise Schedule

```python
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Линейное расписание шума"""
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod

timesteps = 1000
betas, alphas, alphas_cumprod = linear_beta_schedule(timesteps)
```

---

## 🟡 Продвинутый уровень: DDPM

### 2.1 Denoising Diffusion Probabilistic Models

**Обучение:** Нейросеть предсказывает **шум**, добавленный к изображению.

```python
class UNet(nn.Module):
    """Упрощенный U-Net для предсказания шума"""
    def __init__(self, in_channels=3, out_channels=3, time_dim=256):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.enc3 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Decoder
        self.dec3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(64, out_channels, 3, padding=1)
    
    def forward(self, x, t):
        # Encode
        h1 = F.relu(self.enc1(x))
        h2 = F.relu(self.enc2(h1))
        h3 = F.relu(self.enc3(h2))
        
        # Time conditioning (упрощенно)
        t_emb = self.time_mlp(t.float().unsqueeze(-1))
        
        # Decode
        h = F.relu(self.dec3(h3))
        h = F.relu(self.dec2(h))
        out = self.dec1(h)
        
        return out
```

---

### 2.2 Training DDPM

```python
def train_step(model, x0, timesteps=1000):
    # Случайный timestep
    t = torch.randint(0, timesteps, (x0.size(0),))
    
    # Forward diffusion
    xt, noise = forward_diffusion(x0, t, alphas_cumprod)
    
    # Предсказываем шум
    predicted_noise = model(xt, t)
    
    # Loss = MSE между real и predicted noise
    loss = F.mse_loss(predicted_noise, noise)
    
    return loss

# Training loop
model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

for epoch in range(100):
    for x0, _ in dataloader:
        loss = train_step(model, x0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

### 2.3 Sampling (Generation)

Начинаем с чистого шума, постепенно denoising:

```python
@torch.no_grad()
def ddpm_sample(model, shape, timesteps=1000):
    # Начинаем с pure noise
    x = torch.randn(shape)
    
    # Постепенно удаляем шум
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((shape[0],), t, dtype=torch.long)
        
        # Предсказываем шум
        predicted_noise = model(x, t_tensor)
        
        # Удаляем предсказанный шум
        alpha_t = alphas_cumprod[t]
        alpha_t_prev = alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
        
        beta_t = 1 - alpha_t / alpha_t_prev
        
        # Denoising step
        x = (x - beta_t * predicted_noise / torch.sqrt(1 - alpha_t)) / torch.sqrt(1 - beta_t)
        
        # Добавляем небольшой шум (кроме последнего шага)
        if t > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(beta_t) * noise
    
    return x

# Генерация
generated = ddpm_sample(model, shape=(16, 3, 32, 32))
```

---

## 🔴 Экспертный уровень: Stable Diffusion

### 3.1 Latent Diffusion Models

**Проблема:** Diffusion в pixel space медленный и требует много памяти.

**Решение:** Diffusion в **latent space** (как VAE).

```
Image → VAE Encoder → Latent z → Diffusion → VAE Decoder → Image
```

---

### 3.2 Conditioning

**Text-to-Image:** Conditioning на текст через CLIP embeddings.

```python
class ConditionalUNet(nn.Module):
    def forward(self, x, t, text_embedding):
        # Combine x с text_embedding
        # Cross-attention между image features и text
        pass
```

---

### 3.3 Classifier-Free Guidance

Улучшение качества conditional generation:

```python
# Обучаем с и без conditioning
noise_pred_uncond = model(x, t, text_emb=None)
noise_pred_cond = model(x, t, text_emb=text_emb)

# Guidance
guidance_scale = 7.5
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
```

---

## 🎯 Ключевые выводы

1. **Диффузионные модели** постепенно добавляют и удаляют шум
2. **DDPM** обучается предсказывать добавленный шум
3. **Sampling** — iterative denoising процесс (медленный)
4. **Latent diffusion** работает в compressed space для эффективности
5. **Stable Diffusion** = latent diffusion + text conditioning

---

## 📚 Материалы

- [DDPM Paper](https://arxiv.org/abs/2006.11239)
- [Improved DDPM](https://arxiv.org/abs/2102.09672)
- [Latent Diffusion (Stable Diffusion)](https://arxiv.org/abs/2112.10752)
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598)
