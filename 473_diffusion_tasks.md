### **Задачи: Диффузионные модели**

**Цель:** Реализовать и обучить диффузионную модель для генерации изображений.

---

## 🟢 Базовый уровень

### **Задача 1: Forward Diffusion**

Реализуйте forward diffusion process.

```python
def forward_diffusion(x0, t, alphas_cumprod):
    # TODO: добавьте шум согласно timestep t
    pass
```

**Требования:** Визуализируйте изображение на разных timesteps (t=0, 100, 500, 999).

---

### **Задача 2: Noise Schedule**

Реализуйте и сравните разные noise schedules: linear, cosine.

**Требования:** Визуализируйте α_t для обоих schedules.

---

### **Задача 3: Simple Denoising Network**

Обучите простую сеть предсказывать шум.

```python
class SimpleDenoiser(nn.Module):
    def forward(self, x, t):
        # TODO: predict noise
        pass
```

---

## 🟡 Продвинутый уровень

### **Задача 4: DDPM на MNIST**

Реализуйте полный DDPM для генерации MNIST цифр.

**Требования:** Обучите, сгенерируйте 64 изображения, оцените качество.

---

### **Задача 5: U-Net для Diffusion**

Реализуйте U-Net архитектуру с time conditioning.

---

### **Задача 6: DDPM Sampling**

Реализуйте sampling process для генерации новых изображений.

**Требования:** Визуализируйте denoising process (каждые 100 steps).

---

## 🔴 Экспертный уровень

### **Задача 7: DDIM Sampling**

Реализуйте DDIM (Denoising Diffusion Implicit Models) — быстрый sampling.

---

### **Задача 8: Conditional Diffusion**

Реализуйте conditional diffusion для class-guided generation.

---

### **Задача 9: Latent Diffusion**

Реализуйте diffusion в latent space (используйте pre-trained VAE).

---

### **Задача 10: Text-to-Image (упрощенный)**

Реализуйте упрощенную версию text-to-image diffusion.

**Используйте:** Pre-trained text encoder (CLIP), conditional U-Net.

---

## 📚 Ресурсы

- [DDPM Paper](https://arxiv.org/abs/2006.11239)
- [DDIM Paper](https://arxiv.org/abs/2010.02502)
- [Stable Diffusion](https://arxiv.org/abs/2112.10752)
- [Annotated Diffusion](https://huggingface.co/blog/annotated-diffusion)
