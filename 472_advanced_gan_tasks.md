### **Задачи: Продвинутые GAN**

**Цель:** Реализовать и обучить продвинутые GAN архитектуры.

---

## 🟢 Базовый уровень

### **Задача 1: DCGAN Implementation**

Реализуйте DCGAN для генерации изображений MNIST/CIFAR-10.

```python
class DCGANGenerator(nn.Module):
    # TODO: ConvTranspose2d layers
    pass

class DCGANDiscriminator(nn.Module):
    # TODO: Conv2d layers
    pass
```

**Требования:** Обучите, сравните качество с простым GAN.

---

### **Задача 2: DCGAN Hyperparameters**

Поэкспериментируйте с hyperparameters DCGAN.

**Варьируйте:** learning rate, batch size, latent dim, number of filters.

---

### **Задача 3: Conditional DCGAN**

Добавьте conditioning к DCGAN для контролируемой генерации.

---

## 🟡 Продвинутый уровень

### **Задача 4: WGAN Implementation**

Реализуйте WGAN с weight clipping.

**Требования:** Сравните стабильность обучения с vanilla GAN и DCGAN.

---

### **Задача 5: WGAN-GP**

Реализуйте WGAN с gradient penalty вместо weight clipping.

```python
# Gradient penalty
def compute_gradient_penalty(D, real, fake):
    alpha = torch.rand(batch_size, 1, 1, 1)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)
    
    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(d_interpolates, interpolates, ...)[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
```

---

### **Задача 6: CelebA Generation**

Обучите DCGAN на CelebA dataset для генерации лиц.

**Требования:** 64x64 изображения, визуализируйте interpolation в latent space.

---

## 🔴 Экспертный уровень

### **Задача 7: Progressive Growing**

Реализуйте progressive growing GAN (упрощенная версия).

**Этапы:** 4x4 → 8x8 → 16x16 → 32x32

---

### **Задача 8: Style Mixing**

Реализуйте style mixing для StyleGAN-like архитектуры.

---

### **Задача 9: High Resolution Generation**

Обучите GAN для генерации изображений высокого разрешения (128x128+).

---

### **Задача 10: Image-to-Image Translation**

Реализуйте Pix2Pix или CycleGAN (упрощенная версия).

---

## 📚 Ресурсы

- [DCGAN Paper](https://arxiv.org/abs/1511.06434)
- [WGAN Paper](https://arxiv.org/abs/1701.07875)
- [WGAN-GP Paper](https://arxiv.org/abs/1704.00028)
- [StyleGAN Paper](https://arxiv.org/abs/1812.04948)
