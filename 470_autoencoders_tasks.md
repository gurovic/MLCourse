### **Задачи: Autoencoders и VAE**

**Цель:** Реализовать и обучить autoencoder и VAE для различных задач.

---

## 🟢 Базовый уровень

### **Задача 1: Simple Autoencoder**

Реализуйте простой fully-connected autoencoder для MNIST.

```python
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # TODO: создайте encoder и decoder
        pass
```

**Требования:** Обучите, визуализируйте оригиналы и реконструкции, измерьте MSE.

---

### **Задача 2: Convolutional Autoencoder**

Реализуйте convolutional autoencoder для изображений.

**Требования:** Используйте Conv2d/ConvTranspose2d, сравните с fully-connected версией.

---

### **Задача 3: Denoising Autoencoder**

Обучите autoencoder удалять шум из изображений.

```python
# Добавьте шум
noisy_data = data + 0.3 * torch.randn_like(data)
noisy_data = torch.clamp(noisy_data, 0, 1)

# Обучите восстанавливать оригинал из зашумленного
```

---

## 🟡 Продвинутый уровень

### **Задача 4: VAE Implementation**

Реализуйте полный VAE с reparameterization trick.

**Требования:** Правильно вычислите loss (reconstruction + KL), обучите на MNIST.

---

### **Задача 5: VAE Generation**

Используйте обученный VAE для генерации новых изображений.

**Требования:** Сэмплируйте из N(0, 1), сгенерируйте 100 изображений, оцените качество.

---

### **Задача 6: Latent Space Visualization**

Визуализируйте 2D latent space VAE (latent_dim=2).

**Требования:** Scatter plot всех данных в latent space, раскрасьте по классам.

---

## 🔴 Экспертный уровень

### **Задача 7: Conditional VAE**

Реализуйте conditional VAE (CVAE) — генерация с условием (класс цифры).

---

### **Задача 8: β-VAE**

Реализуйте β-VAE (weighted KL divergence) для disentangled representations.

```python
loss = recon_loss + beta * kl_loss  # beta > 1
```

---

### **Задача 9: Anomaly Detection**

Используйте autoencoder для обнаружения аномалий в данных.

**Требования:** Обучите на нормальных данных, измерьте reconstruction error на аномалиях.

---

### **Задача 10: Image Compression**

Используйте autoencoder для сжатия изображений.

**Измерьте:** compression ratio, reconstruction quality (SSIM, PSNR).

---

## 📚 Ресурсы

- [VAE Paper](https://arxiv.org/abs/1312.6114)
- [β-VAE Paper](https://openreview.net/forum?id=Sy2fzU9gl)
