### **Задачи: GAN**

**Цель:** Реализовать и обучить GAN для генерации изображений.

---

## 🟢 Базовый уровень

### **Задача 1: Simple GAN**

Реализуйте простой fully-connected GAN для MNIST.

```python
class Generator(nn.Module):
    # TODO: implement
    pass

class Discriminator(nn.Module):
    # TODO: implement
    pass
```

**Требования:** Обучите 50 epochs, визуализируйте сгенерированные изображения.

---

### **Задача 2: Training Monitoring**

Мониторьте обучение GAN: D loss, G loss, D accuracy на real/fake.

**Требования:** Постройте графики, проанализируйте динамику обучения.

---

### **Задача 3: Conditional GAN**

Реализуйте conditional GAN — генерация цифры по заданному классу.

**Требования:** Сгенерируйте все цифры 0-9 по команде.

---

## 🟡 Продвинутый уровень

### **Задача 4: Mode Collapse Detection**

Обучите GAN и проверьте на mode collapse.

**Требования:** Посчитайте, сколько уникальных классов генерирует модель.

---

### **Задача 5: Stabilization Techniques**

Примените техники стабилизации: label smoothing, one-sided label smoothing.

**Сравните:** обучение с и без стабилизации.

---

### **Задача 6: Inception Score**

Вычислите Inception Score для сгенерированных изображений.

---

## 🔴 Экспертный уровень

### **Задача 7: DCGAN**

Реализуйте Deep Convolutional GAN (будет в следующей главе).

---

### **Задача 8: Latent Space Interpolation**

Интерполируйте в latent space GAN между двумя случайными векторами.

---

### **Задача 9: GAN для цветных изображений**

Обучите GAN на CIFAR-10 (цветные изображения 32x32).

---

### **Задача 10: FID вычисление**

Реализуйте вычисление Fréchet Inception Distance.

---

## 📚 Ресурсы

- [GAN Paper](https://arxiv.org/abs/1406.2661)
- [GAN Hacks](https://github.com/soumith/ganhacks)
