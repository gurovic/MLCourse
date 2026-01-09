### **Задачи: Graph Neural Networks**

**Цель:** Реализовать и обучить GNN для различных задач на графах.

---

## 🟢 Базовый уровень

### **Задача 1: Graph Representation**

Создайте и визуализируйте граф с node features.

```python
import networkx as nx
import matplotlib.pyplot as plt

# TODO: создайте граф
# TODO: добавьте node features
# TODO: визуализируйте
```

**Требования:** Adjacency matrix, node features, визуализация.

---

### **Задача 2: Simple Message Passing**

Реализуйте простой message passing layer вручную.

**Требования:** Aggregate (sum) + Update (linear layer + ReLU).

---

### **Задача 3: Node Classification с GCN**

Обучите GCN для node classification на Cora dataset.

**Требования:** Используйте PyTorch Geometric, достигните accuracy > 70%.

---

## 🟡 Продвинутый уровень

### **Задача 4: GCN от нуля**

Реализуйте полный GCN layer с нормализацией без PyTorch Geometric.

---

### **Задача 5: GAT Implementation**

Реализуйте Graph Attention Network с multi-head attention.

---

### **Задача 6: Сравнение GCN vs GAT**

Сравните GCN и GAT на одном dataset.

**Измерьте:** accuracy, training time, attention weights visualization (для GAT).

---

## 🔴 Экспертный уровень

### **Задача 7: Graph Classification**

Реализуйте GNN для graph-level classification.

**Требования:** Используйте graph pooling (global mean/max/sum).

---

### **Задача 8: Link Prediction**

Используйте GNN для предсказания будущих связей в графе.

---

### **Задача 9: Molecule Property Prediction**

Обучите GNN предсказывать свойства молекул (dataset: QM9 или ZINC).

---

### **Задача 10: Heterogeneous Graphs**

Реализуйте GNN для heterogeneous графов (разные типы вершин/рёбер).

---

## 📚 Ресурсы

- [PyTorch Geometric Tutorials](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html)
- [GCN Paper](https://arxiv.org/abs/1609.02907)
- [GAT Paper](https://arxiv.org/abs/1710.10903)
- [Cora Dataset](https://relational.fit.cvut.cz/dataset/CORA)
