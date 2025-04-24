# 🚀 Early Exit ResNet50 for Heterogeneous Processing-in-Memory (HeteroPIM)

This project implements and evaluates **BranchyNet-style Early Exit architectures** on the **ResNet50 backbone** using the **CIFAR-100 dataset**. The work is part of an ongoing research thesis exploring efficiency tradeoffs in deep learning through Heterogeneous Processing-in-Memory (HeteroPIM) systems.

---

## 📌 Key Features

- **ResNet50 + Early Exit Branches**
  - Implements **3-exit** and **4-exit** variants.
  - Exits placed after major residual blocks.
- **Entropy-based dynamic exit logic**
  - Low entropy = confident → early exit.
- **Joint training with weighted loss**
  - All exits trained simultaneously for regularization.
- **Flexible evaluation across multiple thresholds**
  - Compare exit accuracy and distribution for different entropy thresholds.

---

## 📊 Dataset

- **CIFAR-100**: 100 classes, 50,000 train + 10,000 test samples
- Input images resized to **224×224** for compatibility with ResNet50.

---

## 🧠 Entropy-based Early Exits

Each exit branch computes entropy of its softmax output:

```math
H(y) = -\sum_{i=1}^{C} p_i \log(p_i)
