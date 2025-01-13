# Label Smoothing（标签平滑）

Label smoothing 是一种正则化技术，主要用于分类任务中，目的是防止模型对训练数据中的标签过于自信，从而提升模型的泛化能力。它通过调整标签的分布，使得模型在训练时不会过度拟合训练数据。

## 背景

在标准的分类任务中，通常使用 **one-hot 编码** 来表示标签。例如，对于一个三分类问题，标签可能是 `[1, 0, 0]`，表示样本属于第一类。模型的目标是最大化正确类别的概率，这可能导致模型对预测结果过于自信，尤其是在训练数据有限或存在噪声的情况下。

## Label Smoothing 的原理

Label smoothing 的核心思想是 **软化** one-hot 标签，使其不再是非 0 即 1 的极端分布。具体来说，它会将正确类别的概率稍微降低，同时将其他类别的概率稍微提高。

![image](https://github.com/user-attachments/assets/3a5b7756-a6e5-4b3d-af0d-ccc42f0a68f8)


![image](https://github.com/user-attachments/assets/c961038d-1182-48af-8cda-9748cc7b1401)


## 应用场景

Label smoothing 广泛应用于深度学习中的分类任务，尤其是在图像分类、自然语言处理等领域。例如，在训练 ImageNet 分类模型时，label smoothing 被用来提升模型的泛化能力。

---

**总结**：Label smoothing 是一种简单但有效的正则化方法，通过软化标签分布，帮助模型更好地泛化到未见过的数据。
