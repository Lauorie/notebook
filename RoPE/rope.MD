# ROPE (Rotary Position Embedding) 数学推导

## 从内积角度

设 $\vec{x}_m, \vec{x}_n$ 为 d 维向量，$\Theta$ 为位置编码矩阵，则有：

### 基本定义

$\vec{y}_m = W_q \vec{x}_m e^{im\theta} = (W_q \vec{x}_m) e^{im\theta} = \vec{q}_m e^{im\theta}$ (1)

$\vec{y}_n = W_k \vec{x}_n e^{in\theta} = (W_k \vec{x}_n) e^{in\theta} = \vec{q}_n e^{in\theta}$ (2)

### 内积计算

$\vec{y}_m^T \vec{y}_n = (\vec{q}_m^T \vec{q}_n) \begin{pmatrix} \cos((m-n)\theta) & -\sin((m-n)\theta) \\ \sin((m-n)\theta) & \cos((m-n)\theta) \end{pmatrix} \begin{pmatrix} k_1^T \\ k_2^T \end{pmatrix}$ (3)

展开可得同样的注意力计算公式

### 复数形式表达

$\vec{y}_m = W_q \vec{x}_m e^{im\theta} = (W_q \vec{x}_m) e^{im\theta} = \vec{q}_m e^{im\theta}$

$\therefore \vec{q}_m = \begin{pmatrix} W_q^1 \\ W_q^2 \end{pmatrix} \vec{x}_m = \begin{pmatrix} \vec{q}_m^1 \\ \vec{q}_m^2 \end{pmatrix}$

可以表示为复数形式：$\vec{q}_m = \vec{q}_m^1 + i\vec{q}_m^2$

### 欧拉公式

$(e^{im\theta} + e^{-im\theta}) = \cos(m\theta) + i\sin(m\theta)$

$(W_q \vec{x}_m) e^{im\theta} = \vec{q}_m e^{im\theta} = (\vec{q}_m^1 + i\vec{q}_m^2)(\cos(m\theta) + i\sin(m\theta))$

$= (\vec{q}_m^1 \cos(m\theta) - \vec{q}_m^2 \sin(m\theta)) + i(\vec{q}_m^2 \cos(m\theta) + \vec{q}_m^1 \sin(m\theta))$

$= \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} \vec{q}_m^1 \\ \vec{q}_m^2 \end{pmatrix}$

这就是旋转矩阵形式

### 注意力计算

$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$，V不计算 $QK^T$ 就能得到 (3) 式

## 矩阵形式

$\vec{y}_m = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} \vec{q}_m^1 \\ \vec{q}_m^2 \end{pmatrix}$

### 矩阵分解

$\begin{pmatrix} \cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ \sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & 0 \\ 0 & 0 & \sin m\theta_1 & \cos m\theta_1 & 0 & 0 & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\ 0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1} \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1} \end{pmatrix}$

矩阵可拆分为两个矩阵相乘：

$\begin{pmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1} \end{pmatrix} \otimes \begin{pmatrix} \cos m\theta_0 \\ \cos m\theta_0 \\ \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d/2-1} \\ \cos m\theta_{d/2-1} \end{pmatrix} + \begin{pmatrix} -q_1 \\ q_0 \\ -q_3 \\ q_2 \\ \vdots \\ -q_{d-1} \\ q_{d-2} \end{pmatrix} \otimes \begin{pmatrix} \sin m\theta_0 \\ \sin m\theta_0 \\ \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d/2-1} \\ \sin m\theta_{d/2-1} \end{pmatrix}$

$\theta_i = 10000^{-2i/d}, i \in [0, 1, \cdots, \frac{d}{2}-1]$

$\theta = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))$, float()

恒等式：$e^{-i\omega t} = \cos(\omega t) - i\sin(\omega t)$

实现时需要计算每种频率的 $m\theta_i$，对第 $m$ 个词，$m\theta_0, m\theta_1, \cdots, m\theta_{d/2-1}$ 要双倍，
对应第 $m$ 行，$m\theta_i$ 要双倍

$xb\_theta2 = torch.cat([xb\_theta[:, :, :, None], xb\_theta[:, :, :, None]], 3)$

$cos\_cached = xb\_theta2[:, :, :, 0].cos()[:, None, None, :]$
$sin\_cached = xb\_theta2[:, :, :, 0].sin()[:, None, None, :]$
