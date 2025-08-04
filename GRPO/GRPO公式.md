GRPO 目标函数

$$ 
\mathcal{J}_{\text{GRPO}}(\theta) = 
\mathbb{E}_{\substack{q \sim P(Q), \\ \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O \mid q)}}
\left[
\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
\left(
\min \left[
\frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})} \hat{A}_{i,t},\,
\text{clip}\left( \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}, 1-\varepsilon, 1+\varepsilon \right) \hat{A}_{i,t}
\right]
- \beta D_{\text{KL}}\left( \pi_\theta(\cdot \mid q, o_{i,<t}) \parallel \pi_{\text{ref}} \right)
\right)
\right]
$$

其中优势函数定义为

$ \hat{A}{i,t} = \frac{R_i - \text{mean}({R_j}{j=1}^G)}{\text{std}({R_j}_{j=1}^G)} $

这个目标函数 $\mathcal{J}_{\text{GRPO}}(\theta)$ 是广义强化学习策略优化（Generalized Reinforcement Learning with Policy Optimization, GRPO）的目标函数，通常用于通过最大化期望奖励来优化策略参数 $\theta$。在大型语言模型（LLM）的对齐和微调任务中，这种形式的目标函数非常常见（例如在RLHF中）。

这个目标函数结合了**近端策略优化（PPO）**的思想和**Kullback-Leibler（KL）散度正则化**。

下面我们详细解读这个目标函数的结构和各个组成部分。

### 1. 目标函数的整体结构

$$ 
\mathcal{J}_{\text{GRPO}}(\theta) = 
\mathbb{E}_{\substack{q \sim P(Q), \\ \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O \mid q)}}
\left[
\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
\left(
\dots
\right)
\right]
$$

*   **$\mathbb{E}[\dots]$ (期望):** 目标函数是基于期望值计算的。
*   **采样来源 ($P(Q)$ 和 $\pi_{\theta_{\text{old}}}$):** 期望是针对两个分布采样的：
    *   **$q \sim P(Q)$:** 从查询（Query）分布中采样一个查询 $q$。
    *   **$\{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O \mid q)$:** 在给定查询 $q$ 的条件下，使用旧策略 $\pi_{\theta_{\text{old}}}$ 采样一批 $G$ 个输出（轨迹/响应）$\{o_i\}_{i=1}^G$。
*   **平均计算 ($\frac{1}{G} \sum_{i=1}^G$ 和 $\frac{1}{|o_i|} \sum_{t=1}^{|o_i|}$):** 目标函数计算的是在整个批次中，每个轨迹的每个时间步上的平均目标值。
*   **$\theta$ (新策略参数):** 我们要优化的参数，对应着新策略 $\pi_\theta$。
*   **$\theta_{\text{old}}$ (旧策略参数):** 对应着用于数据采样的旧策略 $\pi_{\theta_{\text{old}}}$。

### 2. 核心目标项的组成

目标函数的核心部分是在每个时间步 $t$ 上的表达式：

$$
\left(
\min \left[
\frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})} \hat{A}_{i,t},\,
\text{clip}\left( \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}, 1-\varepsilon, 1+\varepsilon \right) \hat{A}_{i,t}
\right]
- \beta D_{\text{KL}}\left( \pi_\theta(\cdot \mid q, o_{i,<t}) \parallel \pi_{\text{ref}} \right)
\right)
$$

这个表达式由两大部分组成：一个PPO-like的奖励最大化项，和一个KL正则化项。

#### A. PPO-like 策略梯度项 (第一部分)

这一部分旨在最大化奖励。

$$
\min \left[
\frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})} \hat{A}_{i,t},\,
\text{clip}\left( \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}, 1-\varepsilon, 1+\varepsilon \right) \hat{A}_{i,t}
\right]
$$

*   **$\frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$ (概率比率 $r_t(\theta)$):**
    *   这是新策略 $\pi_\theta$ 生成当前时间步的令牌 $o_{i,t}$ 的概率，与旧策略 $\pi_{\theta_{\text{old}}}$ 生成该令牌的概率之比。
    *   这个比率是重要性采样（Importance Sampling）的关键，它允许我们使用旧策略采集的数据来训练新策略。
*   **$\hat{A}_{i,t}$ (优势函数):**
    *   表示在时间步 $t$ 执行动作 $o_{i,t}$ 相比于平均水平的优势。
*   **`min` 函数与 `clip` 函数 (PPO 剪裁):**
    *   这是 PPO 算法的核心机制。它旨在防止新策略 $\pi_\theta$ 相对于旧策略 $\pi_{\theta_{\text{old}}}$ 变化过大。
    *   **$\text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)$:** 将概率比率 $r_t(\theta)$ 限制在 $[1-\varepsilon, 1+\varepsilon]$ 范围内。$\varepsilon$ 是一个超参数，通常取 0.1 或 0.2。
    *   **剪裁目标:**
        *   如果 $\hat{A}_{i,t} > 0$（优势为正，动作好）：我们希望提高该动作的概率。`min` 函数会选择未剪裁项和剪裁项中较小的一个。如果新策略的概率比率 $r_t(\theta)$ 超过 $1+\varepsilon$，则剪裁会限制目标函数的增长，从而限制策略的变化。
        *   如果 $\hat{A}_{i,t} < 0$（优势为负，动作差）：我们希望降低该动作的概率。`min` 函数会选择未剪裁项和剪裁项中较小的一个。如果新策略的概率比率 $r_t(\theta)$ 低于 $1-\varepsilon$，则剪裁会限制目标函数的下降，从而防止新策略过度降低该动作的概率。
    *   **总结:** 这一项旨在最大化优势，同时通过剪裁来保持新旧策略的接近性，以提高训练稳定性。

##### 深入理解PPO剪裁项
为了深刻理解PPO的剪裁（clip）机制，我们需要一步步拆解，并思考**梯度**是如何作用于策略参数的。

我们先忘掉复杂的公式，记住一个核心目标：

**策略梯度（Policy Gradient）的核心思想是：如果一个动作带来了好的结果（正优势），我们就调整策略，让这个动作的概率变高；如果一个动作带来了坏的结果（负优势），我们就调整策略，让这个动作的概率变低。**

但这里有一个巨大的风险：如果某次采样偶然获得了一个非常高的奖励，一个“天真的”策略梯度算法可能会进行一次巨大的、激进的更新，极大地提高这个动作的概率。这可能导致策略崩溃，因为它会忽略其他所有可能性，变得非常“短视”和不稳定。

PPO的剪裁机制就是为了解决这个问题而设计的。它像一个** “信任区域”的刹车系统 **，允许策略更新，但会阻止更新的步子迈得太大。

---

### 深入理解PPO剪裁项

我们聚焦于这个核心部分：
$$ L_{\text{CLIP}}(\theta) = \min \left[ \underbrace{r_t(\theta) \hat{A}_{i,t}}_{\text{无约束目标 (Unconstrained Objective)}} ,\, \underbrace{\text{clip}\left( r_t(\theta), 1-\varepsilon, 1+\varepsilon \right) \hat{A}_{i,t}}_{\text{有约束目标 (Constrained Objective)}} \right] $$

其中 $ r_t(\theta) = \frac{\pi_\theta(o_{i,t} \mid \dots)}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid \dots)} $ 是新旧策略的概率比。

*   $r_t(\theta) > 1$ 意味着新策略更倾向于选择这个动作。
*   $r_t(\theta) < 1$ 意味着新策略更不倾向于选择这个动作。

现在，我们分两种情况来分析，这两种情况的行为是完全不同的，这也是理解的关键。

---

### Case 1: 优势为正 ($\hat{A}_{i,t} > 0$) —— 这是一个“好”动作

**我们的目标：** 提高这个动作的概率，即**增大 $r_t(\theta)$**。

此时，目标函数 $L_{\text{CLIP}}$ 是两个正数中的较小值。

*   **无约束目标:** $r_t(\theta) \hat{A}_{i,t}$。这是一个线性斜坡。我们越是提高 $r_t(\theta)$，这个目标值就越大，我们获得的“梯度奖励”也越大。这看起来很棒，但也很危险，因为它没有上限。

*   **有约束目标:** $\text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_{i,t}$。这里的 `clip` 函数是关键。
    *   当 $r_t(\theta)$ 在 $[1, 1+\varepsilon]$ 区间内时, `clip` 不起作用，有约束目标等于无约束目标。
    *   当 $r_t(\theta)$ **超过** $1+\varepsilon$ 时, `clip` 会把 $r_t(\theta)$ “摁”回到 $1+\varepsilon$。这时，有约束目标就变成了 $(1+\varepsilon) \hat{A}_{i,t}$，**变成了一个常数！**

**`min` 函数的作用：**

1.  **在“信任区域”内 ($1 \le r_t(\theta) \le 1+\varepsilon$)**:
    两个目标项是相等的，所以 `min` 函数不起作用。目标函数就是 $r_t(\theta) \hat{A}_{i,t}$。梯度会正常地推动 $r_t(\theta)$ 增大。

2.  **当策略更新“过于激进”时 ($r_t(\theta) > 1+\varepsilon$)**:
    *   无约束目标是 $r_t(\theta) \hat{A}_{i,t}$。
    *   有约束目标是 $(1+\varepsilon) \hat{A}_{i,t}$。
    *   因为 $r_t(\theta) > 1+\varepsilon$ 且 $\hat{A}_{i,t} > 0$，所以 $r_t(\theta) \hat{A}_{i,t} > (1+\varepsilon) \hat{A}_{i,t}$。
    *   `min` 函数会选择**有约束目标**，即 $(1+\varepsilon) \hat{A}_{i,t}$。

**深刻的含义：**
一旦新策略相比旧策略对这个“好”动作的偏好超过了 $1+\varepsilon$ 的门槛，目标函数就**不再增长**了。它变成了一个平顶（flat top）。一个平坦的目标函数意味着**梯度为零**。

**这就像一个刹车：** 算法被告知：“你已经足够奖励这个好动作了，再继续提高它的概率不会给你带来更多好处，停下来吧！” 这就有效地阻止了策略因为一次好的经验而进行过大的、破坏性的更新。

**图示 (A > 0):**

```
      Objective Value
            ^
            |
            |          /------------------  <-- Clipped Objective (flat top, zero gradient)
            |         /
            |        /
            |       /   <-- Normal objective increase
            +------/----------------------> r_t(θ)
            1    1+ε
```

---

### Case 2: 优势为负 ($\hat{A}_{i,t} < 0$) —— 这是一个“坏”动作

**我们的目标：** 降低这个动作的概率，即**减小 $r_t(\theta)$**。

此时，目标函数 $L_{\text{CLIP}}$ 是两个负数中的较小值（即绝对值更大的那个）。我们要最大化这个目标，就是要让它**尽可能接近零**。

*   **无约束目标:** $r_t(\theta) \hat{A}_{i,t}$。当我们减小 $r_t(\theta)$ 时，这个负值会变大（更接近零），这符合我们的目标。但同样，它没有下限，我们可能会过度惩罚一个动作。

*   **有约束目标:** $\text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_{i,t}$。
    *   当 $r_t(\theta)$ 在 $[1-\varepsilon, 1]$ 区间内时，`clip` 不起作用。
    *   当 $r_t(\theta)$ **低于** $1-\varepsilon$ 时, `clip` 会把 $r_t(\theta)$ “抬”回到 $1-\varepsilon$。这时，有约束目标就变成了 $(1-\varepsilon) \hat{A}_{i,t}$，**也变成了一个常数！**

**`min` 函数的作用：**

1.  **在“信任区域”内 ($1-\varepsilon \le r_t(\theta) \le 1$)**:
    两个目标项相等，梯度正常地推动 $r_t(\theta)$ 减小。

2.  **当策略更新“过于激进”时 ($r_t(\theta) < 1-\varepsilon$)**:
    *   无约束目标是 $r_t(\theta) \hat{A}_{i,t}$。
    *   有约束目标是 $(1-\varepsilon) \hat{A}_{i,t}$。
    *   因为 $r_t(\theta) < 1-\varepsilon$ 且 $\hat{A}_{i,t} < 0$，所以 $r_t(\theta) \hat{A}_{i,t} > (1-\varepsilon) \hat{A}_{i,t}$ (一个更小的正数乘以一个负数，结果是更接近零的负数)。
    *   `min` 函数会选择那个**更负**的值，即**有约束目标** $(1-\varepsilon) \hat{A}_{i,t}$。

**深刻的含义：**
一旦新策略对这个“坏”动作的厌恶程度超过了 $1-\varepsilon$ 的门槛（即概率比率降得太低），目标函数就会被锁定在一个恒定的惩罚值上。优化器会发现，再继续降低这个动作的概率，并不会让整体目标函数变得更好。

**这就像一个惩罚的下限：** 算法被告知：“你已经足够惩罚这个坏动作了，没必要赶尽杀绝，过度惩罚也不会让你获得更多好处。” 这保留了策略一定的探索性，防止因一次糟糕的经验而完全放弃某个动作。

---

### 总结与类比

**把 PPO 剪裁想象成一个有“护栏”的优化通道：**

*   **对于好动作 (A > 0):** 你可以在通道内自由地往“增加概率”的方向跑，但你不能冲破 `1+ε` 这道上限护栏。一旦撞上护栏，你再跑也没有用了（梯度消失）。
*   **对于坏动作 (A < 0):** 你可以在通道内自由地往“降低概率”的方向跑，但你不能冲破 `1-ε` 这道下限护栏。一旦撞上护栏，惩罚就不会再加重了。

这种机制通过一个简单的 `min` 和 `clip` 函数，非常优雅地实现了对策略更新幅度的限制，使得我们可以用同一批采样数据进行多轮（epoch）的梯度更新，而不用担心策略跑得太偏，从而极大地提高了数据利用效率和训练稳定性。





#### B. KL 散度正则化项 (第二部分)

$$
- \beta D_{\text{KL}}\left( \pi_\theta(\cdot \mid q, o_{i,<t}) \parallel \pi_{\text{ref}} \right)
$$

*   **$D_{\text{KL}}(\cdot \parallel \cdot)$ (KL 散度):**
    *   衡量新策略 $\pi_\theta$ 的分布与参考策略 $\pi_{\text{ref}}$ 的分布之间的差异（距离）。
*   **$\pi_{\text{ref}}$ (参考策略):**
    *   通常是初始模型（例如预训练模型或基线模型），它代表了模型生成高质量、安全、符合人类偏好文本的能力。
*   **$\beta$ (超参数):**
    *   KL 惩罚项的权重。
*   **作用:**
    *   **惩罚偏差:** 这一项旨在惩罚新策略 $\pi_\theta$ 偏离参考策略 $\pi_{\text{ref}}$ 过远。
    *   **保持质量与安全:** 在通过 PPO 优化奖励的同时，KL 正则化项确保模型不会为了追求高奖励而牺牲文本质量或生成不良内容（例如，产生“幻觉”或不连贯的文本）。

### 3. 优势函数 $\hat{A}_{i,t}$

$$
\hat{A}{i,t} = \frac{R_i - \text{mean}(R_j)}{\text{std}(R_j)}
$$

*   **$R_i$ (轨迹奖励):** 轨迹 $o_i$ 获得的最终奖励（例如，由奖励模型计算的得分）。
*   **$\text{mean}(R_j)$ (批次奖励均值):** 当前批次中所有轨迹奖励的平均值。
*   **$\text{std}(R_j)$ (批次奖励标准差):** 当前批次中所有轨迹奖励的标准差。
*   **作用:**
    *   **归一化 (Normalization):** 这是一个归一化的优势函数。它计算的是轨迹 $i$ 的奖励与当前批次平均奖励之间的差异，并除以标准差。
    *   **稳定训练:** 这种归一化有助于稳定训练过程，无论奖励的绝对值大小如何，都能确保优势函数的尺度是合理的。

### 总结

GRPO 目标函数 $\mathcal{J}_{\text{GRPO}}(\theta)$ 是一个用于策略优化的复合目标函数：

1.  **最大化奖励 (PPO 剪裁项):** 旨在通过重要性采样和 PPO 剪裁机制，最大化新策略在采样数据上的期望优势。
2.  **保持策略稳定 (KL 正则化项):** 旨在通过 KL 散度惩罚新策略偏离参考策略的程度，确保优化过程不会损害模型的整体生成质量和安全性。

这个目标函数在 LLM 的对齐训练中非常有效，它平衡了**追求奖励**和**保持生成质量**这两个核心目标。