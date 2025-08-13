# GRPO 目标函数详解

## 目标函数定义

广义强化学习策略优化（Generalized Reinforcement Learning with Policy Optimization, GRPO）的目标函数如下：

<img src="https://latex.codecogs.com/svg.image?\mathcal{J}_{\text{GRPO}}(\theta)%20=%20\mathbb{E}_{\substack{q%20\sim%20P(Q),%20\\%20\{o_i\}_{i=1}^G%20\sim%20\pi_{\theta_{\text{old}}}(O%20\mid%20q)}}%20\left[%20\frac{1}{G}%20\sum_{i=1}^G%20\frac{1}{|o_i|}%20\sum_{t=1}^{|o_i|}%20\left(%20\min%20\left[%20\frac{\pi_\theta(o_{i,t}%20\mid%20q,%20o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}%20\mid%20q,%20o_{i,<t})}%20\hat{A}_{i,t},\,%20\text{clip}\left(%20\frac{\pi_\theta(o_{i,t}%20\mid%20q,%20o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}%20\mid%20q,%20o_{i,<t})},%201-\varepsilon,%201+\varepsilon%20\right)%20\hat{A}_{i,t}%20\right]%20-%20\beta%20D_{\text{KL}}\left(%20\pi_\theta(\cdot%20\mid%20q,%20o_{i,<t})%20\parallel%20\pi_{\text{ref}}%20\right)%20\right)%20\right]" alt="\mathcal{J}_{\text{GRPO}}(\theta)" />

其中，优势函数定义为：

<img src="https://latex.codecogs.com/svg.image?\hat{A}_{i,t}%20=%20\frac{R_i%20-%20\text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}" alt="\hat{A}_{i,t}" />

该目标函数广泛应用于大型语言模型（LLM）的对齐与微调任务（如 RLHF），结合了 **近端策略优化（PPO）** 的稳定性机制与 **KL 散度正则化**，在提升奖励的同时防止策略偏离过大。

---

## 1. 目标函数整体结构

<img src="https://latex.codecogs.com/svg.image?\mathcal{J}_{\text{GRPO}}(\theta)%20=%20\mathbb{E}_{\substack{q%20\sim%20P(Q),%20\\%20\{o_i\}_{i=1}^G%20\sim%20\pi_{\theta_{\text{old}}}(O%20\mid%20q)}}%20\left[%20\frac{1}{G}%20\sum_{i=1}^G%20\frac{1}{|o_i|}%20\sum_{t=1}^{|o_i|}%20\left(%20\dots%20\right)%20\right]" alt="\mathcal{J}_{\text{GRPO}}(\theta)" />

### 关键组成部分

- <img src="https://latex.codecogs.com/svg.image?\mathbb{E}[\cdot]" alt="\mathbb{E}[\cdot]" />：期望值，表示目标函数是对多个查询和响应轨迹的平均。
- <img src="https://latex.codecogs.com/svg.image?q%20\sim%20P(Q)" alt="q \sim P(Q)" />：从查询分布中采样一个输入查询 <img src="https://latex.codecogs.com/svg.image?q" alt="q" />。
- <img src="https://latex.codecogs.com/svg.image?\{o_i\}_{i=1}^G%20\sim%20\pi_{\theta_{\text{old}}}(O%20\mid%20q)" alt="\{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O \mid q)" />：使用旧策略 <img src="https://latex.codecogs.com/svg.image?\pi_{\theta_{\text{old}}}" alt="\pi_{\theta_{\text{old}}}" /> 生成 <img src="https://latex.codecogs.com/svg.image?G" alt="G" /> 个输出轨迹 <img src="https://latex.codecogs.com/svg.image?\{o_i\}" alt="\{o_i\}" />。
- <img src="https://latex.codecogs.com/svg.image?\frac{1}{G}%20\sum_{i=1}^G" alt="\frac{1}{G} \sum_{i=1}^G" />：对 <img src="https://latex.codecogs.com/svg.image?G" alt="G" /> 条轨迹取平均。
- <img src="https://latex.codecogs.com/svg.image?\frac{1}{|o_i|}%20\sum_{t=1}^{|o_i|}" alt="\frac{1}{|o_i|} \sum_{t=1}^{|o_i|}" />：对每条轨迹的每个时间步取平均。
- <img src="https://latex.codecogs.com/svg.image?\theta" alt="\theta" />：待优化的新策略参数。
- <img src="https://latex.codecogs.com/svg.image?\theta_{\text{old}}" alt="\theta_{\text{old}}" />：固定不变的旧策略参数，用于重要性采样。

---

## 2. 核心项分解

目标函数在每个时间步的核心表达式为：

<img src="https://latex.codecogs.com/svg.image?\min%20\left[%20r_t(\theta)%20\hat{A}_{i,t},\,%20\text{clip}\left(%20r_t(\theta),%201-\varepsilon,%201+\varepsilon%20\right)%20\hat{A}_{i,t}%20\right]%20-%20\beta%20D_{\text{KL}}\left(%20\pi_\theta(\cdot%20\mid%20q,%20o_{i,<t})%20\parallel%20\pi_{\text{ref}}%20\right)" alt="core expression" />

其中 <img src="https://latex.codecogs.com/svg.image?r_t(\theta)%20=%20\frac{\pi_\theta(o_{i,t}%20\mid%20q,%20o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}%20\mid%20q,%20o_{i,<t})}" alt="r_t(\theta)" />。

该表达式由两部分组成：

1. **PPO-like 策略梯度项（最大化奖励）**
2. **KL 正则化项（限制策略偏移）**

---

### A. PPO-like 策略梯度项

<img src="https://latex.codecogs.com/svg.image?L^{\text{PPO}}%20=%20\min%20\left[%20r_t(\theta)%20\hat{A}_{i,t},\,%20\text{clip}\left(%20r_t(\theta),%201-\varepsilon,%201+\varepsilon%20\right)%20\hat{A}_{i,t}%20\right]" alt="L^{PPO}" />

#### 关键概念

- <img src="https://latex.codecogs.com/svg.image?r_t(\theta)" alt="r_t(\theta)" />：新旧策略的概率比（重要性权重）。
- <img src="https://latex.codecogs.com/svg.image?\hat{A}_{i,t}" alt="\hat{A}_{i,t}" />：优势函数，衡量动作 <img src="https://latex.codecogs.com/svg.image?o_{i,t}" alt="o_{i,t}" /> 相对于批次平均表现的优劣。
- <img src="https://latex.codecogs.com/svg.image?\text{clip}(r_t,%201-\varepsilon,%201+\varepsilon)" alt="\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)" />：将概率比限制在 <img src="https://latex.codecogs.com/svg.image?[1-\varepsilon,%201+\varepsilon]" alt="[1-\varepsilon, 1+\varepsilon]" /> 范围内。
- <img src="https://latex.codecogs.com/svg.image?\min[\cdot,%20\cdot]" alt="\min[\cdot, \cdot]" />：选择两个目标中较小的一个。

#### PPO 剪裁机制详解

PPO 通过 `min` 和 `clip` 实现“信任区域”控制，防止策略更新过大。

---

#### Case 1: 正优势（<img src="https://latex.codecogs.com/svg.image?\hat{A}_{i,t}%20>%200" alt="\hat{A}_{i,t} > 0" />）——“好”动作

目标：**提高动作概率（增大 <img src="https://latex.codecogs.com/svg.image?r_t(\theta)" alt="r_t(\theta)" />）**

- 无约束项：<img src="https://latex.codecogs.com/svg.image?r_t(\theta)%20\hat{A}_{i,t}" alt="r_t(\theta) \hat{A}_{i,t}" />，随 <img src="https://latex.codecogs.com/svg.image?r_t" alt="r_t" /> 增大而上升。
- 有约束项：<img src="https://latex.codecogs.com/svg.image?\text{clip}(r_t,%201-\varepsilon,%201+\varepsilon)%20\hat{A}_{i,t}" alt="\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \hat{A}_{i,t}" />，当 <img src="https://latex.codecogs.com/svg.image?r_t%20>%201+\varepsilon" alt="r_t > 1+\varepsilon" /> 时变为常数。

`min` 函数会选择更小的目标值：

- 若 <img src="https://latex.codecogs.com/svg.image?r_t%20\leq%201+\varepsilon" alt="r_t \leq 1+\varepsilon" />：使用原始目标，梯度正常。
- 若 <img src="https://latex.codecogs.com/svg.image?r_t%20>%201+\varepsilon" alt="r_t > 1+\varepsilon" />：使用剪裁后常数项，**梯度为零**。

> ✅ **效果**：一旦新策略对“好”动作的偏好超过 $1+\varepsilon$，优化停止进一步提升其概率，防止过度拟合。

**示意图（A > 0）**：
```
目标值
  ↑
  |       /-------------------  ← 剪裁后目标（平坦，梯度为0）
  |      /
  |     / ← 正常增长
  +----/----------------------→ r_t(θ)
      1   1+ε
```

---

#### Case 2: 负优势（<img src="https://latex.codecogs.com/svg.image?\hat{A}_{i,t}%20<%200" alt="\hat{A}_{i,t} < 0" />）——“坏”动作

目标：**降低动作概率（减小 <img src="https://latex.codecogs.com/svg.image?r_t(\theta)" alt="r_t(\theta)" />）**

- 无约束项：<img src="https://latex.codecogs.com/svg.image?r_t(\theta)%20\hat{A}_{i,t}" alt="r_t(\theta) \hat{A}_{i,t}" />，随 <img src="https://latex.codecogs.com/svg.image?r_t" alt="r_t" /> 减小而增大（更接近零）。
- 有约束项：<img src="https://latex.codecogs.com/svg.image?\text{clip}(r_t,%201-\varepsilon,%201+\varepsilon)%20\hat{A}_{i,t}" alt="\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \hat{A}_{i,t}" />，当 <img src="https://latex.codecogs.com/svg.image?r_t%20<%201-\varepsilon" alt="r_t < 1-\varepsilon" /> 时变为常数。

`min` 函数选择更小（更负）的值：

- 若 <img src="https://latex.codecogs.com/svg.image?r_t%20\geq%201-\varepsilon" alt="r_t \geq 1-\varepsilon" />：使用原始目标。
- 若 <img src="https://latex.codecogs.com/svg.image?r_t%20<%201-\varepsilon" alt="r_t < 1-\varepsilon" />：使用剪裁后常数项，**梯度为零**。

> ✅ **效果**：一旦新策略对“坏”动作的惩罚超过 $1-\varepsilon$，优化停止进一步压制，保留探索能力。

**示意图（A < 0）**：
```
目标值
  ↑
  |                             ← 剪裁后目标（平坦）
  |    -------------------\ 
  |                        \
  |                         \ ← 正常下降
  +----------------------------→ r_t(θ)
    1-ε   1
```

---

### 总结：PPO 剪裁的类比

你可以将 PPO 剪裁看作一个有护栏的优化通道：

| 动作类型 | 优化方向 | 护栏作用 |
|--------|--------|--------|
| 好动作（A > 0） | 提高概率 | 不超过 <img src="https://latex.codecogs.com/svg.image?1+\varepsilon" alt="1+\varepsilon" />，防止过拟合 |
| 坏动作（A < 0） | 降低概率 | 不低于 <img src="https://latex.codecogs.com/svg.image?1-\varepsilon" alt="1-\varepsilon" />，防止过度惩罚 |

这种机制允许使用同一批数据进行多轮训练（epochs），显著提高数据利用率和训练稳定性。

---

### B. KL 散度正则化项

<img src="https://latex.codecogs.com/svg.image?-%20\beta%20D_{\text{KL}}\left(%20\pi_\theta(\cdot%20\mid%20q,%20o_{i,<t})%20\parallel%20\pi_{\text{ref}}%20\right)" alt="KL term" />

#### 组成部分

- <img src="https://latex.codecogs.com/svg.image?D_{\text{KL}}(\cdot%20\parallel%20\cdot)" alt="D_{\text{KL}}(\cdot \parallel \cdot)" />：Kullback-Leibler 散度，衡量分布差异。
- <img src="https://latex.codecogs.com/svg.image?\pi_{\text{ref}}" alt="\pi_{\text{ref}}" />：参考策略，通常为初始预训练模型或 SFT 模型。
- <img src="https://latex.codecogs.com/svg.image?\beta" alt="\beta" />：KL 惩罚的强度系数（超参数）。

#### 作用

- ✅ **防止语言退化**：避免模型为了追求高奖励而生成无意义、重复或“幻觉”文本。
- ✅ **保持对齐性**：确保微调后模型仍符合人类语言习惯和安全规范。
- ✅ **稳定训练**：作为正则项，防止策略剧烈震荡。

> 🔔 提示：<img src="https://latex.codecogs.com/svg.image?\beta" alt="\beta" /> 通常较小（如 0.01~0.1），过大可能导致奖励无法提升。

---

## 3. 优势函数 <img src="https://latex.codecogs.com/svg.image?\hat{A}_{i,t}" alt="\hat{A}_{i,t}" />

<img src="https://latex.codecogs.com/svg.image?\hat{A}_{i,t}%20=%20\frac{R_i%20-%20\text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}" alt="\hat{A}_{i,t}" />

### 各项含义

- <img src="https://latex.codecogs.com/svg.image?R_i" alt="R_i" />：第 <img src="https://latex.codecogs.com/svg.image?i" alt="i" /> 条轨迹的总奖励（如来自奖励模型 RM 的打分）。
- <img src="https://latex.codecogs.com/svg.image?\text{mean}(\{R_j\})" alt="\text{mean}(\{R_j\})" />：当前批次所有轨迹奖励的均值。
- <img src="https://latex.codecogs.com/svg.image?\text{std}(\{R_j\})" alt="\text{std}(\{R_j\})" />：当前批次奖励的标准差。

### 作用

- ✅ **归一化**：使优势函数尺度稳定，不受奖励绝对值影响。
- ✅ **去中心化**：奖励高于平均的轨迹获得正优势，反之为负。
- ✅ **提升鲁棒性**：适应不同任务或奖励模型的输出范围。

> ⚠️ 注意：若批次中所有 <img src="https://latex.codecogs.com/svg.image?R_i" alt="R_i" /> 相同（标准差为 0），需加小量 $\epsilon$ 防止除零。

---

## 总结

GRPO 目标函数 <img src="https://latex.codecogs.com/svg.image?\mathcal{J}_{\text{GRPO}}(\theta)" alt="\mathcal{J}_{\text{GRPO}}(\theta)" /> 是一个高效且稳定的策略优化框架，特别适用于 LLM 对齐任务：

| 组件 | 功能 | 目标 |
|------|------|------|
| **PPO 剪裁项** | 基于优势更新策略 | 最大化期望奖励 |
| **KL 正则项** | 限制与参考策略的偏离 | 保持生成质量与安全 |
| **归一化优势** | 批次内相对比较 | 稳定训练过程 |

🎯 **核心思想**：在“追求更高奖励”和“不偏离太远”之间取得平衡。

该设计使得 GRPO 成为现代 LLM 强化学习对齐（如 RLHF、RLOO）中的主流方法之一。
```
