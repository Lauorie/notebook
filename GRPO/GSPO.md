概要
- 问题根源：GRPO 把重要性采样（importance sampling, IS）的比值放在 token 级，并且每个位置只有一个样本权重，违背了 IS 的基本使用前提，导致长序列场景下训练噪声累积、被 clip 机制放大，最终可能训练崩溃，尤其在大模型/MoE 更明显。
- 关键改动：GSPO 把重要性比值改为“序列级似然比”（并做长度归一），clip 也在序列级进行。这样奖励的“度量单位”（序列级）与优化/校正的“单位”（序列级）一致，训练显著更稳、更高效。
- 实践影响：GSPO 在长序列与 MoE 上更稳，省掉了 GRPO 在 MoE 训练中常用的 Routing Replay 复杂策略；同时对基础设施更友好（可直接用推理引擎返回的序列 logprob，避免训练端重算）。

## 1. 预备知识与符号

- 策略（语言模型）记作自回归分布 πθ。给定 query x，响应 y = (y₁,…,y_T) 的序列似然：  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}\pi_\theta(y|x)=\prod_{t=1}^{|y|}\pi_\theta(y_t|x,y_{<t})" alt="\pi_\theta(y\mid x)=\prod_{t=1}^{|y|}\pi_\theta(y_t\mid x, y_{<t})"/>
- 回报/奖励来自一个打分器 r，记作 <img src="https://latex.codecogs.com/svg.image?\dpi{120}r(x,y)\in[0,1]" alt="r(x,y)\in[0,1]"/>。
- 采样通常来自旧策略（off-policy）：<img src="https://latex.codecogs.com/svg.image?\dpi{120}y\sim\pi_{\theta_\text{old}}(\cdot|x)" alt="y\sim \pi_{\theta_\text{old}}(\cdot\mid x)"/>，再优化新策略 πθ，这就需要“近端约束”（clipping 或 KL）保证更新稳定。
- 组内优势（Group advantage）：同一 query 下采样 G 个响应，使用它们的组内标准化作为优势。

## 2. PPO 与 GRPO 回顾

- PPO（忽略 KL 正则项，便于说明）：  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}J_\text{PPO}(\theta)=\mathbb{E}_{x\sim&space;D,\&space;y\sim&space;\pi_{\theta_\text{old}}}\left[\frac{1}{|y|}\sum_{t=1}^{|y|}\min\big(w_t(\theta)A_t^b,\;\text{clip}(w_t(\theta),1-\varepsilon,1+\varepsilon)A_t^b\big)\right]" />
  
  其中 <img src="https://latex.codecogs.com/svg.image?\dpi{120}w_t(\theta)=\frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{\theta_\text{old}}(y_t|x,y_{<t})}" />, <img src="https://latex.codecogs.com/svg.image?\dpi{120}A_t^b"/> 来自价值模型。问题在于价值模型重、准、难扩展。

- GRPO（不用价值模型，直接用组内优势）：  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}J_\text{GRPO}(\theta)=\mathbb{E}_{x,\{y_i\}_{i=1}^G\sim\pi_{\theta_\text{old}}}\left[\frac{1}{G}\sum_{i=1}^G\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\min\big(w_{i,t}(\theta)A_i^b,\;\text{clip}(w_{i,t}(\theta),1-\varepsilon,1+\varepsilon)A_i^b\big)\right]" />
  - 组内优势：  
    <img src="https://latex.codecogs.com/svg.image?\dpi{120}A_i^b=\frac{r(x,y_i)-\text{mean}\{r(x,y_j)\}_{j=1}^G}{\text{std}\{r(x,y_j)\}_{j=1}^G}"/>
  - token 级重要性比：  
    <img src="https://latex.codecogs.com/svg.image?\dpi{120}w_{i,t}(\theta)=\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_\text{old}}(y_{i,t}|x,y_{i,<t})}"/>

## 3. 为什么 GRPO 会不稳？（从 IS 原理看问题）

- 重要性采样的基本式子：  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}\mathbb{E}_{z\sim\pi_\text{tar}}[f(z)]=\mathbb{E}_{z\sim\pi_\text{beh}}\left[\frac{\pi_\text{tar}(z)}{\pi_\text{beh}(z)}f(z)\right]"/>
  要“多样本平均”来降低方差。
- GRPO 在每个 token 位置用一个样本的比值 <img src="https://latex.codecogs.com/svg.image?\dpi{120}w_{i,t}"/> 当校正因子，既没有足够样本平均，又把它乘到梯度里，结果是：
  - 高方差噪声，序列越长累积越大；
  - clip 的非线性进一步放大/形变噪声；
  - 在 MoE 中，一次更新就会导致不同路由，<img src="https://latex.codecogs.com/svg.image?\dpi{120}w_{i,t}"/> 抖动更剧烈，训练不收敛甚至崩溃。
- 本质教训：奖励是“序列级”，优化与 IS 校正也应“序列级”。token 级用 IS 权重不匹配问题单位。

## 4. GSPO：把 IS、clip、奖励全部拉到“序列级”

### 4.1 目标函数与序列级比值

- 组内优势与 GRPO 相同（序列级）：  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}A_i^b=\frac{r(x,y_i)-\text{mean}\{r(x,y_j)\}_{j=1}^G}{\text{std}\{r(x,y_j)\}_{j=1}^G}"/>
- 序列级重要性比（做长度归一，几何平均）：  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}s_i(\theta)=\left[\frac{\pi_\theta(y_i|x)}{\pi_{\theta_\text{old}}(y_i|x)}\right]^{\frac{1}{|y_i|}}=\exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_\text{old}}(y_{i,t}|x,y_{i,<t})}\right)" />
- GSPO 目标：  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}J_\text{GSPO}(\theta)=\mathbb{E}_{x,\{y_i\}\sim\pi_{\theta_\text{old}}}\left[\frac{1}{G}\sum_{i=1}^G\min\big(s_i(\theta)A_i^b,\;\text{clip}(s_i(\theta),1-\varepsilon,1+\varepsilon)A_i^b\big)\right]"/>

长度归一的动机：控制数值范围和方差，避免“少数 token 似然波动把整段比值炸飞”，也避免不同长度需要不同的剪切区间。由于采用了序列级定义，GSPO 的 ε 通常比 GRPO 小很多（数量级不同是正常的）。

### 4.2 GSPO 的梯度推导（去掉 clip 便于看清本质）

- 从  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}\nabla_\theta&space;J_\text{GSPO}\approx&space;\mathbb{E}\left[\frac{1}{G}\sum_i&space;s_i(\theta)A_i^b&space;\cdot&space;\nabla_\theta&space;\log&space;s_i(\theta)\right]"/>
- 注意  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}\log&space;s_i(\theta)=\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log&space;\pi_\theta(y_{i,t}|x,y_{i,<t})\;&space;-&space;\;\underbrace{\frac{1}{|y_i|}\sum_{t}\log&space;\pi_{\theta_\text{old}}(\cdot)}_{\text{常数，对&space;}\theta\text{&space;无梯度}}"/>
- 因而  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}\nabla_\theta\log&space;s_i(\theta)=\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\nabla_\theta\log&space;\pi_\theta(y_{i,t}|x,y_{i,<t})"/>
- 代回去得到  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}\nabla_\theta&space;J_\text{GSPO}\approx&space;\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G&space;s_i(\theta)A_i^b\cdot&space;\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\nabla_\theta&space;\log&space;\pi_\theta(y_{i,t}|x,y_{i,<t})\right]"/>
- 关键信息：同一响应内，每个 token 的梯度被“同一序列权重”乘上，token 间权重一致，避免了 GRPO 中 token 间权重不均导致的累积漂移。

### 4.3 和 GRPO 梯度的对照

- GRPO 的无剪切近似梯度（<img src="https://latex.codecogs.com/svg.image?\dpi{120}A_{i,t}^b=A_i^b"/>）：  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}\nabla_\theta&space;J_\text{GRPO}\approx&space;\mathbb{E}\left[\frac{1}{G}\sum_i&space;A_i^b\cdot&space;\frac{1}{|y_i|}\sum_t&space;\underbrace{\frac{\pi_\theta(y_{i,t}|&space;\cdot)}{\pi_{\theta_\text{old}}(y_{i,t}|&space;\cdot)}}_{\text{token级比值}}&space;\nabla_\theta&space;\log&space;\pi_\theta(y_{i,t}|x,y_{i,<t})\right]"/>
- 差异要点：GRPO 用 token 级比值做“每个 token 的不同权重”，范围会在剪切区间外被截断，但不等于“低方差”；而 GSPO 用“同一序列权重”，更稳。

### 4.4 GSPO-token：保留序列校正，但允许 token 级优势

- 目标：  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}J_\text{GSPO-token}(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_i\frac{1}{|y_i|}\sum_t\min\big(s_{i,t}(\theta)A_{i,t}^b,\;\text{clip}(s_{i,t}(\theta),1-\varepsilon,1+\varepsilon)A_{i,t}^b\big)\right]"/>
  其中  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}s_{i,t}(\theta)=\text{sg}[s_i(\theta)]\cdot\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\text{sg}[\pi_\theta(y_{i,t}|x,y_{i,<t})]}"/>
  sg[·] 表示“数值保留、梯度阻断”（PyTorch 的 detach）。
- 关键性质：数值上 <img src="https://latex.codecogs.com/svg.image?\dpi{120}s_{i,t}(\theta)\equiv&space;s_i(\theta)"/>，但梯度只从 <img src="https://latex.codecogs.com/svg.image?\dpi{120}\pi_\theta"/> 的 logprob 处流，不从 <img src="https://latex.codecogs.com/svg.image?\dpi{120}s_i"/> 的定义里再流第二遍，避免重复依赖。其梯度（无剪切）：
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}\nabla_\theta&space;J_\text{GSPO-token}\approx&space;\mathbb{E}\left[\frac{1}{G}\sum_i&space;s_i(\theta)\cdot&space;\frac{1}{|y_i|}\sum_t&space;A_{i,t}^b\,\nabla_\theta&space;\log&space;\pi_\theta(y_{i,t}|x,y_{i,<t})\right]"/>
- 当设置 <img src="https://latex.codecogs.com/svg.image?\dpi{120}A_{i,t}^b\equiv&space;A_i^b"/>，它与 GSPO 在数值目标、剪切条件、理论梯度上一致；但 GSPO-token 允许更细粒度的 token 级优势（比如多轮对话中对不同片段差异化打分）。

## 5. 实验现象与工程解读

- 稳定性与效率：GSPO 在相同训练算力与相同 query 消耗下，比 GRPO 拿到更高训练回报与更优榜单分数；可以随着算力与长度扩展连续提升性能；训练过程稳定（无崩溃）。
- 剪切比例的“反直觉”现象：论文报告 GSPO 对“被剪切的 token 比例”显著更高（数量级差异），但反而训练更高效。这反证 GRPO 的 token 级梯度本身就很噪，留着也难以有效“利用样本”；而 GSPO 序列级信号更稳定、可用性更高。
- MoE 的特别好处：
  - 现象：在 GRPO 下，MoE 的路由易在每次更新后显著变化，token 级比值剧烈抖动，训练不收敛。此前需要“Routing Replay”：在新策略上复用旧策略的路由，让分子分母在同一激活网络里算，来稳定 <img src="https://latex.codecogs.com/svg.image?\dpi{120}w_{i,t}"/>。但这会带来额外显存、通信开销，还限制了实际可用容量。
  - GSPO 的改善：序列级似然对“个别 token 的路由波动”不那么敏感，MoE 的语言建模能力整体仍稳定，序列似然不会剧烈抖动；因此 GSPO 不需要 Routing Replay 就能稳步收敛，大幅简化工程与资源占用。
- 基础设施上的好处：GSPO 只需要序列级似然（或 logprob 和累计），更能容忍推理/训练引擎间的数值精度差异；很多场景下可以直接用推理引擎返回的序列 logprob，避免在训练引擎端重算，利好“训练-推理解耦”、部分 rollout、多轮 RL 等复杂管线。

## 6. 实践落地建议（从 GRPO 迁到 GSPO）

- 核心实现  
  - 采样仍用 <img src="https://latex.codecogs.com/svg.image?\dpi{120}\pi_{\theta_\text{old}}"/>；对每个 x 生成 G 个响应。
  - 计算每个响应的 reward，做组内标准化得到 <img src="https://latex.codecogs.com/svg.image?\dpi{120}A_i^b"/>。
  - 计算序列 logprob：  
    <img src="https://latex.codecogs.com/svg.image?\dpi{120}\log&space;\pi_\theta(y_i|x)=\sum_t&space;\log&space;\pi_\theta(y_{i,t}|x,y_{i,<t})"/>
    同理算 <img src="https://latex.codecogs.com/svg.image?\dpi{120}\log&space;\pi_{\theta_\text{old}}(y_i|x)"/>；得到  
    <img src="https://latex.codecogs.com/svg.image?\dpi{120}s_i(\theta)=\exp\left(\frac{\log\pi_\theta(y_i|x)-\log\pi_{\theta_\text{old}}(y_i|x)}{|y_i|}\right)"/>
  - 在序列级做 clip，再把该序列的“同一权重”分配给序列内所有 token 的 logprob 梯度。
- 超参数与数值  
  - 剪切范围 ε 的量级会比 GRPO 小很多（论文示例：GSPO 左/右剪切约 3e-4、4e-4；GRPO 为 0.2、0.27）。不要照搬 GRPO 的 ε。
  - 是否加 KL 正则：论文为聚焦主体省略了 KL；工程中可按需加入（对参考策略/旧策略），但注意与序列级剪切的相互作用。
  - 长度归一很关键，既稳又统一不同长度响应的数值尺度。
- 训练管线  
  - 大 batch rollout → 多个 mini-batch 迭代更新（off-policy 不可避免）。GSPO 的序列级 clip 就是为此而生。
  - 监控指标：训练回报、AIME/Code 等外部榜单；序列级比值的分布（均值/方差/剪切比例）；序列长度分布；MoE 的路由稳定性。
- MoE 专项  
  - 不要默认再用 Routing Replay。先试纯 GSPO：序列级比值 + 序列级 clip 通常足够稳。
  - 若仍需控制路由波动，可在采样温度、top-k 等上略保守，或在梯度累积/学习率上做温和设置（但大多数情况 GSPO 已够稳）。
- 推理-训练解耦  
  - 直接用推理引擎返回的 token logprob 求和成序列 logprob，再做长度归一、序列级 clip。无需训练端重算，可减少一次 I/O 与算力开销。
- GSPO-token 用法  
  - 若有细粒度标注（例如多轮对话里不同轮的好坏不同），可用 GSPO-token，把 <img src="https://latex.codecogs.com/svg.image?\dpi{120}A_{i,t}^b"/> 设计为分段或按规则分配；仍由 <img src="https://latex.codecogs.com/svg.image?\dpi{120}s_i(\theta)"/> 提供序列级校正与剪切。

## 7. 与 GRPO 的简明对照

- 权重单位：GRPO 为 token 级，GSPO 为序列级（长度归一）。
- 剪切单位：GRPO 按 token 剪，GSPO 按序列剪。
- 噪声与稳定性：GRPO 在长序列/大模型/ MoE 下噪声累积大，易崩；GSPO 稳定、可扩展。
- 工程复杂度：GRPO 在 MoE 常需 Routing Replay；GSPO 可不用，管线简化、开销减小。
- clipping fraction 观察：GSPO“被剪切的 token 比例”更高，但训练反而更高效，侧面说明 GSPO 的信号更干净、可利用度更高。

## 8. 小结

GSPO 把“奖励-优化-校正”的单位从 token 拉回到序列，遵循了 IS 的基本原则，配合长度归一与序列级剪切，显著提升了稳定性与效率，尤其解决了 MoE 路由波动导致的收敛问题。实证显示，它在算力扩大、长度拉长、数据更新的过程中能持续提升性能，也是近期 Qwen3 模型强化提升的重要支撑。建议在实际 RL 训练中优先采用 GSPO；若需细粒度调控优势，再使用 GSPO-token 变体。

---

## 附录：常用关键公式一览

- 自回归似然：  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}\pi_\theta(y|x)=\prod_{t=1}^{|y|}\pi_\theta(y_t|x,y_{<t})"/>
- 组内优势：  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}A_i^b=\frac{r(x,y_i)-\text{mean}\{r(x,y_j)\}}{\text{std}\{r(x,y_j)\}}"/>
- GRPO 目标：  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}J_\text{GRPO}=\mathbb{E}\left[\frac{1}{G}\sum_i\frac{1}{|y_i|}\sum_t\min\big(w_{i,t}A_i^b,\;\text{clip}(w_{i,t},1-\varepsilon,1+\varepsilon)A_i^b\big)\right]\quad,\quad w_{i,t}=\frac{\pi_\theta(y_{i,t}|\cdot)}{\pi_{\theta_\text{old}}(y_{i,t}|\cdot)}"/>
- GSPO 的序列级比值与目标：  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}s_i=\left[\frac{\pi_\theta(y_i|x)}{\pi_{\theta_\text{old}}(y_i|x)}\right]^{1/|y_i|}\quad,\quad J_\text{GSPO}=\mathbb{E}\left[\frac{1}{G}\sum_i\min\big(s_iA_i^b,\;\text{clip}(s_i,1-\varepsilon,1+\varepsilon)A_i^b\big)\right]"/>
- GSPO 梯度（去剪切）：  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}\nabla_\theta&space;J_\text{GSPO}\approx&space;\mathbb{E}\left[\frac{1}{G}\sum_i&space;s_iA_i^b\cdot&space;\frac{1}{|y_i|}\sum_t&space;\nabla_\theta&space;\log&space;\pi_\theta(y_{i,t}|x,y_{i,<t})\right]"/>
- GSPO-token 与梯度（去剪切）：  
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}J_\text{GSPO-token}=\mathbb{E}\left[\frac{1}{G}\sum_i\frac{1}{|y_i|}\sum_t\min\big(s_{i,t}A_{i,t}^b,\;\text{clip}(s_{i,t},1-\varepsilon,1+\varepsilon)A_{i,t}^b\big)\right]"/>
  <img src="https://latex.codecogs.com/svg.image?\dpi{120}s_{i,t}=\text{sg}[s_i]\cdot\frac{\pi_\theta(y_{i,t}|\cdot)}{\text{sg}[\pi_\theta(y_{i,t}|\cdot)]}
  \quad\Rightarrow\quad
  \nabla_\theta&space;J_\text{GSPO-token}\approx&space;\mathbb{E}\left[\frac{1}{G}\sum_i&space;s_i\cdot&space;\frac{1}{|y_i|}\sum_t&space;A_{i,t}^b\,\nabla_\theta&space;\log&space;\pi_\theta(y_{i,t}|\cdot)\right]"/>
