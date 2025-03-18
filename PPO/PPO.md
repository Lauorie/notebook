# RLHF基础知识

RLHF的核心就是4个模型之间的交互过程

- Actor model（**参与训练**）：经过 SFT 训练后的语言模型，最后一层网络是 nn.Linear(hidden_size, vocab_size)
- Reference model（**不参与训练**）：Actor_model 复制而来
- Reward model（**不参与训练**）：
    - 将传统语言模型的最后一层网络，由 nn.Linear(hidden_size, vocab_size) 替换成 nn.Linear(hidden_size, 1)，也就是说该模型输出的是当前 token 的得分，而不是对下一个token的预测
    - 输入是 prompt + answer， **输出是answer中每个token对应的值，answer中最后一个token对应的值即为这条语料的reward**
    - 该模型是用来计算整条语料的reward
- Critic model（**不参与训练**）：Reward_model 复制而来
    - **该模型是用来计算单个token的reward**

![deepspeedchat](./PPO/deepspeedchat.png)


# **reward model** 和 **critic model** 对比

---

### **1. Reward Model 的作用**
Reward model 的核心任务是**评估一个句子的整体质量**，即它从人类反馈中学习一个评分机制，用来判断生成的句子是否符合人类偏好。  
- **输入**：Reward model 接收的是一个完整的输出（如整个句子或段落）。  
- **输出**：它给这个完整的输出分配一个分数，表示这个输出的质量高低。  
- **目的**：在训练过程中，Reward model 指导 Actor model（主要生成模型），帮助它生成更符合人类偏好和任务需求的输出。

#### 一个比喻
Reward model 就像一个裁判，主要关心的是**最终结果是否符合标准**。比如，在一个写作比赛中，裁判只会看最终完成的文章，并给它打分，而不会关注作者在写每一段话时的表现。

#### 总结
- **作用**：对完整输出（如句子）打分。
- **粒度**：只关心整体表现，而不关心中间过程。

---

### **2. Critic Model 的作用**
Critic model 的核心任务是**评估生成策略的质量**，它的职责更偏向强化学习中的传统角色，计算状态值函数 `V(s)`，即某个状态的“好坏程度”。  
- **输入**：Critic model 接收的是生成的部分序列（如生成到第 `t` 个 token 时的状态 `s_t`）。
- **输出**：它计算从当前状态开始，未来可能获得的总回报（即 `V(s_t)`），帮助优化 Actor model 的策略。  
- **目的**：Critic model 的计算结果可以为每个生成动作（例如选择一个 token）提供更细粒度的反馈，从而指导 Actor model 的优化。

#### 一个比喻
如果 Reward model 是裁判，Critic model 更像是一个教练。教练会在你每一步的表现中给出建议，比如“这句话应该换个词”“这里得注意句子的结构”。它更关注生成过程中的每一步，而不是只看最终结果。

#### 总结
- **作用**：评估生成过程中每一步的状态（策略）的好坏。
- **粒度**：聚焦于生成过程的细节，而不是只看最终输出。

---

### **Reward Model 和 Critic Model 的关键区别**
| 特性               | Reward Model                               | Critic Model                       |
|--------------------|--------------------------------------------|------------------------------------|
| **评估范围**       | 整个句子或完整输出                         | 每一步生成的状态或策略             |
| **评估目标**       | 给最终结果打分                             | 计算每个状态的值函数 `V(s)`        |
| **作用粒度**       | 粗粒度，整体评估                          | 细粒度，逐步评估                   |
| **强化学习中的角色** | 提供奖励信号                              | 指导策略优化                       |

---

### **补充：两者如何协作？**
1. **Reward model** 提供了一个全局的“最终目标”，即希望 Actor model 生成的输出符合人类偏好。
2. **Critic model** 则在生成的过程中，逐步评估当前策略的好坏，为策略改进提供更细致的反馈。
3. **训练过程**：
   - Reward model 的评分通常会被用来构建强化学习中的奖励信号。
   - Critic model 则是强化学习算法中的核心部分，用于计算策略的优化方向。

通过两者的配合，Actor model 能够生成既符合人类偏好的输出，又在生成过程中进行细化调整，逐步提高它的策略质量。

# 强化学习基础知识

#### **1. $Q(s_t, a_t)$：状态-动作值函数**

**定义**：  
$Q(s_t, a_t)$ 表示在某个状态 $s_t$ 下，执行某个动作 $a_t$ 后，最终能够获得的 **累计奖励**（reward）。

**类比**：  
假设我们的语言模型正在生成一个汉字（或 token），当前状态是 `首都`，模型选择的动作是生成 `是`。  
那么，$Q(\text{首都}, \text{是})$ 意味着：当模型在 `首都` 这个状态下，选择生成 `是` 这个动作后，最终能获得的奖励值。

---

#### **2. $V(s_t)$：状态值函数**

**定义**：  
$V(s_t)$ 表示在某个状态 $s_t$ 下，模型接下来按照策略（policy）生成内容时，能够获得的 **累计奖励**（reward）。

**类比**：  
假设当前状态是 `首都是南`，意味着模型已经生成了 `首都是南`，接下来还没选择生成什么。  
$V(\text{首都是南})$ 表示：从这个状态出发，模型继续生成内容时，能够获得的最终奖励值。

---

#### **3. $A(s_t, a_t)$：优势函数**

**定义**：  
$A(s_t, a_t)$ 表示在状态 $s_t$ 下，选择动作 $a_t$ 比起默认策略的平均表现 **有多好**。  
公式：  
$$ A(s_t, a_t) = Q(s_t, a_t) - V(s_t) $$

**类比**：  
如果在 `首都` 这个状态下，选择生成 `是`，然后获得的奖励值很高，那 $A(\text{首都}, \text{是})$ 就是一个正数，说明生成 `是` 比其他动作更优。

---

### 用语言模型生成文本的例子来理解

假设我们有一个问题：  
**Prompt**：`中国的首都是哪里？`  
**Answer**：`首都是南京`

我们从强化学习的角度来分析：

1. **Reward**  
   假设最终的回答 `首都是南京` 被 reward_model 打分为 **-10**，因为这是一个错误答案。  
   语言模型在生成完整的回答后才能知道最终的奖励值。

2. **状态-动作值函数 $Q(s_t, a_t)$**  
   比如：$Q(\text{首都}, \text{是})$ 表示模型在 `首都` 这个状态下，选择生成 `是` 之后，最终能获得的奖励值。  
   如果生成 `是` 是正确答案的一部分，那么这个值会较高。

3. **状态值函数 $V(s_t)$**  
   比如：$V(\text{首都是南})$ 表示模型在生成到 `首都是南` 时，接下来按照策略生成余下内容，最终能获得的奖励值。  
   因为 `首都是南` 已经偏离正确答案，所以这个值可能会较低。

4. **优势函数 $A(s_t, a_t)$**  
   如果在 `首都` 这个状态下，生成 `是` 的效果比模型默认策略平均表现更优，则 $A(\text{首都}, \text{是})$ 为正，表示这是一个好选择。

# RLHF完整流程
有了RLHF 和 RL 的基础知识后，我们来介绍每个模型的作用：

- Actor_model 就是我们要优化的 LLM

- Reference_model 是一个标杆，为的是让我们的 Actor_model 在训练时不要偏离原始模型太远，保证其不会失去原本的说话能力

- Reward_model 负责给 LLM 生成的句子打分

- Critic_model 负责评估 Actor_model 的策略，计算状态值函数，也就是上面提到的`V`函数（**Reward模型只负责给最后一个token或者说整个句子打分，给之前token打分的重任靠Critic_model 完成**）



## RLHF的第一个环节：让模型生成答案，并对其打分

- 给定 batch_size 条 prompt
- 调用 actor_model 生成 answer，并进行 token 化，得到一个 B * L 的矩阵；
- reward_model 对 answer 进行打分，得到一个 B * 1 的矩阵；
- critic_model 对每个 token 进行打分，得到一个 B * L 的矩阵；
- actor_model 和 reference_model 对生成的句子进行一遍正向传播，保存 output.logits，得到两个 B * L * V 的矩阵
- 利用 gather_log_probs() 函数，只保存目标 token 的 logit 值，得到两个 B * L 的矩阵

```python
{
    'prompts': prompts,
    'input_ids': seq,
    "attention_mask": attention_mask
    'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),        # batch_size  * (seq_len - 1)
    'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,1:]), # batch_size  * (seq_len - 1)
    'value': values,                                                    # batch_size * seq_len
    'rewards': reward_score,                                            # torch.Size([batch_size])
}

def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)
```

以下是对这段代码的逐步拆解和解释：

---

### **代码的整体结构**
```python
{
    'prompts': prompts,
    'input_ids': seq,
    "attention_mask": attention_mask,
    'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),        # batch_size  * (seq_len - 1)
    'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,1:]), # batch_size  * (seq_len - 1)
    'value': values,                                                    # batch_size * seq_len
    'rewards': reward_score,                                            # torch.Size([batch_size])
}
```

这是一个 Python 字典，包含多个关键值，分别是训练中涉及的输入和输出数据，具体含义如下：

1. **`prompts`**:
   - 输入的 prompt（提示），即模型生成句子的起始上下文。

2. **`input_ids` (即 `seq`)**:
   - 输入序列的 token IDs，表示模型生成的整个序列。

3. **`attention_mask`**:
   - 用于标记序列中的有效 token，告诉模型哪些 token 是需要处理的，哪些是 padding。

4. **`logprobs`**:
   - 使用 `gather_log_probs` 函数计算的 log 概率，代表生成序列中每一个 token 的对数概率（由模型计算）。

5. **`ref_logprobs`**:
   - 基于参考模型（`logits_ref`）计算的 log 概率，通常用于评估生成序列和参考模型之间的差异。

6. **`value`**:
   - 由价值网络（Critic 模型）计算的每个 token 的价值估计，形状为 `[batch_size, seq_len]`。

7. **`rewards`**:
   - 对生成的句子整体评分（通常由 Reward 模型计算），是一个标量，形状为 `[batch_size]`。

---

### **`gather_log_probs` 函数**
这个函数用于计算生成序列中每个 token 的对数概率，并提取与输入序列中真实 token 对应的 log 概率。

#### 函数实现：
```python
def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)
```

#### **解析**：

1. **`F.log_softmax(logits, dim=-1)`**:
   - 将 `logits` 转化为 log 概率，`log_softmax` 是 softmax 的对数版本，数值更稳定。
   - `logits` 的形状为 `[batch_size, seq_len, vocab_size]`，因此 log 概率的形状也为 `[batch_size, seq_len, vocab_size]`。

2. **`log_probs.gather(dim=-1, index=labels.unsqueeze(-1))`**:
   - `labels` 是输入序列中对应的 token IDs，形状为 `[batch_size, seq_len]`。
   - `labels.unsqueeze(-1)` 将其变成形状 `[batch_size, seq_len, 1]`，以便与 `log_probs` 的最后一个维度对齐。
   - `gather(dim=-1, index=...)` 提取 `log_probs` 中与 `labels` 对应的 log 概率（即每个 token 的生成概率），结果形状为 `[batch_size, seq_len, 1]`。

3. **`squeeze(-1)`**:
   - 将最后一维去掉，结果形状变为 `[batch_size, seq_len]`，即每个 token 的 log 概率。

#### 总结：
`gather_log_probs` 的功能是从 log 概率分布中提取每个 token 的 log 概率值，返回 `[batch_size, seq_len]` 的张量。

---

### **重点字段的理解**
1. **`logprobs` 和 `ref_logprobs`**:
   - `logprobs`: 当前模型生成的序列中每个 token 的 log 概率。
   - `ref_logprobs`: 参考模型生成的序列中每个 token 的 log 概率。
   - 这两个值用于比较 Actor 模型和参考模型的生成差异，通常是计算 KL 散度的基础。

   **形状**：
   - `[batch_size, seq_len - 1]`：因为需要对生成的序列进行对齐（跳过第一个 token）。

2. **`value`**:
   - Critic 模型（价值网络）对输入序列每个 token 的价值估计，形状为 `[batch_size, seq_len]`。
   - 用于计算策略梯度时的优势函数（Advantage Function）。

3. **`rewards`**:
   - 由 Reward 模型对整个句子进行打分，形状为 `[batch_size]`。
   - 通常是一个标量，表示生成句子的整体质量。

---

### **代码背后的逻辑**
这段代码是强化学习中数据处理的一部分：
1. 模型生成序列（通过 `logits` 和 `logits_ref`）。
2. 计算生成序列的 log 概率（`logprobs` 和 `ref_logprobs`）。
3. 使用价值网络（`value`）和 Reward 模型（`rewards`）的得分，结合策略梯度方法（例如 PPO），优化生成模型（Actor 模型）。

最终目的是调整 Actor 模型的策略，使其生成的序列更符合人类偏好或特定目标。

---

### **总结**
这段代码中的核心是：
- **`gather_log_probs`** 提取每个 token 的 log 概率。
- **`logprobs` 和 `ref_logprobs`** 用于计算生成分布和参考分布的差异。
- **`value` 和 `rewards`** 是强化学习训练中价值函数和奖励信号的重要部分。
整体流程遵循 RLHF 的标准框架，结合 log 概率和奖励信号优化生成模型的策略。



## RLHF的第二个环节：修正reward

前面提到，我们不能让 actor_model 偏离 reference_model 太远，因此我们要给rewards矩阵添加一个惩罚项，compute_rewards() 函数的返回是：每个token修正后的rewards：

- 最后一个token的计算方法是 Reward_score + KL_penalty
- 前面的所有的token 的计算方法是 0 + KL_penalty （除了最后一个token，前置token的reward初始值都是0，但是要加上惩罚项）

结合代码看的时候，要始终记住这个变换： log(A/B) = log(A) - log(B)

```python
prompts = inputs['prompts']
log_probs = inputs['logprobs']
ref_log_probs = inputs['ref_logprobs']
reward_score = inputs['rewards']
values = inputs['value']
attention_mask = inputs['attention_mask']
seq = inputs['input_ids']
start = prompts.size()[-1] - 1
action_mask = attention_mask[:, 1:]
old_values = values
old_rewards = self.compute_rewards(prompts, log_probs, ref_log_probs, reward_score, action_mask)
ends = start + action_mask[:, start:].sum(1) + 1

# 计算reward
def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score, action_mask):
    kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
    rewards = kl_divergence_estimate
    start = prompts.shape[1] - 1
    ends = start + action_mask[:, start:].sum(1) + 1
    reward_clip = torch.clamp(reward_score, -self.clip_reward_value, self.clip_reward_value)
    batch_size = log_probs.shape[0]
    for j in range(batch_size):
        rewards[j, start:ends[j]][-1] += reward_clip[j]

    return rewards
```

这段代码实现了强化学习（RL）中的 **奖励计算函数**，它结合了 KL 散度、奖励裁剪等机制，作为模型优化的重要部分。以下是对代码的逐步拆解和解释。

---

### **整体逻辑**

1. **`inputs` 字典的含义**
   - 这里的输入 `inputs` 包含了模型生成所需的各种数据，如：
     - `prompts`: 初始提示（Prompt）。
     - `log_probs`: 当前模型生成的每个 token 的 log 概率。
     - `ref_log_probs`: 参考模型生成的每个 token 的 log 概率。
     - `reward_score`: 最终由 Reward 模型计算的奖励分数（标量），对生成句子的整体评分。
     - `values`: Critic 模型对每个 token 的价值估计。
     - `attention_mask`: 用于标记 padding 的有效 token。
     - `seq`: 输入序列（token IDs）。

2. **奖励计算的上下文**
   - **起始点 `start`**:
     - 计算从 prompts 的最后一个 token 开始的索引，即 `start = prompts.size()[-1] - 1`。

   - **终止点 `ends`**:
     - 计算每个序列的结束点，考虑 `action_mask`（生成过程中有效 token 的掩码）。
     - 公式：`ends = start + action_mask[:, start:].sum(1) + 1`。

   - **`compute_rewards` 函数**:
     - 根据 KL 散度、裁剪后的奖励和其他机制计算每个 token 的细粒度奖励。
     - 结果 `rewards` 是一个形状为 `[batch_size, seq_len]` 的张量，表示每个 token 的奖励。

---

### **`compute_rewards` 函数详解**

#### **输入参数**
```python
def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score, action_mask):
```
1. **`prompts`**:
   - 初始提示的 token 序列，形状为 `[batch_size, prompt_len]`。

2. **`log_probs` 和 `ref_log_probs`**:
   - 当前模型和参考模型生成 token 的 log 概率，形状为 `[batch_size, seq_len]`。

3. **`reward_score`**:
   - 对整个句子的整体评分（标量），形状为 `[batch_size]`。

4. **`action_mask`**:
   - 用于标记生成序列中有效 token 的掩码，形状为 `[batch_size, seq_len]`。

---

#### **步骤解析**

##### 1. **计算 KL 散度估计**
```python
kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
rewards = kl_divergence_estimate
```
- 公式：`kl_divergence_estimate = -λ * (log_probs - ref_log_probs)`。
  - `log_probs - ref_log_probs` 是生成的分布与参考分布之间的 log 概率差。
  - `self.kl_ctl` 是一个控制 KL 散度影响的超参数（通常是一个正值）。
  - 最终结果是一个形状为 `[batch_size, seq_len]` 的张量，表示每个 token 的 KL 散度贡献。

- **赋值给 `rewards`**:
  - 初始 `rewards` 设置为 KL 散度估计值。

---

##### 2. **计算起止索引**
```python
start = prompts.shape[1] - 1
ends = start + action_mask[:, start:].sum(1) + 1
```
- `start`:
  - 计算生成序列的起始位置，跳过 prompts 的部分，即 `start = prompts.shape[1] - 1`。

- `ends`:
  - 计算每个序列的结束位置，`action_mask[:, start:]` 表示从 `start` 开始的有效 token 掩码。
  - `action_mask[:, start:].sum(1)` 表示每个序列中有效 token 的数量。

---

##### 3. **裁剪奖励分数**
```python
reward_clip = torch.clamp(reward_score, -self.clip_reward_value, self.clip_reward_value)
```
- 使用 `torch.clamp` 对 `reward_score` 的值进行裁剪，限制其范围在 `[-clip_reward_value, clip_reward_value]`。
- 目的：避免因为奖励值过大或过小导致训练不稳定。

---

##### 4. **调整末尾的奖励值**
```python
batch_size = log_probs.shape[0]
for j in range(batch_size):
    rewards[j, start:ends[j]][-1] += reward_clip[j]
```
- 遍历每个样本的序列（`batch_size` 次循环）。
- 对当前序列的 `rewards` 末尾 token（即 `[-1]`）添加裁剪后的 `reward_clip[j]`。
  - `start:ends[j]` 是序列的生成部分。
  - `[-1]` 表示序列最后一个 token 的奖励。

---

#### **输出**
```python
return rewards
```
- 返回每个 token 的奖励值，形状为 `[batch_size, seq_len]`。

---

### **代码主要逻辑总结**

1. **KL 散度惩罚**:
   - 通过 KL 散度鼓励生成的分布接近参考模型，同时允许一定程度的探索。

2. **裁剪奖励**:
   - 对整体奖励进行裁剪，防止异常值影响训练。

3. **生成序列末尾奖励调整**:
   - 对序列的最后一个 token 加上裁剪后的整体奖励，体现对句子整体质量的评价。

4. **返回 token 级别奖励**:
   - 输出每个 token 的奖励，用于后续的策略优化（例如 PPO 算法）。

---

### **在 RLHF 中的作用**
- 这段代码的核心是结合 KL 散度和 Reward 模型的分数，为 Actor 模型的生成策略提供细粒度的奖励信号。
- 最终奖励信号用于计算优势函数（Advantage Function），进而更新策略网络（Actor 模型）。