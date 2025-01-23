import torch
import math

class RoPE:
    def __init__(self, dim):
        """
        初始化RoPE类，计算逆频率（inv_freq），该逆频率用于构建旋转位置编码。

        参数：
        - dim: 输入向量的维度，必须是偶数，因为位置编码是针对偶数索引和奇数索引分别进行的。
        """
        self.dim = dim
        # 计算逆频率inv_freq，维度为`dim // 2`。`torch.arange(0, dim, 2)`生成从0开始到`dim`的偶数索引。
        # `10000`是常见的基数，`torch.arange(0, dim, 2).float() / dim`将偶数索引归一化到[0, 1]的区间。
        # 计算得到的逆频率会用于后续的正弦和余弦函数构造位置编码。
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    
    def apply_rotary_pos_emb(self, x, seq_len):
        """
        应用旋转位置编码到输入张量x上。

        参数：
        - x: 输入的张量，形状为(batch_size, seq_len, dim)，表示批量大小、序列长度和维度。
        - seq_len: 序列的长度。

        返回：
        - 返回一个与输入x形状相同的张量，应用了旋转位置编码。
        """
        
        # 生成位置索引，形状为(seq_len, 1)，在序列长度的维度上进行扩展。
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1)
        
        # 将位置索引与逆频率相乘，得到角度，形状为(seq_len, dim // 2)。
        angles = positions * self.inv_freq
        
        # 计算正弦和余弦编码，分别扩展到维度`dim`大小。
        # `sin_emb`和`cos_emb`的形状为(seq_len, dim)。
        sin_emb = torch.sin(angles).repeat(1, 2)  # 重复以适应dim维度
        cos_emb = torch.cos(angles).repeat(1, 2)

        # 使用正弦和余弦编码对输入张量进行旋转。
        # x * cos_emb对输入张量进行缩放，而rotate_half(x) * sin_emb对输入张量进行旋转。
        # 最终结果是将这些值求和得到旋转后的张量。
        x_rotated = x * cos_emb + self.rotate_half(x) * sin_emb
        return x_rotated
    
    def rotate_half(self, x):
        """
        将张量的后一半进行旋转。具体来说，将张量的奇数索引部分变为负值，并交换偶数索引部分和奇数索引部分。

        参数：
        - x: 形状为(batch_size, seq_len, dim)的输入张量。

        返回：
        - 返回一个与输入张量形状相同的张量，但后半部分进行了旋转。
        """
        # 将输入张量的偶数索引部分和奇数索引部分分别取出。
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        # 将奇数索引部分x2变负，并与偶数索引部分x1拼接，交换它们的位置。
        # 最终返回的张量形状仍然是(batch_size, seq_len, dim)。
        return torch.cat([-x2, x1], dim=-1)

# 示例使用
batch_size = 2
seq_len = 6
dim = 4

# 随机生成一个形状为(batch_size, seq_len, dim)的输入张量x。
x = torch.randn(batch_size, seq_len, dim)

# 创建RoPE对象，指定维度dim。
rope = RoPE(dim)

# 应用旋转位置编码到输入张量x上。
x_with_rope = rope.apply_rotary_pos_emb(x, seq_len)

# 打印结果张量的形状，应该与输入张量的形状相同。
print(x_with_rope.shape)
