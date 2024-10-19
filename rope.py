import torch
import math

class RoPE:
    def __init__(self, dim):
        self.dim = dim
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    
    def apply_rotary_pos_emb(self, x, seq_len):

        positions = torch.arange(seq_len, device=x.device).unsqueeze(1)
        
 
        angles = positions * self.inv_freq
        

        sin_emb = torch.sin(angles).repeat(1, 2)
        cos_emb = torch.cos(angles).repeat(1, 2)
        
  
        x_rotated = x * cos_emb + self.rotate_half(x) * sin_emb
        return x_rotated
    
    def rotate_half(self, x):
      
        x1, x2 = x[..., ::2], x[..., 1::2]

        return torch.cat([-x2, x1], dim=-1)

# Example usage
batch_size = 2
seq_len = 6
dim = 4


x = torch.randn(batch_size, seq_len, dim)


rope = RoPE(dim)


x_with_rope = rope.apply_rotary_pos_emb(x, seq_len)

print(x_with_rope.shape)  
