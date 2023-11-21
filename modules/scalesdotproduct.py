from torch import nn
import torch

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # q, k, v: [(batch_size * n_heads) x seq_len x hidden_size]
        
        # Calculate attention scores
        attn = torch.bmm(q, k.transpose(1, 2))  # (batch_size * n_heads) x seq_len x seq_len
        attn = attn / self.temperature  # Scale by temperature
        
        # Apply mask (if provided)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))  # Set masked positions to -inf for softmax
        
        # Perform softmax to get attention weights
        attn = self.softmax(attn)  # (batch_size * n_heads) x seq_len x seq_len
        
        # Apply dropout
        attn = self.dropout(attn)
        
        # Weight values by the attention scores
        output = torch.bmm(attn, v)  # (batch_size * n_heads) x seq_len x hidden_size
        
        return output, attn