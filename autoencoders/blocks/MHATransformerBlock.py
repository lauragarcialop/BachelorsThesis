import torch
import torch.nn as nn
import torch.nn.functional as F

from performer_pytorch import SelfAttention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, embed_dim=None, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        
        if embed_dim is None:
            self.embed_dim = input_dim
        else:
            self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert self.embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = self.embed_dim // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(input_dim, self.embed_dim * 3)
        self.out_proj = nn.Linear(self.embed_dim, input_dim)  # Output projection

        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute attention scores and return weighted values.
        """
        d_k = Q.shape[-1]  # Dimension per head
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # Softmax over last dimension
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)  # Apply attention weights
        
        return output, attn_weights

    def forward(self, x, mask=None):
        batch_size, _ = x.shape
        
        # Linear projection into Q, K, V
        qkv = self.qkv_proj(x)  # Shape: (batch, 3 * embed_dim)
        Q, K, V = torch.chunk(qkv, 3, dim=-1)  # Split into (batch, embed_dim)
        
        # Reshape into multiple heads: (batch, num_heads, head_dim)
        Q = Q.view(batch_size, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)
        
        # Compute attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Merge attention heads back: (batch, seq_len, embed_dim)
        attn_output = attn_output.contiguous().view(batch_size, self.embed_dim)
        
        # Final linear projection
        output = self.out_proj(attn_output)
        
        return output, attn_weights



class TransformerBlock(nn.Module):
    def __init__(self, params):
        super(TransformerBlock, self).__init__()

        # Multi-head self-attention
        if params['transformer'] == 'multihead_attention':
            self.attention = MultiHeadSelfAttention(params['input_dim'], params['num_heads'], embed_dim=params['intrinsic_dim'], dropout=params['dropout'])
        else: # params['transformer'] == 'performer':
            self.attention = SelfAttention(params['intrinsic_dim'], depth=1, heads=params['num_heads'], causal=True)

        # Feedforward layers
        self.ffn = nn.Sequential(
            nn.Linear(params['input_dim'], params['intrinsic_dim']),
            nn.ReLU(),
            nn.Linear(params['intrinsic_dim'], params['input_dim'])
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(params['input_dim'])
        self.norm2 = nn.LayerNorm(params['input_dim'])
        
        # Dropout layers
        self.dropout1 = nn.Dropout(params['dropout'])
        self.dropout2 = nn.Dropout(params['dropout'])
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_output, _ = self.attention(x)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feedforward network with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x