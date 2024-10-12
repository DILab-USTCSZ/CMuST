import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianDistribution:

    def __init__(self, parameters: torch.Tensor):
        # Split mean and log of variance
        self.mean, log_var = torch.chunk(parameters, 2, dim=1)
        # Clamp the log of variances
        self.log_var = torch.clamp(log_var, -30.0, 20.0)
        # Calculate standard deviation
        self.std = torch.exp(0.5 * self.log_var)

    def sample(self):
        # Sample from the distribution
        return self.mean + self.std * torch.randn_like(self.std)
    

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout=0, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Autoencoder(nn.Module):
    
    def __init__(self, encoder, decoder, emb_channels: int, z_channels: int):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # Convolution to map from embedding space to
        # quantized embedding space moments (mean and log variance)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * emb_channels, 1)
        # Convolution to map from quantized embedding space back to
        # embedding space
        self.post_quant_conv = nn.Conv2d(emb_channels, z_channels, 1)

    def encode(self, x: torch.Tensor) :
        # Get embeddings with shape `[batch_size, z_channels * 2, z_height, z_height]`
        z = self.encoder(x)
        # Get the moments in the quantized embedding space
        moments = self.quant_conv(z)
        # Return the distribution
        return GaussianDistribution(moments)

    def decode(self, z: torch.Tensor):
        """
        Decode from latent representation
        """
        # Map to embedding space from the quantized representation
        z = self.post_quant_conv(z)
        return self.decoder(z)


class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
class PositionWiseFeedForward(nn.Module):
    """
    Implements a position-wise feed-forward network with ReLU activation and dropout.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, atten_dim, n_heads=8, mask=False):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.atten_dim = atten_dim
        self.mask = mask

        # Ensure dimensions are divisible by number of heads
        assert atten_dim % n_heads == 0, "Attention dimension must be divisible by number of heads"
        self.head_dim = atten_dim // n_heads

        self.Wq = nn.Linear(q_dim, atten_dim)
        self.Wk = nn.Linear(k_dim, atten_dim)
        self.Wv = nn.Linear(v_dim, atten_dim)

        self.dense = nn.Linear(atten_dim, v_dim)

    def split_heads(self, x, size):
        """ Split the last dimension into (n_heads, depth) and transpose to (batch_size, n_heads, seq_len, depth) """
        x = x.view(size, -1, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(self, query, key, value):
        """ Calculate the attention weights. """
        matmul_qk = torch.matmul(query, key.transpose(-2, -1))

        # Scale matmul_qk
        dk = key.size()[-1]
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        if self.mask:
            mask = torch.tril(torch.ones(scaled_attention_logits.size()))
            scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        output = torch.matmul(attention_weights, value)
        return output, attention_weights

    def forward(self, query, key, value):
        batch_size = query.size(0)
        length = query.size(1)

        query = query.reshape(-1, query.size(-2), query.size(-1))
        key = key.reshape(-1, key.size(-2), key.size(-1))
        value = value.reshape(-1, value.size(-2), value.size(-1))
        
        # Linear projections
        query = self.Wq(query)
        key = self.Wk(key)
        value = self.Wv(value)

        # Split and transpose
        query = self.split_heads(query, batch_size*length)
        key = self.split_heads(key, batch_size*length)
        value = self.split_heads(value, batch_size*length)

        # Apply scaled dot-product attention
        scaled_attention, _ = self.scaled_dot_product_attention(query, key, value)

        # Transpose and reshape
        scaled_attention = scaled_attention.transpose(1, 2).contiguous()
        concat_attention = scaled_attention.view(batch_size*length, -1, self.atten_dim)

        # Final linear layer
        output = self.dense(concat_attention)
        output = output.view(batch_size, length, -1, self.v_dim)

        return output
    
    
class SelfAttentionLayer(nn.Module):
    """
    Implements self-attention followed by a feed-forward network, including normalization and optional dropout.
    """
    def __init__(self, x_dim, atten_dim, ffn_dim=2048, n_heads=8, dropout=0, mask=False):
        super(SelfAttentionLayer, self).__init__()

        self.x_dim = x_dim

        self.attn = MultiHeadAttentionLayer(x_dim, x_dim, x_dim, atten_dim, n_heads, mask)
        self.ffn = PositionWiseFeedForward(x_dim, ffn_dim, dropout)

        self.ln1 = nn.LayerNorm(x_dim)
        self.ln2 = nn.LayerNorm(x_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_start, dim=-2):
        _x = x  # original input
        x = x.transpose(dim, -2)  # transpose input for processing
        residual = x[..., x_start:x_start + self.x_dim]  # residual for later addition
        x_qkv = x[..., x_start:x_start + self.x_dim]  # query, key, and value from input

        attn_out = self.attn(x_qkv, x_qkv, x_qkv)  # self-attention mechanism
        out1 = self.ln1(residual + self.dropout(attn_out))  # layer normalization, and add residual
        out2 = self.ln2(out1 + self.ffn(out1) )  # layer normalization, and add residual

        _out = out2.transpose(dim, -2)  # transpose output back to original shape
        out = torch.cat((_x[..., :x_start], _out, _x[..., x_start + self.x_dim:]), dim=-1)

        return out


class CrossAttentionLayer(nn.Module):
    """
    Implements a cross-attention layer followed by a feed-forward network, including normalization and optional dropout.
    This layer allows for attention across different representations (queries, keys, and values from potentially different sources).
    """
    def __init__(self, q_dim, k_dim, v_dim, atten_dim, ffn_dim=2048, n_heads=8, dropout=0, mask=False):
        super(CrossAttentionLayer, self).__init__()

        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.attn = MultiHeadAttentionLayer(q_dim, k_dim, v_dim, atten_dim, n_heads, mask)
        self.ffn = PositionWiseFeedForward(v_dim, ffn_dim, dropout)

        self.ln1 = nn.LayerNorm(v_dim)
        self.ln2 = nn.LayerNorm(v_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, q_start, k_start, v_start, dim=-2):
        _x = x  # original input
        x = x.transpose(dim, -2)  # transpose input for processing

        residual = x[..., v_start:v_start + self.v_dim]

        query = x[..., q_start:q_start + self.q_dim]
        key = x[..., k_start:k_start + self.k_dim]
        value = x[..., v_start:v_start + self.v_dim]

        attn_out = self.attn(query, key, value)
        out1 = self.ln1(residual + self.dropout(attn_out))
        out2 = self.ln2(out1 + self.ffn(out1))

        _out = out2.transpose(dim, -2)
        out = torch.cat((_x[..., :v_start], _out, _x[..., v_start + self.v_dim:]), dim=-1)

        return out


class MSTI(nn.Module):
    """
    Combines self-attention and cross-attention mechanisms to process data that includes spatial and temporal components.
    Utilizes self-attention for temporal and spatial features independently and cross-attention to integrate observations with
    spatial and temporal embeddings.
    """

    def __init__(self, 
                 obser_start, s_start, t_start, 
                 d_obser, d_s, d_t, h_dim, 
                 self_atten_dim, cross_atten_dim, ffn_dim=2048, n_heads=8, dropout=0, mask=False):
        super(MSTI, self).__init__()

        # Start indices for different features
        self.obser_start = obser_start
        self.s_start = s_start
        self.t_start = t_start

        # Cross-attention layers for integrating observations with temporal and spatial embeddings
        self.cross_attn_x_t = CrossAttentionLayer(
            d_t, d_obser, d_obser, 
            cross_atten_dim, ffn_dim, n_heads, dropout, mask
        )
        self.cross_attn_t_x = CrossAttentionLayer(
            d_obser, d_t, d_t, 
            cross_atten_dim, ffn_dim, n_heads, dropout, mask
        )
        self.cross_attn_x_s = CrossAttentionLayer(
            d_s, d_obser, d_obser, 
            cross_atten_dim, ffn_dim, n_heads, dropout, mask
        )
        self.cross_attn_s_x = CrossAttentionLayer(
            d_obser, d_s, d_s, 
            cross_atten_dim, ffn_dim, n_heads, dropout, mask
        )

        # Self-attention layers for temporal and spatial processing
        self.self_attn_t_t = SelfAttentionLayer(
            h_dim, self_atten_dim, ffn_dim, n_heads, dropout, mask
        )
        self.self_attn_t_t1 = SelfAttentionLayer(
            h_dim, self_atten_dim, ffn_dim, n_heads, dropout, mask
        )
        self.self_attn_s_s = SelfAttentionLayer(
            h_dim, self_atten_dim, ffn_dim, n_heads, dropout, mask
        )
        self.self_attn_s_s1 = SelfAttentionLayer(
            h_dim, self_atten_dim, ffn_dim, n_heads, dropout, mask
        )

    def forward(self, x):
        """
        Forward pass through multiple attention layers.

        Parameters:
        - x: Input tensor

        Returns:
        - Output tensor after processing through multiple attention layers
        """
        # Cross-attention to integrate different feature dimensions
        x = self.cross_attn_x_t(x, q_start=self.t_start, k_start=self.obser_start, v_start=self.obser_start, dim=1)
        x = self.cross_attn_t_x(x, q_start=self.obser_start, k_start=self.t_start, v_start=self.t_start, dim=1)
        x = self.cross_attn_x_s(x, q_start=self.s_start, k_start=self.obser_start, v_start=self.obser_start, dim=2)
        x = self.cross_attn_s_x(x, q_start=self.obser_start, k_start=self.s_start, v_start=self.s_start, dim=2)

        # Self-attention for temporal and spatial dimensions
        x = self.self_attn_t_t(x, x_start=self.obser_start, dim=1)
        x = self.self_attn_t_t1(x, x_start=self.obser_start, dim=1)
        x = self.self_attn_s_s(x, x_start=self.obser_start, dim=2)
        x = self.self_attn_s_s1(x, x_start=self.obser_start, dim=2)

        return x
