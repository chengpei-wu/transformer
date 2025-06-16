import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    """
    transform input tokens into embeddings
    """

    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # to ensure that the embedding is scaled by sqrt(d_model)
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    positional encoding
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Compute the positional encodings once in log space.
        # an implementation trick: a^x === exp(x * log(a))
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # add a batch dimension
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # position encoding is added to the input embeddings, and it is not trainable
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    """
    Scaled Dot-Product Attention
    """

    # note that query, key, and value are all 3D (4D) tensors (their shape is (b, s, d) or (b, h, s, d_k))
    d_k = query.size(-1)

    # compute attention scores: scores = QK^T / sqrt(d_k)
    # multiplying in matmul() is based on the last two dimensions
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    """

    def __init__(self, num_head, d_model, drop=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head

        # d_model is the dimension of the input embeddings, and it should be divisible by num_head in this implementation
        assert d_model % num_head == 0

        # d_k is the dimension of each head's query, key, and value
        self.d_k = d_model // num_head

        # for each w, it mathematically equals to num_heads of nn.Linear(output_dim, d_k),
        # this is a parallel computation trick to speed up the computation
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=drop)

    def forward(self, query, key, value, mask=None):
        b = value.size(0)

        if mask is not None:
            # assure mask can be applied to all h heads.
            mask = mask.unsqueeze(1)

        # split q, k, and v into num_heads of sub- q, k, and v
        query = self.w_q(query).view(b, -1, self.num_head, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(b, -1, self.num_head, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(b, -1, self.num_head, self.d_k).transpose(1, 2)

        # compute (masked) attention
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(b, -1, self.d_model)

        return self.w_o(x)


class LayerNorm(nn.Module):
    def __init__(self, feat_size, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(feat_size))
        self.b = nn.Parameter(torch.zeros(feat_size))
        self.eps = eps

    def forward(self, x):
        # shape of x: (b, s, d)
        # where b is batch size, s is the length of input sequence, d is the feature dimension

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # shape of mean and std: (b, s, 1)

        return self.w * (x - mean) / (std + self.eps) + self.b


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class Encoder(nn.Module):
    def __init__(self, num_layer, size, num_head, d_model, d_ff):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(size, num_head, d_model, d_ff) for _ in range(num_layer)]
        )

    def forward(self, x, mask):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layer, size, num_head, d_model, d_ff):
        super().__init__()
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(size, num_head, d_model, d_ff) for _ in range(num_layer)]
        )

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, size, num_head, d_model, d_ff):
        super().__init__()
        self.mul_att = MultiHeadAttention(num_head=num_head, d_model=d_model)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.layer_norm1 = LayerNorm(feat_size=size)
        self.layer_norm2 = LayerNorm(feat_size=size)

    def forward(self, x, mask):
        x = self.layer_norm1(x + self.mul_att(x, x, x, mask))
        x = self.layer_norm2(x + self.ff(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, size, num_head, d_model, d_ff):
        super().__init__()
        self.self_mul_att = MultiHeadAttention(num_head=num_head, d_model=d_model)
        self.src_mul_att = MultiHeadAttention(num_head=num_head, d_model=d_model)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.layer_norm1 = LayerNorm(feat_size=size)
        self.layer_norm2 = LayerNorm(feat_size=size)
        self.layer_norm3 = LayerNorm(feat_size=size)

    def forward(self, x, memory, src_mask, tgt_mask):
        encoder_x = memory
        x = self.layer_norm1(x + self.self_mul_att(x, x, x, tgt_mask))
        x = self.layer_norm2(x + self.src_mul_att(x, encoder_x, encoder_x, src_mask))
        x = self.layer_norm2(x + self.ff(x))
        return x


class OutputGenerator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
