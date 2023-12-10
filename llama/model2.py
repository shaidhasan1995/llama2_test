import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_head_size = int(embedding_dim / num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size

        self.query = nn.Linear(embedding_dim, self.all_head_size)
        self.key = nn.Linear(embedding_dim, self.all_head_size)
        self.value = nn.Linear(embedding_dim, self.all_head_size)

        self.out = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        query = self.query(query).view(batch_size, -1, self.num_heads, self.attention_head_size).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.attention_head_size).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.attention_head_size).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, -1, self.all_head_size)
        attention_output = self.out(context_layer)
        return attention_output

# Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.linear2 = nn.Linear(embedding_dim * 4, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# Transformer Layer
class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(embedding_dim, n_heads)
        self.ffn = FeedForward(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.norm1(x)
        x = x + self.dropout(self.attention(x1, x1, x1))
        x2 = self.norm2(x)
        x = x + self.dropout(self.ffn(x2))
        return x

# Simple Transformer Model
class SimpleTransformer(nn.Module):
    def __init__(self, params, embedding_dim = 768, n_heads = 8, dropout=0):
        super(SimpleTransformer, self).__init__()
        self.params = params
        self.embedding = nn.Embedding(params.vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, params.max_seq_len)
        self.layers = nn.ModuleList([TransformerLayer(embedding_dim, n_heads, dropout) for _ in range(params.n_layers)])
        self.output_layer = nn.Linear(embedding_dim, params.vocab_size)

    def weight_initialization(self):
        def init_func(m):
            if isinstance(m, nn.Linear):
                # Using Xavier normal initialization
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # Kaiming initialization is typically used for Conv2d layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # Add other initializations if needed

        self.apply(init_func)

    def forward(self, tokens, prev_pos = 0, learning = None): # so can use same as other model
        x = self.embedding(tokens)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x)

        logits = self.output_layer(x)
        return logits
