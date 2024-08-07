import torch
import torch.nn as nn
from einops import rearrange
import copy  # copy 모듈 import 추가

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert self.head_dim * n_heads == hidden_dim, "Warning: Check Head, Embed_Size"

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]).to(device))

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = rearrange(Q, 'b t (h d) -> b h t d', h=self.n_heads)
        K = rearrange(K, 'b t (h d) -> b h t d', h=self.n_heads)
        V = rearrange(V, 'b t (h d) -> b h t d', h=self.n_heads)

        attention_score = Q @ K.transpose(2, 3) / self.scale
        attention_weights = torch.softmax(attention_score, dim=-1)
        attention = attention_weights @ V
        attention = attention.contiguous()
        x = rearrange(attention, 'b h t d -> b t (h d)')
        return self.fc_o(x), attention_weights

class FeedForward(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout_ratio):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, pf_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(pf_dim, hidden_dim)
        )

    def forward(self, x):
        x = self.linear(x)
        return x

class TransformerEncoderLayer_PostResidualLN(nn.Module):
    def __init__(self, hidden_dim, pf_dim, n_heads, dropout_ratio, device):
        super().__init__()
        self.self_atten = MultiHeadAttention(hidden_dim, n_heads, dropout_ratio, device)
        self.FF = FeedForward(hidden_dim, pf_dim, dropout_ratio)
        self.self_atten_LN = nn.LayerNorm(hidden_dim)
        self.FF_LN = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        residual, atten_enc = self.self_atten(x, x, x)
        residual = self.dropout(residual)
        x = x + residual
        x = self.self_atten_LN(x)
        residual = self.dropout(self.FF(x))
        x = x + residual
        x = self.FF_LN(x)
        return x, atten_enc

class TransformerEncoder(nn.Module):
    def __init__(self, layer, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, x):
        output = x
        for layer in self.layers:
            output, _ = layer(output)
        return output

class FeatureTokenizer(nn.Module):
    def __init__(self, num_numerical_cols, hidden_dim, device):
        super().__init__()
        self.num_numerical_cols = num_numerical_cols
        self.hidden_dim = hidden_dim
        self.numerical_projection = nn.ModuleList([nn.Linear(1, hidden_dim) for _ in range(num_numerical_cols)])
        self.mask_embedding = nn.Parameter(torch.randn(1, hidden_dim))
        self.device = device

    def forward(self, x_numerical, mask=None):
        x_numerical = x_numerical.to(self.device)
        numerical_features = [emb_num(x_numerical[:, i].unsqueeze(1)).unsqueeze(1) for i, emb_num in enumerate(self.numerical_projection)]
        numerical_features = torch.cat(numerical_features, dim=1)

        if mask is not None:
            masked_features = numerical_features * (1 - mask.unsqueeze(2)) + self.mask_embedding * mask.unsqueeze(2)
        else:
            masked_features = numerical_features

        return masked_features

class FTTransformer(nn.Module):
    def __init__(self, num_numerical_cols, hidden_dim, pf_dim, num_heads, num_layers, output_dim, dropout_ratio, device):
        super().__init__()
        self.feature_tokenizer = FeatureTokenizer(num_numerical_cols, hidden_dim, device)
        encoder_layer = TransformerEncoderLayer_PostResidualLN(hidden_dim, pf_dim, num_heads, dropout_ratio, device)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.output_layer1 = nn.Linear(hidden_dim, 1)

    def forward(self, x_numerical, mask=None):
        x = self.feature_tokenizer(x_numerical, mask)
        x = self.transformer_encoder(x)
        x = self.output_layer1(x)
        x = x.squeeze(-1)
        return x

class FTTransformerNew(nn.Module):
    def __init__(self, num_numerical_cols, hidden_dim, pf_dim, num_heads, num_layers, dropout_ratio, device):
        super().__init__()
        self.feature_tokenizer = FeatureTokenizer(num_numerical_cols, hidden_dim, device)
        encoder_layer = TransformerEncoderLayer_PostResidualLN(hidden_dim, pf_dim, num_heads, dropout_ratio, device)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_dim * num_numerical_cols, 1)

    def forward(self, x_numerical, mask=None):
        x = self.feature_tokenizer(x_numerical, mask)
        x = self.transformer_encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
