import torch.nn as nn


# Simple linear project head with LayerNorm
class LinearHead(nn.Module):
    # Initializer
    def __init__(self, hidden_dim, out_dim):
        super(LinearHead, self).__init__()

        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.norm(x)
        return self.linear(x)


# MLP head with LayerNorm and GELU activation
class MLPHead(nn.Module):
    # Initializer
    def __init__(self, hidden_dim, out_dim, r, dropout):
        super(MLPHead, self).__init__()

        reduced_dim = hidden_dim // r
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, reduced_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(reduced_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)
