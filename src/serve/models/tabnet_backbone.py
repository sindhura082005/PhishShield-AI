import torch
import torch.nn as nn

class SimpleTabNet(nn.Module):
    """
    Compact TabNet-like MLP for tabular signals.
    Keep layer sizes modest for fast inference in a service worker loop.
    """
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)  # binary
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
