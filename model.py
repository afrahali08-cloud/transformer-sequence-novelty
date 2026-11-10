"""A small transformer encoder for fixed-length, univariate time series."""
import torch
from torch import nn

CLASS_NAMES = ["normal", "cyclic", "increasing trend", "decreasing trend", "upward shift"]


class SequenceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Linear(6, 32)
        self.position = nn.Parameter(torch.randn(1, 10, 32) * .02)
        layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64, dropout=.1,
            activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        # TransformerEncoder clones the prototype. Initialize layers independently.
        for block in self.encoder.layers:
            for parameter in block.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
        self.head = nn.Linear(32, len(CLASS_NAMES))

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != 60 or not torch.isfinite(x).all():
            raise ValueError("Expected finite float sequences of shape (batch, 60)")
        tokens = self.patch(x.reshape(-1, 10, 6)) + self.position
        embedding = self.encoder(tokens).mean(dim=1)
        return self.head(embedding), embedding


def novelty_reference(embeddings, labels):
    centers = torch.stack([embeddings[labels == k].mean(0) for k in range(5)])
    residuals = embeddings - centers[labels]
    variance = residuals.square().mean(0).clamp_min(.01)
    return centers, variance


def novelty_score(embeddings, centers, variance):
    """Minimum standardized squared embedding distance to a known class center."""
    return ((embeddings[:, None] - centers[None]).square() / variance).mean(2).min(1).values
