import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int, n_classes: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
