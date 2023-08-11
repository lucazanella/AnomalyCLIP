import torch.nn as nn
import torch.nn.init as torch_init


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        dropout_prob: float = 0.7,
    ):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
        )
        # self.apply(weight_init)

    def forward(self, x):
        x = self.fc_layers(x)
        return x
