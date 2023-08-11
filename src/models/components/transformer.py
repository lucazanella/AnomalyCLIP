import torch
import torch.nn.functional as F
import torch.nn.init as torch_init
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor, nn
from torchvision.transforms import Compose, Resize, ToTensor


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, L: int = 4, drop_p: float = 0.3):
        super().__init__(
            nn.Linear(emb_size, L * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(L * emb_size, emb_size),
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 13, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.scaling = (self.emb_size // num_heads) ** -0.5

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum(
            "bhqd, bhkd -> bhqk", queries, keys
        )  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        att = F.softmax(energy, dim=-1) * self.scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size: int = 768,
        drop_p: float = 0.3,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.3,
        **kwargs
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, L=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__()
        self.reduce = Reduce("b n e -> b e", reduction="mean")
        self.layer_norm = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = x[:, 0, :]  # select only the cls token
        x = x[:, 1:, :]  # select all but the cls token
        x = self.layer_norm(x)
        x = self.linear(x)
        x = self.sigmoid(x)

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, seq_length, input_size: int = 13, emb_size: int = 64):
        super().__init__()

        self.projection = nn.Linear(input_size, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(seq_length + 1, emb_size))
        # self.positions = nn.Parameter(torch.randn(seq_length, emb_size))

    def forward(self, x):
        b, _, _ = x.shape
        x = self.projection(x)

        cls_tokens = repeat(self.cls_token, "() n e -> b n e", b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions

        return x


class Transformer(nn.Sequential):
    def __init__(
        self,
        seq_length: int = 64 * 32,
        input_size: int = 13,
        emb_size: int = 64,
        depth: int = 3,
        n_classes: int = 2,
        **kwargs
    ):
        # apply self.apply(weight_init) to initialize the weights
        super().__init__(
            PatchEmbedding(seq_length, input_size, emb_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes),
        )
        # self.apply(weight_init)
