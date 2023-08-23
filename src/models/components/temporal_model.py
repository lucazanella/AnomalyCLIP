from axial_attention import AxialImageTransformer
from einops import rearrange
from torch import nn

from src.models.components.classification_head import ClassificationHead


class TemporalModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        emb_size: int,
        output_size: int,
        heads: int,
        dim_heads: int,
        depth: int,
        num_segments: int,
        seg_length: int,
    ):
        super().__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.output_size = output_size
        self.heads = heads
        self.dim_heads = dim_heads
        self.depth = depth
        self.num_segments = num_segments
        self.seg_length = seg_length

        self.projection = nn.Linear(self.input_size, self.emb_size)
        self.axial_attn = AxialImageTransformer(
            dim=self.emb_size,
            depth=self.depth,
            heads=self.heads,
            dim_heads=self.dim_heads,
            reversible=True,
            axial_pos_emb_shape=(self.num_segments, self.seg_length),
        )
        self.classifier = ClassificationHead(self.emb_size, output_size)

    def forward(self, features, segment_size, test_mode):
        features = self.projection(features)

        if test_mode:
            features = rearrange(
                features,
                "(b n s l) d -> b n s l d",
                n=self.num_segments,
                s=segment_size,
                l=self.seg_length,
            )
            features = rearrange(features, "b n s l d -> (b s) n l d")
        else:
            features = rearrange(
                features,
                "(b n l) d -> b n l d",
                n=self.num_segments,
                l=self.seg_length,
            )

        features = rearrange(features, "b n l d -> b d n l")

        features = self.axial_attn(
            features
        )  # (batch_size, num_segments, seg_length, num_classes - 1)
        features = rearrange(features, "b d n l -> b n l d")

        if test_mode:
            features = rearrange(features, "(b s) n l d -> b n s l d", s=segment_size)
            features = rearrange(features, "b n s l d -> (b n s l) d")
        else:
            features = rearrange(features, "b n l d -> (b n l) d ")

        scores = self.classifier(features)

        return scores
