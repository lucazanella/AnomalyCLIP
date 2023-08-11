from axial_attention import AxialImageTransformer
from einops import rearrange
from torch import nn

from src.models.components.classification_head import ClassificationHead
from src.models.components.mlp import MLP
from src.models.components.mtn import Aggregate
from src.models.components.transformer import Transformer


class TemporalModel(nn.Module):
    def __init__(
        self,
        temporal_module: str,
        input_size: int,
        emb_size: int,
        output_size: int,
        dropout_prob: float,
        heads: int,
        dim_heads: int,
        depth: int,
        num_segments: int,
        seg_length: int,
    ):
        super().__init__()

        assert temporal_module in [
            "axial",
            "transf_short",
            "transf_long",
            "mlp",
            "mtn",
        ], "Invalid temporal module"

        self.temporal_module = temporal_module
        self.input_size = input_size
        self.emb_size = emb_size
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.heads = heads
        self.dim_heads = dim_heads
        self.depth = depth
        self.num_segments = num_segments
        self.seg_length = seg_length

        if self.temporal_module == "axial":
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

        elif self.temporal_module == "transf_short" or self.temporal_module == "transf_long":
            seq_length = (
                self.seg_length if self.temporal_module == "transf_short" else self.num_segments
            )
            self.transformer = Transformer(
                seq_length=seq_length,
                input_size=input_size,
                emb_size=self.emb_size,
                depth=self.depth,
                n_classes=output_size,
            )

        elif self.temporal_module == "mlp":
            mlp_hidden_size = self.emb_size
            self.mlp = MLP(
                input_size,
                output_size,
                hidden_size=mlp_hidden_size,
                dropout_prob=self.dropout_prob,
            )

        elif self.temporal_module == "mtn":
            self.mtn = Aggregate(len_feature=input_size)
            self.classifier = ClassificationHead(input_size, output_size)

    def forward(self, features, segment_size, test_mode):
        if self.temporal_module == "axial":
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

        elif self.temporal_module == "transf_short":
            if test_mode:
                features = rearrange(
                    features,
                    "(b n s l) d -> (b n s) l d",
                    n=self.num_segments,
                    s=segment_size,
                    l=self.seg_length,
                )
            else:
                features = rearrange(
                    features,
                    "(b n l) d -> (b n) l d",
                    n=self.num_segments,
                    l=self.seg_length,
                )
            # Common steps for all cases
            scores = self.transformer(features)

        elif self.temporal_module == "transf_long":
            if test_mode:
                features = rearrange(
                    features,
                    "(b n s l) d -> (b s) n l d",
                    n=self.num_segments,
                    s=segment_size,
                    l=self.seg_length,
                )
            else:
                features = rearrange(
                    features,
                    "(b n l) d -> b n l d",
                    n=self.num_segments,
                    l=self.seg_length,
                )
            # Common steps for all cases
            features = features.mean(dim=2)
            scores = self.transformer(features)
            if test_mode:
                scores = rearrange(
                    scores,
                    "(b s) n d -> (b n s) d",
                    n=self.num_segments,
                    s=segment_size,
                )
            else:
                scores = rearrange(scores, "b n d -> (b n) d")
            scores = scores.repeat_interleave(self.seg_length, dim=0)

        elif self.temporal_module == "mlp":
            scores = self.mlp(features)

        elif self.temporal_module == "mtn":
            if test_mode:
                features = rearrange(
                    features,
                    "(b n s l) d -> (b s) n l d",
                    n=self.num_segments,
                    s=segment_size,
                    l=self.seg_length,
                )
            else:
                features = rearrange(
                    features,
                    "(b n l) d -> b n l d",
                    n=self.num_segments,
                    l=self.seg_length,
                )
            features = features.mean(dim=2)
            features = self.mtn(features)

            if test_mode:
                features = rearrange(features, "(b s) n d -> (b n s) d", s=segment_size)
            else:
                features = rearrange(features, "b n d -> (b n) d")

            scores = self.classifier(features)
            scores = scores.repeat_interleave(self.seg_length, dim=0)

        return scores
