import pandas as pd
import torch
from dotmap import DotMap
from einops import rearrange
from torch import nn

from src import utils
from src.models.components.clip import clip
from src.models.components.coop import PromptLearner
from src.models.components.selector_model import SelectorModel
from src.models.components.temporal_model import TemporalModel
from src.models.components.text_encoder import TextEncoder, TextEncoderZeroshot
from src.models.components.zeroshot_classifier import zeroshot_classifier

log = utils.get_pylogger(__name__)


class AnomalyCLIP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        config = DotMap(**kwargs)

        (
            self.arch,
            self.labels_file,
            self.emb_size,
            self.depth,
            self.heads,
            self.dim_heads,
            self.num_segments,
            self.seg_length,
            self.concat_features,
            self.normal_id,
            self.stride,
            self.load_from_features,
            self.select_idx_dropout_topk,
            self.select_idx_dropout_bottomk,
            self.ncrops,
            self.num_topk,
            self.num_bottomk,
            self.selector_module,
            self.temporal_module,
            self.dropout_prob,
            self.direction_module,
            self.batch_norm,
            self.feature_size,
            self.use_similarity_as_features,
        ) = (
            config.arch,
            config.labels_file,
            config.emb_size,
            config.depth,
            config.heads,
            config.dim_heads,
            config.num_segments,
            config.seg_length,
            config.concat_features,
            config.normal_id,
            config.stride,
            config.load_from_features,
            config.select_idx_dropout_topk,
            config.select_idx_dropout_bottomk,
            config.ncrops,
            config.num_topk,
            config.num_bottomk,
            config.selector_module,
            config.temporal_module,
            config.dropout_prob,
            config.direction_module,
            config.batch_norm,
            config.feature_size,
            config.use_similarity_as_features,
        )

        clip_model, preprocess = clip.load(
            self.arch,
            device="cpu",
        )  # Must set jit=False for training  ViT-B/32

        # CLIP's default precision is fp16
        clip_model.float()

        classes_df = pd.read_csv(self.labels_file)
        classnames = sorted(c for i, c in classes_df.values.tolist())

        self.embedding_dim = clip_model.ln_final.weight.shape[0]
        self.image_encoder = clip_model.visual
        self.token_embedding = clip_model.token_embedding

        if self.direction_module == "learned_no_encoder":
            prompts = torch.randn(len(classnames), self.embedding_dim, dtype=clip_model.dtype)
            self.prompts = nn.Parameter(prompts)
            self.text_encoder = None
        elif self.direction_module == "engineered":
            self.text_encoder = TextEncoderZeroshot(clip_model)
            self.zeroshot_weights = zeroshot_classifier(self.text_encoder, classnames)
        elif self.direction_module.startswith("learned_encoder"):
            self.prompt_learner = PromptLearner(config, classnames, clip_model)
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts
            self.text_encoder = TextEncoder(clip_model)

        self.selector_model = SelectorModel(
            selector_module=self.selector_module,
            classnames=classnames,
            normal_id=self.normal_id,
            logit_scale=clip_model.logit_scale,
            batch_norm=self.batch_norm,
            num_segments=self.num_segments,
            seg_length=self.seg_length,
            select_idx_dropout_topk=self.select_idx_dropout_topk,
            select_idx_dropout_bottomk=self.select_idx_dropout_bottomk,
            num_topk=self.num_topk,
            num_bottomk=self.num_bottomk,
        )

        additional_concat = len(classnames) - 1
        input_size = self.embedding_dim + (len(classnames) - 1) * self.concat_features
        input_size = additional_concat if self.use_similarity_as_features else input_size
        output_size = 1

        self.temporal_model = TemporalModel(
            temporal_module=self.temporal_module,
            input_size=self.feature_size,  # 512 or 2048 ?
            emb_size=self.emb_size,
            output_size=output_size,
            dropout_prob=self.dropout_prob,
            heads=self.heads,
            dim_heads=self.dim_heads,
            depth=self.depth,
            num_segments=self.num_segments,
            seg_length=self.seg_length,
            i3d=True,
        )

        self.feature_projection = (
            nn.Linear(self.feature_size, input_size)
            if self.feature_size != self.embedding_dim
            else nn.Identity()
        )

    def forward(
        self,
        image_features,
        image_features_i3d,
        labels,
        ncentroid,
        segment_size=1,
        test_mode=False,
    ):
        ncentroid = ncentroid.to(image_features.device)

        if test_mode:
            if not self.load_from_features:
                b, t, c, h, w = image_features.size()
                image_features = image_features.view(-1, c, h, w)
                image_features = self.image_encoder(
                    image_features
                )  # (b * self.num_segments * self.seg_lenght, 512 )
                image_features = rearrange(
                    image_features,
                    "(b ncrops n s l) d -> b ncrops (n s l) d",
                    ncrops=self.ncrops,
                    n=self.num_segments,
                    s=segment_size,
                    l=self.seg_length,
                )
            b, ncrops, t, d = image_features.shape

            image_features = image_features.view(-1, t, d)

            text_features = self.get_text_features()

            image_features_i3d = torch.squeeze(image_features_i3d, dim=1)
            image_features_i3d = image_features_i3d.view(-1, image_features_i3d.shape[-1])

            similarity = self.selector_model(
                image_features,
                text_features,
                labels,
                ncentroid,
                test_mode,
            )

            # Re-center the features
            # TODO: ablate the transformation
            if self.selector_module != "cosine":
                image_features = image_features - ncentroid

            features = self.get_temporal_model_input(image_features_i3d, similarity)

            scores = self.temporal_model(features, segment_size, test_mode)

            similarity = similarity.repeat_interleave(self.stride, dim=0)
            scores = scores.view(-1)
            scores = scores.repeat_interleave(self.seg_length, dim=0)

            return similarity, scores

        else:
            if not self.load_from_features:
                (b, t, c, h, w) = image_features.size()
                image_features = image_features.view(-1, c, h, w)

                chunk_size = 16
                image_features_list = []

                for i in range(0, image_features.shape[0], chunk_size):
                    chunk = image_features[i : i + chunk_size]
                    encoded_chunk = self.image_encoder(chunk)
                    image_features_list.append(encoded_chunk)

                image_features = torch.cat(image_features_list, dim=0)

                image_features = rearrange(
                    image_features,
                    "(b ncrops n l) d -> b ncrops (n l) d",
                    ncrops=self.ncrops,
                    n=self.num_segments,
                    l=self.seg_length,
                )

            (
                b,
                ncrops,
                t,
                d,
            ) = image_features.shape  # batch_size, ncrops, num_segments * seg_length, 512

            image_features = torch.squeeze(
                image_features
            )  # batch_size, num_segments * seg_length, 512
            image_features = image_features.view(-1, d)

            text_features = self.get_text_features()

            image_features_i3d = torch.squeeze(image_features_i3d)
            image_features_i3d = image_features_i3d.view(-1, image_features_i3d.shape[-1])

            (
                logits,
                logits_topk,
                logits_bottomk,
                idx_topk_abn,
                idx_topk_nor,
                idx_bottomk_abn,
            ) = self.selector_model(
                image_features,
                text_features,
                labels,
                ncentroid,
                test_mode,
            )

            # Re-center the features
            # TODO: ablate the transformation
            if self.selector_module != "cosine":
                image_features = image_features - ncentroid

            features = self.get_temporal_model_input(image_features_i3d, logits)
            scores = self.temporal_model(features, segment_size, test_mode)

            scores = scores.view(-1)  # (batch*num_segments*seg_length)
            scores = scores.repeat_interleave(self.seg_length, dim=0)
            return (
                logits,
                logits_topk,
                logits_bottomk,
                scores,
                idx_topk_abn,
                idx_topk_nor,
                idx_bottomk_abn,
            )

    def get_text_features(self):
        if self.direction_module == "learned_no_encoder":
            text_features = self.prompts
        elif self.direction_module == "engineered":
            text_features = self.zeroshot_weights
        elif self.direction_module.startswith("learned_encoder"):
            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)  # num_classes, 512
        return text_features

    def get_temporal_model_input(self, image_features, similarity):
        image_features = image_features.view(-1, image_features.shape[-1])

        # Concatenate similarity scores with image features if needed
        combined_features = (
            torch.cat((similarity, image_features), dim=-1)
            if self.concat_features
            else image_features
        )

        # Use similarity scores as features if specified, otherwise keep the combined_features features
        features = similarity if self.use_similarity_as_features else combined_features

        return features


if __name__ == "__main__":
    _ = AnomalyCLIP()
