import pandas as pd
import torch
from axial_attention import AxialImageTransformer
from dotmap import DotMap
from einops import rearrange
from torch import nn

from src import utils
from src.models.components.classification_head import ClassificationHead
from src.models.components.clip import clip
from src.models.components.coop import PromptLearner
from src.models.components.text_encoder import TextEncoder

log = utils.get_pylogger(__name__)


class AnomalyCLIP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        config = DotMap(**kwargs)

        (
            self.arch,
            self.labels_file,
            self.emb_size,
            self.attn_layers,
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
            self.classification_head,
            self.dropout_prob,
        ) = (
            config.arch,
            config.labels_file,
            config.emb_size,
            config.attn_layers,
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
            config.classification_head,
            config.dropout_prob,
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
        self.prompt_learner = PromptLearner(config, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)

        self.image_encoder = clip_model.visual
        self.token_embedding = clip_model.token_embedding

        self.bn_layer = nn.BatchNorm1d(len(classnames) - 1, affine=False)

        input_size = self.embedding_dim
        if self.concat_features:
            input_size = input_size + len(classnames) - 1

        self.projection = nn.Linear(input_size, self.emb_size)

        self.axial_attn = AxialImageTransformer(
            dim=self.emb_size,
            depth=self.attn_layers,
            heads=self.heads,
            dim_heads=self.dim_heads,
            reversible=True,
            axial_pos_emb_shape=(self.num_segments, self.seg_length),
        )

        output_size = 1
        self.classifier = ClassificationHead(self.emb_size, output_size)

    def forward(
        self,
        image_features,
        labels,
        ncentroid,
        segment_size=1,
        test_mode=False,
    ):
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

            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)  # num_classes, 512

            text_features_except_normal = torch.cat(
                (
                    text_features[: self.normal_id],
                    text_features[(self.normal_id + 1) :],
                ),
                dim=0,
            )

            ncentroid = ncentroid.to(image_features.device)

            text_features = text_features_except_normal - ncentroid  # num_classes - 1, 512

            image_features = torch.reshape(
                image_features, (-1, image_features.shape[-1])
            )  # (ncrops * ncrops * t // 16, n_cls)

            image_features = image_features - ncentroid

            # Normalization
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )  # num_classes - 1, num_classes - 1

            logits = (
                image_features @ text_features.T
            )  # (batch * ncrops * num_segments, num_classes - 1)

            similarity = self.bn_layer(logits)

            logits = logits.view(-1, logits.shape[-1])
            similarity = similarity.view(-1, similarity.shape[-1])

            features = image_features.view(-1, image_features.shape[-1])
            if self.concat_features:
                features = torch.cat((similarity, features), dim=-1)

            features = self.projection(features)
            features = features.view(
                -1,
                self.num_segments,
                segment_size,
                self.seg_length,
                features.shape[-1],
            )

            # Axial Transformer
            features = rearrange(features, "b n s l d -> (b s) n l d")
            features = rearrange(features, "b n l d -> b d n l")
            features = self.axial_attn(
                features
            )  # (batch_size, num_segments, seg_length, num_classes - 1)
            features = rearrange(features, "b d n l -> b n l d")
            features = rearrange(features, "(b s) n l d -> b n s l d", s=segment_size)
            features = rearrange(features, "b n s l d -> (b n s l) d")

            scores = self.classifier(features)

            logits = logits.repeat_interleave(self.stride, dim=0)
            similarity = similarity.repeat_interleave(self.stride, dim=0)
            scores = scores.repeat_interleave(self.stride, dim=0)

            scores = scores.view(-1)

            return logits, similarity, scores

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

            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)  # num_classes, 512

            text_features_except_normal = torch.cat(
                (
                    text_features[: self.normal_id],
                    text_features[(self.normal_id + 1) :],
                ),
                dim=0,
            )

            ncentroid = ncentroid.to(image_features.device)

            text_features = text_features_except_normal - ncentroid  # num_classes - 1, 512

            image_features = image_features - ncentroid
            image_features = image_features.view(
                -1, image_features.shape[-1]
            )  # (batch * num_segments * seg_length, 512)

            # Normalization
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )  # num_classes - 1, num_classes - 1

            logits = (
                image_features @ text_features.T
            )  # (batch * num_segments * seg_length, num_classes - 1)

            logits = self.bn_layer(logits)

            logits = logits.view(
                -1, self.ncrops * self.num_segments * self.seg_length, logits.shape[-1]
            )  # (batch, num_segments*seg_length, n_cls)

            # Generate a mask with the desired percentage of zeros for each row
            select_idx = torch.ones((logits.shape[0], self.num_segments))
            topk_select_idx = select_idx * (1 - self.select_idx_dropout_topk)
            bottomk_select_idx = select_idx * (1 - self.select_idx_dropout_bottomk)

            topk_mask = torch.bernoulli(topk_select_idx)
            bottomk_mask = torch.bernoulli(bottomk_select_idx)
            topk_mask = topk_mask.unsqueeze(dim=2).expand(
                [-1, -1, logits.shape[-1]]
            )  # (batch, num_segments, n_cls)
            bottomk_mask = bottomk_mask.unsqueeze(dim=2).expand(
                [-1, -1, logits.shape[-1]]
            )  # (batch, num_segments, n_cls)
            topk_mask = topk_mask.to(image_features.device)
            bottomk_mask = bottomk_mask.to(image_features.device)
            if self.select_idx_dropout_topk == self.select_idx_dropout_bottomk:
                topk_mask = bottomk_mask

            logits_topk, idx_topk = self.select_topk(logits, labels, topk_mask)
            idx_topk_abn, idx_topk_nor = (
                idx_topk[: idx_topk.shape[0] // 2],
                idx_topk[idx_topk.shape[0] // 2 :],
            )

            logits_bottomk, idx_bottomk = self.select_bottomk(logits, labels, bottomk_mask)
            idx_bottomk_abn = idx_bottomk[: idx_bottomk.shape[0] // 2]

            logits = logits.view(-1, logits.shape[-1])
            logits_topk = logits_topk.view(-1, logits_topk.shape[-1])

            features = image_features.view(
                -1, image_features.shape[-1]
            )  # (batch*num_segments*seg_length, 512)
            if self.concat_features:
                features = torch.cat((logits.view(-1, logits.shape[-1]), features), dim=-1)

            features = self.projection(features)
            features = features.view(-1, self.num_segments, self.seg_length, features.shape[-1])

            # Axial Transformer
            features = rearrange(features, "b n l d -> b d n l")
            features = self.axial_attn(features)  # (batch_size, num_segments, seg_length, d)
            features = rearrange(features, "b d n l -> b n l d")
            features = rearrange(features, "b n l d -> (b n l) d ")

            scores = self.classifier(features)

            logits = logits.view(-1, logits.shape[-1])
            scores = scores.view(-1)  # (batch*num_segments*seg_length)

            return (
                logits,
                logits_topk,
                scores,
                idx_topk_abn,
                idx_topk_nor,
                idx_bottomk_abn,
            )

    def select_topk(self, logits, labels, mask):
        b, t, num_classes = logits.shape

        logits_sum = logits.view(
            -1, self.num_segments, self.seg_length, num_classes
        )  # (batch, num_segments, seg_length, n_cls)
        logits_sum = torch.sum(logits_sum, dim=2)
        # Set the values to a high value where the mask is zero and leave the values unchanged where the mask is one
        min_value = -1e6
        logits_drop = torch.where(
            mask == 0, torch.ones_like(logits_sum) * min_value, logits_sum
        )  # (batch, num_segments, n_cls)

        alogits_drop = logits_drop[: b // 2]

        # For each abnormal video, it returns the indices of the k_abn snippet most similar to their textual prompt
        alabels = labels[: b // 2]  # (batch//2, 1)
        alabels = torch.where(alabels > self.normal_id, alabels - 1, alabels)

        idx_topk_abn = []
        for alogits_i, alabels_i in zip(alogits_drop, alabels):
            idx_topk_abn_i = torch.topk(
                alogits_i[:, alabels_i],
                self.num_topk,
                dim=0,
                largest=True,
            )[
                1
            ]  # (k_abn)
            idx_topk_abn_i = idx_topk_abn_i.unsqueeze(0)  # (1, num_topk)
            idx_topk_abn.append(idx_topk_abn_i)
        idx_topk_abn = torch.cat(idx_topk_abn)  # (batch / 2, k_abn)

        idx_topk_abn_logits = idx_topk_abn.unsqueeze(2).expand(
            [-1, -1, num_classes]
        )  # (batch/2, k_abn, num_classes)

        alogits = logits[: logits.shape[0] // 2]
        alogits = alogits.view(
            -1, self.num_segments, self.seg_length, num_classes
        )  # (batch/2, num_segments, seg_length, n_cls)
        # expand indices of idx_most_abn_logits to (batch/2, num_topk, seg_length, n_cls)
        idx_topk_abn_logits = idx_topk_abn_logits.unsqueeze(2).expand(
            [-1, -1, self.seg_length, -1]
        )  # (batch/2, num_topk, seg_length, n_cls)

        total_select_abn_logits = []
        for a_logit, a_topk_idx in zip(alogits, idx_topk_abn_logits):
            # a_logit (num_segments, d)
            # a_topk_idx (k_abn, d)
            # Gathers values of a_logit along axis 0 with indices a_idx_most
            logit_topk_abn = torch.gather(
                a_logit, 0, a_topk_idx
            )  # 3 most abnormal snippets in abnormal bag  (k_abn, n_cls)
            total_select_abn_logits.append(logit_topk_abn)
        total_select_abn_logits = torch.cat(
            total_select_abn_logits
        )  # (batch/2*k_abn, seg_length, n_cls)

        nlogits_drop = logits_drop[logits_drop.shape[0] // 2 :]
        nlogits_drop = torch.sum(nlogits_drop, dim=2)  # (batch, num_segments)
        idx_topk_nor = torch.topk(nlogits_drop, k=self.num_topk, dim=1, largest=True)[
            1
        ]  # (batch / 2, k_abn)

        idx_topk_nor_logits = idx_topk_nor.unsqueeze(2).expand(
            [-1, -1, num_classes]
        )  # (batch/2, num_topk, 13)

        nlogits = logits[
            logits.shape[0] // 2 :
        ]  # normal feature logits (batch//2, num_segments * seg_length, n_cls) (batch_size//2, 32 , 14)
        nlogits = nlogits.view(
            -1, self.num_segments, self.seg_length, num_classes
        )  # (batch/2, num_segments, seg_length, n_cls)
        # expand indices of idx_most_abn_logits to (batch/2, k_abn, seg_length, n_cls)
        idx_topk_nor_logits = idx_topk_nor_logits.unsqueeze(2).expand(
            [-1, -1, self.seg_length, -1]
        )  # (batch/2, k_abn, seg_length, n_cls)

        total_select_nor_logits = []
        for n_logit, n_topk_idx in zip(nlogits, idx_topk_nor_logits):
            # n_logit (num_segments, d)
            # n_idx_least (k_abn, d)
            # Gathers values of n_logit along axis 0 with indices n_idx_least
            logit_topk_nor = torch.gather(
                n_logit, 0, n_topk_idx
            )  # 3 most abnormal snippets in normal bag  (k_abn, n_cls)
            total_select_nor_logits.append(logit_topk_nor)
        total_select_nor_logits = torch.cat(
            total_select_nor_logits
        )  # (batch/2*k_abn, seg_length, n_cls)

        total_select_logits = torch.cat(
            (total_select_abn_logits, total_select_nor_logits)
        )  # (batch * k, seg_length, n_cls)

        idx_logits = torch.cat((idx_topk_abn, idx_topk_nor), dim=0)

        return total_select_logits, idx_logits

    def select_bottomk(self, logits, labels, mask):
        b, t, num_classes = logits.shape

        logits_sum = logits.view(
            -1, self.num_segments, self.seg_length, num_classes
        )  # (batch, num_segments, seg_length, n_cls)
        logits_sum = torch.sum(logits_sum, dim=2)
        # Set the values to a high value where the mask is zero and leave the values unchanged where the mask is one
        max_value = 1e6
        logits_drop = torch.where(
            mask == 0, torch.ones_like(logits_sum) * max_value, logits_sum
        )  # (batch, num_segments, n_cls)

        alogits_drop = logits_drop[: b // 2]

        # For each abnormal video, it returns the indices of the k_abn snippet most similar to their textual prompt
        alabels = labels[: b // 2]  # (batch//2, 1)
        alabels = torch.where(alabels > self.normal_id, alabels - 1, alabels)

        idx_bottomk_abn = []
        for alogits_i, alabels_i in zip(alogits_drop, alabels):
            idx_bottomk_abn_i = torch.topk(
                alogits_i[:, alabels_i],
                self.num_bottomk,
                dim=0,
                largest=False,
            )[
                1
            ]  # (num_bottomk)
            idx_bottomk_abn_i = idx_bottomk_abn_i.unsqueeze(0)  # (1, num_topk)
            idx_bottomk_abn.append(idx_bottomk_abn_i)
        idx_bottomk_abn = torch.cat(idx_bottomk_abn)  # (batch / 2, num_topk)

        idx_bottomk_abn_logits = idx_bottomk_abn.unsqueeze(2).expand(
            [-1, -1, num_classes]
        )  # (batch/2, k_abn, num_classes)

        alogits = logits[: logits.shape[0] // 2]
        alogits = alogits.view(
            -1, self.num_segments, self.seg_length, num_classes
        )  # (batch/2, num_segments, seg_length, n_cls)
        # expand indices of idx_most_abn_logits to (batch/2, num_topk, seg_length, n_cls)
        idx_bottomk_abn_logits = idx_bottomk_abn_logits.unsqueeze(2).expand(
            [-1, -1, self.seg_length, -1]
        )  # (batch/2, num_topk, seg_length, n_cls)

        total_select_abn_logits = []
        for a_logit, a_bottomk_idx in zip(alogits, idx_bottomk_abn_logits):
            # a_logit (num_segments, d)
            # a_bottomk_idx (num_bottomk, d)
            # Gathers values of a_logit along axis 0 with indices a_idx_most
            logit_bottomk_abn = torch.gather(
                a_logit, 0, a_bottomk_idx
            )  # 3 most abnormal snippets in abnormal bag  (k_abn, n_cls)
            total_select_abn_logits.append(logit_bottomk_abn)
        total_select_abn_logits = torch.cat(
            total_select_abn_logits
        )  # (batch/2*k_abn, seg_length, n_cls)

        nlogits_drop = logits_drop[b // 2 :]
        nlogits_drop = torch.sum(nlogits_drop, dim=2)  # (batch, num_segments)
        idx_bottomk_nor = torch.topk(nlogits_drop, k=self.num_bottomk, dim=1, largest=False)[
            1
        ]  # (batch / 2, num_bottomk)

        idx_bottomk_nor_logits = idx_bottomk_nor.unsqueeze(2).expand(
            [-1, -1, num_classes]
        )  # (batch/2, num_topk, 13)

        nlogits = logits[
            logits.shape[0] // 2 :
        ]  # normal feature logits (batch//2, num_segments * seg_length, n_cls) (batch_size//2, 32 , 14)
        nlogits = nlogits.view(
            -1, self.num_segments, self.seg_length, num_classes
        )  # (batch/2, num_segments, seg_length, n_cls)
        # expand indices of idx_most_abn_logits to (batch/2, num_bottomk, seg_length, n_cls)
        idx_bottomk_nor_logits = idx_bottomk_nor_logits.unsqueeze(2).expand(
            [-1, -1, self.seg_length, -1]
        )  # (batch/2, num_bottomk, seg_length, n_cls)

        total_select_nor_logits = []
        for n_logit, n_bottomk_idx in zip(nlogits, idx_bottomk_nor_logits):
            # n_logit (num_segments, d)
            # n_bottomk_idx (num_bottomk, d)
            # Gathers values of n_logit along axis 0 with indices n_idx_least
            logit_bottomk_nor = torch.gather(
                n_logit, 0, n_bottomk_idx
            )  # 3 most abnormal snippets in normal bag  (num_bottomk, n_cls)
            total_select_nor_logits.append(logit_bottomk_nor)
        total_select_nor_logits = torch.cat(
            total_select_nor_logits
        )  # (batch/2*num_bottomk, seg_length, n_cls)

        total_select_logits = torch.cat(
            (total_select_abn_logits, total_select_nor_logits)
        )  # (batch * k, seg_length, n_cls)

        idx_logits = torch.cat((idx_bottomk_abn, idx_bottomk_nor), dim=0)

        return total_select_logits, idx_logits


if __name__ == "__main__":
    _ = AnomalyCLIP()
