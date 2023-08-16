import json
import os
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
from sklearn.cluster import KMeans
from torchmetrics import (
    AUROC,
    ROC,
    AveragePrecision,
    ConfusionMatrix,
    F1Score,
    MeanMetric,
    PrecisionRecallCurve,
)
from torchmetrics.classification import Accuracy, MulticlassAUROC, Precision

from src import utils
from src.models.components.loss import ComputeLoss

log = utils.get_pylogger(__name__)


class AnomalyCLIPModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss: ComputeLoss,
        **kwargs,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.criterion = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        # freezing backbone
        for p in self.net.image_encoder.parameters():
            p.requires_grad = False
        if self.net.text_encoder is not None:
            for p in self.net.text_encoder.parameters():
                p.requires_grad = False
            if self.net.direction_module == "learned_encoder_finetune":
                self.net.text_encoder.text_projection.requires_grad = True
        for p in self.net.token_embedding.parameters():
            p.requires_grad = False

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.dir_abn_topk_loss = MeanMetric()
        self.dir_abn_bottomk_loss = MeanMetric()
        self.dir_nor_loss = MeanMetric()
        self.topk_abn_loss = MeanMetric()
        self.bottomk_abn_loss = MeanMetric()
        self.topk_nor_loss = MeanMetric()
        self.smooth_loss = MeanMetric()
        self.sparse_loss = MeanMetric()

        self.roc = ROC(task="binary")
        self.auroc = AUROC(task="binary")
        self.pr_curve = PrecisionRecallCurve(task="binary")
        self.average_precision = AveragePrecision(task="binary")
        self.f1 = F1Score(task="binary")
        self.confmat = ConfusionMatrix(
            task="multiclass", num_classes=self.hparams.num_classes, normalize="true"
        )
        self.pr = Precision(task="binary")
        self.top1_accuracy = Accuracy(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            top_k=1,
            average=None,
        )
        self.top5_accuracy = Accuracy(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            top_k=5,
            average=None,
        )
        self.mc_auroc = MulticlassAUROC(
            num_classes=self.hparams.num_classes, average=None, thresholds=None
        )
        self.mc_aupr = AveragePrecision(
            task="multiclass", num_classes=self.hparams.num_classes, average=None
        )

        self.labels = []
        self.abnormal_scores = []
        self.class_probs = []

    def forward(
        self,
        image_features: torch.Tensor,
        labels,
        ncentroid: torch.Tensor,
        segment_size: int = 1,
        test_mode: bool = False,
        val_mode: bool = False,
    ):
        return self.net(
            image_features,
            labels,
            ncentroid,
            segment_size,
            test_mode,
            val_mode,
        )

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        save_dir = Path(self.hparams.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        ncentroid_file = Path(save_dir / "ncentroid.pt")
        if ncentroid_file.is_file():
            # file exists, load
            self.ncentroid = torch.load(ncentroid_file)
        else:
            with torch.no_grad():
                loader = self.trainer.datamodule.train_dataloader_test_mode()

                # Initialize variables to accumulate the sum of embeddings and the total count
                embedding_sum = torch.zeros(self.net.feature_size)

                count = 0

                for nimage_features, nlabels, _, _ in loader:
                    nimage_features = nimage_features.view(-1, nimage_features.shape[-1])
                    nimage_features = nimage_features[: len(nlabels.squeeze())]
                    embedding_sum += nimage_features.sum(dim=0)
                    count += nimage_features.shape[0]

                # Compute and save the average embedding
                self.ncentroid = embedding_sum / count
                torch.save(self.ncentroid, ncentroid_file)

    def model_step(self, batch: Any):
        nbatch, abatch = batch
        nimage_features, nlabel, _ = nbatch
        aimage_features, alabel, _ = abatch
        image_features = torch.cat((aimage_features, nimage_features), 0)
        labels = torch.cat((alabel, nlabel), 0)

        (
            logits,
            logits_topk,
            logits_bottomk,
            scores,
            idx_topk_abn,
            idx_topk_nor,
            idx_bottomk_abn,
        ) = self.forward(
            image_features,
            labels,
            ncentroid=self.ncentroid,
        )  # forward

        return (
            logits,
            logits_topk,
            logits_bottomk,
            labels,
            scores,
            idx_topk_abn,
            idx_topk_nor,
            idx_bottomk_abn,
        )

    def training_step(self, batch: Any, batch_idx: int):
        # Forward pass
        (
            similarity,
            similarity_topk,
            similarity_bottomk,
            labels,
            scores,
            idx_topk_abn,
            idx_topk_nor,
            idx_bottomk_abn,
        ) = self.model_step(batch)

        # Compute loss
        (
            loss,
            ldir_abn_topk,
            ldir_abn_bottomk,
            ldir_nor,
            ltopk_abn,
            lbottomk_abn,
            ltopk_nor,
            lsmooth,
            lsparse,
        ) = self.criterion(
            similarity,
            similarity_topk,
            similarity_bottomk,
            labels,
            scores,
            idx_topk_abn,
            idx_topk_nor,
            idx_bottomk_abn,
        )

        # update and log metrics
        self.train_loss(loss)
        self.dir_abn_topk_loss(ldir_abn_topk)
        self.dir_abn_bottomk_loss(ldir_abn_bottomk)
        self.dir_nor_loss(ldir_nor)
        self.topk_abn_loss(ltopk_abn)
        self.bottomk_abn_loss(lbottomk_abn)
        self.topk_nor_loss(ltopk_nor)
        self.smooth_loss(lsmooth)
        self.sparse_loss(lsparse)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/dir_abn_topk_loss",
            self.dir_abn_topk_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/dir_abn_bottomk_loss",
            self.dir_abn_bottomk_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/dir_nor_loss",
            self.dir_nor_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/topk_abn_loss",
            self.topk_abn_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/bottomk_abn_loss",
            self.bottomk_abn_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/topk_nor_loss",
            self.topk_nor_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/smooth_loss",
            self.smooth_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/sparse_loss",
            self.sparse_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # return loss or backpropagation will fail
        return {"loss": loss}

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        image_features, labels, label, segment_size = batch
        image_features = image_features.to(self.device)
        labels = labels.squeeze(0).to(self.device)

        save_dir = Path(self.hparams.save_dir)

        ncentroid_file = Path(save_dir / "ncentroid.pt")
        if ncentroid_file.is_file():
            # file exists, load
            self.ncentroid = torch.load(ncentroid_file)
        else:
            raise FileNotFoundError(f"ncentroid file {ncentroid_file} not found")

        # Forward pass
        similarity, abnormal_scores = self.forward(
            image_features,
            labels,
            self.ncentroid,
            segment_size,
            test_mode=True,
        )

        # Compute conditional probabilities
        softmax_similarity = torch.softmax(similarity, dim=1)

        # Compute joint probabilities
        class_probs = softmax_similarity * abnormal_scores.unsqueeze(1)

        # Remove padded frames
        num_labels = labels.shape[0]
        class_probs = class_probs[:num_labels]
        abnormal_scores = abnormal_scores[:num_labels]

        self.labels.extend(labels)
        self.class_probs.extend(class_probs)
        self.abnormal_scores.extend(abnormal_scores)

    def on_validation_epoch_end(self):
        labels = torch.stack(self.labels)
        class_probs = torch.stack(self.class_probs)
        abnormal_scores = torch.stack(self.abnormal_scores)

        num_classes = self.trainer.datamodule.num_classes
        normal_idx = self.trainer.datamodule.hparams.normal_id

        # add normal probability to the class probabilities
        normal_probs = 1 - abnormal_scores
        normal_probs = normal_probs.unsqueeze(1)  # Add a new dimension to match class_probs shape
        class_probs = torch.cat(
            (
                class_probs[:, :normal_idx],
                normal_probs,
                class_probs[:, normal_idx:],
            ),
            dim=1,
        )

        labels_binary = torch.where(labels == normal_idx, 0, 1)

        fpr, tpr, thresholds = self.roc(abnormal_scores, labels_binary)
        auc_roc = self.auroc(abnormal_scores, labels_binary)

        optimal_idx = np.argmax(tpr.cpu().data.numpy() - fpr.cpu().data.numpy())
        optimal_threshold = thresholds[optimal_idx]

        precision, recall, thresholds = self.pr_curve(abnormal_scores, labels_binary)
        auc_pr = self.average_precision(abnormal_scores, labels_binary)

        mc_auroc = self.mc_auroc(class_probs, labels)
        mc_aupr = self.mc_aupr(class_probs, labels)

        mc_auroc_without_normal = torch.cat((mc_auroc[:normal_idx], mc_auroc[normal_idx + 1 :]))
        mc_auroc_without_normal[mc_auroc_without_normal == 0] = torch.nan
        mean_mc_auroc = torch.nanmean(mc_auroc_without_normal)

        mc_aupr_without_normal = torch.cat((mc_aupr[:normal_idx], mc_aupr[normal_idx + 1 :]))
        mc_aupr_without_normal[mc_aupr_without_normal == 0] = torch.nan
        mean_mc_aupr = torch.nanmean(mc_aupr_without_normal)

        self.log("test/AUC", auc_roc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/AP", auc_pr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mAUC", mean_mc_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mAP", mean_mc_aupr, on_step=False, on_epoch=True, prog_bar=True)

        ###
        with torch.no_grad():
            # Top-k precision
            test_dataloader_val_mode = self.trainer.datamodule.test_dataloader_val_mode()

            topk_abn_targets = list()
            bottomk_abn_targets = list()

            for image_features, video_labels, image_labels in test_dataloader_val_mode:
                if video_labels[0].item() != normal_idx:
                    image_features = image_features.to(self.device)
                    video_labels = video_labels.to(self.device)
                    image_labels = image_labels.to(self.device)
                    num_repeats = 2
                    image_features_repeated = image_features.repeat(num_repeats, 1, 1, 1)
                    video_labels_repeated = video_labels.repeat(num_repeats)

                    # Perform forward pass
                    (
                        logits,
                        logits_topk,
                        logits_bottomk,
                        scores,
                        idx_topk_abn,
                        idx_topk_nor,
                        idx_bottomk_abn,
                    ) = self.forward(
                        image_features_repeated,
                        video_labels_repeated,
                        ncentroid=self.ncentroid,
                        val_mode=True,
                    )

                    for (
                        topk_abn_idx_i,
                        bottomk_abn_idx_i,
                        video_labels_i,
                        image_labels_i,
                    ) in zip(
                        idx_topk_abn,
                        idx_bottomk_abn,
                        video_labels,
                        image_labels,
                    ):
                        vid_label = video_labels_i.item()
                        im_labels = image_labels_i.view(self.net.num_segments, self.net.seg_length)
                        # print(f"vid_label: {vid_label}")
                        # print(f"im_labels: {im_labels}")

                        # Process topk targets
                        topk_abn_target = torch.index_select(im_labels, 0, topk_abn_idx_i)  # 3, 16
                        topk_abn_target = torch.any(topk_abn_target == vid_label, 1)
                        topk_abn_targets.append(topk_abn_target)

                        # Process bottomk targets
                        bottomk_abn_target = torch.index_select(
                            im_labels, 0, bottomk_abn_idx_i
                        )  # 3, 16
                        bottomk_abn_target = torch.all(bottomk_abn_target == normal_idx, 1)
                        bottomk_abn_targets.append(bottomk_abn_target)

            topk_abn_targets = torch.cat(topk_abn_targets, 0).int()
            bottomk_abn_targets = torch.cat(bottomk_abn_targets, 0).int()
            topk_abn_preds = torch.ones_like(topk_abn_targets)
            bottomk_abn_preds = torch.ones_like(bottomk_abn_targets)
            topk_abn_precision = self.pr(topk_abn_preds, topk_abn_targets)
            bottomk_abn_precision = self.pr(bottomk_abn_preds, bottomk_abn_targets)

            self.log("test/top_k_abn_precision", topk_abn_precision, on_step=False, on_epoch=True)
            self.log(
                "test/bottom_k_abn_precision", bottomk_abn_precision, on_step=False, on_epoch=True
            )
        ###

        metrics = {
            "epoch": self.trainer.current_epoch,
            "auc_roc": auc_roc.item(),
            "auc_pr": auc_pr.item(),
            "mean_mc_auroc": mean_mc_auroc.item(),
            "mean_mc_aupr": mean_mc_aupr.item(),
            "mc_auroc": mc_auroc.tolist(),
            "mc_aupr": mc_aupr.tolist(),
            "optimal_threshold": optimal_threshold.item(),
            "topk_abn_precision": topk_abn_precision.item(),
            "bottomk_abn_precision": bottomk_abn_precision.item(),
        }

        save_dir = Path(self.hparams.save_dir)

        with open(save_dir / f"metrics_{self.trainer.current_epoch}.json", "w") as fp:
            json.dump(metrics, fp, indent=4, sort_keys=True)

        self.labels.clear()
        self.class_probs.clear()
        self.abnormal_scores.clear()

    @rank_zero_only
    def test_step(self, batch: Any, batch_idx: int):
        image_features, labels, label, segment_size = batch
        image_features = image_features.to(self.device)
        labels = labels.squeeze(0).to(self.device)

        ckpt_path = Path(self.trainer.ckpt_path)
        save_dir = os.path.normpath(ckpt_path.parent).split(os.path.sep)[-1]
        save_dir = Path(os.path.join("/usr/src/app/logs/train/runs", str(save_dir)))

        ncentroid_file = Path(save_dir / "ncentroid.pt")

        if ncentroid_file.is_file():
            # file exists, load
            self.ncentroid = torch.load(ncentroid_file)
        else:
            raise FileNotFoundError(f"ncentroid file {ncentroid_file} not found")

        # Forward pass
        similarity, abnormal_scores = self.forward(
            image_features,
            labels,
            self.ncentroid,
            segment_size,
            test_mode=True,
        )

        # Compute conditional probabilities
        softmax_similarity = torch.softmax(similarity, dim=1)

        # Compute joint probabilities
        class_probs = softmax_similarity * abnormal_scores.unsqueeze(1)

        # Remove padded frames
        num_labels = labels.shape[0]
        class_probs = class_probs[:num_labels]
        abnormal_scores = abnormal_scores[:num_labels]

        return {
            "abnormal_scores": abnormal_scores,
            "labels": labels,
            "class_probs": class_probs,
        }

    @rank_zero_only
    def test_epoch_end(self, outputs: List[Any]):
        abnormal_scores = torch.cat([o["abnormal_scores"] for o in outputs])
        labels = torch.cat([o["labels"] for o in outputs])
        class_probs = torch.cat([o["class_probs"] for o in outputs])

        num_classes = self.trainer.datamodule.num_classes
        normal_idx = self.trainer.datamodule.hparams.normal_id

        # add normal probability to the class probabilities
        normal_probs = 1 - abnormal_scores
        normal_probs = normal_probs.unsqueeze(1)  # Add a new dimension to match class_probs shape
        class_probs = torch.cat(
            (
                class_probs[:, :normal_idx],
                normal_probs,
                class_probs[:, normal_idx:],
            ),
            dim=1,
        )

        labels_binary = torch.where(labels == normal_idx, 0, 1)

        fpr, tpr, thresholds = self.roc(abnormal_scores, labels_binary)
        auc_roc = self.auroc(abnormal_scores, labels_binary)

        optimal_idx = np.argmax(tpr.cpu().data.numpy() - fpr.cpu().data.numpy())
        optimal_threshold = thresholds[optimal_idx]

        precision, recall, thresholds = self.pr_curve(abnormal_scores, labels_binary)
        auc_pr = self.average_precision(abnormal_scores, labels_binary)

        class_probs_without_normal = torch.cat(
            (class_probs[:, :normal_idx], class_probs[:, normal_idx + 1 :]),
            dim=1,
        )

        # select predictions based on abnormal score and class probabilities
        y_pred = []
        for i in range(len(abnormal_scores)):
            if abnormal_scores[i] < optimal_threshold:
                y_pred.append(normal_idx)
            else:
                pred = torch.argmax(class_probs_without_normal[i])
                if pred >= normal_idx:
                    pred += 1
                y_pred.append(pred)
        y_pred = torch.tensor(y_pred).to(self.device)

        # compute top1, top5, and auc roc for each class
        top1_accuracy = torch.zeros(num_classes)
        top5_accuracy = torch.zeros(num_classes)

        top1_preds = torch.max(class_probs_without_normal, dim=1)[1]
        top1_preds = torch.where(top1_preds >= normal_idx, top1_preds + 1, top1_preds)
        top1_preds = torch.where(y_pred == normal_idx, normal_idx, top1_preds)
        top5_preds = torch.topk(class_probs_without_normal, k=5, dim=1)[1]
        top5_preds = torch.where(top5_preds >= normal_idx, top5_preds + 1, top5_preds)
        # if y_pred == normal_idx, then top5_preds = [normal_idx, top5_preds[0], top5_preds[1], top5_preds[2], top5_preds[3]], else top5_preds = top5_preds
        top5_preds = torch.where(
            y_pred.unsqueeze(1) == normal_idx,
            torch.cat(
                (
                    torch.tensor([normal_idx])
                    .unsqueeze(0)
                    .expand(top5_preds.shape[0], -1)
                    .to(self.device),
                    top5_preds[:, :4],
                ),
                dim=1,
            ),
            top5_preds,
        )

        for class_idx in range(num_classes):
            class_mask = (labels == class_idx).bool()
            class_preds = top1_preds[class_mask]
            class_labels = labels[class_mask]
            top1_accuracy[class_idx] = (class_preds == class_labels).float().mean()
            top5_accuracy[class_idx] = (
                (top5_preds[class_mask] == class_labels.view(-1, 1)).any(dim=1).float().mean()
            )

        mc_auroc = self.mc_auroc(class_probs, labels)
        mc_aupr = self.mc_aupr(class_probs, labels)

        mc_auroc_without_normal = torch.cat((mc_auroc[:normal_idx], mc_auroc[normal_idx + 1 :]))
        mc_auroc_without_normal[mc_auroc_without_normal == 0] = torch.nan
        mean_mc_auroc = torch.nanmean(mc_auroc_without_normal)

        mc_aupr_without_normal = torch.cat((mc_aupr[:normal_idx], mc_aupr[normal_idx + 1 :]))
        mc_aupr_without_normal[mc_aupr_without_normal == 0] = torch.nan
        mean_mc_aupr = torch.nanmean(mc_aupr_without_normal)

        with torch.no_grad():
            # Top-k precision
            test_dataloader_val_mode = self.trainer.datamodule.test_dataloader_val_mode()

            topk_abn_targets = list()
            bottomk_abn_targets = list()

            for image_features, video_labels, image_labels in test_dataloader_val_mode:
                if video_labels[0].item() != normal_idx:
                    image_features = image_features.to(self.device)
                    video_labels = video_labels.to(self.device)
                    image_labels = image_labels.to(self.device)
                    num_repeats = 2
                    image_features_repeated = image_features.repeat(num_repeats, 1, 1, 1)
                    video_labels_repeated = video_labels.repeat(num_repeats)

                    # Perform forward pass
                    (
                        logits,
                        logits_topk,
                        logits_bottomk,
                        scores,
                        idx_topk_abn,
                        idx_topk_nor,
                        idx_bottomk_abn,
                    ) = self.forward(
                        image_features_repeated,
                        video_labels_repeated,
                        ncentroid=self.ncentroid,
                        val_mode=True,
                    )

                    for (
                        topk_abn_idx_i,
                        bottomk_abn_idx_i,
                        video_labels_i,
                        image_labels_i,
                    ) in zip(
                        idx_topk_abn,
                        idx_bottomk_abn,
                        video_labels,
                        image_labels,
                    ):
                        vid_label = video_labels_i.item()
                        im_labels = image_labels_i.view(self.net.num_segments, self.net.seg_length)
                        # Process topk targets
                        topk_abn_target = torch.index_select(im_labels, 0, topk_abn_idx_i)  # 3, 16
                        topk_abn_target = torch.any(topk_abn_target == vid_label, 1)
                        topk_abn_targets.append(topk_abn_target)

                        # Process bottomk targets
                        bottomk_abn_target = torch.index_select(
                            im_labels, 0, bottomk_abn_idx_i
                        )  # 3, 16
                        bottomk_abn_target = torch.all(bottomk_abn_target == normal_idx, 1)
                        bottomk_abn_targets.append(bottomk_abn_target)

            topk_abn_targets = torch.cat(topk_abn_targets, 0).int()
            bottomk_abn_targets = torch.cat(bottomk_abn_targets, 0).int()
            topk_abn_preds = torch.ones_like(topk_abn_targets)
            bottomk_abn_preds = torch.ones_like(bottomk_abn_targets)
            topk_abn_precision = self.pr(topk_abn_preds, topk_abn_targets)
            bottomk_abn_precision = self.pr(bottomk_abn_preds, bottomk_abn_targets)

        ckpt_path = Path(self.trainer.ckpt_path)
        save_dir = os.path.normpath(ckpt_path.parent).split(os.path.sep)[-1]
        save_dir = Path(os.path.join("/usr/src/app/logs/eval/runs", str(save_dir)))
        if not save_dir.is_dir():
            save_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving results to {save_dir}")

        labels_df = pd.read_csv(self.trainer.datamodule.hparams.labels_file)
        classes = labels_df["id"].tolist()
        class_names = labels_df["name"].tolist()

        metrics = {
            "epoch": self.trainer.current_epoch,
            "auc_roc": auc_roc.item(),
            "auc_pr": auc_pr.item(),
            "mean_mc_auroc": mean_mc_auroc.item(),
            "mean_mc_aupr": mean_mc_aupr.item(),
            "mc_auroc": mc_auroc.tolist(),
            "mc_aupr": mc_aupr.tolist(),
            "top1_accuracy": top1_accuracy.tolist(),
            "top5_accuracy": top5_accuracy.tolist(),
            "optimal_threshold": optimal_threshold.item(),
            "topk_abn_precision": topk_abn_precision.item(),
            "bottomk_abn_precision": bottomk_abn_precision.item(),
        }

        with open(save_dir / "metrics.json", "w") as fp:
            json.dump(metrics, fp, indent=4, sort_keys=True)

        f1_scores = {}
        far = {}
        for i in range(10):
            thresh = (i + 1) / 10
            y_pred_binary = torch.where(abnormal_scores < thresh, 0, 1)
            f1_scores[thresh] = self.f1(y_pred_binary, labels_binary)

        # PR-Curve plot
        recall = recall.cpu().data.numpy()
        precision = precision.cpu().data.numpy()
        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        plt.style.use("ggplot")
        plt.ylim(0, 1.1)
        plt.plot(recall, precision, color="red")
        plt.title(f"PR Curve: {auc_pr*100:.2f}")
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        fig_file = save_dir / "PR.png"
        plt.savefig(fig_file)
        plt.close()

        # ROC-Curve plot
        fpr = fpr.cpu().data.numpy()
        tpr = tpr.cpu().data.numpy()
        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        plt.style.use("ggplot")
        plt.ylim(0, 1.1)
        plt.plot(fpr, tpr, color="blue")
        plt.title(f"ROC Curve: {auc_roc*100:.2f}")
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        fig_file = save_dir / "ROC.png"
        plt.savefig(fig_file)
        plt.close()

        # F1 score curve
        x = [(i + 1) / 10 for i in range(10)]
        y = [f1_scores[xx].cpu().data.numpy() for xx in x]
        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        plt.style.use("ggplot")
        # plt.ylim(0, 1.1)
        plt.plot(x, y, color="blue")
        plt.title(f"F1@0.5: {f1_scores[0.5]*100:.2f}")
        plt.ylabel("F1")
        plt.xlabel("threshold")
        fig_file = save_dir / "F1.png"
        plt.savefig(fig_file)
        plt.close()

        confmat = self.confmat(y_pred, labels)
        fig = plt.figure(figsize=(20, 18))
        ax = plt.subplot()  # /np.sum(cm)
        f = sns.heatmap(confmat.cpu().data.numpy(), annot=True, ax=ax, fmt=".2%", cmap="Blues")

        # labels, title and ticks
        ax.set_xlabel("Predicted", fontsize=20)
        ax.xaxis.set_label_position("bottom")
        plt.xticks(rotation=90)
        ax.xaxis.set_ticklabels(class_names, fontsize=15)
        ax.xaxis.tick_bottom()
        #
        ax.set_ylabel("True", fontsize=20)
        ax.yaxis.set_ticklabels(class_names, fontsize=15)
        plt.yticks(rotation=0)

        fig_file = save_dir / "confusion_matrix.png"
        plt.savefig(fig_file)
        plt.close()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        param_list = []

        param_list.append(
            {
                "params": self.net.selector_model.parameters(),
                "lr": self.hparams.solver.lr * self.hparams.solver.selector_model_ratio,
                "name": "selector_model",
            }
        )

        param_list.append(
            {
                "params": self.net.temporal_model.parameters(),
                "lr": self.hparams.solver.lr * self.hparams.solver.temporal_model_ratio,
                "name": "temporal_model",
            }
        )

        if self.net.direction_module == "learned_no_encoder":
            param_list.append(
                {
                    "params": self.net.prompts,
                    "lr": self.hparams.solver.lr * self.hparams.solver.prompts_ratio,
                    "name": "prompts",
                }
            )
        elif self.net.direction_module.startswith("learned_encoder"):
            param_list.append(
                {
                    "params": self.net.prompt_learner.parameters(),
                    "lr": self.hparams.solver.lr * self.hparams.solver.prompt_learner_ratio,
                    "name": "prompt_learner",
                }
            )
            if self.net.direction_module == "learned_encoder_finetune":
                param_list.append(
                    {
                        "params": self.net.text_encoder.text_projection,
                        "lr": self.hparams.solver.lr * self.hparams.solver.text_projection_ratio,
                        "name": "text_projection",
                    }
                )

        if self.net.feature_size != self.net.embedding_dim:
            param_list.append(
                {
                    "params": self.net.feature_projection.parameters(),
                    "lr": self.hparams.solver.lr * self.hparams.solver.feature_projection_ratio,
                    "name": "feature_projection",
                }
            )

        optimizer = self.optimizer(params=param_list)
        if self.scheduler is not None:
            successor = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(self.trainer.max_epochs)
            )
            scheduler = self.scheduler(optimizer=optimizer, successor=successor)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = AnomalyCLIPModule(None, None, None, None)
