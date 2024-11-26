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
from src.utils.visualizer import Visualizer

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
        for p in self.net.text_encoder.parameters():
            p.requires_grad = False
        self.net.text_encoder.text_projection.requires_grad = True
        for p in self.net.token_embedding.parameters():
            p.requires_grad = False

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.dir_abn_loss = MeanMetric()
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
    ):
        return self.net(
            image_features,
            labels,
            ncentroid,
            segment_size,
            test_mode,
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
                embedding_sum = torch.zeros(self.net.embedding_dim).to(self.device)
                count = 0

                if self.trainer.datamodule.hparams.load_from_features:
                    for nimage_features, nlabels, _, _ in loader:
                        nimage_features = nimage_features.view(-1, nimage_features.shape[-1])
                        nimage_features = nimage_features[: len(nlabels.squeeze())]
                        nimage_features = nimage_features.to(self.device)
                        embedding_sum += nimage_features.sum(dim=0)
                        count += nimage_features.shape[0]
                else:
                    for nimages, nlabels, _, _ in loader:
                        b, t, c, h, w = nimages.size()
                        nimages = nimages.view(-1, c, h, w)
                        nimages = nimages[: len(nlabels.squeeze())]
                        nimages = nimages.to(self.device)
                        nimage_features = self.net.image_encoder(nimages)
                        embedding_sum += nimage_features.sum(dim=0)
                        count += nimage_features.shape[0]

                # Compute and save the average embedding
                self.ncentroid = embedding_sum / count
                torch.save(self.ncentroid, ncentroid_file)

    def model_step(self, batch: Any):
        nbatch, abatch = batch
        nimage_features, nlabel = nbatch
        aimage_features, alabel = abatch
        image_features = torch.cat((aimage_features, nimage_features), 0)
        labels = torch.cat((alabel, nlabel), 0)

        (
            logits,
            logits_topk,
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
            labels,
            scores,
            idx_topk_abn,
            idx_topk_nor,
            idx_bottomk_abn,
        ) = self.model_step(batch)

        # Compute loss
        (
            loss,
            ldir_abn,
            ldir_nor,
            ltopk_abn,
            lbottomk_abn,
            ltopk_nor,
            lsmooth,
            lsparse,
        ) = self.criterion(
            similarity,
            similarity_topk,
            labels,
            scores,
            idx_topk_abn,
            idx_topk_nor,
            idx_bottomk_abn,
        )

        # update and log metrics
        self.train_loss(loss)
        self.dir_abn_loss(ldir_abn)
        self.dir_nor_loss(ldir_nor)
        self.topk_abn_loss(ltopk_abn)
        self.bottomk_abn_loss(lbottomk_abn)
        self.topk_nor_loss(ltopk_nor)
        self.smooth_loss(lsmooth)
        self.sparse_loss(lsparse)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/dir_abn_loss",
            self.dir_abn_loss,
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

        metrics = {
            "epoch": self.trainer.current_epoch,
            "auc_roc": auc_roc.item(),
            "auc_pr": auc_pr.item(),
            "mean_mc_auroc": mean_mc_auroc.item(),
            "mean_mc_aupr": mean_mc_aupr.item(),
            "mc_auroc": mc_auroc.tolist(),
            "mc_aupr": mc_aupr.tolist(),
            "optimal_threshold": optimal_threshold.item(),
        }

        save_dir = Path(self.hparams.save_dir)

        with open(save_dir / f"metrics_{self.trainer.current_epoch}.json", "w") as fp:
            json.dump(metrics, fp, indent=4, sort_keys=True)

        self.labels.clear()
        self.class_probs.clear()
        self.abnormal_scores.clear()

    def on_test_start(self):
        ckpt_path = Path(self.trainer.ckpt_path)
        save_dir = os.path.normpath(ckpt_path.parent).split(os.path.sep)[-1]
        save_dir = Path(os.path.join("/usr/src/app/logs/train/runs", str(save_dir)))
        if not save_dir.is_dir():
            save_dir.mkdir(parents=True, exist_ok=True)

        ncentroid_file = Path(save_dir / "ncentroid.pt")

        if ncentroid_file.is_file():
            # file exists, load
            self.ncentroid = torch.load(ncentroid_file)
        else:
            with torch.no_grad():
                loader = self.trainer.datamodule.train_dataloader_test_mode()

                # Initialize variables to accumulate the sum of embeddings and the total count
                embedding_sum = torch.zeros(self.net.embedding_dim).to(self.device)
                count = 0

                if self.trainer.datamodule.hparams.load_from_features:
                    for nimage_features, nlabels, _, _, _ in loader:
                        nimage_features = nimage_features.view(-1, nimage_features.shape[-1])
                        nimage_features = nimage_features[: len(nlabels.squeeze())]
                        nimage_features = nimage_features.to(self.device)
                        embedding_sum += nimage_features.sum(dim=0)
                        count += nimage_features.shape[0]
                else:
                    for nimages, nlabels, _, _, _ in loader:
                        b, t, c, h, w = nimages.size()
                        nimages = nimages.view(-1, c, h, w)
                        nimages = nimages[: len(nlabels.squeeze())]
                        nimages = nimages.to(self.device)
                        nimage_features = self.net.image_encoder(nimages)
                        embedding_sum += nimage_features.sum(dim=0)
                        count += nimage_features.shape[0]

            # Compute and save the average embedding
            self.ncentroid = embedding_sum / count
            torch.save(self.ncentroid, ncentroid_file)

        if self.trainer.datamodule.hparams.visualize:
            self.visualizer = Visualizer(
                self.trainer.datamodule.hparams.normal_id,
                self.trainer.datamodule.hparams.labels_file,
                self.trainer.datamodule.hparams.image_tmpl,
                save_dir,
                self.device,
            )
        else:
            self.visualizer = None

    @rank_zero_only
    def test_step(self, batch: Any, batch_idx: int):
        image_features, labels, label, segment_size, path = batch
        image_features = image_features.to(self.device)
        labels = labels.squeeze(0).to(self.device)

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
        softmax_similarity = softmax_similarity[:num_labels]

        if self.visualizer is not None:
            self.visualizer.process_video(
                abnormal_scores,
                class_probs,
                softmax_similarity,
                labels,
                path,
            )

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
        param_list.append(
            {
                "params": self.net.prompt_learner.parameters(),
                "lr": self.hparams.solver.lr * self.hparams.solver.prompt_learner_ratio,
                "name": "prompt_learner",
            }
        )
        param_list.append(
            {
                "params": self.net.text_encoder.text_projection,
                "lr": self.hparams.solver.lr * self.hparams.solver.text_projection_ratio,
                "name": "text_projection",
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
