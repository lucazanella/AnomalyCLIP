import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import gridspec


class Visualizer:
    def __init__(self, normal_idx, labels_csv_path, image_tmpl, save_dir, device):
        self.normal_idx = normal_idx
        self.labels_csv_path = labels_csv_path
        self.image_tmpl = image_tmpl
        self.save_dir = save_dir
        self.device = device

    def setup_directories(self, path):
        dir_path = path[0].split(os.path.sep)
        dir_path[-3] = "Anomaly-Frames"
        dir_path[-1] = dir_path[-1].replace(".npy", "")
        video_name = dir_path[-1]

        dir_path = os.path.sep.join(dir_path)
        save_dir_var = Path(self.save_dir / "qualitatives_var")
        save_dir_var.mkdir(parents=True, exist_ok=True)

        return dir_path, save_dir_var, video_name

    def compute_predictions(self, abnormal_scores, class_probs, softmax_similarity, threshold=0.5):
        y_pred = []
        for i in range(len(abnormal_scores)):
            if abnormal_scores[i] < threshold:
                y_pred.append(self.normal_idx)
            else:
                pred = torch.argmax(class_probs[i])
                if pred >= self.normal_idx:
                    pred += 1
                y_pred.append(pred)
        y_pred = torch.tensor(y_pred).to(self.device)

        top3_preds = torch.topk(class_probs, k=3, dim=1)[1]
        top3_probs = torch.topk(class_probs, k=3, dim=1)[0]
        top3_preds = torch.where(top3_preds >= self.normal_idx, top3_preds + 1, top3_preds)
        top3_preds = torch.where(
            y_pred.unsqueeze(1) == self.normal_idx,
            torch.cat(
                (
                    torch.tensor([self.normal_idx])
                    .unsqueeze(0)
                    .expand(top3_preds.shape[0], -1)
                    .to(self.device),
                    top3_preds[:, :2],
                ),
                dim=1,
            ),
            top3_preds,
        )

        normal_probs = 1 - abnormal_scores.unsqueeze(1).expand(abnormal_scores.shape[0], 3)

        top3_probs = torch.where(top3_preds == self.normal_idx, normal_probs, top3_probs)

        top3_preds = torch.topk(softmax_similarity, k=3, dim=1)[1]
        top3_probs = torch.topk(softmax_similarity, k=3, dim=1)[0]
        top3_preds = top3_preds.cpu().numpy()
        top3_probs = top3_probs.cpu().numpy()

        return y_pred, top3_preds, top3_probs

    def generate_video(self, figs, video_path, video_fps=30):
        fig_size = figs[0].get_size_inches() * figs[0].dpi
        video_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (int(fig_size[0]), int(fig_size[1])),
        )

        for fig in figs:
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            video_writer.write(img)

        video_writer.release()

        plt.close("all")

    def create_figure(
        self,
        i,
        img,
        abnormal_scores,
        top3_preds,
        softmax_similarity,
        labels,
        class_names,
        title,
        threshold,
    ):
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

        # Subplot 1: Video frame
        ax1 = plt.subplot(gs[0, 0])
        if abnormal_scores[i] < threshold:
            box_color = (255, 0, 0)  # blue
        else:
            box_color = (0, 0, 255)  # red
        cv2.rectangle(img, (0, 0), (320, 240), box_color, 5)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax1.imshow(img_rgb)
        ax1.axis("off")

        # Subplot 2: Bar chart of class probabilities
        ax2 = plt.subplot(gs[0, 1])
        x_pos = np.arange(len(softmax_similarity))
        ax2.bar(x_pos, softmax_similarity, color=(0.5, 0.5, 0.5), align="center")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(class_names, rotation=90, fontsize=14)
        ax2.set_ylabel("P(c|A)", fontsize=14)
        ax2.set_ylim([0, 1])
        ax2.axhline(y=0.2, color="grey", linestyle="--", linewidth=0.8)
        ax2.axhline(y=0.4, color="grey", linestyle="--", linewidth=0.8)
        ax2.axhline(y=0.6, color="grey", linestyle="--", linewidth=0.8)
        ax2.axhline(y=0.8, color="grey", linestyle="--", linewidth=0.8)

        if abnormal_scores[i] >= threshold:
            # Highlight the predicted class on the bar chart with a different color
            highlight_idx = top3_preds
            # red if the class is abnormal
            highlight_color = (1, 0, 0)
            ax2.bar(
                highlight_idx,
                [softmax_similarity[idx] for idx in highlight_idx],
                color=highlight_color,
                align="center",
            )

        # Subplot 3: Abnormal score over frames
        ax3 = plt.subplot(gs[1, :])
        x = np.arange(len(abnormal_scores))
        ax3.plot(x, abnormal_scores, color="#4e79a7", linewidth=1)
        ymin = 0
        ymax = 1
        xmin = 0
        xmax = len(abnormal_scores)
        ax3.set_xlim([xmin, xmax])
        ax3.set_ylim([ymin, ymax])
        start_idx = None
        for frame_idx in range(labels.size(0)):
            if labels[frame_idx] != self.normal_idx and start_idx is None:
                start_idx = frame_idx
            elif labels[frame_idx] == self.normal_idx and start_idx is not None:
                rect = plt.Rectangle(
                    (start_idx, ymin),
                    frame_idx - start_idx,
                    ymax - ymin,
                    color="#e15759",
                    alpha=0.5,
                )
                ax3.add_patch(rect)
                start_idx = None
        if (
            start_idx is not None
        ):  # handle case where abnormality extends to the end of the sequence
            rect = plt.Rectangle(
                (start_idx, ymin),
                labels.size(0) - start_idx,
                ymax - ymin,
                color="#e15759",
                alpha=0.5,
            )
            ax3.add_patch(rect)

        ax3.text(0.02, 0.90, title, fontsize=14, transform=ax3.transAxes)
        ax3.axhline(y=0.25, color="grey", linestyle="--", linewidth=0.8)
        ax3.axhline(y=0.5, color="grey", linestyle="--", linewidth=0.8)
        ax3.axhline(y=0.75, color="grey", linestyle="--", linewidth=0.8)
        ax3.set_yticks([0.25, 0.5, 0.75])
        ax3.tick_params(axis="y", labelsize=14)
        ax3.set_ylabel("p(A)", fontsize=14)
        ax3.set_xlabel("Frame Number", fontsize=14)
        ax3.axvline(x=i, color="red")

        fig.tight_layout()
        return fig

    def generate_figures(
        self,
        dir_path,
        abnormal_scores,
        top3_preds,
        softmax_similarity,
        labels,
        class_names,
        num_frames,
        title,
        threshold=0.5,
    ):
        figs = []
        for i in range(num_frames):
            img_path = os.path.join(dir_path, self.image_tmpl.format(i))
            img = cv2.imread(img_path)
            fig = self.create_figure(
                i,
                img,
                abnormal_scores,
                top3_preds[i],
                softmax_similarity[i],
                labels,
                class_names,
                title,
                threshold,
            )
            figs.append(fig)
        return figs

    def process_video(self, abnormal_scores, class_probs, softmax_similarity, labels, path):
        dir_path, save_dir_var, video_name = self.setup_directories(path)

        video_path = save_dir_var / f"{video_name}.mp4"
        if video_path.exists():
            return

        abnormal_scores_np = abnormal_scores.detach().cpu().numpy()
        softmax_similarity_np = softmax_similarity.cpu().numpy()

        labels_df = pd.read_csv(self.labels_csv_path)
        class_names = labels_df["name"].tolist()
        class_names_except_normal = class_names.copy()
        class_names_except_normal.remove("Normal")
        class_names_except_normal = [
            name.replace("RoadAccidents", "RoadAcc.") for name in class_names_except_normal
        ]

        title = os.path.splitext(path[0].split("/")[-1])[0]

        y_pred, top3_preds, top3_probs = self.compute_predictions(
            abnormal_scores, class_probs, softmax_similarity
        )
        figs = self.generate_figures(
            dir_path,
            abnormal_scores_np,
            top3_preds,
            softmax_similarity_np,
            labels,
            class_names_except_normal,
            len(abnormal_scores),
            title,
        )

        self.generate_video(figs, video_path)
