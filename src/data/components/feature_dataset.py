import math
import os
import os.path
from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src import utils

log = utils.get_pylogger(__name__)


def round_to_nearest(number: float, X: int) -> int:
    """Rounds a number to the smallest upper integer that is divisible by a given integer X.

    Args:
    number: A float or integer to be rounded to the nearest multiple of X.
    X: An integer that the rounded number should be divisible by.

    Returns:
    The smallest upper integer that is divisible by X and is closest to the input number.
    """
    return math.ceil(number / X) * X


def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)

    r = np.linspace(0, len(feat), length + 1, dtype=np.int)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i] : r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat


class VideoRecord:
    """Helper class for class VideoFrameDataset. This class represents a video sample's metadata.

    Args:
        root_datapath: the system path to the root folder
                       of the videos.
        row: A list with four or more elements where 1) The first
             element is the path to the video sample's frames excluding
             the root_datapath prefix 2) The  second element is the starting frame id of the video
             3) The third element is the inclusive ending frame id of the video
             4) The fourth element is the label index.
             5) any following elements are labels in the case of multi-label classification
    """

    def __init__(self, row, root_datapath, spatialannotationdir_path=None):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])
        if spatialannotationdir_path:
            filename = row[0].split("/")[1].replace("_x264", "")
            self._spatialannotationdir_path = Path(
                spatialannotationdir_path, filename
            ).with_suffix(".txt")
            self._spatialannotationdir_path = (
                self._spatialannotationdir_path
                if self._spatialannotationdir_path.is_file()
                else None
            )
        else:
            self._spatialannotationdir_path = None

    @property
    def path(self) -> str:
        return self._path + ".npy"

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1  # +1 because end frame is inclusive

    @property
    def start_frame(self) -> int:
        return int(self._data[1])

    @property
    def end_frame(self) -> int:
        return int(self._data[2])

    @property
    def label(self) -> Union[int, List[int]]:
        # just one label_id
        if len(self._data) == 4:
            return int(self._data[3])
        # sample associated with multiple labels
        else:
            return [int(label_id) for label_id in self._data[3:]]

    @property
    def tbox(self) -> List[Tuple[int]]:
        if self._spatialannotationdir_path:
            anno_df = pd.read_csv(
                self._spatialannotationdir_path, delim_whitespace=True, header=None
            )
            anno_df.columns = [
                "Track",  # All rows with the same ID belong to the same path.
                "xmin",  # The top left x-coordinate of the bounding box.
                "ymin",  # The top left y-coordinate of the bounding box.
                "xmax",  # The bottom right x-coordinate of the bounding box.
                "ymax",  # The bottom right y-coordinate of the bounding box.
                "frame",  # The frame that this annotation represents.
                "lost",  # If 1, the annotation is outside of the view screen.
                "occluded",  # If 1, the annotation is occluded.
                "generated",  # If 1, the annotation was automatically interpolated.
                "label",  # The label for this annotation, enclosed in quotation marks.
            ]
            anno_df = anno_df.loc[
                (anno_df["frame"] >= self.start_frame) & (anno_df["frame"] <= self.end_frame)
            ]
            anomaly_frames = (1 - anno_df["lost"].values).tolist()
        else:
            anomaly_frames = [0] * self.num_frames
        return anomaly_frames


class VideoFrameDataset(torch.utils.data.Dataset):
    r"""A highly efficient and adaptable dataset class for videos. Instead of loading every frame of
    a video, loads x RGB frames of a video (sparse temporal sampling) and evenly chooses those
    frames from start to end of the video, returning a list of x PIL images or ``FRAMES x CHANNELS
    x HEIGHT x WIDTH`` tensors where FRAMES=x if the ``ImglistToTensor()`` transform is used.

    More specifically, the frame range [START_FRAME, END_FRAME] is divided into NUM_SEGMENTS
    segments and FRAMES_PER_SEGMENT consecutive frames are taken from each segment.

    Note:
        A demonstration of using this class can be seen
        in ``demo.py``
        https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch

    Note:
        This dataset broadly corresponds to the frame sampling technique
        introduced in ``Temporal Segment Networks`` at ECCV2016
        https://arxiv.org/abs/1608.00859.


    Note:
        This class relies on receiving video data in a structure where
        inside a ``ROOT_DATA`` folder, each video lies in its own folder,
        where each video folder contains the frames of the video as
        individual files with a naming convention such as
        img_001.jpg ... img_059.jpg.
        For enumeration and annotations, this class expects to receive
        the path to a .txt file where each video sample has a row with four
        (or more in the case of multi-label, see README on Github)
        space separated values:
        ``VIDEO_FOLDER_PATH     START_FRAME      END_FRAME      LABEL_INDEX``.
        ``VIDEO_FOLDER_PATH`` is expected to be the path of a video folder
        excluding the ``ROOT_DATA`` prefix. For example, ``ROOT_DATA`` might
        be ``home\data\datasetxyz\videos\``, inside of which a ``VIDEO_FOLDER_PATH``
        might be ``jumping\0052\`` or ``sample1\`` or ``00053\``.

    Args:
        root_path: The root path in which video folders lie.
                   this is ROOT_DATA from the description above.
        annotationfile_path: The .txt annotation file containing
                             one row per video sample as described above.
        num_segments: The number of segments the video should
                      be divided into to sample frames from.
        frames_per_segment: The number of frames that should
                            be loaded per segment. For each segment's
                            frame-range, a random start index or the
                            center is chosen, from which frames_per_segment
                            consecutive frames are loaded.
        imagefile_template: The image filename template that video frame files
                            have inside of their video folders as described above.
        transform: Transform pipeline that receives a list of PIL images/frames.
        test_mode: If True, frames are taken from the center of each
                   segment, instead of a random location in each segment.
    """

    def __init__(
        self,
        root_path: str,
        annotationfile_path: str,
        normal_id: int,
        num_segments: int = 32,
        frames_per_segment: int = 16,
        imagefile_template: str = "{:06d}.jpg",
        transform=None,
        test_mode: bool = False,
        val_mode: bool = False,
        ncrops: int = 1,
        temporal_annotation_file: str = None,
        labels_file: str = None,
        stride: int = 1,
        spatialannotationdir_path: str = None,
    ):
        super().__init__()

        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.normal_id = normal_id
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.test_mode = test_mode
        self.val_mode = val_mode
        self.ncrops = ncrops
        self.temporal_annotation_file = temporal_annotation_file
        self.labels_file = labels_file
        self.stride = stride
        self.spatialannotationdir_path = spatialannotationdir_path

        self._parse_labelsfile()
        self._parse_annotationfile()
        if self.test_mode or self.val_mode:
            self.annotations = self._temporal_testing_annotations()

    def _load_image(self, directory: str, idx: int) -> Image.Image:
        return Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert(
            "RGB"
        )

    def _parse_labelsfile(self):
        self.labels = pd.read_csv(self.labels_file) if self.labels_file else None

    def _parse_annotationfile(self):
        self.video_list = [
            VideoRecord(x.strip().split(), self.root_path, self.spatialannotationdir_path)
            for x in open(self.annotationfile_path)
        ]

    def _temporal_testing_annotations(self):
        annotations = {}
        if self.temporal_annotation_file:
            with open(self.temporal_annotation_file) as annotations_f:
                lines = annotations_f.readlines()
                annotations = {
                    str(Path(line.strip().split()[0]).stem): line.strip().split()[2:]
                    for line in lines
                }
        return annotations

    def _get_start_indices(self, record: VideoRecord) -> "np.ndarray[int]":
        """For each segment, choose a start index from where frames are to be loaded from.

        Args:
            record: VideoRecord denoting a video sample.
        Returns:
            List of indices of where the frames of each
            segment are to be loaded from.
        """
        if self.test_mode:
            end_frame = round_to_nearest(
                record.num_frames,
                self.num_segments * self.frames_per_segment * self.stride,
            )
            start_indices = np.arange(end_frame / (self.frames_per_segment * self.stride)) * (
                self.frames_per_segment * self.stride
            )
        else:
            lower_bound = self.num_segments * self.frames_per_segment * self.stride
            if record.num_frames >= lower_bound:
                distance_between_indices = (
                    record.num_frames - self.frames_per_segment + 1
                ) // self.num_segments
            else:
                distance_between_indices = (
                    lower_bound - self.frames_per_segment + 1
                ) // self.num_segments

            start_indices = np.multiply(
                list(range(self.num_segments)), distance_between_indices
            ) + np.random.randint(
                (distance_between_indices + 1) - self.frames_per_segment + 1,
                size=self.num_segments,
            )

        return start_indices

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple["torch.Tensor[num_frames, channels, height, width]", Union[int, List[int]]],
        Tuple[Any, Union[int, List[int]]],
    ]:
        """For video with id idx, loads self.NUM_SEGMENTS * self.FRAMES_PER_SEGMENT frames from
        evenly chosen locations across the video.

        Args:
            idx: Video sample index.
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either 1) a list of PIL images if no transform is used
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
            if the transform "ImglistToTensor" is used
            3) or anything else if a custom transform is used.
        """
        record: VideoRecord = self.video_list[idx]

        frame_start_indices: "np.ndarray[int]" = self._get_start_indices(record)

        return self._get(record, frame_start_indices)

    def _get(
        self, record: VideoRecord, frame_start_indices: "np.ndarray[int]"
    ) -> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple["torch.Tensor[num_frames, channels, height, width]", Union[int, List[int]]],
        Tuple[Any, Union[int, List[int]]],
    ]:
        """Loads the frames of a video at the corresponding indices.

        Args:
            record: VideoRecord denoting a video sample.
            frame_start_indices: Indices from which to load consecutive frames from.
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either 1) a list of PIL images if no transform is used
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
            if thetra transform "ImglistToTensor" is used
            3) or anything else if a custom transform is used.
        """
        im_feature = np.load(record.path, allow_pickle=True)
        im_feature = torch.tensor(im_feature)

        if self.test_mode or self.val_mode:
            labels = list()
            video_name = Path(record.path).stem

            if self.annotations:
                start_indices = self.annotations[video_name][::2]
                stop_indices = self.annotations[video_name][1::2]
            else:
                start_indices = []
                stop_indices = []

            for i in range(im_feature.shape[0] // self.ncrops):
                label = self.normal_id
                for start_idx, end_idx in zip(start_indices, stop_indices):
                    if int(start_idx) <= i + record.start_frame <= int(end_idx):
                        label = record.label
                labels.append(label)

        im_feature = im_feature.view(-1, self.ncrops, im_feature.shape[-1])  # (t, ncrops, 512)
        im_feature = torch.transpose(im_feature, 0, 1)  # (ncrops, t, 512)
        im_feature = torch.permute(im_feature, (1, 0, 2))  # (t, ncrops, 512)

        if self.transform is not None:
            im_feature = self.transform(im_feature)

        # from each start_index, load self.frames_per_segment
        # consecutive frames
        features = list()
        val_labels = list()

        for start_index in frame_start_indices:
            # load self.frames_per_segment frames sequentially from frame_index (inclusive) every self.stride frames
            for i in range(self.frames_per_segment):
                frame_index = (int(start_index) + i * self.stride) % im_feature.shape[0]
                feature = im_feature[frame_index]
                features.append(feature)

                if self.val_mode:
                    val_labels.append(labels[frame_index])

        features = torch.cat(features)
        features = features.view(-1, self.ncrops, features.shape[-1])
        features = torch.permute(features, (1, 0, 2))

        if self.test_mode:
            # Determine the number of frames per segment
            segment_size = len(frame_start_indices) // self.num_segments
            return features, np.asarray(labels), record.label, segment_size, record.path
        elif self.val_mode:
            return features, record.label, np.asarray(val_labels)
        else:
            return features, record.label

    def __len__(self):
        return len(self.video_list)
