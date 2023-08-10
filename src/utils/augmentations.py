import torch
import torchvision
from PIL import Image
from torchvision.transforms import Lambda

from src.data.components.gtransforms import (
    GroupCenterCrop,
    GroupNormalize,
    GroupScale,
    GroupToTensor,
)

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def get_augmentations(input_size, ncrops):
    # return transform
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    transform = torchvision.transforms.Compose(
        [
            GroupScale(input_size, interpolation=BICUBIC),
            GroupCenterCrop(input_size),
            GroupToTensor(),
            GroupNormalize(input_mean, input_std),
            Lambda(lambda x: torch.stack(x)),  # returns a 4D tensor
        ]
    )
    return transform
