from enum import Enum
from typing import NamedTuple

import torch


class ColourModel(Enum):
    GRAYSCALE = 1
    RGB = 3
    RGBA = 4


class Set(NamedTuple):
    features: torch.Tensor
    labels: torch.Tensor


class Dataset(NamedTuple):
    training: Set
    validation: Set
    test: Set
