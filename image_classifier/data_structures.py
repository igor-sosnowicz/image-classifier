from enum import Enum
from typing import NamedTuple, Optional

import torch
from dataclasses import dataclass
from pathlib import Path


class ColourModel(Enum):
    GRAYSCALE = 1
    RGB = 3
    RGBA = 4


class Set(NamedTuple):
    features: torch.Tensor
    labels: torch.Tensor


class Dataset(NamedTuple):
    test: Set
    training: Optional[Set] = None
    validation: Optional[Set] = None


@dataclass
class DataSourceDirectory:
    path: Path
    colour_model: ColourModel
    class_label: int
    loading_message: str
    extension: str
