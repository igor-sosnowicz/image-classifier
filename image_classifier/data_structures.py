"""The module with project-wide data structures, including: enumerations, named tuples, and data classes."""

from enum import Enum
from typing import NamedTuple, Optional
from dataclasses import dataclass
from pathlib import Path

import torch


class ColourModel(Enum):
    """
    The enumeration mapping colour models for images to a corresponding
    number of channels.
    """

    GRAYSCALE = 1
    RGB = 3
    RGBA = 4


class Set(NamedTuple):
    """
    The named tuple constituting set, a part of an entire dataset.
    It consists of input features (X) and class labels (Y).
    """

    features: torch.Tensor
    labels: torch.Tensor


class Dataset(NamedTuple):
    """
    The named tuple representing a dataset containing test,
    training (optionally), and validation (optionally) sets.
    """

    test: Set
    training: Optional[Set] = None
    validation: Optional[Set] = None


@dataclass
class DataSourceDirectory:
    """
    Configuration for a directory containing image data.
    Specifies the location, image format, colour model, class label,
    and a message to display while loading. Used to define which
    directories to load data from and how to interpret their contents.

    Attributes
    ----------
    path: Path
        A path to a directory with images.
    colour_model: ColourModel
        A colour model for all images in the directory.
    class_label: int
        A numeric value indicating, to which class a sample belongs.
        In other words, it is the label of the sample.
        1 means `fire` class, 0 means `nofire`.
    loading_message: str
        A text displayed by an interactive progress bar while
        loading a directory.
    extension: str
        A file extension that should be used to load images from the directory.
        For example, `jpg` or `png`.
    """

    path: Path
    colour_model: ColourModel
    class_label: int
    loading_message: str
    extension: str
