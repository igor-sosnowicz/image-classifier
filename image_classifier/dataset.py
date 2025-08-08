"""
The module for a dataset loading from a filesystem and the dataset
manipulations, such as splitting into sub-sets.
"""

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from image_classifier.data_structures import (
    ColourModel,
    DataSourceDirectory,
    Dataset,
    Set,
)
from image_classifier.logger import logger
from image_classifier.device import device


def split_set(
    entire_set: Set,
    first_set_percentage: float,
    randomise: bool = True,
) -> tuple[Set, Set]:
    """
    Split a set into two sub-sets.

    Parameters
    ----------
    entire_set : Set
        An initial set containing samples to be splitted.
    first_set_percentage : float
        A value in the range (0, 1) what part of the initial set
        should go the first sub-set.
    randomise : bool, optional
        Whether samples should be assigned randomly. By default, True.

    Returns
    -------
    tuple[Set, Set]
        A tuple containing two sub-sets.
    """
    features, labels = entire_set[0], entire_set[1]
    num_samples = features.shape[0]
    split_idx = int(num_samples * first_set_percentage)
    if randomise:
        indices = torch.randperm(num_samples)
    else:
        indices = torch.arange(num_samples)
    first_indices = indices[:split_idx]
    second_indices = indices[split_idx:]

    first_set = Set(features=features[first_indices], labels=labels[first_indices])
    second_set = Set(features=features[second_indices], labels=labels[second_indices])
    return first_set, second_set


def _load_and_preprocess_from_directories(
    data_source_directories: Sequence[DataSourceDirectory],
) -> Set:
    """
    Load a set from one or more directories.

    Parameters
    ----------
    data_source_directories : Sequence[DataSourceDirectory]
        An iterable containing entries, which describe directories to
        load the data from.

    Returns
    -------
    Set
        A set made of combined data from all directories indicated
        in `data_source_directories`.
    """
    images = []
    image_classes = []

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    for config in data_source_directories:
        image_paths = list(config.path.glob(f"*.{config.extension}"))
        print("image paths", config.path)
        for img_path in tqdm(image_paths, desc=config.loading_message, unit="image"):
            img = Image.open(img_path).convert(config.colour_model.name)
            img_tensor = transform(img)
            images.append(img_tensor)
            image_classes.append(config.class_label)

        logger.info("Loaded %i images from %s", len(image_paths), str(config.path))

    # Stack tensors and move to GPU if available
    features = torch.stack(images).to(device)
    labels = torch.tensor(image_classes, dtype=torch.float32).unsqueeze(1).to(device)

    return Set(features, labels)


def load_set(set_directory: Path) -> Set:
    """
    Load a set of the Wildfire Detection Image Data dataset.

    Parameters
    ----------
    set_directory : Path
        _description_

    Returns
    -------
    Set
        _description_
    """
    fire_config = DataSourceDirectory(
        set_directory / "fire",
        ColourModel.RGB,
        1,
        "Loading images depicting fire",
        "jpg",
    )
    fireless_config = DataSourceDirectory(
        set_directory / "nofire",
        ColourModel.RGB,
        0,
        "Loading images not depicting fire",
        "jpg",
    )

    return _load_and_preprocess_from_directories((fire_config, fireless_config))


def load_dataset(
    dataset_path: Path,
    validation_percentage: float = 0.1,
    test_set_only: bool = False,
) -> Dataset:
    """
    Load the Wildfire Detection Image Data dataset from a directory.

    Parameters
    ----------
    dataset_path : Path
        A path to the root directory of the Wildfire Detection Image Data dataset.
    validation_percentage : float, optional
        A value in the range (0, 1), defining what part of
        the training+validation set should go the validation set.
        The validation and training set are combined in this dataset.
        Defaults to 0.1.
    test_set_only : bool, optional
        Whether to load only a test set. It is not necessary to load
        the entire dataset for the sole evaluation on the test set.
        Defaults to False.

    Returns
    -------
    Dataset
        The preprocessed dataset loaded into memory.

    Raises
    ------
    FileNotFoundError
        Raised if there is no dataset under the provided location.
    """
    TRAINING_AND_VALIDATION = "Training and Validation"
    TEST = "Testing"
    dataset_path = dataset_path.resolve().absolute()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Cannot load a dataset from the non-existent location: {str(dataset_path)}"
        )

    test_set = load_set(dataset_path / TEST)

    if test_set_only:
        return Dataset(
            test=test_set,
            training=None,
            validation=None,
        )

    training_and_validation_set = load_set(dataset_path / TRAINING_AND_VALIDATION)

    training_set, validation_set = split_set(
        entire_set=training_and_validation_set,
        first_set_percentage=1 - validation_percentage,
        randomise=True,
    )

    return Dataset(
        training=training_set,
        validation=validation_set,
        test=test_set,
    )
