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
    Splits the given features and labels into two sets according to the given percentage.
    Returns (first_set, second_set).
    If randomise is True, shuffles the data before splitting.
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


def _load_set_from_directories(
    data_source_directories: Sequence[DataSourceDirectory],
) -> Set:
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

    return _load_set_from_directories((fire_config, fireless_config))


def load_dataset(
    dataset_path: Path,
    validation_percentage: float = 0.1,
    test_set_only: bool = False,
) -> Dataset:
    # Training & validation sets share the common set in this database.
    TRAINING_AND_VALIDATION = "Training and Validation"
    TEST = "Testing"
    dataset_path = dataset_path.resolve().absolute()
    if not dataset_path.exists():
        raise FileExistsError(
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
