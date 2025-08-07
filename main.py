"""The main module being an entrypoint to the application."""

from pathlib import Path

import torch

from image_classifier.dataset import load_dataset
from image_classifier.evaluation import evaluate, showcase
from image_classifier.logger import logger
from image_classifier.metric import Accuracy, Precision, Recall
from image_classifier.network import BinaryImageClassifier
from image_classifier.device import device
from image_classifier.training import train
from image_classifier.data_structures import (
    ColourModel,
)


def main() -> None:
    """
    Run the training, inference, and evaluation.

    Raises
    ------
    RuntimeError
        Raised if a dataset is missing or failed to load.
    """
    epochs = 10
    dataset_location = Path("forest_fire")
    model_location = Path("model_weights.pth")

    logger.info("Using %s as the device.", device)

    if model_location.exists() and model_location.is_file():
        model: torch.nn.Module = BinaryImageClassifier(
            image_size=(250, 250), colour_model=ColourModel.RGB
        )

        logger.info("Loaded will be loaded from %s", str(model_location))
        model.load_state_dict(torch.load(model_location, weights_only=True))
        model.eval()

        logger.info(
            "Only the test set will be loaded, because pretrained weights has been loaded."
        )
        try:
            dataset = load_dataset(dataset_location, test_set_only=True)
        except FileNotFoundError as e:
            raise RuntimeError("The dataset is missing.") from e

    else:
        try:
            dataset = load_dataset(
                dataset_location, test_set_only=False, validation_percentage=0.1
            )
        except FileNotFoundError as e:
            raise RuntimeError("The dataset is missing.") from e

        if dataset.training is None or dataset.validation is None:
            logger.error(
                "Either a training set or a validation set is empty. "
                "Ensure both of them have been loaded properly."
            )
            raise RuntimeError(
                (
                    "Load the database with `test_set_only=False` to load "
                    "all three sets: validation, training, and test set."
                )
            )
        model = train(
            epochs=epochs,
            training_data=dataset.training,
            validation_data=dataset.validation,
        )
        torch.save(model.state_dict(), model_location)

    evaluate(
        model=model,
        test_set=dataset.test,
        metric_types=[
            Accuracy,
            Precision,
            Recall,
        ],
    )
    showcase(
        model=model, test_set_path=dataset_location / "Testing", samples_per_class=3
    )


if __name__ == "__main__":
    main()
