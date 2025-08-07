"""
The module for evaluation and showcasing the trained image classifier capabilities.
"""

from pathlib import Path
from random import shuffle

import torch
import torchvision
from PIL import Image

from image_classifier.device import device
from image_classifier.data_structures import (
    ColourModel,
    Set,
)
from image_classifier.metric import Metric


def evaluate(
    model: torch.nn.Module, metric_types: list[type[Metric]], test_set: Set
) -> None:
    """
    Evaluate the trained model with Mean Squared Error (MSE) loss and other metrics.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be evaluated.
    metric_types : list[type[Metric]]
        A list of classes representing metrics that should be calculated.
        These metric classes should be imported from `images_classifier.metric` module.
    test_set : Set
        A test set, on which the metrics should be calculated.
    """
    loss_fn = torch.nn.MSELoss()
    model.eval()

    with torch.inference_mode():
        features = test_set.features.to(device)
        labels = test_set.labels.to(device)
        predictions = model(features).to(device)

        loss = loss_fn(predictions, labels)

        for Metric in metric_types:
            metric = Metric(actual=labels, predicted=predictions)
            print(f"{metric.get_metric_name()}: {metric.get_result()}")

    print(f"Loss: {loss}")


def assess_images(
    model: torch.nn.Module,
    image_paths: list[Path],
    true_label: str,
    colour_model: ColourModel,
) -> None:
    """
    Assess the model manually by printing its predictions and actual labels for provided images.

    All the images have to share the label and colour model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be evaluated.
    image_paths : list[Path]
        A list of paths to images.
    true_label : str
        A common label for all images.
    colour_model : ColourModel
        A common colour model for all images.
    """
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    model.eval()

    for img_path in image_paths:
        img = Image.open(img_path).convert(colour_model.name)
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.inference_mode():
            pred = model(img_tensor)

        print(
            f"Path: {img_path} | True label: {true_label} | "
            f"Predicted label: {'fire' if pred.item() > 0.5 else 'no fire'} |"
            f" Model output: {pred.item():.3f}"
        )


def showcase(
    test_set_path: Path,
    model: torch.nn.Module,
    samples_per_class: int = 5,
    randomise_order: bool = True,
    colour_model: ColourModel = ColourModel.RGB,
) -> None:
    """
    Showcase the capabilities of a model by printing real labels
    and model predictions for samples from a test set.

    Parameters
    ----------
    test_set_path : Path
        A path to a test set.
    model : torch.nn.Module
        The model to be evaluated.
    samples_per_class : int, optional
        A number of samples from each class to show, by default 5.
    randomise_order : bool
        Whether shown samples should chosen randomly.
        If False, the samples are the first samples,
        sorted by a alphanumerical sorting provided by the filesystem.
        Default to True.
    colour_model : ColourModel
        A colour model of all images in a test set.
    """
    class_labels = (
        "fire",
        "nofire",
    )
    for label in class_labels:
        data_source_directory = test_set_path / label
        image_files = list(data_source_directory.glob("*.jpg"))[:samples_per_class]
        if randomise_order:
            shuffle(image_files)
        assess_images(
            model=model,
            image_paths=image_files,
            true_label=label,
            colour_model=colour_model,
        )
