from pathlib import Path

import torch
import torchvision
from PIL import Image

from image_classifier.device import device
from image_classifier.data_structures import (
    Set,
)
from image_classifier.metric import Metric


def evaluate(
    model: torch.nn.Module, metric_types: list[type[Metric]], test_set: Set
) -> None:
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

    # TODO: Implement the following metrics: confusion matrix, F1 score, precision, accuracy.

    print(f"Total loss on the test set: {loss}")


def assess_images(
    model: torch.nn.Module, image_paths: list[Path], true_label: str
) -> None:
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    model.eval()

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.inference_mode():
            pred = model(img_tensor)

        print(
            f"Path: {img_path} | True label: {true_label} | "
            f"Predicted label: {'fire' if pred.item() > 0.5 else 'no fire'} |"
            f" Model output: {pred.item():.3f}"
        )


def showcase(
    test_set_path: Path, model: torch.nn.Module, samples_per_class: int = 5
) -> None:
    # Load a few images from the test set and show their paths, infer their labels with the model.
    class_labels = (
        "fire",
        "nofire",
    )
    for label in class_labels:
        data_source_directory = test_set_path / label
        image_files = list(data_source_directory.glob("*.jpg"))[:samples_per_class]
        assess_images(model=model, image_paths=image_files, true_label=label)
