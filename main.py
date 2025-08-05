# pylint: disable=missing-docstring

from enum import Enum
from pathlib import Path
from typing import NamedTuple

import torch
import numpy

from image_classifier.logger import logger


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


class BinaryImageClassifier(torch.nn.Module):
    def __init__(
        self, image_size: tuple[int, int], colour_model: ColourModel, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        channels = {
            "input": colour_model.value,
            "conv_1": 8,
            "conv_2": 16,
            "conv_3": 32,
        }
        padding = "same"
        stride = 1
        self.conv_1 = torch.nn.Conv2d(
            in_channels=channels["input"],
            out_channels=channels["conv_1"],
            kernel_size=(3, 3),
            padding=padding,
            stride=stride,
        )
        self.relu_1 = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv2d(
            in_channels=channels["conv_1"],
            out_channels=channels["conv_2"],
            kernel_size=(3, 3),
            padding=padding,
            stride=stride,
        )
        self.relu_2 = torch.nn.ReLU()
        self.conv_3 = torch.nn.Conv2d(
            in_channels=channels["conv_2"],
            out_channels=channels["conv_3"],
            kernel_size=(3, 3),
            padding=padding,
            stride=stride,
        )
        self.relu_3 = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        in_features = channels["conv_3"] * image_size[0] * image_size[1]
        self.dense = torch.nn.Linear(in_features=in_features, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_1(x)
        y = self.relu_1(y)
        y = self.conv_2(y)
        y = self.relu_2(y)
        y = self.conv_3(y)
        y = self.relu_3(y)
        y = self.flatten(y)
        y = self.dense(y)
        return self.sigmoid(y)


def train(
    epochs: int,
    training_data: tuple[torch.Tensor, torch.Tensor],
    validation_data: tuple[torch.Tensor, torch.Tensor],
) -> BinaryImageClassifier:
    # We expect the data in the NCHW format.
    image_size = (training_data[0].shape[2], training_data[0].shape[3])
    model = BinaryImageClassifier(colour_model=ColourModel.RGB, image_size=image_size)

    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters())

    X_train, Y_train = training_data
    X_valid, Y_valid = validation_data

    model.train()
    for epoch in range(epochs):
        model.zero_grad()

        predictions = model(X_train)
        loss = loss_fn(predictions, Y_train)
        loss.backward()
        optimiser.step()

        if epoch % 100:
            model.eval()
            current_predictions = model(X_valid)
            current_loss = loss_fn(current_predictions, Y_valid)
            logger.info("Epoch %i: %f", epoch + 1, current_loss)
            model.train()

    return model


def load_dataset(dataset_path: Path) -> Dataset:
    def random_set():
        n = 16  # Fixed batch size
        features = torch.rand((n, 3, 250, 250))  # NCHW: N, C=3, H=250, W=250
        labels = torch.randint(0, 2, (n, 1)).float()  # Binary labels
        return Set(features, labels)

    return Dataset(
        training=random_set(),
        validation=random_set(),
        test=random_set(),
    )


def evaluate(model: BinaryImageClassifier, test_set: Set) -> None:
    # print evaluation
    pass


def main():
    epochs = 10_000
    dataset_location = Path()

    dataset = load_dataset(dataset_location)
    model = train(
        epochs=epochs,
        training_data=dataset.training,
        validation_data=dataset.validation,
    )
    evaluate(model, dataset.test)


if __name__ == "__main__":
    main()
