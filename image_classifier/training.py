"""The module for training a binary image classifier."""

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from image_classifier.device import device
from image_classifier.logger import logger
from image_classifier.network import BinaryImageClassifier
from image_classifier.data_structures import (
    ColourModel,
    Set,
)


def train(
    epochs: int,
    training_data: Set,
    validation_data: Set,
) -> torch.nn.Module:
    """
    Train a binary image classifier using the provided training and validation datasets.

    Parameters
    ----------
    epochs : int
        Number of epochs to train the model.
    training_data : Set
        Training dataset containing features and labels in NCHW format.
    validation_data : Set
        Validation dataset containing features and labels in NCHW format.

    Returns
    -------
    torch.nn.Module
        The trained binary image classifier model.
    """
    image_size = (training_data[0].shape[2], training_data[0].shape[3])
    model: torch.nn.Module = BinaryImageClassifier(
        colour_model=ColourModel.RGB, image_size=image_size
    )
    model = model.to(device)

    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters())

    training_set = TensorDataset(training_data.features, training_data.labels)
    training_dataloader: DataLoader = DataLoader(
        training_set, batch_size=64, shuffle=True
    )

    for epoch in tqdm(range(epochs), desc="Epochs"):
        for batch in tqdm(training_dataloader, desc="Batches", leave=False):
            features, label = batch

            model.zero_grad()
            predictions = model(features)
            loss = loss_fn(predictions, label)

            loss.backward()
            optimiser.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            predictions = model(validation_data.features)
            loss = loss_fn(predictions, validation_data.labels)
            logger.info("Epoch %i: Validation loss: %f", epoch + 1, loss)
            model.train()

    return model
