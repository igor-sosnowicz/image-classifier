# pylint: disable=missing-docstring

from pathlib import Path

from image_classifier.data_structures import ColourModel, Dataset, Set
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from PIL import Image

from image_classifier.logger import logger
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        pooled_h = BinaryImageClassifier._calculate_post_pooling_size(image_size[0])
        pooled_w = BinaryImageClassifier._calculate_post_pooling_size(image_size[1])
        in_features = channels["conv_3"] * pooled_h * pooled_w

        self.conv_1 = torch.nn.Conv2d(
            in_channels=channels["input"],
            out_channels=channels["conv_1"],
            kernel_size=(3, 3),
            padding=padding,
            stride=stride,
        )
        self.pool_1 = torch.nn.AvgPool2d(kernel_size=(3, 3), stride=2)
        self.relu_1 = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv2d(
            in_channels=channels["conv_1"],
            out_channels=channels["conv_2"],
            kernel_size=(3, 3),
            padding=padding,
            stride=stride,
        )
        self.pool_2 = torch.nn.AvgPool2d(kernel_size=(3, 3), stride=2)
        self.relu_2 = torch.nn.ReLU()
        self.conv_3 = torch.nn.Conv2d(
            in_channels=channels["conv_2"],
            out_channels=channels["conv_3"],
            kernel_size=(3, 3),
            padding=padding,
            stride=stride,
        )
        self.pool_3 = torch.nn.AvgPool2d(kernel_size=(3, 3), stride=2)
        self.relu_3 = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.dense = torch.nn.Linear(in_features=in_features, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_1(x)
        y = self.pool_1(y)
        y = self.relu_1(y)
        y = self.conv_2(y)
        y = self.pool_2(y)
        y = self.relu_2(y)
        y = self.conv_3(y)
        y = self.pool_3(y)
        y = self.relu_3(y)
        y = self.flatten(y)
        y = self.dense(y)
        return self.sigmoid(y)

        # Calculate output size after 3 poolings with stride 2 and kernel 3

    @staticmethod
    def _calculate_post_pooling_size(
        size: int, num_pools: int = 3, kernels: int = 3, stride: int = 2
    ) -> int:
        for _ in range(num_pools):
            size = (size + 2 * 0 - kernels) // stride + 1  # padding=0 for avgpool
        return int(size)


def train(
    epochs: int,
    training_data: Set,
    validation_data: Set,
) -> BinaryImageClassifier:
    # We expect the data in the NCHW format.
    image_size = (training_data[0].shape[2], training_data[0].shape[3])
    model = BinaryImageClassifier(colour_model=ColourModel.RGB, image_size=image_size)
    model = model.to(device)  # Move model to GPU if available

    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters())

    # X_train, Y_train = training_data
    X_valid, Y_valid = validation_data

    # Move data to GPU if available
    # X_train = X_train.to(device)
    # Y_train = Y_train.to(device)
    # X_valid = X_valid.to(device)
    # Y_valid = Y_valid.to(device)

    training_set = TensorDataset(training_data.features, training_data.labels)
    # validation_set = TensorDataset(validation_data.features, validation_data.labels)

    training_dataloader: DataLoader = DataLoader(
        training_set, batch_size=64, shuffle=True
    )
    # validation_dataloader: DataLoader = DataLoader(
    #     validation_data, batch_size=64, shuffle=True
    # )

    for epoch in tqdm(range(epochs), desc="Epochs"):
        for batch in tqdm(training_dataloader, desc="Batches", leave=False):
            features, label = batch

            model.train()
            model.zero_grad()
            prediction = model(features)
            loss = loss_fn(prediction, label)

            loss.backward()
            optimiser.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            validation_predictions = model(X_valid)
            validation_loss = loss_fn(validation_predictions, Y_valid)
            logger.info("Epoch %i: Validation loss: %f", epoch + 1, validation_loss)
            model.train()

    return model


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


def load_set(set_directory: Path) -> Set:
    path_to_images_with_fire = set_directory / "fire"
    path_to_images_without_fire = set_directory / "nofire"

    # load All images from path_to_images_with_fire and set labels to 1

    image_paths = list(path_to_images_with_fire.glob("*"))
    images = []
    labels = []

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    for img_path in tqdm(image_paths, desc="Loading fire images"):
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)
        images.append(img_tensor)
        labels.append(1)

    logger.info(
        "Loaded %i images from %s", len(image_paths), str(path_to_images_with_fire)
    )

    # Repeat for nofire images, set labels to 0.
    nofire_paths = list(path_to_images_without_fire.glob("*.jpg"))
    for img_path in tqdm(nofire_paths, desc="Loading nofire images"):
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)
        images.append(img_tensor)
        labels.append(0)

    logger.info(
        "Loaded %i images from %s", len(image_paths), str(path_to_images_without_fire)
    )

    # Stack tensors and move to GPU if available
    features = torch.stack(images).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)

    return Set(features, labels)


def load_dataset(dataset_path: Path, validation_percentage: float = 0.1) -> Dataset:
    # Training & validation sets share the common set in this database.
    TRAINING_AND_VALIDATION = "Training and Validation"
    TEST = "Testing"
    dataset_path = dataset_path.resolve().absolute()
    if not dataset_path.exists():
        raise FileExistsError(
            f"Cannot load a dataset from the non-existent location: {str(dataset_path)}"
        )

    training_and_validation_set = load_set(dataset_path / TRAINING_AND_VALIDATION)
    test_set = load_set(dataset_path / TEST)

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


def evaluate(model: BinaryImageClassifier, test_set: Set) -> None:
    loss_fn = torch.nn.MSELoss()
    model.eval()

    with torch.inference_mode():
        predictions = model(test_set.features)
        loss = loss_fn(predictions, test_set.labels)

    print(f"Total loss on the training set: {loss}")


def assess_images(
    model: torch.nn.Module, image_paths: list[Path], true_label: str
) -> None:
    transform = transforms.Compose([transforms.ToTensor()])
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


def main() -> None:
    epochs = 10
    dataset_location = Path("forest_fire")
    logger.info("Using %s as the device.", device)

    dataset = load_dataset(dataset_location)
    model = train(
        epochs=epochs,
        training_data=dataset.training,
        validation_data=dataset.validation,
    )
    evaluate(model, dataset.test)
    showcase(
        model=model, test_set_path=dataset_location / "Testing", samples_per_class=3
    )


if __name__ == "__main__":
    main()
