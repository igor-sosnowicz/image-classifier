import torch
from image_classifier.data_structures import ColourModel


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
        # Ensure input and model are on the same device.
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)

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

    @staticmethod
    def _calculate_post_pooling_size(
        size: int, num_pools: int = 3, kernels: int = 3, stride: int = 2
    ) -> int:
        for _ in range(num_pools):
            size = (size + 2 * 0 - kernels) // stride + 1  # padding=0 for avgpool
        return int(size)
