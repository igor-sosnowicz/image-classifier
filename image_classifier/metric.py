"""
The module with definitions of machine learning metrics,
such as: accuracy, precision, F1 score and other.
"""

import abc
from typing import Union

import torch


class Metric(abc.ABC):
    """
    An abstract base class defining a structure of a metric.
    Its sub-classes are used for assessing an artificial neural
    network, for example, accuracy, F1 score, precision,
    confusion matrix and other.
    """

    def __init__(self, actual: torch.Tensor, predicted: torch.Tensor) -> None:
        """
        Assign predictions and ground truths to member attributes.

        Parameters
        ----------
        actual : torch.Tensor
            The tensor with ground truth labels.
        predicted : torch.Tensor
            The tensor with predicted labels.
        """
        super().__init__()
        self.actual = actual
        self.predicted = predicted

    @abc.abstractmethod
    def get_result(self) -> Union[int, float]:
        """
        Retrieve the latest cached version of the metric.

        Returns
        -------
        Union[int, float]
            The value of the metric.
        """
        pass

    @abc.abstractmethod
    def get_metric_name(self) -> str:
        """
        Get name of the metric.

        Returns
        -------
        str
            The textual name of the metric.
        """
        pass

    @abc.abstractmethod
    def _calculate(self) -> None:
        """Calculate and store a cached value of the metric."""
        pass


class Accuracy(Metric):
    """The implementation of accuracy metric."""

    def get_result(self) -> float:
        if not hasattr(self, "_precision"):
            self._calculate()
        return self._accuracy

    def get_metric_name(self) -> str:
        return "Accuracy"

    def _calculate(self) -> None:
        predicted_classes = (self.predicted > 0.5).to(dtype=torch.int)
        correct = (self.actual == predicted_classes).sum().item()
        total = self.actual.numel()
        self._accuracy = correct / total if total > 0 else 0.0


class Precision(Metric):
    """The implementation of precision metric."""

    def get_result(self) -> float:
        if not hasattr(self, "_precision"):
            self._calculate()
        return self._precision

    def get_metric_name(self) -> str:
        return "Precision"

    def _calculate(self) -> None:
        predicted_classes = (self.predicted > 0.5).to(dtype=torch.int)
        true_positives = ((predicted_classes == 1) & (self.actual == 1)).sum().item()
        predicted_positives = (predicted_classes == 1).sum().item()
        self._precision = (
            true_positives / predicted_positives if predicted_positives > 0 else 0.0
        )


class Recall(Metric):
    """The implementation of recall metric."""

    def get_result(self) -> float:
        if not hasattr(self, "_recall"):
            self._calculate()
        return self._recall

    def get_metric_name(self) -> str:
        return "Recall"

    def _calculate(self) -> None:
        predicted_classes = (self.predicted > 0.5).int()
        true_positives = ((predicted_classes == 1) & (self.actual == 1)).sum().item()
        actual_positives = (self.actual == 1).sum().item()
        self._recall = (
            true_positives / actual_positives if actual_positives > 0 else 0.0
        )
