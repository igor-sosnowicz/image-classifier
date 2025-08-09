"""The module with test for calculating metrics."""

import inspect
import pytest
import torch

from image_classifier import metric


def get_all_metrics() -> list[type[metric.Metric]]:
    """
    Get all available metric classes.

    Returns
    -------
    list[type[metric.Metric]]
        A list containing each metric class.
    """
    return [
        cls
        for name, cls in inspect.getmembers(metric, inspect.isclass)
        if issubclass(cls, metric.Metric)
        and cls is not metric.Metric
        and cls.__module__ == metric.__name__
    ]


@pytest.fixture
def perfectly_matched_labels() -> tuple[torch.Tensor, torch.Tensor]:
    """Fixture providing a tuple of actual and predicted labels matching perfectly."""
    actual = torch.concat(
        [
            torch.zeros(size=(5,)),
            torch.ones(size=(5,)),
        ]
    )
    predicted = torch.linspace(0, 0.9, steps=10) + 0.1

    assert actual.shape == predicted.shape
    return (actual, predicted)


@pytest.mark.parametrize(
    "metric_cls",
    get_all_metrics(),
)
def test_metrics_on_perfect_predictions(
    metric_cls: type[metric.Metric], perfectly_matched_labels
) -> None:
    """
    Test if values of metrics are equal to 1.0 on a set
    of labels and predictions matching perfectly.
    """
    metric_instance = metric_cls(
        actual=perfectly_matched_labels[0], predicted=perfectly_matched_labels[1]
    )
    assert metric_instance.get_result() == 1.0
