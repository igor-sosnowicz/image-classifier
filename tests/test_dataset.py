"""Module for testing splitting, preprocessing, and loading a dataset."""

import torch
import pytest

from main import Set, split_set


@pytest.fixture
def sample_set() -> Set:
    random_features = torch.randn(
        (
            100,
            3,
            250,
            250,
        )
    )
    random_labels = torch.randint(0, 2, (100, 1, 1, 1))

    return Set(random_features, random_labels)


@pytest.mark.parametrize(
    "randomise",
    (
        False,
        True,
    ),
)
def test_splitting_set(sample_set, randomise: bool) -> None:
    """Test splitting a larger set into two smaller sets."""
    training_set, validation_set = split_set(
        entire_set=sample_set, first_set_percentage=0.7, randomise=randomise
    )

    total = len(training_set.features) + len(validation_set.features)
    assert total == 100
    assert len(training_set.features) == 70
    assert len(validation_set.features) == 30
