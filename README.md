# Binary Image Classifier

A Python-based binary image classifier using PyTorch, designed to detect wildfires in images.

## Dataset

This project uses the [Wildfire Detection Image Data](https://www.kaggle.com/datasets/brsdincer/wildfire-detection-image-data) from Kaggle.

**Before running the project:**

1. Download the dataset from Kaggle.
2. Unzip the contents.
3. Place the `forest_fire` directory in the root of this repository.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) – Python version and dependency manager

## Installation

Install dependencies with:

```bash
uv sync
```

## Usage

Run the classifier with:

```bash
uv run main.py
```

On the first run, the model is trained, and inference is performed.  
On subsequent runs, the weights are loaded and only inference is performed.
