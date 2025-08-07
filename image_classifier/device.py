"""
The module providing an unified location to import a device,
where the training and inference should be performed.
"""

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = ["device"]
