"""Code global settings.

Includes:
	`seed`: global seed
	`generator`: once-defined random number generator with aforementioned seed
	`torch.set_default_device`: use CUDA if available
"""


import numpy
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"  # use CUDA if available

torch.set_default_device(device)

seed: int = 0

numpy.random.seed(seed)
torch.manual_seed(seed)

generator = torch.Generator(device).manual_seed(seed)
