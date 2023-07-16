"""Custom `torch.utils.data.DataLoader` for the Animals with Attributes 2 dataset.

[homepage](https://cvml.ista.ac.at/AwA2/)
[download](https://cvml.ista.ac.at/AwA2/AwA2-data.zip)

Includes:
	`torch.utils.data.AnimalsWithAttributesDataLoader`
"""


from __future__ import annotations

from typing import Union

import torch.utils.data

from ...torchvision.datasets import AnimalsWithAttributesDataset


class AnimalsWithAttributesDataLoader(torch.utils.data.DataLoader):
	"""A custom dataloader on images from the Animals with Attributes dataset.

	Combines a dataset and a sampler, and provides an iterable over the given dataset.

	"""

	def __init__(self, dataset: torch.utils.data.Dataset, generator: torch.Generator,
		batch_size: int = 1,
	**kwargs):
		"""Initialize the directory containing the images.

		Arguments:
			dataset: dataset from which to load the data.
			batch_size: how many samples per batch to load (default: `1`)
		"""
		super(AnimalsWithAttributesDataLoader, self).__init__(dataset,
			batch_size=batch_size,
			shuffle=True,
		#	sampler=None,
		#	batch_sampler=None,
		#	num_workers=0,
		#	collate_fn=None,
		#	pin_memory=False,
			drop_last=True,
		#	timeout=0,
		#	worker_init_fn=None,
		#	multiprocessing_context=None,
			generator=generator,
		#	prefetch_factor=None,
		#	persistent_workers=False,
		#	pin_memory_device='',
		**kwargs)
