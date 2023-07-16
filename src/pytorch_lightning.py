"""Pytorch lightning wrappers.

Includes:
	`AnimalsWithAttributesDataModule`: based on `AnimalsWithAttribulesDataLoader` on a `AnimalsWithAttributesDataset`
	`AnimalsWithAttributesModule`: composing several `torch.nn.Module` with a loss and an optimizer
"""


from typing import Optional, Union

import torch
import torchvision
import pytorch_lightning

from .torch.utils.data import AnimalsWithAttributesDataLoader
from .torchvision.datasets import AnimalsWithAttributesDataset
from .zeroshot.datasets import ZeroshotAnimalsWithAttributesDataset
from globals import generator


class AnimalsWithAttributesDataModule(pytorch_lightning.LightningDataModule):
	def __init__(self,
		dataset: Union[AnimalsWithAttributesDataset, ZeroshotAnimalsWithAttributesDataset],
		batch_size: int = 1,
	):
		super(AnimalsWithAttributesDataModule, self).__init__()

		self.dataset = dataset
		self.batch_size = batch_size

	def setup(self, stage: str):
		(
			self.data_train,
			self.data_devel,
			self.data_valid,
		) = self.dataset.random_split()

	def train_dataloader(self):
		return AnimalsWithAttributesDataLoader(self.data_train, generator, batch_size=self.batch_size)

	def val_dataloader(self):
		return AnimalsWithAttributesDataLoader(self.data_devel, generator, batch_size=self.batch_size)

	def test_dataloader(self):
		return AnimalsWithAttributesDataLoader(self.data_valid, generator, batch_size=self.batch_size)

	def predict_dataloader(self):
		return AnimalsWithAttributesDataLoader(self.data_valid, generator, batch_size=self.batch_size)
