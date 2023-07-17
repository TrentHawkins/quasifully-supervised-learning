"""Pytorch lightning wrappers.

Includes:
	`AnimalsWithAttributesDataModule`: based on `AnimalsWithAttribulesDataLoader` on a `AnimalsWithAttributesDataset`
	`AnimalsWithAttributesModule`: composing several `torch.nn.Module` with a loss and an optimizer
"""


from typing import Optional, Union

import torch
import torch.utils.data
import torchvision
import pytorch_lightning

from .torch.utils.data import AnimalsWithAttributesDataLoader
from .torchvision.datasets import AnimalsWithAttributesDataset
from chartools import separator
from globals import generator


class AnimalsWithAttributesDataModule(pytorch_lightning.LightningDataModule):
	def __init__(self,
		source_path: str = "trainvalclasses.txt",
		target_path: str = "testclasses.txt",
		batch_size: int = 1,
		zeroshot: bool = False,
		transductive: bool = False,
	**kwargs):
		super(AnimalsWithAttributesDataModule, self).__init__(**kwargs)

	#	Label subset paths:
		self.source_path = source_path
		self.target_path = target_path

	#	Batch size:
		self.batch_size = batch_size

	#	Setting:
		self.zeroshot = zeroshot
		self.transductive = zeroshot

	def prepare_data(self):
		separator(3)
		separator(2, f"Animals with Attributes 2: zeroshot")

	#	Images with source label:
		self.source = AnimalsWithAttributesDataset(
			labels_path=self.source_path,
		)

		separator(1, str(self.source))

	#	Images with target label:
		self.target = AnimalsWithAttributesDataset(
			labels_path=self.target_path,
		)

		separator(2, str(self.target))

		self._totals = torch.utils.data.ConcatDataset(
			[
				self.source,
				self.target,
			]
		)

	def setup(self, stage: str):
		if self.zeroshot:
			(
				source_train_images,
				source_devel_images,
				source_valid_images,
			) = self.source.random_split()
			(
				target_train_images,
				target_devel_images,
				target_valid_images,
			) = self.target.random_split()

			if self.transductive:
				if stage == "fit":
					self.train_images = torch.utils.data.ConcatDataset(
						[
							source_train_images,
							target_train_images,
						]
					)
				if stage == "validate":
					self.devel_images = torch.utils.data.ConcatDataset(
						[
							source_devel_images,
							target_devel_images,
						]
					)

				if stage == "test" or stage == "predict":
					self.valid_images = torch.utils.data.ConcatDataset(
						[
							source_valid_images,
							target_valid_images,
						]
					)

	def train_dataloader(self):
		return AnimalsWithAttributesDataLoader(self.train_images, generator, batch_size=self.batch_size)

	def val_dataloader(self):
		return AnimalsWithAttributesDataLoader(self.devel_images, generator, batch_size=self.batch_size)

	def test_dataloader(self):
		return AnimalsWithAttributesDataLoader(self.valid_images, generator, batch_size=self.batch_size)

	def predict_dataloader(self):
		return AnimalsWithAttributesDataLoader(self.valid_images, generator, batch_size=self.batch_size)
