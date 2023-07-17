"""Pytorch lightning wrappers.

Includes:
	`AnimalsWithAttributesDataModule`: based on `AnimalsWithAttribulesDataLoader` on a `AnimalsWithAttributesDataset`
	`AnimalsWithAttributesModule`: composing several `torch.nn.Module` with a loss and an optimizer
"""


from typing import Optional, Union

from .chartools import separator
from .globals import generator

import torch
import torch.utils.data

from src.torch.utils.data import AnimalsWithAttributesDataLoader
from src.torchvision.datasets import AnimalsWithAttributesDataset

import pytorch_lightning


torch.utils.data.ConcatDataset.__str__ = torch.utils.data.Dataset.__str__


class AnimalsWithAttributesDataModule(pytorch_lightning.LightningDataModule):
	"""A lightning data module encapsulating datasplitting and loading depending on context.

	Contexts:
		normal training:
		-	all labels known during training
		-	all labels known during  testing

		generalized_zeroshot training:
		-	only source labels known during training
		-	all         labels known during  testing

		generalized_zeroshot training in the transductive setting:
		-	source labels     known during training
		-	target labels not known during training but samples with such are known and unlabelled
		-	all    labels     known during  testing
	"""

	def __init__(self,
		labels_path: str = "allclasses.txt",
		source_path: str = "trainvalclasses.txt",
		target_path: str = "testclasses.txt",
		batch_size: int = 1,
		generalized_zeroshot: bool = False,
		transductive_setting: bool = False,
	**kwargs):
		"""Set `AnimalsWithAttributesDataModule` metadata.

		Arguments:
			`source_path`: path to source labels (usually known during training)
			`target_path`: path to target labels (usually known during  testing)

			`batch_size`: for training/validating/testing/predicting

			`generalized_zeroshot`: set to `True` to separate dataset based on source/target labels
			`transductive_setting`: set to `True` to allocate samples with a target label as unlabelled during training
		"""
		super(AnimalsWithAttributesDataModule, self).__init__(**kwargs)

	#	Label subset paths:
		self.labels_path = labels_path
		self.source_path = source_path
		self.target_path = target_path

	#	Batch size:
		self.batch_size = batch_size

	#	Setting:
		self.generalized_zeroshot = generalized_zeroshot
		self.transductive_setting = transductive_setting if self.generalized_zeroshot else False

	def prepare_data(self):
		"""Create datasets for accessing Animals with Attributes data based on label.

		Attributes:
			`source`: subset containing examples of source label
			`target`: subset containing examples of target label

			`totals`: a concatenated view of `source` and `target` for access as a whole dataset
		"""
		separator(3)
		separator(2, f"Animals with Attributes 2: reading")

	#	Images with source label:
		self.source = AnimalsWithAttributesDataset(
			labels_path=self.source_path,
		)

		separator(1, str(self.source))

	#	Images with target label:
		self.target = AnimalsWithAttributesDataset(
			labels_path=self.target_path,
		)

		separator(1, str(self.target))

	#	Images with anyall label:
		self.totals = AnimalsWithAttributesDataset(
			labels_path=self.labels_path,
		)
		separator(2, str(self.totals))

	def setup(self, stage: str):
		"""Create subsets for training/validating/testing/predicting.

		Arguments:
			`stage`: the staging/`stage`/name correpsondence is:
			-	training  :`'fit'`     :`train`
			-	validation:`'validate'`:`devel`
			-	testing   :`'test'`    :`valid`
			-	prediction:`'predict'` :`valid`
		"""
		separator(2, f"Animals with Attributes 2: spliting")

	#	If training with unknown labels:
		if self.generalized_zeroshot:
			(
				source_train_images,
				source_devel_images,
				source_valid_images,
			) = self.source.random_split(
				len(self.source),
				len(self.target),
			)
			(
				target_train_images,
				target_devel_images,
				target_valid_images,
			) = self.target.random_split(
				len(self.source),
				len(self.target),
			)

		#	If unlabelled samples allowed during training:
			if self.transductive_setting:
				separator(2, f"Animals with Attributes 2: data to {stage} in generalized zeroshot transductive setting")

				if stage == "fit":
					self.train_images = torch.utils.data.ConcatDataset(
						[
							source_train_images,
							target_train_images,
						]
					)

					separator(3, str(self.train_images))

				if stage == "validate":
					self.devel_images = torch.utils.data.ConcatDataset(
						[
							source_devel_images,
							target_devel_images,
						]
					)

					separator(3, str(self.devel_images))

				if stage == "test" or stage == "predict":
					self.valid_images = torch.utils.data.ConcatDataset(
						[
							source_valid_images,
							target_valid_images,
						]
					)

					separator(3, str(self.valid_images))

		#	if only source labels allowed during training:
			else:
				separator(2, f"Animals with Attributes 2: data to {stage} in generalized zeroshot")

				if stage == "fit":
					self.train_images = torch.utils.data.ConcatDataset(
						[
							source_train_images,
						]
					)

					separator(3, str(self.train_images))

				if stage == "validate":
					self.devel_images = torch.utils.data.ConcatDataset(
						[
							source_devel_images,
						]
					)

					separator(3, str(self.devel_images))

				if stage == "test" or stage == "predict":
					self.valid_images = torch.utils.data.ConcatDataset(
						[
							target_train_images,
							target_devel_images,
							source_valid_images,
							target_valid_images,
						]
					)

					separator(3, str(self.valid_images))

	#	If training normally:
		else:
			(
				train_images,
				devel_images,
				valid_images,
			) = self.totals.random_split(
				len(self.source),
				len(self.target),
			)

			separator(2, f"Animals with Attributes 2: data to {stage}")

			if stage == "fit":
				self.train_images = train_images

				separator(3, str(self.train_images))

			if stage == "validate":
				self.devel_images = devel_images

				separator(3, str(self.devel_images))

			if stage == "test" or stage == "predict":
				self.valid_images = valid_images

				separator(3, str(self.valid_images))

	def train_dataloader(self) -> AnimalsWithAttributesDataLoader:
		return AnimalsWithAttributesDataLoader(self.train_images, generator, batch_size=self.batch_size)

	def val_dataloader(self) -> AnimalsWithAttributesDataLoader:
		return AnimalsWithAttributesDataLoader(self.devel_images, generator, batch_size=self.batch_size)

	def test_dataloader(self) -> AnimalsWithAttributesDataLoader:
		return AnimalsWithAttributesDataLoader(self.valid_images, generator, batch_size=self.batch_size)

	def predict_dataloader(self) -> AnimalsWithAttributesDataLoader:
		return AnimalsWithAttributesDataLoader(self.valid_images, generator, batch_size=self.batch_size)
