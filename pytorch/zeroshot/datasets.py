"""Custom `torch.utils.data.Dataset` for the Animals with Attributes 2 dataset.

[homepage](https://cvml.ista.ac.at/AwA2/)
[download](https://cvml.ista.ac.at/AwA2/AwA2-data.zip)

Includes:
	`torchvision.datasets.ZeroshotAnimalsWithAttributesDataset`
	`torchvision.datasets.TransductiveZeroshotAnimalsWithAttributesDataset`
"""

from __future__ import annotations

from typing import Optional

import torch.utils.data

from ..chartools import separator
from ..torchvision.datasets import AnimalsWithAttributesDataset


class ZeroshotAnimalsWithAttributesDataset(torch.utils.data.ConcatDataset):
	"""Animals with Attributes 2.

	Semi-transductive generalized zeroshot setting.

	Attributes:
		`_source`: source-labelled images
		`_taregt`: target-labelled images

	Methods:
		`random_split`: the images into proportionate training, development and validation subsets
	"""

	def __init__(self,
		source_path: str = "trainvalclasses.txt",
		target_path: str = "testclasses.txt",
	**kwargs):
		"""Initialize two separate datasets by source and target labels.

		Arguments:
			`labels_path`: not used
			`source_path`: relative path to file with source labels (default: source labels)
			`target_path`: relative path to file with target labels (default: target labels)
		"""
		separator(3)
		separator(2, "Animals with Attributes 2: directory look-up")

		kwargs.pop("labels_path", None)

	#	Images with source label:
		self._source = AnimalsWithAttributesDataset(
			labels_path=source_path,
		**kwargs)

		separator(1, str(self._source))

	#	Images with target label:
		self._target = AnimalsWithAttributesDataset(
			labels_path=target_path,
		**kwargs)

		separator(3, str(self._target))

	#	Concatenated view of the datasets forming the original full dataset:
		super(ZeroshotAnimalsWithAttributesDataset, self).__init__(
			[
				self._source,
				self._target,
			]
		)

	def random_split(self,
		len_source: Optional[int] = None,
		len_target: Optional[int] = None,
	) -> tuple[
		torch.utils.data.ConcatDataset,
		torch.utils.data.ConcatDataset,
		torch.utils.data.ConcatDataset,
	]:
		"""Randomly split the images into proportionate training, development and validation subsets.

		Returns:
			`torch.utils.data.Subset` including:
				`train_images`: source label images only for training
				`devel_images`: source label images only for development
				`valid_images`: source label images only for testing plus
					target label images for training
					target label images for develpment
					target label images for testing
				as target label images are axiomatically unknown during training in the semi-tranductive setting
		"""
		(
			source_train_images,
			source_devel_images,
			source_valid_images,
		) = self._source.random_split(
			len_source or len(self._source),
			len_target or len(self._target),
		)
		(
			target_train_images,
			target_devel_images,
			target_valid_images,
		) = self._target.random_split(
			len_source or len(self._source),
			len_target or len(self._target),
		)

	#	Only source-labelled data can be used in training:
		train_images = torch.utils.data.ConcatDataset(
			[
				source_train_images,
			]
		)
		devel_images = torch.utils.data.ConcatDataset(
			[
				source_devel_images,
			]
		)
		valid_images = torch.utils.data.ConcatDataset(
			[
				source_valid_images,
				target_train_images,
				target_devel_images,
				target_valid_images,
			]
		)

		return (
			train_images,
			devel_images,
			valid_images,
		)


class TransductiveZeroshotAnimalsWithAttributesDataset(ZeroshotAnimalsWithAttributesDataset):
	"""Animals with Attributes 2.

	Transductive generalized zeroshot setting.

	Attributes:
		`_source`: source-labelled images
		`_taregt`: target-labelled images

	Methods:
		`random_split`: the images into proportionate training, development and validation subsets
	"""

	def random_split(self,
		len_source: Optional[int] = None,
		len_target: Optional[int] = None,
	) -> tuple[
		torch.utils.data.ConcatDataset,
		torch.utils.data.ConcatDataset,
		torch.utils.data.ConcatDataset,
	]:
		"""Randomly split the images into proportionate training, development and validation subsets.

		Returns:
			`torch.utils.data.Subset` including:
				`train_images`: source label images and target label images unlabelled only for training
				`devel_images`: source label images and target label images unlabelled only for development
				`valid_images`: source label images and target label images unlabelled only for testing plus
			as target label images are axiomatically known but without label during training in the tranductive setting
		"""
		(
			source_train_images,
			source_devel_images,
			source_valid_images,
		) = self._source.random_split(
			len(self._source),
			len(self._target),
		)
		(
			target_train_images,
			target_devel_images,
			target_valid_images,
		) = self._target.random_split(
			len(self._source),
			len(self._target),
		)

	#	Only source-labelled data and target-labelled data but unlabelled can be used in training:
		train_images = torch.utils.data.ConcatDataset(
			[
				source_train_images,
				target_train_images,
			]
		)
		devel_images = torch.utils.data.ConcatDataset(
			[
				source_devel_images,
				target_devel_images,
			]
		)
		valid_images = torch.utils.data.ConcatDataset(
			[
				source_valid_images,
				target_valid_images,
			]
		)

		return (
			train_images,
			devel_images,
			valid_images,
		)
