"""Custom dataset and dataloader for the Animals with Attributes 2 dataset.

Includes:
	`torch.utils.data.Dataset`
	`torch.utils.data.DataLoader`
"""

from __future__ import annotations

from shutil import get_terminal_size
from sys import float_info
from typing import Callable, Optional, Union

import os
import pandas
import torch.utils.data
import torchvision


def print_separator(char: Union[str, int] = 0,
	title: Optional[str] = None,
):
	"""Repeat character across full terminal width.

	Arguments:
		`char`: single character to repeat
			0: " "
			1: "─"
			2: "═"
			3: "━"

	Keyword Arguments:
		`title`: optional text to display before separator
	"""
	if isinstance(char, int):
		char = {
			0: " ",
			1: "─",
			2: "═",
			3: "━",
		}[char]

	if title is not None:
		print(title)

	print(char * get_terminal_size((96, 96)).columns)


class Dataset(torch.utils.data.Dataset):
	"""
	"""
	def __init__(self,
		images_path: str = "datasets/animals_with_attributes",
		labels_path: str = "standard_split",
		*,
		seed: int = 0,
	):
		"""Initialize paths for dataset, then generate dataset from said paths.

		Arguments:
			images_path: absolute
				default: assumes root directory
			labels_path: absolute
				default: assumes root directory

		Keyword Arguments:
			seed: global seed setting for dataset
				default: 0
		"""
		self._images_path: str = images_path
		self._labels_path: str = labels_path

		self._labels: pandas.Series[int] = pandas.read_csv(
			os.path.join(self._images_path, "classes.txt"),
			sep=r"\s+",
			names=[
				"index"
			],
			index_col=1,
			dtype={
				0: int,
				1: str,
			},
		).squeeze() - 1
		self._alphas: pandas.Series[int] = pandas.read_csv(
			os.path.join(self._images_path, "predicates.txt"),
			sep=r"\s+",
			names=[
				"index"
			],
			index_col=1,
			dtype={
				0: int,
				1: str,
			},
		).squeeze()

		print_separator(3)
		print_separator(3, "Animals with Attributes 2: directory look-up")

	#	Transform image path and label data to pandas series.
		self._images: pandas.Series[int] = pandas.Series(dict(zip(image_paths, labels)))

	#	Split ratios to be used:
		self._target_source: float = len(self.images("testclasses.txt")) / len(self.images("trainvalclasses.txt"))
		self._target_totals: float = len(self.images("testclasses.txt")) / len(self.images())

	#	Set global seed for dataset:
		self.seed = seed

	def __len__(self):
		return len(self._images)

	def _read(self, selection: Union[pandas.Series, str]) -> list:
		"""Read items either from a file or a series into a list.

		Arguments:
			selection: either a `pandas.Series` or a text file

		Returns:
			list with items in selection
		"""
		if isinstance(selection, str):
			with open(os.path.join(self._images_path, self._labels_path, selection)) as labels_file:
				return [label.strip() for label in labels_file]

		if isinstance(selection, pandas.Series):
			return selection.tolist()

	def labels(self,
		selection: Union[pandas.Series, str] = "allclasses.txt",
	) -> pandas.Series:
		"""Get a label set from dataset.

		Arguments:
			selection: text file containing the labels to be listed
				default: list all labels in dataset

		Returns:
			label set indexed from 0 (contrary to the vanilla index starting from 1)
		"""
		return self._labels.filter(self._read(selection), axis="index")

	def alphas(self,
		selection: Union[pandas.Series, str] = "allclasses.txt",
		binary: bool = False,
		logits: bool = False,
	) -> pandas.DataFrame:
		"""Get label predicates.

		Argumebts:
			selection: text file containing the labels to be listed
				default: list all labels in dataset
			binary: continuous if `False`
				default: continuous
			logits: modify probabilistic range to logits
				default: not

		Returns:
			predicate `pandas.DataFrame` indexed with labels and named with predicates
		"""
		alpha_matrix = pandas.read_csv(
			os.path.join(self._images_path, "predicate-matrix-binary.txt") if binary else
			os.path.join(self._images_path, "predicate-matrix-continuous.txt"),
			sep=r"\s+",
			names=self._alphas.index.tolist(),
			dtype=float,
		).set_index(self.labels().index)

	#	Normalize continuous predicates.
		if not binary:
			alpha_matrix /= 100

	#	logit(0) == -inf
	#	logit(1) == +inf
		if logits:
			alpha_matrix = alpha_matrix\
				.replace(0., 0. + float_info.epsilon)\
				.replace(1., 1. - float_info.epsilon).applymap(scipy.special.logit)  # type: ignore

		return alpha_matrix.filter(self._read(selection), axis="index")

	def images(self,
		selection: Union[pandas.Series, str] = "allclasses.txt",
	) -> pandas.Series:
		"""Get images and a label set from dataset.

		Arguments:
			selection: text file containing the labels to be listed
				default: list all labels in dataset

		Returns:
			label `pandas.Series` indexed with image paths
		"""
		return self._images[self._images.isin(self.labels(selection))]

	def split(self,
		labels: Union[pandas.Series, str] = "allclasses.txt",
		*,
		image_size: int = 224,
		batch_size: int = 1,
	) -> tuple[
		tensorflow.data.Dataset,
		tensorflow.data.Dataset,
		tensorflow.data.Dataset,
	]:
		"""Get shuffled split by label and subset.

		Arguments:
			labels: get examples only from these
			subset: after splitting dataset get this

		Keyword_arguments:
			image_size: to rescale fetched examples to (square ratio)
			batch_size: to batch   fetched examples

		Returns:
			An optimized (cached and prefeched) TensorFlow dataset.
		"""
		_total_images = self.images(selection=labels)

	#	Split validation subset off total data. Use a larger chunk than what corresponds to the source/target labels.
		_train_images, _valid_images = sklearn.model_selection.train_test_split(_total_images,
			test_size=self._target_source,  # slightly more validation examples
			random_state=self.seed,
			shuffle=True,
			stratify=_total_images,
		)

	#	Split development subset off training data. Use a smaller chunk than what corresponds to the source/target labels.
		_train_images, _devel_images = sklearn.model_selection.train_test_split(_train_images,
			test_size=self._target_totals,  # slightly less validation examples
			random_state=self.seed,
			shuffle=True,
			stratify=_train_images,
		)
