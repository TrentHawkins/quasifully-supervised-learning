"""Custom `torch.utils.data.Dataset` for the Animals with Attributes 2 dataset.

[homepage](https://cvml.ista.ac.at/AwA2/)
[download](https://cvml.ista.ac.at/AwA2/AwA2-data.zip)

Includes:
	`torchvision.datasets.AnimalsWithAttributesDataset`
"""


from __future__ import annotations

from os import path
from sys import float_info
from typing import Callable, Optional, Union

import matplotlib.pyplot
import matplotlib.ticker
import numpy
import pandas
import scipy.special
import seaborn
import torch.utils.data
import torchvision

from ..globals import generator as default_generator
from ..similarities import dotDataFrame


class AnimalsWithAttributesDataset(torchvision.datasets.ImageFolder):
	"""A custom dataset on images from the Animals with Attributes dataset.

	Attributes:
		`_labels`: `pandas.Series` of label index indexed by label name
		`_alphas`: `pandas.Series` of integer listing of class predicates by name
		`_images`: `pandas.Series` of image label index indexed by image path

	Methods:
		`labels`: `pandas.Series` of label index indexed by label name (optionally filtered)
		`alphas`: `pandas.Dataframe` of predicate values indexed by label name (optionally filtered) and predicate name
		`images`: `pandas.Series` of image label index (optionally filtered) indexed by image path

		`random_split`: the images into proportionate training, development and validation subsets

		`plot_labels`: plot label statistics
		`plot_alphas`: plot predicates heatmap against labels

		`plot_label_correlation`: plot label correlation on predicates heatmap using dot product, optinally on logits
	"""

	def __init__(self,
		images_path: str = "datasets/animals_with_attributes",
		splits_path: str = "standard_split",
		labels_path: str = "allclasses.txt",
		target_size: int = 224,
		transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
		target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
		generator: Optional[torch.Generator] = None,
	):
		"""Initialize the directory containing the images.

		Arguments:
			`images_path`: relative path to dataset (default: assumes root directory)
			`splits_path`: relative path to definition of data splitting (default: standard)
			`labels_path`: relative path to file with labels (default: all labels)
			`target_size`: scale images to size `(target_size, target_size)` via center cropping
			`transform`: a function/transform that takes in an PIL image and returns a transformed version (optional)
			`target_transform`: A function/transform that takes in the target and transforms it (optional)
			`generator`: random number generator to pass around for reproducibility (default: one with seed 0)
		"""
		self._images_path = path.join(
			images_path,
		)
		self._splits_path = path.join(
			images_path,
			splits_path,
		)
		self._labels_path = path.join(
			images_path,
			splits_path,
			labels_path,
		)
		self._target_size = target_size

	#	Data transforms:
		self.transform = transform or torchvision.transforms.Compose(
			[
				torchvision.transforms.Resize(self._target_size),
				torchvision.transforms.CenterCrop(self._target_size),
			]
		)
		self.target_transform = target_transform or torch.nn.functional.one_hot

	#	Labels (animal labels):
		self._labels: pandas.Series[int] = pandas.read_csv(
			path.join(self._images_path, "classes.txt"),
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

	#	Alphas (semantic data):
		self._alphas: pandas.Series[int] = pandas.read_csv(
			path.join(self._images_path, "predicates.txt"),
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

	#	Instantiate `torchvision.datasets.ImageFolder`:
		super(AnimalsWithAttributesDataset, self).__init__(
			path.join(self._images_path, "JPEGImages"),  # `self.root` overwriten later but the same
			transform=transform,
			target_transform=target_transform,
			loader=torchvision.io.read_image,
		#	is_valid_file=None,
		)

	#	Images (paths and labels):
		keys, values = zip(*self.imgs)
		self._images: pandas.Series[int] = pandas.Series(values, keys)

	#	Set global seed for dataset:
		self._generator = generator or default_generator

	def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
		"""Find the class labels in the Animals with Attributes dataset.

		This method is overridden:
		-	to consider subsets of classes based on a given labels file
		-	to keep the ordering provided by the dataset labels files

		Arguments:
			`directory`: labels file

		Raises:
			`FileNotFoundError`: If `directory` has no class labels.

		Returns:
			list of all classes and dictionary mapping each class to an index
		"""
		with open(path.join(self._labels_path)) as labels_file:
			classes = [label.strip() for label in labels_file]

		if not classes:
			raise FileNotFoundError(f"Couldn't find any class labels in `{directory}`.")

		class_to_idx = self._labels.filter(classes, axis="index").to_dict()

		return classes, class_to_idx

	def _read(self, selection: Union[pandas.Series, str]) -> list:
		"""Read items either from a file or a series into a list.

		Arguments:
			`selection`: either a `pandas.Series` or a text file

		Returns:
			`list` with items in selection
		"""
		if isinstance(selection, str):
			with open(path.join(self._splits_path, selection)) as labels_file:
				return [label.strip() for label in labels_file]

		if isinstance(selection, pandas.Series):
			return selection.tolist()

	def labels(self,
		selection: Union[pandas.Series, str] = "allclasses.txt",
	) -> pandas.Series:
		"""Get a label set from dataset.

		Arguments:
			`selection`: text file containing the labels to be listed (default: list all labels in dataset)

		Returns:
			label `pandas.Series` indexed from 0 (contrary to the vanilla index starting from 1)
		"""
		return self._labels.filter(self._read(selection), axis="index")

	def alphas(self,
		selection: Union[pandas.Series, str] = "allclasses.txt",
		binary: bool = False,
		logits: bool = False,
	) -> pandas.DataFrame:
		"""Get label predicates.

		Argumebts:
			`selection`: text file containing the labels to be listed (default: list all labels in dataset)
			`binary`: continuous if `False` (default: continuous)
			`logits`: modify probabilistic range to logits (default: not)

		Returns:
			predicate `pandas.DataFrame` indexed with labels and named with predicates
		"""
		alpha_matrix = pandas.read_csv(
			path.join(self._images_path, "predicate-matrix-binary.txt") if binary else
			path.join(self._images_path, "predicate-matrix-continuous.txt"),
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
				.replace(1., 1. - float_info.epsilon).applymap(scipy.special.logit)

		return alpha_matrix.filter(self._read(selection), axis="index")

	def images(self,
		selection: Union[pandas.Series, str] = "allclasses.txt",
	) -> pandas.Series:
		"""Get images and a label set from dataset.

		Arguments:
			`selection`: text file containing the labels to be listed (default: list all labels in dataset)

		Returns:
			label `pandas.Series` indexed with image paths
		"""
		return self._images[self._images.isin(self.labels(selection))]

	def random_split(self,
		len_source: Optional[int] = None,
		len_target: Optional[int] = None,
	) -> tuple[
		torch.utils.data.Subset,
		torch.utils.data.Subset,
		torch.utils.data.Subset,
	]:
		"""Randomly split the images into proportionate training, development and validation subsets.

		Arguments:
			`len_source`: size of training
			`len_target`: size of testing

		Returns:
			`torch.utils.data.Subset` including:
				`train_images`: images used for training
				`devel_images`: images used for hyper-tuning and or regularization
				`valid_images`: images used for testing
		"""
		len_source = len_source or len(self.images("trainvalclasses.txt"))
		len_target = len_target or len(self.images("testclasses.txt"))

	#	Split validation subset off total data. Use a larger chunk than what corresponds to the source/target labels:
		_train_images, _valid_images = torch.utils.data.random_split(self,
			[1. - len_target / len_source, len_target / len_source],
			generator=self._generator,
		)

	#	Split development subset off training data. Use a smaller chunk than what corresponds to the source/target labels:
		_train_images, _devel_images = torch.utils.data.random_split(_train_images,
			[1. - len_target / (len_source + len_target), len_target / (len_source + len_target)],
			generator=self._generator,
		)

		return (
			_train_images,
			_devel_images,
			_valid_images,
		)

	def plot_labels(self):
		"""Plot label statistics.

		Label distribution in dataset in both linear and logarigthmic scales.
		"""
		fig, ax = matplotlib.pyplot.subplots(
			figsize=(
				9.,
				6.,
			)
		)
		fig.set_tight_layout(True)

		labels = self.labels()
		target = self.labels(selection="testclasses.txt")

		labels_reversed = pandas.Series(labels.index, index=labels)
		images_labelled = self.images().replace(labels_reversed).value_counts(
			normalize=False,
			sort=False,
			ascending=False,
			dropna=True,
		).reindex(labels.index)

		ax.set_facecolor("#000000")
		ax.xaxis.set_tick_params(length=0)

		images_labelled.plot(
			kind="bar",
			ax=ax,
			title="image class distribution",
			xlabel="class label",
			ylim=(
				1,
				2000,
			),
			logy=True,
			width=.9,
			color=["#AA5500" if label in target.index else "#0055AA" for label in labels.index],
		).tick_params("y",
			labelcolor="#555555",
		)

		ax = ax.twinx()

		images_labelled.plot(
			kind="bar",
			ax=ax,
			title="image class distribution",
			xlabel="class label",
			ylim=(
				0,
				2000,
			),
			logy=False,
			width=.7,
			color=["#FFAA55" if label in target.index else "#55AAFF" for label in labels.index],
		).tick_params("y",
			labelcolor="#AAAAAA",
		)

		matplotlib.pyplot.savefig(path.join(self._images_path, "classes.pdf"))

	def plot_alphas(self,
		binary: bool = False,
	):
		"""Plot predicates heatmap against labels.

		Arguments:
			binary: continuous if `False` (default: continuous)
		"""
		alpha_range = "-binary" if binary else "-continuous"

		fig, ax = matplotlib.pyplot.subplots(
			figsize=(
				15.,
				10.
			)
		)
		fig.set_tight_layout(True)

		ax.xaxis.set_tick_params(length=0)
		ax.yaxis.set_tick_params(length=0)

		alphas = self.alphas(binary=binary)

		seaborn.heatmap(alphas,
			vmin=0.,
			vmax=1.,
			cmap="gnuplot",
			linewidths=1,
			linecolor="black",
			cbar=False,
			square=True,
			xticklabels=True,  # type: ignore
			yticklabels=True,  # type: ignore
			ax=ax,
		)

		ax.set_title(f"class {alpha_range} predicate matrix")
		ax.set_xlabel("predicates")
		ax.set_ylabel("class label")

		matplotlib.pyplot.savefig(path.join(self._images_path, f"predicate-matrix{alpha_range}.pdf"))

	def plot_label_correlation(self, alter_dot: Callable = numpy.dot,
		binary: Optional[bool] = None,
		logits: bool = False,
		softmx: bool = False,
	):
		"""Plot label correlation on predicates heatmap using dot product, optinally on logits.

		Arguments:
			`binary`: continuous if `False` (default: continuous)
			`logits`: modify probabilistic range to logits (default: not
			`softmx`: apply softmax to predictions
				default not
		"""
		alpha_range = "-binary" if binary else ""
		alpha_field = "-logits" if logits else ""

		alpha_normalization = "-softmax" if softmx else ""

		fig, ax = matplotlib.pyplot.subplots(
			figsize=(
				10.,
				10.
			)
		)
		fig.set_tight_layout(True)

		ax.xaxis.set_tick_params(length=0)
		ax.yaxis.set_tick_params(length=0)

	#	Mix continuous and binary semantic representations, emulating sigmoid predictions against binary truth:
		if binary is not None:
			label_correlation = dotDataFrame(
				self.alphas(
					binary=binary,
					logits=logits,
				),
				self.alphas(
					binary=binary,
					logits=logits,
				).transpose(), alter_dot=alter_dot
			)

	#	Pure binary or continuous semantic representation correlation:
		else:
			label_correlation = dotDataFrame(
				self.alphas(
					binary=True,
					logits=logits,
				),
				self.alphas(
					binary=False,
					logits=logits,
				).transpose(), alter_dot=alter_dot
			)

	#	Optinally dress correlation with a softmax:
		if softmx:
			label_correlation = label_correlation.transform(scipy.special.softmax)

		seaborn.heatmap(label_correlation,
			vmin=0. if softmx or alter_dot != numpy.dot else None,
			vmax=1. if softmx or alter_dot != numpy.dot else None,
			cmap="gnuplot",
			robust=not softmx and alter_dot == numpy.dot,
			linewidths=0,
			linecolor="black",
			cbar=False,
			square=True,
			xticklabels=True,  # type: ignore
			yticklabels=True,  # type: ignore
			ax=ax,
		)

		ax.set_title(f"class label-predicate correlation matrix")
		ax.set_xlabel("predicted")
		ax.set_ylabel("true")

		altered_dot = f".{alter_dot.__name__}" if alter_dot != numpy.dot else ""

		matplotlib.pyplot.savefig(
			path.join(self._images_path,
				f"class-correlation{alpha_range}{alpha_field}{alpha_normalization}{altered_dot}.pdf"
			)
		)
