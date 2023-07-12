"""Custom dataset and dataloader for the Animals with Attributes 2 dataset.

[homepage](https://cvml.ista.ac.at/AwA2/)
[download](https://cvml.ista.ac.at/AwA2/AwA2-data.zip)

Includes:
	`torch.utils.data.Dataset`
	`torch.utils.data.DataLoader`
"""

from __future__ import annotations

from shutil import get_terminal_size
from sys import float_info
from typing import Callable, Iterable, Optional, Union

import os
import pandas
import scipy.special
import seaborn
import sklearn.model_selection
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


class Dataset(torchvision.datasets.ImageFolder):
	"""A custom dataset loading images from the Animals with Attributes dataset.

	Attributes:
		`_labels`: `pandas.Series` of label index indexed by label name
		`_alphas`: `pandas.Series` of integer listing of class predicates by name
		`_images`: `pandas.Series` of image label index indexed by image path
	"""

	def __init__(self,
		base: str = "datasets/animals_with_attributes",
		transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
		target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
	):
		"""Initialize the directory containing the images.

		Arguments:
			`base`: absolute path to dataset (default: assumes root directory)
			`transform`: a function/transform that takes in an PIL image and returns a transformed version
			`target_transform`: A function/transform that takes in the target and transforms it
		"""
		self.base: str = os.path.normpath(base)
		self.root = os.path.join(self.base, "JPEGImages")  # `self.root` overwriten later but the same

	#	instantiate `torchvision.datasets.ImageFolder`:
		super(Dataset, self).__init__(
			self.root,
			transform=transform,
			target_transform=target_transform,
		#	loader=torchvision.io.read_image,
		#	is_valid_file=None,
		)

	#	labels (animal labels):
		self._labels: pandas.Series[int] = pandas.read_csv(
			os.path.join(self.base, "classes.txt"),
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

	#	alphas (semantic data):
		self._alphas: pandas.Series[int] = pandas.read_csv(
			os.path.join(self.base, "predicates.txt"),
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

	#	images (paths and labels):
		self._images: pandas.Series[int] = pandas.Series(dict(*zip(self.imgs)))

	#	set global seed for dataset:
		self.seed = 0

	def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
		"""Find the class labels in the Animals with Attributes dataset:

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
		with open(directory) as labels_file:
			classes = [label.strip() for label in labels_file]

		if not classes:
			raise FileNotFoundError(f"Couldn't find any class labels in `{directory}`.")

		class_to_idx = self._labels.filter(classes, axis="index").to_dict()

		return classes, class_to_idx

#		TODO:
#		-	concat datasets
#		-	random spliting

















































	def _read(self, selection: Union[pandas.Series, str]) -> list:
		"""Read items either from a file or a series into a list.

		Arguments:
			`selection`: either a `pandas.Series` or a text file

		Returns:
			`list` with items in selection
		"""
		if isinstance(selection, str):
			with open(os.path.join(self.base, self._labels_path, selection)) as labels_file:
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
			os.path.join(self.base, "predicate-matrix-binary.txt") if binary else
			os.path.join(self.base, "predicate-matrix-continuous.txt"),
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

	def split(self,
		labels: Union[pandas.Series, str] = "allclasses.txt",
		*,
		image_size: int = 224,
		batch_size: int = 1,
	) -> tuple[
		torch.utils.data.Dataset,
		torch.utils.data.Dataset,
		torch.utils.data.Dataset,
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

	#	Return dataset from paths and labels with preferred settings.
		def paths_and_labels_to_dataset(_images: pandas.Series,
			training: bool = False,
		) -> torch.utils.data.Dataset:
			images: torch.utils.data.Dataset = keras.utils.image_dataset.paths_and_labels_to_dataset(
				image_paths=_images.index,
				image_size=(
					image_size,
					image_size,
				),
				num_channels=3,
				labels=_images,
				label_mode="categorical",
				num_classes=len(self._labels),
				interpolation="bilinear",
				crop_to_aspect_ratio=True,
			)

		#	Cache subset in RAM for performance.
		#	images = images.cache()

		#	Shuffle elements by fixed seed, but set subset for reshuffling after each epoch.
			images = images.shuffle(ceil(sqrt(len(images))),
				seed=self.seed,
				reshuffle_each_iteration=training,
			)

		#	If batch size is set, batch subset.
			images = images.batch(batch_size,
				num_parallel_calls=tensorflow.data.AUTOTUNE,
				deterministic=True,
			)

		#	Prefetch the first examples of subset.
		#	images = images.prefetch(tensorflow.data.AUTOTUNE)

			return images

	#	Build datasets.
		train_images: torch.utils.data.Dataset = paths_and_labels_to_dataset(_train_images,
			training=True,
		)
		devel_images: torch.utils.data.Dataset = paths_and_labels_to_dataset(_devel_images,
			training=False,
		)
		valid_images: torch.utils.data.Dataset = paths_and_labels_to_dataset(_valid_images,
			training=False,
		)

		return (
			train_images,
			devel_images,
			valid_images,
		)

	def plot_labels(self):
		"""Plot label statistics."""
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

		matplotlib.pyplot.savefig(os.path.join(self.base, "classes.pdf"))

	def plot_alphas(self,
		binary: bool = False,
	):
		"""Plot predicates heatmap against labels.

		Arguments:
			binary: continuous if `False`
				default: continuous
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

		matplotlib.pyplot.savefig(os.path.join(self.base, f"predicate-matrix{alpha_range}.pdf"))

	def plot_label_correlation(self, alter_dot: Callable = numpy.dot,
		binary: Optional[bool] = None,
		logits: bool = False,
		softmx: bool = False,
	):
		"""Plot label correlation on predicates heatmap using dot product, optinally on logits.

		Arguments:
			binary: continuous if `False`
				default: continuous
			logits: modify probabilistic range to logits
				default: not
			softmx: apply softmax to predictions
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

	#	Mix continuous and binary semantic representations, emulating sigmoid predictions against binary truth.
		if binary is not None:
			alphas = self.alphas(
				binary=binary,
				logits=logits,
			)
			label_correlation = dotDataFrame(
				alphas,
				alphas.transpose(), alter_dot=alter_dot
			)

	#	Pure binary or continuous semantic representation correlation.
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

	#	Optinally dress correlation with a softmax.
		if softmx:
			label_correlation = label_correlation.apply(tensorflow.nn.softmax,
				axis="index",
			)

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
			os.path.join(self.base,
				f"class-correlation{alpha_range}{alpha_field}{alpha_normalization}{altered_dot}.pdf"
			)
		)
