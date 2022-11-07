"""Create image data loading pipeline for the Animals with Attributes 2 dataset using TensorFlow."""


from glob import glob
from os import path
from sys import float_info
from typing import Callable

import matplotlib.pyplot
import matplotlib.ticker
import keras.utils.dataset_utils
import keras.utils.image_dataset
import numpy
import pandas
import scipy.special
import seaborn
import tensorflow

from ..seed import SEED
from ..similarity import dotDataFrame


class Dataset:
	"""Animals with Attributes 2.

	All label and image loc data are generated dynamically on demand to allow filtering fexibility while maingaining readability.
	Assumes that internal file structure of the AWA2 (equipped with the standard splits) dataset is left unaltered.

	Attributes:
		images_path: absolute
		labels_path: relative to `images_path` used by default if left unspecified

	Methods:
		labels:	a selection of labels given by `labels_path`
		alphas: the predicate matrix for the selection of labels given by `labels_path`
		images:	the images filtered by the selection of labels given by `labels_path`
	"""

	def __init__(self,
		images_path: str = "datasets/animals_with_attributes",
		labels_path: str = "standard_split",
	):
		"""Initialize paths for dataset, then generate dataset from said paths.

		Arguments:
			images_path: absolute
				default: assumes root directory
			labels_path: absolute
				default: assumes root directory
		"""
		self._images_path: str = images_path
		self._labels_path: str = labels_path

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

	#	Use TensorFlows's image look-up.
		image_paths, labels, class_names = keras.utils.dataset_utils.index_directory(
			directory=path.join(self._images_path, "JPEGImages"),
			labels="inferred",
			formats=(
				".jpg",
			),
			class_names=self._labels.index.tolist(),
			shuffle=False,  # delegate shuffling to the corresponding splits
		)

	#	Transform image path and label data to pandas series.
		self._images: pandas.Series[int] = pandas.Series(dict(zip(image_paths, labels)))

	def read(self, selection: pandas.Series | str) -> list:
		"""Read items either from a file or a series into a list.

		Arguments:
			selection: either a `pandas.Series` or a text file

		Returns:
			list with items in selection
		"""
		if isinstance(selection, str):
			with open(path.join(self._images_path, self._labels_path, selection)) as labels_file:
				return [label.strip() for label in labels_file]

		if isinstance(selection, pandas.Series):
			return selection.tolist()

	def labels(self,
		selection: pandas.Series | str = "allclasses.txt",
	) -> pandas.Series:
		"""Get a label set from dataset.

		Arguments:
			selection: text file containing the labels to be listed
				default: list all labels in dataset

		Returns:
			label set indexed from 0 (contrary to the vanilla index starting from 1)
		"""
		return self._labels.filter(self.read(selection), axis="index")

	def alphas(self,
		selection: pandas.Series | str = "allclasses.txt",
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

		return alpha_matrix.filter(self.read(selection), axis="index")

	def images(self,
		selection: pandas.Series | str = "allclasses.txt",
	) -> pandas.Series:
		"""Get images and a label set from dataset.

		Arguments:
			selection: text file containing the labels to be listed
				default: list all labels in dataset

		Returns:
			label `pandas.Series` indexed with image paths
		"""
		return self._images[self._images.isin(self.labels(selection))]

	def plot_labels(self):
		"""Plot label statistics."""
		fig, ax = matplotlib.pyplot.subplots(
			figsize=(
				9.,
				6.
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

		matplotlib.pyplot.savefig(path.join(self._images_path, f"predicate-matrix{alpha_range}.pdf"))

	def plot_label_correlation(self, alter_dot: Callable = numpy.dot,
		binary: bool | None = None,
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
		alpha_range = "-binary" if binary else "-continuous"
		alpha_field = "-logits" if logits else "-probabilities"

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
			path.join(self._images_path,
				f"class-correlation{alpha_range}{alpha_field}{alpha_normalization}{altered_dot}.pdf"
			)
		)
