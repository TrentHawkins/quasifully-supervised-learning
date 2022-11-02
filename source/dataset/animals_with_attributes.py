"""Create image data loading pipeline for the Animals with Attributes 2 dataset using TensorFlow."""


from glob import glob
from os import path
from sys import float_info
from typing import Callable

import matplotlib.pyplot
import matplotlib.ticker
import numpy
import pandas
import scipy.special
import seaborn
import tensorflow

from ..similarity import dotDataFrame, cosine, jaccard


class Dataset:
	"""Animals with Attributes 2.

	All label and image loc data are generated dynamically on demand to allow filtering fexibility while maingaining readability.
	Assumes that internal file structure of the AWA2 (equipped with the standard splits) dataset is left unaltered.

	Attributes:
		images_path: absolute
		labels_path: relative to `images_path` used by default if left unspecified

	Methods:
		labels:	a selection of labels given by `labels_path`
		predicates: the predicate matrix for the selection of labels given by `labels_path`
		images:	the images filtered by the selection of labels given by `labels_path`
	"""

	def __init__(self,
		images_path: str = "datasets/animals_with_attributes",
		labels_path: str = "standard_split",
	):
		"""Initialize paths for dataset.

		Arguments:
			images_path: absolute
				default: assumes root directory
		"""
		self.images_path = images_path
		self.labels_path = labels_path

	def read(self, selection: list | pandas.Series | str) -> list:
		"""Read items either from a file or a series into a list.

		Arguments:
			selection: either a `pandas.Series` or a text file

		Returns:
			list with items in selection
		"""
		if isinstance(selection, pandas.Series):
			selection = selection.tolist()

		if isinstance(selection, str):
			with open(path.join(self.images_path, self.labels_path, selection)) as labels_file:
				selection = [label.strip() for label in labels_file]

		return selection

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
		labels: pandas.Series = pandas.read_csv(
			path.join(self.images_path, "classes.txt"),
			sep=r"\s+",
			names=[
				"index"
			],
			index_col=1,
			dtype={
				0: int,
				1: str,
			},
		).squeeze(axis="columns")

	#	Reset index to start from 0.
		labels -= 1

		return labels.filter(self.read(selection), axis="index")

	def predicates(self,
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
		predicates: pandas.Series = pandas.read_csv(
			path.join(self.images_path, "predicates.txt"),
			sep=r"\s+",
			names=[
				"index"
			],
			index_col=1,
			dtype={
				0: int,
				1: str,
			},
		).squeeze(axis="columns")

	#	Form predicate matrix, with predicates as columns, labels are rows.
		predicate_matrix = pandas.read_csv(
			path.join(self.images_path, "predicate-matrix-binary.txt") if binary else
			path.join(self.images_path, "predicate-matrix-continuous.txt"),
			sep=r"\s+",
			names=predicates.index.tolist(),
			dtype=float,
		).set_index(self.labels().index)

	#	Normalize continuous predicates.
		if not binary:
			predicate_matrix /= 100

	#	logit(0) == -inf
	#	logit(1) == +inf
		if logits:
			predicate_matrix = predicate_matrix\
				.replace(0., 0. + float_info.epsilon)\
				.replace(1., 1. - float_info.epsilon)\
				.applymap(scipy.special.logit)

		return predicate_matrix.filter(self.read(selection), axis="index")

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
		images_path = glob(path.join(self.images_path, "JPEGImages/*/*.jpg"))

	#	Get image labels.
		images = pandas.Series([path.basename(path.dirname(image)) for image in images_path],
			index=images_path,
			name="labels",
			dtype=str,
		)

		return images[images.isin(self.read(selection))].replace(self.labels(selection))

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

		matplotlib.pyplot.savefig(path.join(self.images_path, "classes.pdf"))

	def plot_predicates(self,
		binary: bool = False,
	):
		"""Plot predicates heatmap against labels.

		Arguments:
			binary: continuous if `False`
				default: continuous
		"""
		predicate_range = "binary" if binary else "continuous"

		fig, ax = matplotlib.pyplot.subplots(
			figsize=(
				15.,
				10.
			)
		)
		fig.set_tight_layout(True)

		ax.xaxis.set_tick_params(length=0)
		ax.yaxis.set_tick_params(length=0)

		predicates = self.predicates(binary=binary)

		seaborn.heatmap(predicates,
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

		ax.set_title(f"class {predicate_range} predicate matrix")
		ax.set_xlabel("predicates")
		ax.set_ylabel("class label")

		matplotlib.pyplot.savefig(path.join(self.images_path, f"predicate-matrix-{predicate_range}.pdf"))

	def plot_label_correlation(self,
		binary: bool = False,
		logits: bool = False,
		softmx: bool = False, alter_dot: Callable = numpy.dot
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
		fig, ax = matplotlib.pyplot.subplots(
			figsize=(
				10.,
				10.
			)
		)
		fig.set_tight_layout(True)

		ax.xaxis.set_tick_params(length=0)
		ax.yaxis.set_tick_params(length=0)

		predicates = self.predicates(
			binary=binary,
			logits=logits,
		)
		label_correlation = dotDataFrame(
			predicates,
			predicates.transpose(), alter_dot=alter_dot
		)

		if softmx:
			label_correlation = label_correlation.apply(tensorflow.nn.softmax,
				axis="index",
			)

		seaborn.heatmap(label_correlation,
			vmin=0. if softmx or alter_dot != numpy.dot else None,
			vmax=1. if softmx or alter_dot != numpy.dot else None,
			cmap="gnuplot",
			robust=not softmx and alter_dot == numpy.dot,
			linewidths=1,
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

		atered_dot = f".{alter_dot.__name__}" if alter_dot != numpy.dot else ""

		matplotlib.pyplot.savefig(path.join(self.images_path, f"class-correlation{atered_dot}.pdf"))


if __name__ == "__main__":
	dataset = Dataset()

	dataset.plot_labels()
	dataset.plot_predicates()
	dataset.plot_predicates(binary=True)

	kwargs = {
	#	"binary": True,
	#	"logits": True,
	#	"softmx": True,
	}
	dataset.plot_label_correlation(**kwargs)
	dataset.plot_label_correlation(**kwargs, alter_dot=jaccard)
	dataset.plot_label_correlation(**kwargs, alter_dot=cosine)
