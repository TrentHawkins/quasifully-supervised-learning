"""Create image data loading pipeline for the Animals with Attributes 2 dataset using TensorFlow."""


import glob
import os

import pandas as pd


class Dataset:
	"""Animals with Attributes 2:

	All label and image loc data are generated dynamically on demand to allow filtering fexibility while maingaining readability.
	Assumes that internal file structure of the AWA2 (equipped with the standard splits) dataset is left unaltered.

	Attributes:
		images_path: absolute
		labels_path: relative to `images_path` used by default if left unspecified

	Methods:
		labels:	a selection of labels given by `labels_path`
		predicates: the predicate matrix for the selection of labels given by `labels_path`
		images:	the images filtered by the selection of labels given by `labels_path`
		standard_split: provides a train/test split based on the standard label split of the dataset
	"""

	def __init__(self,
		images_path: str = "datasets/animals_with_attributes",
	):
		"""Initialize paths for dataset.

		Arguments:
			images_path: absolute
				default: assumes dataset is in the current working directory
		"""

		self.images_path = images_path

	def read(self, selection: list | pd.Series | str) -> list:
		"""Read items either from a file or a series into a list.

		Arguments:
			selection: either a `pd.Series` or a text file

		Returns:
			list with items in selection
		"""

	#	Prepare filter iterable, either from another series, or
		if isinstance(selection, pd.Series):
			selection = selection.tolist()

	#	from another labels file.
		if isinstance(selection, str):
			with open(os.path.join(self.images_path, selection)) as labels_file:
				selection = [label.strip() for label in labels_file]

		return selection

	def labels(self,
		selection: pd.Series | str = "standard_split/allclasses.txt",
	) -> pd.Series:
		"""Get a label set from dataset.

		Arguments:
			selection: text file containing the labels to be listed
				default: list all labels in dataset

		Returns:
			label set indexed from 0 (contrary to the vanilla index starting from 1)
		"""

	#	Read labels from file with all labels numbered. Shift numbering to start from 0.
		labels: pd.Series = pd.read_csv(
			os.path.join(self.images_path, "classes.txt"),
			sep=r"\s+",
			names=[
				"index"
			],
			index_col=1,
			dtype={
				0: int,
				1: str,
			},
		).squeeze().apply(lambda index: index - 1)

		return labels.filter(self.read(selection),
			axis="index",
		)

	def predicates(self,
		selection: pd.Series | str = "standard_split/allclasses.txt",
		binary: bool = False,
	) -> pd.DataFrame:
		"""Get label predicates (features).

		Argumebts:
			selection: text file containing the labels to be listed
				default: list all labels in dataset
			binary: continuous if `False`
				default: continuous

		Returns:
			predicate `pandas.DataFrame` indexed with labels and named with predicates
		"""

	#	Get the predicates from file that lists them incrementally. Ignore the numbering.
		with open(os.path.join(self.images_path, "predicates.txt")) as predicates_file:
			predicates = [predicate.split()[1] for predicate in predicates_file]

	#	Form predicate matrix, with predicates as columns, labels are rows.
		predicate_matrix = pd.read_csv(
			os.path.join(self.images_path, "predicate-matrix-binary.txt") if binary else
			os.path.join(self.images_path, "predicate-matrix-continuous.txt"),
			sep=r"\s+",
			names=predicates,
			dtype=float,
		).set_index(self.labels().index)

		return predicate_matrix.filter(self.read(selection),
			axis="index",
		)

	def images(self,
		selection: pd.Series | str = "standard_split/allclasses.txt",
	) -> pd.Series:
		"""Get images and a label set from dataset.

		Arguments:
			selection: text file containing the labels to be listed
				default: list all labels in dataset

		Returns:
			label `pandas.Series` indexed with image paths
		"""

	#	Get all image paths first.
		images_path = glob.glob(os.path.join(self.images_path, "JPEGImages/*/*.jpg"))

	#	Load images.
		images = pd.Series([os.path.basename(os.path.dirname(image)) for image in images_path],
			index=images_path,
			name="labels",
			dtype=str,
		)

		return images[images.isin(self.read(selection))].replace(self.labels(selection))
