"""Create image data loading pipeline for the Animals with Attributes 2 dataset using TensorFlow."""


import glob
import os

import pandas as pd
import tensorflow as tf


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
		images_path: str = "./datasets/animals_with_attributes",
		labels_path: str = "./standard_split",
	):
		"""Initialize paths for dataset.

		Arguments:
			images_path: absolute
				default: assumes dataset is in the current working directory
			labels_path: relative to `images_path` used by default if left unspecified
				default: all the classes (best leave this as is)
		"""

		self.images_path = images_path
		self.labels_path = labels_path

	def labels(self,
		select: pd.Series | str = "allclasses.txt",
	) -> pd.Series:
		"""Get a label set from dataset.

		Arguments:
			select: text file containing the labels to be listed
				default: list all labels in dataset

		Returns:
			label set indexed from 0 (contrary to the vanilla index starting from 1)
		"""

		label_pack = pd.read_csv(
			os.path.join(self.images_path, "classes.txt"),
			sep=r"\s+",
		#	delimiter=None,
		#	header='infer',
			names=[
				"index"
			],
			index_col=1,
		#	usecols=None,
		#	prefix=NoDefault.no_default,
		#	mangle_dupe_cols=True,
			dtype={
				0: int,
				1: str,
			},
		#	engine=None,
		#	converters=None,
		#	true_values=None,
		#	false_values=None,
		#	skipinitialspace=False,
		#	skiprows=None,
		#	skipfooter=0,
		#	nrows=None,
		#	na_values=None,
		#	keep_default_na=True,
		#	na_filter=True,
		#	verbose=False,
		#	skip_blank_lines=True,
		#	parse_dates=False,
		#	infer_datetime_format=False,
		#	keep_date_col=False,
		#	date_parser=None,
		#	dayfirst=False,
		#	cache_dates=True,
		#	iterator=False,
		#	chunksize=None,
		#	compression='infer',
		#	thousands=None,
		#	decimal='.',
		#	lineterminator=None,
		#	quotechar='"',
		#	quoting=0,
		#	doublequote=True,
		#	escapechar=None,
		#	comment=None,
		#	encoding=None,
		#	encoding_errors='strict',
		#	dialect=None,
		#	error_bad_lines=None,
		#	warn_bad_lines=None,
		#	on_bad_lines=None,
		#	delim_whitespace=False,
			low_memory=False,
		#	memory_map=False,
		#	float_precision=None,
		#	storage_options=None,
		).squeeze().apply(lambda index: index - 1)

		if isinstance(select, pd.Series):
			labels = select.tolist()

		else:
			with open(os.path.join(self.images_path, self.labels_path, select)) as labels_file:
				labels = [label.strip() for label in labels_file]

		return label_pack.filter(
			items=labels,
		#	like=None,
		#	regex=None,
		#	axis=None,
		)

	def predicates(self,
		select: pd.Series | str = "allclasses.txt",
		binary: bool = False,
	) -> pd.DataFrame:
		"""Get label predicates (features).

		Argumebts:
			select: text file containing the labels to be listed
				default: list all labels in dataset
			binary: continuous if `False`
				default: continuous

		Returns:
			predicate `pandas.DataFrame` indexed with labels and named with predicates
		"""

	#	with open(os.path.join(self.images_path,"classes.txt")) as labels_file:
	#		labels=[label.split()[1] for label in labels_file]

		with open(os.path.join(self.images_path, "predicates.txt")) as predicates_file:
			predicates = [predicate.split()[1] for predicate in predicates_file]

		predicates = pd.read_csv(
			os.path.join(self.images_path, "predicate-matrix-binary.txt") if binary else
			os.path.join(self.images_path, "predicate-matrix-continuous.txt"),
			sep=r"\s+",
		#	delimiter=None,
		#	header='infer',
			names=predicates,
		#	index_col=None,
		#	usecols=None,
		#	prefix=NoDefault.no_default,
		#	mangle_dupe_cols=True,
			dtype=float,
		#	engine=None,
		#	converters=None,
		#	true_values=None,
		#	false_values=None,
		#	skipinitialspace=False,
		#	skiprows=None,
		#	skipfooter=0,
		#	nrows=None,
		#	na_values=None,
		#	keep_default_na=True,
		#	na_filter=True,
		#	verbose=False,
		#	skip_blank_lines=True,
		#	parse_dates=False,
		#	infer_datetime_format=False,
		#	keep_date_col=False,
		#	date_parser=None,
		#	dayfirst=False,
		#	cache_dates=True,
		#	iterator=False,
		#	chunksize=None,
		#	compression='infer',
		#	thousands=None,
		#	decimal='.',
		#	lineterminator=None,
		#	quotechar='"',
		#	quoting=0,
		#	doublequote=True,
		#	escapechar=None,
		#	comment=None,
		#	encoding=None,
		#	encoding_errors='strict',
		#	dialect=None,
		#	error_bad_lines=None,
		#	warn_bad_lines=None,
		#	on_bad_lines=None,
		#	delim_whitespace=False,
			low_memory=False,
		#	memory_map=False,
		#	float_precision=None,
		#	storage_options=None,
		).set_index(self.labels(select).index,
		#	drop=True,
		#	append=False,
		#	inplace=False,
			verify_integrity=True,
		)

	#	Normalize percentages to unity.
		if not binary:
			predicates /= 100

		return predicates

	def images(self,
		select: pd.Series | str = "allclasses.txt",
		images: pd.Series | None = None,
	) -> pd.Series:
		"""Get images and a label set from dataset.

		Arguments:
			select: text file containing the labels to be listed
				default: list all labels in dataset
			images: use optional image subset instead of all images
				default: use every image

		Returns:
			label `pandas.Series` indexed with image paths
		"""

	#	subset mask
		if isinstance(select, str):
			with open(os.path.join(self.images_path, self.labels_path, select)) as labels_file:
				labels = [label.strip() for label in labels_file]

		else:
			labels = select.tolist()

	#	get all image paths first
		if images is None:
			image_paths = glob.glob(os.path.join(self.images_path, "JPEGImages/*/*.jpg"))
			images = pd.Series(
				data=[os.path.basename(os.path.dirname(image)) for image in image_paths],
				index=image_paths,
				name="labels",
				dtype=str,
			#	copy=None
			)

		return images[images.isin(labels)].replace(self.labels(select))


if __name__ == "__main__":
	animals_with_attributes = Dataset()
	animals_with_attributes_loader = tf.keras.utils.image_dataset_from_directory(
		"datasets/animals_with_attributes/JPEGImages",
	#	labels="inferred",
		label_mode="categorical",
		class_names=animals_with_attributes.labels().index.tolist(),
	#	color_mode="rgb",
		batch_size=1,
		image_size=(
			600,
			600,
		),
	#	shuffle=True,
		seed=0,
		validation_split=None,
		subset=None,
	#	interpolation="bilinear",
	#	follow_links=False,
		crop_to_aspect_ratio=True,
	)
