"""Classifier specialized to transductive generalized zeroshot learning with the quasifully supervised learning loss."""

from dataclasses import dataclass, field
from math import ceil, sqrt
from typing import Iterable

import tensorflow

from ..keras.classifiers import Classifier
from .losses import ZeroshotCategoricalCrossentropy, QuasifullyZeroshotCategoricalCrossentropy


@dataclass
class CategoricalClassifier(Classifier):
	"""Specialize generic `source.keras.classifiers.Classifier` instance.

	Includes:
	-	adam optimizer with adjustable learning rate,
	-	categorical cross-entory loss,
	-	accuracy metric on top of loss for evaluation
	-	callbacks:
		-	early stopping for regularization with patience a small fraction of the number of epochs,
		-	reduce learning rate on plateau with patience a fraction of the early stopping patience,
	"""

	def compile(self, learning_rate: float | None = None):
		"""Configure the model for training.

		Keyword Arguments:
			learning_rate: The learning rate.
				Is:
				-	a `tensorflow.Tensor`,
				-	floating point value, or
				-	a schedule that is a `tf.keras.optimizers.schedules.LearningRateSchedule`, or
				-	a callable that takes no arguments and returns the actual value to use.

				Defaults to the inverse of the training data size.
		"""
		super(CategoricalClassifier, self).compile(
			optimizer=tensorflow.keras.optimizers.Adam(
				learning_rate=learning_rate or 1 / len(self.train),
			#	beta_1=.9,
			#	beta_2=.999,
			#	epsilon=1e-7,
				amsgrad=True,
			#	name="Adam",
			),
			loss=tensorflow.keras.losses.CategoricalCrossentropy(
			#	from_logits=False,
			#	label_smoothing=0.,
			#	axis=-1,
			#	name='categorical_crossentropy'
			),
			metrics=[
				tensorflow.keras.metrics.CategoricalAccuracy(),
			],
		)


@dataclass
class ZeroshotCategoricalClassifier(CategoricalClassifier):
	"""Specialize generic `CategoricalClassifier` instance with zeroshot loss.

	Includes:
	-	adam optimizer with adjustable learning rate,
	-	trimmed categorical cross-entroy loss on definable source labels
	-	accuracy metric on top of loss for evaluation
	-	callbacks:
		-	early stopping for regularization with patience a small fraction of the number of epochs,
		-	reduce learning rate on plateau with patience a fraction of the early stopping patience,
	"""

	source: tensorflow.Tensor | Iterable[int] = field()

	def compile(self, learning_rate: float | None = None):
		"""Configure the model for training.

		Keyword Arguments:
			learning_rate: The learning rate.
				Is:
				-	a `tensorflow.Tensor`,
				-	floating point value, or
				-	a schedule that is a `tf.keras.optimizers.schedules.LearningRateSchedule`, or
				-	a callable that takes no arguments and returns the actual value to use.

				Defaults to the inverse of the training data size.
		"""
		super(CategoricalClassifier, self).compile(
			optimizer=tensorflow.keras.optimizers.Adam(
				learning_rate=learning_rate or 1 / len(self.train),
			#	beta_1=.9,
			#	beta_2=.999,
			#	epsilon=1e-7,
				amsgrad=True,
			#	name="Adam",
			),
			loss=ZeroshotCategoricalCrossentropy(
				self.source,
			#	from_logits=False,
			#	label_smoothing=0.,
			#	axis=-1,
			#	name='categorical_crossentropy'
			),
			metrics=[
				tensorflow.keras.metrics.CategoricalAccuracy(),
			],
		)


@dataclass
class QuasifullyZeroshotCategoricalClassifier(ZeroshotCategoricalClassifier):
	"""Augment generic `ZeroshotCategoricalClassifier` instance with quasifully supervised learning bias.

	Includes:
	-	adam optimizer with adjustable learning rate,
	-	trimmed categorical cross-entroy loss on definable source labels quasifully biased on target labels
	-	accuracy metric on top of loss for evaluation
	-	callbacks:
		-	early stopping for regularization with patience a small fraction of the number of epochs,
		-	reduce learning rate on plateau with patience a fraction of the early stopping patience,
	"""

	target: tensorflow.Tensor | Iterable[int] = field()

#	Set bias as powers of 2.
	bias: int = field(default=1, kw_only=True)

	def __post_init__(self):
		"""Set bias as powers of 2."""
		self.bias = 2 ** self.bias

	def compile(self, learning_rate: float | None = None):
		"""Configure the model for training.

		Keyword Arguments:
			learning_rate: The learning rate:
				-	a `tensorflow.Tensor`, or
				-	floating point value, or
				-	a schedule that is a `tensorflow.keras.optimizers.schedules.LearningRateSchedule`, or
				-	a callable that takes no arguments and returns the actual value to use.

				Defaults to the inverse of the training data size.
		"""
		super(CategoricalClassifier, self).compile(
			optimizer=tensorflow.keras.optimizers.Adam(
				learning_rate=learning_rate or 1 / len(self.train),
			#	beta_1=.9,
			#	beta_2=.999,
			#	epsilon=1e-7,
				amsgrad=True,
			#	name="Adam",
			),
			loss=QuasifullyZeroshotCategoricalCrossentropy(
				self.source,
				self.target, bias=self.bias,
			),
			metrics=[
				tensorflow.keras.metrics.CategoricalAccuracy(),
			],
		)
