"""Classifier specialized to transductive generalized zeroshot learning with the quasifully supervised learning loss."""

from dataclasses import dataclass
from typing import Iterable

import tensorflow

from ..keras.classifiers import Classifier
from .losses import ZeroshotCategoricalCrossentropy, QuasifullyGeneralizedZeroshotCategoricalCrossentropy
from .metrics import ZeroshotCategoricalAccuracy


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

	def __post_init__(self):
		"""Set meaningful default compiling parameters for model underlying classifier.

		Includes:
		-	adam optimizer with adjustable learning rate,
		-	categorical cross-entory loss,
		-	accuracy metric on top of loss for evaluation
		"""
		super(CategoricalClassifier, self).__post_init__()

	#	optimizer
		self.optimizer: tensorflow.keras.optimizers.Optimizer = tensorflow.keras.optimizers.Adam(
		#	learning_rate=1e-3,
		#	beta_1=.9,
		#	beta_2=.999,
		#	epsilon=1e-7,
			amsgrad=True,
		#	name="Adam",
		)

	#	loss
		self.loss: tensorflow.keras.losses.Loss = tensorflow.keras.losses.CategoricalCrossentropy(
		#	from_logits=False,
		#	label_smoothing=0.,
		#	axis=-1,
		#	name="categorical_crossentropy"
		)

	#	metrics (basic)
		self.metrics: list[tensorflow.keras.metrics.Metric | str] = [
			tensorflow.keras.metrics.CategoricalAccuracy(
				name="n",
			),
		]

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
		self.optimizer.learning_rate = learning_rate or 10 / len(self.train)

		super(CategoricalClassifier, self).compile(self.optimizer, self.loss, self.metrics)


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

#	labels seen during training and testing
	source: tensorflow.Tensor | Iterable[int]

	def __post_init__(self):
		"""Set label zeroshot filter and update metrics.

		Arguments:
			source: A selection of labels known during training.
		"""
		super(ZeroshotCategoricalClassifier, self).__post_init__()

	#	labels seen during training and testing
		self.source = tensorflow.constant(self.source, dtype=tensorflow.int32)

	#	zeroshot loss (trimming to source labels)
		self.loss: tensorflow.keras.losses.Loss = ZeroshotCategoricalCrossentropy(
			self.source,
		#	from_logits=False,
		#	label_smoothing=0.,
		#	axis=-1,
		#	name="categorical_crossentropy"
		)

	#	zeroshot accuracy (trimmed to source labels)
		self.metrics.append(
			ZeroshotCategoricalAccuracy(self.source,
				name="s",
			)
		)


@dataclass
class GeneralizedZeroshotCategoricalClassifier(ZeroshotCategoricalClassifier):
	"""Augment generic `ZeroshotCategoricalClassifier` instance with target label evaluators.

	Includes:
	-	adam optimizer with adjustable learning rate,
	-	trimmed categorical cross-entroy loss on definable source labels quasifully biased on target labels
	-	accuracy metric on top of loss for evaluation
	-	callbacks:
		-	early stopping for regularization with patience a small fraction of the number of epochs,
		-	reduce learning rate on plateau with patience a fraction of the early stopping patience,
	"""

#	labels seen during testing
	target: tensorflow.Tensor | Iterable[int]

	def __post_init__(self):
		"""Set label zeroshot filter and update metrics.

		Arguments:
			source: A selection of labels known during testing and training.
			target: A selection of labels known during testing.
		"""
		super(GeneralizedZeroshotCategoricalClassifier, self).__post_init__()

	#	labels seen during testing
		self.target = tensorflow.constant(self.target, dtype=tensorflow.int32)

	#	generalized zeroshot accuracy (adding one trimmed to target labels)
		self.metrics.append(
			ZeroshotCategoricalAccuracy(self.target,
				name="t",
			),
		)


@dataclass
class QuasifullyGeneralizedZeroshotCategoricalClassifier(GeneralizedZeroshotCategoricalClassifier):
	"""Augment generic `GeneralizedZeroshotCategoricalClassifier` instance with quasifully supervised learning bias.

	Includes:
	-	adam optimizer with adjustable learning rate,
	-	trimmed categorical cross-entroy loss on definable source labels quasifully biased on target labels
	-	accuracy metric on top of loss for evaluation
	-	callbacks:
		-	early stopping for regularization with patience a small fraction of the number of epochs,
		-	reduce learning rate on plateau with patience a fraction of the early stopping patience,
	"""

#	bias towards unlabelled examples
	bias: float = 0.

	def __post_init__(self):
		"""Set label zeroshot filter and update metrics.

		Arguments:
			source: A selection of labels known during training.
		"""
		super(QuasifullyGeneralizedZeroshotCategoricalClassifier, self).__post_init__()

	#	zeroshot loss (trimming to source labels)
		self.loss: tensorflow.keras.losses.Loss = QuasifullyGeneralizedZeroshotCategoricalCrossentropy(
			self.source,
			self.target,
		#	from_logits=False,
		#	label_smoothing=0.,
		#	axis=-1,
		#	name="categorical_crossentropy"
		)
