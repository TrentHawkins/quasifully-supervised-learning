"""Classifier specialized to transductive generalized zeroshot learning with the quasifully supervised learning loss."""


from dataclasses import dataclass
from typing import Iterable

import tensorflow

from ..keras.classifiers import Classifier
from .losses import QuasifullyBiasLoss, TransductiveGeneralizedZeroshotLoss, ZeroshotLoss
from .metrics import ZeroshotMetric


@dataclass
class GeneralizedZeroshotClassifier(Classifier):
	"""Specialize generic `source.keras.classifiers.Classifier` instance.

	Requires:
		source: A selection of labels known during testing and training.
		target: A selection of labels known during testing.

	Includes:
	-	adam optimizer with adjustable learning rate,
	-	zeroshot loss filtered by source labels,
	-	zeroshot metric,
		-	one filtered by source labels
		-	one filtered by target labels
		-	the harmonic mean of the two
	"""

	source: Iterable[int]
	target: Iterable[int]

	def compile(self, loss: tensorflow.keras.losses.Loss, metric: tensorflow.keras.metrics.Metric,
		learning_rate: float | None = None,
	):
		"""Configure the model for training.

		Arguments:
			loss: A `tensorflow.keras.losses.Loss` instance used as base for a `ZeroshotLoss` instance.
			metric: A `tensorflow.keras.metric.Metric` instance usedused as base for `ZeroshotMetric` instances.

		Keyword Arguments:
			learning_rate: The learning rate.
				Is:
				-	a `tensorflow.Tensor`,
				-	floating point value, or
				-	a schedule that is a `tf.keras.optimizers.schedules.LearningRateSchedule`, or
				-	a callable that takes no arguments and returns the actual value to use.

				Defaults to the inverse of the training data size.
		"""
		optimizer = tensorflow.keras.optimizers.Adam(
			learning_rate=learning_rate or 10 / len(self.train),
		#	beta_1=.9,
		#	beta_2=.999,
		#	epsilon=1e-7,
			amsgrad=True,
		#	name="Adam",
		)

	#	Filter loss by source labels:
		loss = ZeroshotLoss(loss, self.source)

	#	Filter metric by source and target labels:
		metrics = [
			ZeroshotMetric(metric, self.source, name="s"),
			ZeroshotMetric(metric, self.target, name="t"),
		]

		super(GeneralizedZeroshotClassifier, self).compile(optimizer, loss, metrics)  # type: ignore


@dataclass
class QuasifullyGeneralizedZeroshotClassifier(GeneralizedZeroshotClassifier):
	"""Expand `GeneralizedZeroshotClassifier` to include a quasifully loss bias.

	Includes:
	-	adam optimizer with adjustable learning rate,
	-	zeroshot loss filtered by source labels including a loss bias filtered by target labels,
	-	zeroshot metric,
		-	one filtered by source labels
		-	one filtered by target labels
		-	the harmonic mean of the two
	"""

	def compile(self, loss: tensorflow.keras.losses.Loss, metric: tensorflow.keras.metrics.Metric,
		learning_rate: float | None = None,
		log2_bias: int = 0,
		weight: float = 1.,
	):
		"""Configure the model for training.

		Arguments:
			loss: A `tensorflow.keras.losses.Loss` instance used as base for a `ZeroshotLoss` instance.
			metric: A `tensorflow.keras.metric.Metric` instance usedused as base for `ZeroshotMetric` instances.

		Keyword Arguments:
			learning_rate: The learning rate.
				Is:
				-	a `tensorflow.Tensor`,
				-	floating point value, or
				-	a schedule that is a `tf.keras.optimizers.schedules.LearningRateSchedule`, or
				-	a callable that takes no arguments and returns the actual value to use.

				Defaults to the inverse of the training data size.

			log2_bias: A hyper-coeffient adjusting the strength of the bias as a power of 2.
				Defaults to 0 for a coefficient of 1.

			weight: A probability balancing the quasifully cross-entropic bias with its binary cross-entropic complement.
				Defaults to 1 masking the complementary probabilities.
		"""
		optimizer = tensorflow.keras.optimizers.Adam(
			learning_rate=learning_rate or 10 / len(self.train),
		#	beta_1=.9,
		#	beta_2=.999,
		#	epsilon=1e-7,
			amsgrad=True,
		#	name="Adam",
		)

	#	Check for loss compatibility with quasifully bias:
		if not isinstance(loss,
			(
				tensorflow.keras.losses.CategoricalCrossentropy,
				tensorflow.keras.losses.BinaryCrossentropy,
			)
		):
			raise TypeError("The loss provided is incompatible with the quasifully loss bias.")

		bias = QuasifullyBiasLoss(log2_bias, weight)
		loss = TransductiveGeneralizedZeroshotLoss(
			ZeroshotLoss(loss, self.source),
			ZeroshotLoss(bias, self.target),
		)

	#	Filter metric by source and target labels:
		metrics = [
			ZeroshotMetric(metric, self.source, name="s"),
			ZeroshotMetric(metric, self.target, name="t"),
		]

		super(GeneralizedZeroshotClassifier, self).compile(optimizer, loss, metrics)  # type: ignore
