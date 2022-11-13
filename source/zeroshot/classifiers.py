"""Classifier specialized to transductive generalized zeroshot learning with the quasifully supervised learning loss."""

from dataclasses import dataclass
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

	def fit(self,
		train: tensorflow.data.Dataset | None = None,
		devel: tensorflow.data.Dataset | None = None,
		*,
		epochs: int = 1,
	):
		"""Train the model for a fixed number of epochs (iterations on a dataset).

		Unpacking behavior for iterator-like inputs:
			A common pattern is to pass a
				`tensorflow.data.Dataset`,
				`generator`, or
				`tensorflow.keras.utils.Sequence`
			which will in fact yield not only features but optionally targets and sample weights.

			Keras requires that the output of such iterator-likes be unambiguous.

		In this wrapper `tensorflow.data.Dataset` are used.

		Arguments:
			train: Input data as a `tensorflow.data dataset`.
				Should return a `tuple` of either `(inputs, targets)` or `(inputs, targets, sample_weights)`.

				Defaults to the `tensorflow.data dataset` set at initialization.

			devel: Data as a `tensorflow.data dataset` to evaluate the loss and any model metrics at the end of each epoch on.
				The model will not be trained on this data.
				Thus, note the fact that the validation loss is not affected by regularization layers like noise and dropout.

				Defaults to the `tensorflow.data dataset` set at initialization.

		Keyword Arguments:
			epochs: Integer.
				Number of epochs to train the model.

				An epoch is an iteration over the entire data provided.

				Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch".
				The model is not trained for a number of iterations given by `epochs`,
				but merely until the epoch of index epochs is reached.

				Defaults to 1.

			callbacks: List of `tensorflow.keras.callbacks.Callback` instances.
				List of callbacks to apply during training.
				See tensorflow.keras.callbacks.

				Note:
				-	`tensorflow.keras.callbacks.ProgbarLogger` and
				-	`tensorflow.keras.callbacks.History`
				callbacks are created automatically and need not be passed into `model.fit`.

				`tensorflow.keras.callbacks.ProgbarLogger` is created or not based on verbose argument to `model.fit`.

				Callbacks with batch-level calls are currently unsupported with:
				-	`tensorflow.distribute.experimental.ParameterServerStrategy`,
				and users are advised to implement epoch-level calls instead with an appropriate `steps_per_epoch` value.

		Returns:
			A `History` object:
				Its `History.history` attribute is a record of training loss values and metrics values at successive epochs,
				as well as validation loss values and validation metrics values (if applicable).

		Raises:
			RuntimeError: If the model was never compiled or `model.fit` is wrapped in `tensorflow.function`.

			ValueError:	If mismatch between the provided input data and what the model expects or when the input data is empty.
		"""
		patience_early_stopping = ceil(sqrt(epochs))
		patience_reduce_learning_rate_on_plateau = ceil(sqrt(patience_early_stopping))

		return super(CategoricalClassifier, self).fit(
			train,
			devel,
			epochs=epochs,
			callbacks=[
				tensorflow.keras.callbacks.ReduceLROnPlateau(
				#	monitor="val_loss",
				#	factor=1e-1,
					patience=patience_reduce_learning_rate_on_plateau,
				#	verbose=0,
				#	mode="auto",
				#	min_delta=1e-4,
					cooldown=patience_reduce_learning_rate_on_plateau,
				#	min_lr=0,
				),
				tensorflow.keras.callbacks.EarlyStopping(
					monitor="val_loss",
				#	min_delta=0,
					patience=patience_early_stopping,
				#	verbose=0,
				#	mode="auto",
				#	baseline=None,
					restore_best_weights=True,
				)
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

	source: tensorflow.Tensor | Iterable[int]

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

	target: tensorflow.Tensor | Iterable[int]

#	Set bias as powers of 2.
	bias: int = 1

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
