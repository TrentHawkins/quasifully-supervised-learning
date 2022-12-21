"""Zeroshot loss functions."""


from typing import Iterable

import tensorflow
import pandas

from ..chartools import from_string, to_string


@tensorflow.keras.utils.register_keras_serializable("source>zeroshot>losses")
class ZeroshotCategoricalCrossentropy(tensorflow.keras.losses.CategoricalCrossentropy):
	"""Computes the crossentropy loss between the labels and predictions.

	Use this crossentropy loss function when there are two or more label classes.
	We expect labels to be provided in a `one_hot` representation.
	If you want to provide labels as integers, please use `SparseCategoricalCrossentropy` loss.
	There should be `# classes` floating point values per feature.

	In the snippet below, there is `# classes` floating pointing values per example.
	The shape of both `y_pred` and `y_true` are `[batch_size, num_classes]`.

	NOTE: This modified version only sums over a subset of the (seen) labels, ignoring other (unseen) labels.

	Standalone usage:
	```
	>>> y_true = [[0, 1, 0], [0, 0, 1]]
	>>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
	>>> # Using 'auto'/'sum_over_batch_size' reduction type.
	>>> cce = tf.keras.losses.CategoricalCrossentropy()
	>>> cce(y_true, y_pred).numpy()
	1.177
	```
	```
	>>> # Calling with 'sample_weight'.
	>>> cce(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7])).numpy()
	0.814
	```
	```
	>>> # Using 'sum' reduction type.
	>>> cce = tf.keras.losses.CategoricalCrossentropy(
	...     reduction=tf.keras.losses.Reduction.SUM)
	>>> cce(y_true, y_pred).numpy()
	2.354
	```
	```
	>>> # Using 'none' reduction type.
	>>> cce = tf.keras.losses.CategoricalCrossentropy(
	...     reduction=tf.keras.losses.Reduction.NONE)
	>>> cce(y_true, y_pred).numpy()
	array([0.0513, 2.303], dtype=float32)
	```

	Usage with the `compile()` API:
	```python
	model.compile(
		optimizer='sgd',
		loss=tf.keras.losses.CategoricalCrossentropy()
	)
	```
	"""

	def __init__(self, source: tensorflow.Tensor | Iterable[int],
		axis: int = -1,
		name: str = "zeroshot_categorical_crossentropy",
	**kwargs):
		"""Initialize `ZeroshotCategoricalCrossentropy` instance.

		Arguments:
			source: The labels seen during zeroshot training.
				Must be a subset of the labels used in the respective problem.

				Unlike the usage in the dataset and this loss functions, they must be in sparse categorical form.
				Used to trim categorical `one_hot` vectors to only the source labels.

		Keyword Arguments:
			from_logits: Whether `y_pred` is expected to be a logits tensor.
				We assume that `y_pred` encodes a probability distribution.

			label_smoothing: Float in [0, 1].
				When > 0, label values are smoothed, meaning the confidence on label values are relaxed.
				We do not smoothen labels.

				For example, if `0.1`,
				-	use `0.1 / num_classes` for non-target labels and
				-	`0.9 + 0.1 / num_classes` for target labels.

			axis: The axis along which to compute crossentropy (the features axis).
				Defaults to -1.

			reduction: Type of `tf.keras.losses.Reduction` to apply to loss.
				Set value is `AUTO`.

				`AUTO` indicates that the reduction option will be determined by the usage context.
				For almost all cases this defaults to `SUM_OVER_BATCH_SIZE`.
				When used with `tf.distribute.Strategy`,
				outside of built-in training loops such as `tf.keras` `compile` and `fit`,
				using `AUTO` or `SUM_OVER_BATCH_SIZE` will raise an error.

			name: Optional name for the instance.
				Defaults to "zeroshot_categorical_crossentropy".
		"""
		super(ZeroshotCategoricalCrossentropy, self).__init__(
			axis=kwargs.pop("axis", None) or axis,
			name=kwargs.pop("name", None) or name,
		**kwargs)

	#	save axis:
		self.axis = axis

	#	labels seen during training:
		self.source = tensorflow.convert_to_tensor(source, dtype=tensorflow.int32)

	def call(self,
		y_true,
		y_pred,
	):
		"""Invoke the `Loss` instance.

		Argumentss:
			y_true: Ground truth values.
				shape = `[batch_size, d0, ..., dN]`, except sparse loss functions such as sparse categorical crossentropy where
				shape = `[batch_size, d0, ..., dN-1]`

			y_pred: The predicted values.
				shape = `[batch_size, d0, ..., dN]`

		Returns:
			Loss values with the shape `[batch_size, d0, ..., dN-1]`.
		"""
		y_pred_source = tensorflow.gather(y_pred, self.source,
			axis=self.axis,
		)
		y_true_source = tensorflow.gather(y_true, self.source,
			axis=self.axis,
		)

		return super(ZeroshotCategoricalCrossentropy, self).call(
			y_true_source,
			y_pred_source,
		)


@tensorflow.keras.utils.register_keras_serializable("source>zeroshot>losses")
class QuasifullyZeroshotCategoricalCrossentropy(ZeroshotCategoricalCrossentropy):
	"""Computes the crossentropy loss between the labels and predictions.

	Use this crossentropy loss function when there are two or more label classes.
	We expect labels to be provided in a `one_hot` representation.
	If you want to provide labels as integers, please use `SparseCategoricalCrossentropy` loss.
	There should be `# classes` floating point values per feature.

	In the snippet below, there is `# classes` floating pointing values per example.
	The shape of both `y_pred` and `y_true` are `[batch_size, num_classes]`.

	NOTE: This modified version adds an entropic sum corresponding to unlabelled examples.

	Standalone usage:
	```
	>>> y_true = [[0, 1, 0], [0, 0, 1]]
	>>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
	>>> # Using 'auto'/'sum_over_batch_size' reduction type.
	>>> cce = tf.keras.losses.CategoricalCrossentropy()
	>>> cce(y_true, y_pred).numpy()
	1.177
	```
	```
	>>> # Calling with 'sample_weight'.
	>>> cce(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7])).numpy()
	0.814
	```
	```
	>>> # Using 'sum' reduction type.
	>>> cce = tf.keras.losses.CategoricalCrossentropy(
	...     reduction=tf.keras.losses.Reduction.SUM)
	>>> cce(y_true, y_pred).numpy()
	2.354
	```
	```
	>>> # Using 'none' reduction type.
	>>> cce = tf.keras.losses.CategoricalCrossentropy(
	...     reduction=tf.keras.losses.Reduction.NONE)
	>>> cce(y_true, y_pred).numpy()
	array([0.0513, 2.303], dtype=float32)
	```

	Usage with the `compile()` API:
	```python
	model.compile(
		optimizer='sgd',
		loss=tf.keras.losses.CategoricalCrossentropy()
	)
	```
	"""

	def __init__(self,
		source: tensorflow.Tensor | Iterable[int],
		target: tensorflow.Tensor | Iterable[int],
		bias: float = 0.,
		axis: int = -1,
		name: str = "quasifully_categorical_crossentropy",
	**kwargs):
		"""Initialize `QuasifullyCategoricalCrossentropy` instance.

		Arguments:
			source: The labels seen during zeroshot training.
				Must be a subset of the labels used in the respective problem.

				Unlike the usage in the dataset and this loss functions, they must be in sparse categorical form.
				Used to trim categorical `one_hot` vectors to only the source labels.

			target: The labels not seen during zeroshot training.
				Must be a subset of the labels used in the respective problem.

				Unlike the usage in the dataset and this loss functions, they must be in sparse categorical form.
				Used to trim categorical `one_hot` vectors to only the source labels.

			bias: The influence of the quasifully bias term to the overall loss.
				Defaults to no bias at all.

		Keyword Arguments:
			from_logits: Whether `y_pred` is expected to be a logits tensor.
				By default, we assume that `y_pred` encodes a probability distribution.

			label_smoothing: Float in [0, 1].
				When > 0, label values are smoothed, meaning the confidence on label values are relaxed.

			For example, if `.1`, use `.1 / num_classes` for non-target labels and `.9 + .1 / num_classes` for target labels.

			axis: The axis along which to compute crossentropy (the features axis).
				Defaults to -1.

			reduction: Type of `tf.keras.losses.Reduction` to apply to loss.
				Default value is `AUTO`.

				`AUTO` indicates that the reduction option will be determined by the usage context.
				For almost all cases this defaults to `SUM_OVER_BATCH_SIZE`.
				When used with `tf.distribute.Strategy`,
				outside of built-in training loops such as `tf.keras` `compile` and `fit`,
				using `AUTO` or `SUM_OVER_BATCH_SIZE` will raise an error.

			name: Optional name for the instance.
				Defaults to "quasifully_categorical_crossentropy".
		"""
		super(QuasifullyZeroshotCategoricalCrossentropy, self).__init__(source,
			axis=kwargs.pop("axis", None) or axis,
			name=kwargs.pop("name", None) or name,
		**kwargs)

	#	labels not seen during training
		self.target = tensorflow.convert_to_tensor(target, dtype=tensorflow.int32)

	#	bias coefficient of quasifully supervised loss influence
		self.bias = bias

	def call(self,
		y_true,
		y_pred,
	):
		"""Invoke the `Loss` instance.

		Argumentss:
			y_true: Ground truth values.
				shape = `[batch_size, d0, ..., dN]`, except sparse loss functions such as sparse categorical crossentropy where
				shape = `[batch_size, d0, ..., dN-1]`

			y_pred: The predicted values.
				shape = `[batch_size, d0, ..., dN]`

		Returns:
			Loss values with the shape `[batch_size, d0, ..., dN-1]`.
		"""
		y_pred_target = tensorflow.gather(y_pred, self.source,
			axis=self.axis,
		)
	#	y_true_target = tensorflow.gather(y_true, self.source,
	#		axis=self.axis,
	#	)

		return super(QuasifullyZeroshotCategoricalCrossentropy, self).call(
			y_true,
			y_pred,
		) - tensorflow.math.log(
			tensorflow.reduce_sum(y_pred_target,
				axis=self.axis,
			)
		) * self.bias  # type: ignore  # Pylance miss-identifies the return type of `tensorflow.math.log` for some reason

	@classmethod
	def from_config(cls, config: dict):
		"""Instantiate a `Loss` from its config (output of `get_config()`).

		Decode label subset from string storage format before loading loss.

		Arguments:
			config: Output of `get_config()`.

		Returns:
			A `Loss` instance.
		"""
		config["source"] = pandas.Series(from_string(config["source"]))
		config["target"] = pandas.Series(from_string(config["target"]))

		loss = super(QuasifullyZeroshotCategoricalCrossentropy, cls).from_config(config)

		return loss

	def get_config(self) -> dict:
		"""Return the config dictionary for a `Loss` instance.

		Encode label subset to string storage format after saving loss.

		Returns:
			The config dictionary for a `Loss` instance.
		"""
		config = super(ZeroshotCategoricalCrossentropy, self).get_config()

		config.update(
			{
				"source": to_string(self.source),
				"target": to_string(self.target), "bias": self.bias,
			}
		)

		return config
