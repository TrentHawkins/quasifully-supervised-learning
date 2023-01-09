"""Zeroshot loss functions.

NOTE: There is a lot of repeating code. Consider abstracting the interface to apply label trimiing on any loss function given.
"""


from copy import deepcopy
from math import log2
from typing import Any, Iterable

import pandas
import tensorflow

from ..chartools import from_string, to_string


@tensorflow.keras.utils.register_keras_serializable("source>zeroshot>losses")
class ZeroshotLoss(tensorflow.keras.losses.Loss):
	"""Wrapper class for a zeroshot loss.

	NOTE: This means that `y_true`, `y_pred` are trimmed to a subset of `filter` of the full label set, known during training.
	The zeroshot filter is applied to any `tensorflow.keras.losses.Loss` instance provided.
	"""

	def __init__(self, loss: tensorflow.keras.losses.Loss, filter: Iterable[int],
		name: str = "zeroshot_loss",
	**kwargs):
		"""Initialize a `tensorflow.keras.losses.Loss` instance with a zeroshot label filter.

		Arguments:
			loss: A `tensorflow.keras.losses.Loss` instance to apply zeroshot label filter to.
			filter: An Iterable[int] of labels in sparce format to trim `y_true` and `y_pred` to.

		Keyword Argumdents:
			axis: Dimension of reduction of `y_true` and `y_pred`.
				Ignored when given `tensorflow.keras.losses.Loss` provides a reduction axis.

			name: Optional name for the instance, prepended by 'zeroshot_'.
				Defaults to 'zeroshot_loss'.
		"""
		super(ZeroshotLoss, self).__init__(
			name=name,
		**kwargs)

	#	deepcopy the losses to avoid sharing the same losses with other wrapped zeroshot losses
		self.loss = deepcopy(loss)

	#	reduction filter
		self.filter = tensorflow.constant(filter,
			dtype=tensorflow.int32,
		)

	def call(self,
		y_true: tensorflow.Tensor,
		y_pred: tensorflow.Tensor,
	):
		"""Invoke the given `Loss` instance modified with the zeroshot label filter.

		Arguments:
			y_true: Ground truth values.
				shape = `[batch_size, d0, .. dN]`, except sparse loss functions such as sparse categorical crossentropy where
				shape = `[batch_size, d0, .. dN-1]`

			y_pred: The predicted values.
				shape = `[batch_size, d0, .. dN]`

		Returns:
			Loss float `Tensor`.
				If `reduction` is `NONE`, this has shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar.
				(`dN-1` because all loss functions reduce by 1 dimension, usually axis=-1.)

		Raises:
			ValueError: If the shape of `sample_weight` is invalid.
		"""
		y_true_filter = tensorflow.gather(y_true, self.filter,
			axis=-1,
		)
		y_pred_filter = tensorflow.gather(y_pred, self.filter,
			axis=-1,
		)

		return self.loss.call(
			y_true_filter,
			y_pred_filter,
		)

	@classmethod
	def from_config(cls, config):
		"""Instantiate a `Loss` from its config (output of `get_config()`).

		NOTE: Deserializing the given `Loss` instance is necessary for the saving and loading of the model equipping wrapped loss.

		Arguments:
			config: Output of `get_config()`.

		Returns:
			A `Loss` instance.
		"""
		config["loss"] = tensorflow.keras.losses.deserialize(config["loss"])
		config["filter"] = pandas.Series(from_string(config["filter"]))

		return super(ZeroshotLoss, cls).from_config(config)

	def get_config(self):
		"""Return the config dictionary of the given a `Loss` instance.

		NOTE: Serializing the given `Loss` instance will allow the saving and loading of the model equipping the wrapped loss.
		"""
		config = super(ZeroshotLoss, self).get_config()
		config.update(
			{
				"loss": tensorflow.keras.losses.serialize(self.loss),
				"filter": to_string(self.filter.numpy()),
			}
		)

		return config


@tensorflow.keras.utils.register_keras_serializable("source>zeroshot>losses")
class TransductiveGeneralizedZeroshotLoss(tensorflow.keras.losses.Loss):
	"""Wrapper class for a generalized zeroshot loss in the transductive setting.

	NOTE: This supposedly uses the target labels provided.
	"""

	def __init__(self,
		loss: ZeroshotLoss,
		bias: ZeroshotLoss, name: str = "transductive_generalized_zeroshot_loss",
	**kwargs):
		"""Initialize a `tensorflow.keras.losses.Loss` instance with a complementary zeroshot label filter.

		Arguments:
			loss: A `tensorflow.keras.losses.Loss` instance to apply zeroshot label filter to.
			bias: A `tensorflow.keras.losses.Loss` instance to augment original loss with.
		"""
		super(TransductiveGeneralizedZeroshotLoss, self).__init__(
			name=name,
		**kwargs)

	#	deepcopy the losses to avoid sharing the same losses with other wrapped zeroshot losses
		self.loss = deepcopy(loss)
		self.bias = deepcopy(bias)

	def call(self,
		y_true: tensorflow.Tensor,
		y_pred: tensorflow.Tensor,
	):
		"""Invoke the given `Loss` instance filtered by source labels and quasifully biased by target labels.

		Arguments:
			y_true: Ground truth values.
				shape = `[batch_size, d0, .. dN]`, except sparse loss functions such as sparse categorical crossentropy where
				shape = `[batch_size, d0, .. dN-1]`

			y_pred: The predicted values.
				shape = `[batch_size, d0, .. dN]`

		Returns:
			Loss float `Tensor`.
				If `reduction` is `NONE`, this has shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar.
				(`dN-1` because all loss functions reduce by 1 dimension, usually axis=-1.)
		"""
		return \
			self.loss.call(y_true, y_pred) + \
			self.bias.call(y_true, y_pred)

	@classmethod
	def from_config(cls, config):
		"""Instantiate a `Loss` from its config (output of `get_config()`).

		NOTE: Deserializing the given `Loss` instance is necessary for the saving and loading of the model equipping wrapped loss.

		Arguments:
			config: Output of `get_config()`.

		Returns:
			A `Loss` instance.
		"""
		config["loss"] = tensorflow.keras.losses.deserialize(config["loss"])
		config["bias"] = tensorflow.keras.losses.deserialize(config["bias"])

		return super(TransductiveGeneralizedZeroshotLoss, cls).from_config(config)

	def get_config(self) -> dict:
		"""Return the config dictionary of the given a `Loss` instance.

		NOTE: Serializing the given `Loss` instance will allow the saving and loading of the model equipping the wrapped loss.
		"""
		config = super(TransductiveGeneralizedZeroshotLoss, self).get_config()
		config.update(
			{
				"loss": tensorflow.keras.losses.serialize(self.loss),
				"bias": tensorflow.keras.losses.serialize(self.bias),
			}
		)

		return config


@tensorflow.keras.utils.register_keras_serializable("source>zeroshot>losses")
class QuasifullyBiasLoss(tensorflow.keras.losses.Loss):
	"""Quasifully supervised loss bias based on unlabelled examples, assuming given labels are unknown during training.

	Jie Song, Chengchao Shen, Yezhou Yang, Yang Liu, Mingli Song
	Transductive Unbiased Embedding for Zero-Shot Learning
	CVPR2018
	[arXiv:1803.11320](https://arxiv.org/abs/1803.11320)
	"""

	def __init__(self,
		log2_bias: int = 0,
		weight: float = 1,
		name: str = "quasifully_bias_loss",
	**kwargs):
		"""Initialize a quasifully bias loss `tensorflow.keras.losses.Loss` instance.

		This loss is compatible in a transductive generalized zeroshot learning setting only with (subclasses of):
		-	`tensorflow.keras.losses.CategoricalCrossentropy`
		-	`tensorflow.keras.losses.BinaryCrossentropy`

		Arguments:
			log2_bias: A hyper-coeffient adjusting the strength of the bias as a power of 2.
				Defaults to 0 for a coefficient of 1.

			weight: A probability balancing the quasifully cross-entropic bias with its binary cross-entropic complement.
				Defaults to 1 masking the complementary probabilities.

		Keyword Argumdents:
			name: Optional name for the instance, prepended by 'quasifully_bias_'.
				Defaults to 'quasifully_bias_loss'.
		"""
		super(QuasifullyBiasLoss, self).__init__(
			name=name,
		**kwargs)

	#	bias coefficient as a power of 2
		self.bias: float = 2 ** log2_bias

	#	probabilistic balancer
		self.weight = weight

	def call(self,
		y_true: tensorflow.Tensor,
		y_pred: tensorflow.Tensor,
	):
		"""Invoke the given `Loss` instance modified with the zeroshot filter.

		NOTE: The formula is modified to include a complementary binary cross-entropic term.
		Takes into account all labels given.

		Arguments:
			y_true: Ground truth values.
				shape = `[batch_size, d0, .. dN]`, except sparse loss functions such as sparse categorical crossentropy where
				shape = `[batch_size, d0, .. dN-1]`

			y_pred: The predicted values.
				shape = `[batch_size, d0, .. dN]`

		Returns:
			Loss float `Tensor`.
				If `reduction` is `NONE`, this has shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar.
				(`dN-1` because all loss functions reduce by 1 dimension, usually axis=-1.)
		"""
		return -self.bias * (
			self.weight * tensorflow.math.log(
				tensorflow.reduce_sum(y_pred,
					axis=-1,
				)  # type: ignore  # `*` not supported with `tensorflow.math.log` return type but in fact is
			) + (1 - self.weight) * tensorflow.math.log(
				1 - tensorflow.reduce_sum(y_pred,
					axis=-1,
				)  # type: ignore  # `*` not supported with `tensorflow.math.log` return type but in fact is
			)
		)

	def get_config(self) -> dict:
		"""Return the config dictionary of the given a `Loss` instance.

		NOTE: Serializing the given `Loss` instance will allow the saving and loading of the model equipping the wrapped loss.
		"""
		config = super(QuasifullyBiasLoss, self).get_config()
		config.update(
			{
				"log2_bias": int(log2(self.bias)),
				"weight": self.weight,
			}
		)

		return config
