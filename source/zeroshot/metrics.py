"""Zeroshot metrics."""


from __future__ import annotations

from typing import Iterable, Optional

import numpy
import pandas
import tensorflow

from ..chartools import from_string, to_string


@tensorflow.keras.utils.register_keras_serializable("source>zeroshot>metrics")
class ZeroshotCategoricalAccuracy(tensorflow.keras.metrics.CategoricalAccuracy):
	"""Calculate how often predictions match one-hot labels.

	You can provide logits of classes as `y_pred`, since argmax of logits and probabilities are same.

	This metric creates two local variables, `total` and `count`,
	that are used to compute the frequency with which `y_pred` matches `y_true`.

	This frequency is ultimately returned as `categorical accuracy`:
		an idempotent operation that simply divides `total` by `count`.

	`y_pred` and `y_true` should be passed in as vectors of probabilities, rather than as labels.
	If necessary, use `tf.one_hot` to expand `y_true` as a vector.

	If `sample_weight` is `None`, weights default to 1.
	Use `sample_weight` of 0 to mask values.

	This modification trims some labels off to account for missing labels during training.

	For example if 40 out of 50 labels in a clasification task are picked,
	all 50-one-hot representations will be trimmed to a 40-one-hot representation corresponding to the picked labels.
	"""

	def __init__(self, filter: Iterable, **kwargs):
		"""Store label selection for trimming output.

		Arguments:
			filter: A selection to account for when evaluating accuracy.
		"""
		super(ZeroshotCategoricalAccuracy, self).__init__(**kwargs)

		self.filter: tensorflow.Tensor = tensorflow.constant(filter, dtype=tensorflow.int32)

	def update_state(self,
		y_true: tensorflow.Tensor,
		y_pred: tensorflow.Tensor, sample_weight: Optional[tensorflow.Tensor] = None
	):
		"""Accumulate metric statistics.

		`y_true` and `y_pred` should have the same shape.

		Arguments:
			y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
			y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
			sample_weight: Optional `sample_weight` acts as a coefficient for the metric.
				If a scalar is provided, then the metric is simply scaled by the given value.
				If `sample_weight` is a tensor of size `[batch_size]`,
				then the metric for each sample of the batch is rescaled by the corresponding element in `sample_weight` vector.

			If the shape of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted to this shape),
			then each metric element of `y_pred` is scaled by the corresponding value of `sample_weight`.
			(Note on `dN-1`: all metric functions reduce by 1 dimension, usually the last axis (-1)).
		"""
		y_true_filter = tensorflow.gather(y_true, self.filter,
			axis=-1,
		)
		y_pred_filter = tensorflow.gather(y_pred, self.filter,
			axis=-1,
		)

		super(ZeroshotCategoricalAccuracy, self).update_state(
			y_true_filter,
			y_pred_filter, sample_weight
		)

	@classmethod
	def from_config(cls, config: dict):
		"""Instantiate a `Metric` from its config (output of `get_config()`).

		Decode label subset from string storage format before loading loss.

		Arguments:
			config: Output of `get_config()`.

		Returns:
			A `Metric` instance.
		"""
		config["filter"] = pandas.Series(from_string(config["filter"]))
		accuracy = super(ZeroshotCategoricalAccuracy, cls).from_config(config)

		return accuracy

	def get_config(self) -> dict:
		"""Return the config dictionary for a `Metric` instance.

		Encode label subset to string storage format after saving loss.

		Returns:
			The config dictionary for a `Metric` instance.
		"""
		config = super(ZeroshotCategoricalAccuracy, self).get_config()
		config.update(
			{
				"filter": to_string(self.filter.numpy()),  # type: ignore https://github.com/microsoft/pylance-release/issues/2871
			}
		)

		return config
