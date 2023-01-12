"""Zeroshot metrics."""


from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import pandas
import tensorflow

from ..chartools import from_string, to_string


@tensorflow.keras.utils.register_keras_serializable("source>zeroshot>metrics")
class ZeroshotMetric(tensorflow.keras.metrics.Mean):
	"""Wrapper class for a zeroshot metric.

	NOTE: This means that `y_true`, `y_pred` are trimmed to a subset of `filter` of the full label set, known during training.
	The zeroshot filter is applied to any `tensorflow.keras.metrics.Metric` instance provided.
	"""

	def __init__(self, metric: tensorflow.keras.metrics.Metric, filter: Iterable[int],
		name: str = "zeroshot_metric",
	**kwargs):
		"""Initialize a `tensorflow.keras.metrics.Metric` instance with a zeroshot label filter.

		Arguments:
			metric: A `tensorflow.keras.metrics.Metric` instance to apply zeroshot label filter to.
			filter: An Iterable[int] of labels in sparce format to trim `y_true` and `y_pred` to.

		Keyword Argumdents:
			axis: Dimension of reduction of `y_true` and `y_pred`.
				Ignored when given `tensorflow.keras.metrics.Metric` provides a reduction axis.

			name: Optional name for the instance, prepended by 'zeroshot_'.
				Defaults to 'zeroshot_loss'.
		"""
		super(ZeroshotMetric, self).__init__(
			name=name,
		**kwargs)

	#	deepcopy the metric to avoid sharing the same metric with other wrapped zeroshot metrics
		self.metric = deepcopy(metric)

	#	reduction filter
		self.filter = tensorflow.constant(filter,
			dtype=tensorflow.int32,
		)

	def update_state(self,
		y_true: tensorflow.Tensor,
		y_pred: tensorflow.Tensor, sample_weight: tensorflow.Tensor | None = None,
	):
		"""Accumulate metric statistics.

		`y_true` and `y_pred` should have the same shape.

		Arguments:
			y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
				Trimmed by zeroshot label filter.

			y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
				Trimmed by zeroshot label filter.

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

		return self.metric.update_state(
			y_true_filter,
			y_pred_filter, sample_weight,
		)

	def result(self):
		"""Compute and return the scalar metric value tensor or a dict of scalars.

		Result computation is an idempotent operation that simply calculates the metric value using the state variables.

		Returns:
			A scalar tensor, or a dictionary of scalar tensors.
		"""
		return self.metric.result()

	@classmethod
	def from_config(cls, config: dict):
		"""Instantiate a `Metric` from its config (output of `get_config()`).

		Decode label subset from string storage format before loading loss.

		Arguments:
			config: Output of `get_config()`.

		Returns:
			A `Metric` instance.
		"""
		config["metric"] = tensorflow.keras.metrics.deserialize(config["metric"])
		config["filter"] = pandas.Series(from_string(config["filter"]))

		return super(ZeroshotMetric, cls).from_config(config)

	def get_config(self) -> dict:
		"""Return the config dictionary for a `Metric` instance.

		Encode label subset to string storage format after saving loss.

		Returns:
			The config dictionary for a `Metric` instance.
		"""
		config = super(ZeroshotMetric, self).get_config()
		config.update(
			{
				"metric": tensorflow.keras.metrics.serialize(self.metric),
				"filter": to_string(self.filter.numpy()),
			}
		)

		return config
