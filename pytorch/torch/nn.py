"""Custom modules.

Compound dense layer containing:
-	dropout
-	dense sublayer

Basic attention layer assigning a trainable weight for each input in an array of inputs.

Metric sigmoid-like non-linear dense layers using similarity metrics in their definition:
-	based on the jaccard similarity metric (sigmoid-like only on sigmoid-like features)
-	based on the cosine similarity metric (sigmoid-like always)

Dense layer stack.
-	Basically a sequential of dense layers with uniform settings across, apart for the top.

Dense layer stack array.
-	An colelction of dense sequentials used in parallel.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import tensorflow
import torch


class DropoutDense(torch.nn.Linear):
	"""Custom compound dense layer.

	Attribures:
		dense: the dense sublayer
	"""

	def __init__(self, input_size: int, output_size: int,
		dropout_rate: float = .5,
	**kwargs):
		"""Hyrparametrize base layer with dense topping.

		Arguments:
			units: number of neurons in layer

		Keyword arguments:
			dropout: dropout factor applied on input of the layer
				default: half
		"""
		super(DropoutDense, self).__init__(input_size, output_size,
		#	bias=True,
		**kwargs)

	#	dropout:
		assert dropout_rate >= 0. and dropout_rate <= 1.
		self.dropout = torch.nn.Dropout(dropout_rate)

	#	activation:
		self.activation = torch.nn.SiLU()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Call the model on new inputs.

		In this case call just reapplies all ops in the graph to the new inputs.

		Arguments:
			inputs: a tensor or list of tensors

		Returns:
			output of layer
		"""
		return self.activation(super(DropoutDense, self).forward(self.dropout(x)))


"""What follows are special types of dense layers."""


class AttentionDense(torch.nn.Linear):
	"""Wrapper for dense layer operating on a stacks of input to recombine them with attention.

	Such a layer is expected to have no bias and be trainable with no dropout.
	Other dense features include activation only.

	Call:
		stack: stack inputs horizontally
		dense: one weight for each input
		squeeze: eliminate redudant dims on output
	"""

	def __init__(self, threads: int, **kwargs):
		"""Hyperparametrize recombination layer.

		Keyword arguments:
			activation: to apply on output of decision
		"""
		super(AttentionDense, self).__init__(threads, 1,
			bias=False,
		**kwargs)

	def call(self, x: torch.Tensor) -> torch.Tensor:
		"""Call the model on new inputs.

		In this case call just reapplies all ops in the graph to the new inputs.

		Arguments:
			inputs: a tensor or list of tensors

		Returns:
			output of layer
		"""
		return super(AttentionDense, self).forward(x).squeeze(
			dim=-1,
		)


class MetricDense(tensorflow.keras.layers.Dense):
	"""A non-linear dense layer emulating the action of a metric and outputing in sigmoid range.

	Such a modified layer has no bias explicitely.
	"""

	def __init__(self, units,
		activation: Optional[Union[Callable, str]] = None,
		kernel: Optional[tensorflow.Tensor] = None,
		name: str = "metric",
	**kwargs):
		"""Hyperparametrize recombination layer.

		No activation is needed as these metric layers are manifestly non-linear to begin with.
		Option is left (and ignored) for compatibility with other dense-like layers.

		Keyword arguments:
			activation: to apply on output of decision
			kernel: weight values to begin with
		"""
		super(MetricDense, self).__init__(kwargs.pop("units", None) or units,
			activation=kwargs.pop("activation", None) or activation,
			use_bias=kwargs.pop("use_bias", None) or False,
		#	kernel_initializer=tensorflow.keras.initializers.Constant(kernel),  # type: ignore
		#	bias_initializer="zeros",
		#	kernel_regularizer=regularizer,
		#	bias_regularizer=regularizer,
		#	activity_regularizer=regularizer,
		#	kernel_constraint=constraint,
		#	bias_constraint=constraint,
			name=kwargs.pop("name", None) or name,
		**kwargs)

		self.initial_kernel = kernel

	def build(self, input_shape: tensorflow.TensorShape):
		"""Custtomize kernel."""
		super(MetricDense, self).build(input_shape)

		kernel, *weights = self.get_weights()

		if self.initial_kernel is not None and kernel.shape == self.initial_kernel.shape:
			self.set_weights([self.initial_kernel, *weights])


class CosineDense(MetricDense):
	"""A dense layer that perform the cosine operation per input and kernel vector instead of a dot product.

	Such a modified layer has no bias explicitely.
	"""

	def call(self, inputs):
		"""Call the model on new inputs.

		In this case call just reapplies all ops in the graph to the new inputs.

		Arguments:
			inputs: a tensor or list of tensors

		Returns:
			output of layer
		"""
		inputs_kernel = super(CosineDense, self).call(inputs)

	#	the norms of inputs vectors:
		inputs_inputs = tensorflow.expand_dims(
			tensorflow.einsum("...i, ...i -> ...",
				inputs,
				inputs,
			), -1
		)

	#	the norms of kernel vectors (diagonal):
		kernel_kernel = tensorflow.broadcast_to(
			tensorflow.einsum("...ji, ...ji -> ...i",
				self.kernel,
				self.kernel,
			), tensorflow.shape(inputs_kernel)
		)

	#	denominator:
		denominator = tensorflow.math.sqrt(
			inputs_inputs *
			kernel_kernel
		)

		return inputs_kernel / denominator


class JaccardDense(MetricDense):
	"""A dense layer that perform the Jaccard operation per input and kernel vector instead of a dot product.

	Such a modified layer has no bias explicitely.
	"""

	def call(self, inputs):
		"""Call the model on new inputs.

		In this case call just reapplies all ops in the graph to the new inputs.

		Arguments:
			inputs: a tensor or list of tensors

		Returns:
			output of layer
		"""
		inputs_kernel = super(JaccardDense, self).call(inputs)

	#	the norms of inputs vectors:
		inputs_inputs = tensorflow.expand_dims(
			tensorflow.einsum("...i, ...i -> ...",
				inputs,
				inputs,
			), -1
		)

	#	the norms of kernel vectors (diagonal):
		kernel_kernel = tensorflow.broadcast_to(
			tensorflow.einsum("...ji, ...ji -> ...i",
				self.kernel,
				self.kernel,
			), tensorflow.shape(inputs_kernel)
		)

	#	denominator:
		denominator = (
			inputs_inputs +
			kernel_kernel -
			inputs_kernel
		)

		return inputs_kernel / denominator


class DiceDense(MetricDense):
	"""A dense layer that perform the Dice operation per input and kernel vector instead of a dot product.

	Such a modified layer has no bias explicitely.
	"""

	def call(self, inputs):
		"""Call the model on new inputs.

		In this case call just reapplies all ops in the graph to the new inputs.

		Arguments:
			inputs: a tensor or list of tensors

		Returns:
			output of layer
		"""
		inputs_kernel = super(DiceDense, self).call(inputs)

	#	the norms of inputs vectors:
		inputs_inputs = tensorflow.expand_dims(
			tensorflow.einsum("...i, ...i -> ...",
				inputs,
				inputs,
			), -1
		)

	#	the norms of kernel vectors (diagonal):
		kernel_kernel = tensorflow.broadcast_to(
			tensorflow.einsum("...ji, ...ji -> ...i",
				self.kernel,
				self.kernel,
			), tensorflow.shape(inputs_kernel)
		)

	#	denominator:
		denominator = (
			inputs_inputs +
			kernel_kernel
		) / 2

		return inputs_kernel / denominator


class RandDense(MetricDense):
	"""A dense layer that perform the Rand operation per input and kernel vector instead of a dot product.

	Such a modified layer has no bias explicitely.
	"""

	def call(self, inputs):
		"""Call the model on new inputs.

		In this case call just reapplies all ops in the graph to the new inputs.

		Arguments:
			inputs: a tensor or list of tensors

		Returns:
			output of layer
		"""
		inputs_kernel = super(RandDense, self).call(inputs)
		inputs_kernel_complement = 1 - inputs_kernel

	#	the norms of inputs vectors:
		inputs_inputs = tensorflow.expand_dims(
			tensorflow.einsum("...i, ...i -> ...",
				inputs,
				inputs,
			), -1
		)

	#	the norms of kernel vectors (diagonal):
		kernel_kernel = tensorflow.broadcast_to(
			tensorflow.einsum("...ji, ...ji -> ...i",
				self.kernel,
				self.kernel,
			), tensorflow.shape(inputs_kernel)
		)

	#	numerator:
		numerator = (
			inputs_kernel +
			inputs_kernel_complement
		)

	#	denominator:
		denominator = (
			inputs_inputs +
			kernel_kernel -
			inputs_kernel +
			inputs_kernel_complement
		)

		return numerator / denominator
