"""Custom keras layers for model building.

Compound dense layer containing:
-	dropout
-	dense sublayer

Basic attention layer assigning a trainable weight for each input in an array of inputs.

Metric sigmoid-like non-linear dense layers using similarity metrics in their definition:
-	based on the jaccard similarity metric (sigmoid-like only on sigmoid-like features)
-	based on the cosine similarity metric (sigmoid-like always)
"""


from typing import Callable

import tensorflow

from .utils.layer_utils import print_summary


@tensorflow.keras.utils.register_keras_serializable("source>keras>layers")
class DropoutDense(tensorflow.keras.layers.Dense):
	"""Custom compound dense layer.

	Attribures:
		dense: the dense sublayer
	"""

	def __init__(self, units: int,
		activation: Callable | str | None = None,
	#	regularizer: tensorflow.keras.regularizers.Regularizer | str | None = None,
	#	constraint: tensorflow.keras.constraints.Constraint | str | None = None,
		dropout_rate: float = .5,
		seed: int = 0,
		name: str = "base_dense",
	**kwargs):
		"""Hyrparametrize base layer with dense topping.

		Arguments:
			units: number of neurons in layer

		Keyword arguments:
			activation: of layer
				default: none
			regularizer: on the weights of the layer
				default: Glorot uniform
			constraint: on the weights of the layer
				default: none
			dropout: dropout factor applied on input of the layer
				default: half
			seed: dropout seed
				default: 0
		"""
		super(DropoutDense, self).__init__(kwargs.pop("units", None) or units,
			activation=kwargs.pop("activation", None) or activation,
		#	use_bias=True,
		#	kernel_initializer="glorot_uniform",
		#	bias_initializer="zeros",
		#	kernel_regularizer=regularizer,
		#	bias_regularizer=regularizer,
		#	activity_regularizer=regularizer,
		#	kernel_constraint=constraint,
		#	bias_constraint=constraint,
			name=kwargs.pop("name", None) or name,
		**kwargs)

	#	dropout:
		assert dropout_rate >= 0. and dropout_rate <= 1.
		self.dropout = tensorflow.keras.layers.Dropout(dropout_rate,
			noise_shape=None,
			seed=seed,  # None,
			name=f"dropout_{name}",  # None
		)

	def build(self, input_shape: tensorflow.TensorShape):
		"""Create the variables of the layer (optional, for subclass implementers).

		This is a method that implementers of subclasses of `Layer` or `Model` can override
		if they need a state-creation step in-between layer instantiation and layer call.

		It is invoked automatically before the first execution of `call()`.

		This is typically used to create the weights of `Layer` subclasses (at the discretion of the subclass implementer).

		Arguments:
			input_shape: instance of `TensorShape` or list of instances of `TensorShape` if the layer expects a list of inputs
		"""
		super(DropoutDense, self).build(input_shape)

	#	dropout:
		self.dropout.build(input_shape)

	def call(self, inputs: tensorflow.Tensor,
		training: bool | None = None,
	) -> tensorflow.Tensor:
		"""Call the model on new inputs.

		In this case call just reapplies all ops in the graph to the new inputs.

		Arguments:
			inputs: a tensor or list of tensors

		Keyword arguments:
			training: boolean indicating whether to run the network in training mode or inference mode

		Returns:
			output of layer
		"""
		x = inputs

	#	dropout:
		x = self.dropout(x,
			training=training,
		)

		return super(DropoutDense, self).call(x)


"""What follows are special types of dense layers."""


@tensorflow.keras.utils.register_keras_serializable("source>keras>layers")
class AttentionDense(tensorflow.keras.layers.Dense):
	"""Wrapper for dense layer operating on a stacks of input to recombine them with attention.

	Such a layer is expected to have no bias and be trainable with no dropout.
	Other dense features include activation only.

	Call:
		stack: stack inputs horizontally
		dense: one weight for each input
		squeeze: eliminate redudant dims on output
	"""

	def __init__(self,
		activation: Callable | str | None = None,
		name: str = "attention",
	**kwargs):
		"""Hyperparametrize recombination layer.

		Keyword arguments:
			activation: to apply on output of decision
		"""
		super(AttentionDense, self).__init__(kwargs.pop("units", None) or 1,
			activation=kwargs.pop("activation", None) or activation,
			use_bias=kwargs.pop("use_bias", None) or False,
			kernel_initializer=kwargs.pop("kernel_initializer", None) or "zeros",
			bias_initializer=kwargs.pop("bias_initializer", None) or "zeros",
		#	kernel_regularizer=regularizer,
		#	bias_regularizer=regularizer,
		#	activity_regularizer=regularizer,
		#	kernel_constraint=constraint,
		#	bias_constraint=constraint,
			name=kwargs.pop("name", None) or name,
		**kwargs)

	def call(self, inputs):
		"""Call the model on new inputs.

		In this case call just reapplies all ops in the graph to the new inputs.

		Arguments:
			inputs: a tensor or list of tensors

		Returns:
			output of layer
		"""
		return tensorflow.squeeze(super(AttentionDense, self).call(inputs),
			axis=-1,
		)


@tensorflow.keras.utils.register_keras_serializable("source>keras>layers")
class MetricDense(tensorflow.keras.layers.Dense):
	"""A non-linear dense layer emulating the action of a metric and outputing in sigmoid range.

	Such a modified layer has no bias explicitely.
	"""

	def __init__(self, units,
		activation: Callable | str | None = None,
		kernel: tensorflow.Tensor | None = None,
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


@tensorflow.keras.utils.register_keras_serializable("source>keras>layers")
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


@tensorflow.keras.utils.register_keras_serializable("source>keras>layers")
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


@tensorflow.keras.utils.register_keras_serializable("source>keras>layers")
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


@tensorflow.keras.utils.register_keras_serializable("source>keras>layers")
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
