"""Custom keras layers for model building.

Compound dense layer containing:
-	batch normalization
-	dropout
-	dense sublayer

Basic attention layer assigning a trainable weight for each input in an array of inputs.

Metric sigmoid-like non-linear dense layers using similarity metrics in their definition:
-	based on the jaccard similarity metric (sigmoid-like only on sigmoid-like features)
-	based on the cosine similarity metric (sigmoid-like always)
"""


from typing import Callable

import tensorflow

from ..seed import SEED
from .utils.layer_utils import print_summary


class Dense(tensorflow.keras.layers.Dense):
	"""Custom compound dense layer.

	Attribures:
		dense: the dense sublayer
	"""

	def __init__(self, units: int,
		activation: Callable | str | None = None,
	#	regularizer: tensorflow.keras.regularizers.Regularizer | str | None = None,
	#	constraint: tensorflow.keras.constraints.Constraint | str | None = None,
		normalization: bool = False,
		dropout: float = .5,
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
			normalization: whether to batch-nosmalize or not
				default: no batch-normalization
			dropout: dropout factor applied on input of the layer
				default: half
		"""
		super(Dense, self).__init__(units,
				activation=activation,
			#	use_bias=True,
			#	kernel_initializer="glorot_uniform",
			#	bias_initializer="zeros",
			#	kernel_regularizer=regularizer,
			#	bias_regularizer=regularizer,
			#	activity_regularizer=regularizer,
			#	kernel_constraint=constraint,
			#	bias_constraint=constraint,
				name=name,  # None
		**kwargs)

	#	batch-normalization
		if normalization:
			self.normalization = tensorflow.keras.layers.BatchNormalization(
			#	axis=-1,
			#	momentum=0.99,
			#	epsilon=0.001,
			#	center=True,
			#	scale=True,
			#	beta_initializer="zeros",
			#	gamma_initializer="ones",
			#	moving_mean_initializer="zeros",
			#	moving_variance_initializer="ones",
			#	beta_regularizer=regularizer,
			#	gamma_regularizer=regularizer,
			#	beta_constraint=constraint,
			#	gamma_constraint=constraint,
				name=f"normalization_{name}",
			**kwargs)

	#	dropout
		assert dropout >= 0. and dropout <= 1.
		self.dropout = tensorflow.keras.layers.Dropout(dropout,
			noise_shape=None,
			seed=SEED,  # None,
			name=f"dropout_{name}",  # None
		**kwargs)

	def build(self, input_shape):
		"""Create the variables of the layer (optional, for subclass implementers).

		This is a method that implementers of subclasses of `Layer` or `Model` can override
		if they need a state-creation step in-between layer instantiation and layer call.

		It is invoked automatically before the first execution of `call()`.

		This is typically used to create the weights of `Layer` subclasses (at the discretion of the subclass implementer).

		Arguments:
			input_shape: instance of `TensorShape` or list of instances of `TensorShape` if the layer expects a list of inputs
		"""
		super(Dense, self).build(input_shape)

	#	batch-normalization
		try:
			self.normalization.build(input_shape)

		except AttributeError:
			pass

	#	dropout
		self.dropout.build(input_shape)

	def call(self, inputs: tensorflow.Tensor,
		training: bool | None = None,
	):
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

	#	batch-normalization
		try:
			x = self.normalization(x,
				training=training,
			)

		except AttributeError:
			pass

	#	dropout
		x = self.dropout(x,
			training=training,
		)

		return super(Dense, self).call(x)


"""What follows are special types of dense layers."""


class Attention(tensorflow.keras.layers.Dense):
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
		super(Attention, self).__init__(1,
			activation=activation,
			use_bias=False,
		#	kernel_initializer="glorot_uniform",
		#	bias_initializer="zeros",
		#	kernel_regularizer=regularizer,
		#	bias_regularizer=regularizer,
		#	activity_regularizer=regularizer,
		#	kernel_constraint=constraint,
		#	bias_constraint=constraint,
			name=name,
		**kwargs)

	def call(self, inputs):
		"""Call the model on new inputs.

		In this case call just reapplies all ops in the graph to the new inputs.

		Arguments:
			inputs: a tensor or list of tensors

		Returns:
			output of layer
		"""
		return tensorflow.squeeze(super(Attention, self).call(inputs),
			axis=-1,
		)


class Metric(tensorflow.keras.layers.Dense):
	"""A non-linear dense layer emulating the action of a metric and outputing in sigmoid range.

	Such a modified layer has no bias explicitely.
	"""

	def __init__(self, units,
		activation: Callable | str | None = None,
		kernel_initializer: tensorflow.keras.initializers.Initializer | str = "glorot_uniform",
		name: str = "metric",
	**kwargs):
		"""Hyperparametrize recombination layer.

		No activation is needed as these metric layers are manifestly non-linear to begin with.
		Option is left (and ignored) for compatibility with other dense-like layers.

		Keyword arguments:
			activation: to apply on output of decision
			kernel_initializer: weight values to begin with
		"""
		super(Metric, self).__init__(units,
			activation=activation,
			use_bias=False,
			kernel_initializer=kernel_initializer,  # type: ignore
			bias_initializer="zeros",
		#	kernel_regularizer=regularizer,
		#	bias_regularizer=regularizer,
		#	activity_regularizer=regularizer,
		#	kernel_constraint=constraint,
		#	bias_constraint=constraint,
			name=name,
		**kwargs)

	def build(self, input_shape):
		"""Create the variables of the layer (optional, for subclass implementers).

		This is a method that implementers of subclasses of `Layer` or `Model` can override
		if they need a state-creation step in-between layer instantiation and layer call.

		It is invoked automatically before the first execution of `call()`.

		This is typically used to create the weights of `Layer` subclasses (at the discretion of the subclass implementer).

		Arguments:
			input_shape: instance of `TensorShape` or list of instances of `TensorShape` if the layer expects a list of inputs
		"""
		super(Metric, self).build(input_shape)

	#	the norms of kernel vectors (diagonal)
		self.kernel_kernel = tensorflow.einsum("...ji, ...ji -> ...i",
			self.kernel,
			self.kernel,
		)


class Jaccard(Metric):
	"""A dense layer that perform the jaccard operation per input and kernel vector instead of a dot product.

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
		inputs_kernel = super(Jaccard, self).call(inputs)

	#	the norms of inputs vectors
		inputs_inputs = tensorflow.einsum("...i, ...i -> ...",
			inputs,
			inputs,
		)

		return inputs_kernel / (
			tensorflow.expand_dims(inputs_inputs, -1) +
			tensorflow.broadcast_to(self.kernel_kernel, tensorflow.shape(inputs_kernel)) +
			inputs_kernel
		)


class Cosine(Metric):
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
		inputs_kernel = super(Cosine, self).call(inputs)

	#	the norms of inputs vectors
		inputs_inputs = tensorflow.einsum("...i, ...i -> ...",
			inputs,
			inputs,
		)

		return inputs_kernel / tensorflow.math.sqrt(
			tensorflow.expand_dims(inputs_inputs, -1) *
			tensorflow.broadcast_to(self.kernel_kernel, tensorflow.shape(inputs_kernel))
		)
