"""Custom keras layers for model building.

Compound dense layer containing:
-	batch normalization
-	dropout
-	dense sublayer

Basic attention layer assigning a trainable weight for each input in an array of inputs.
"""


from typing import Callable

import tensorflow

from ..seed import SEED


class BaseLayer(tensorflow.keras.layers.Layer):
	"""Custom base layer equipped with (optional) batch normalization and dropout.

	Attributes:
		normalization: batch normalization of layer
		dropout: layer with optionally fixed random seeding
		layer: to modify with normalization and dropout
	"""

	def __init__(self, layer: tensorflow.keras.layers.Layer,
		normalization: bool = False,
		dropout: float = .5,
		name: str = "base_layer",
	**kwargs):
		"""Hyperparametrize custom base layer.

		Arguments:
			layer: to modify with normalization and dropout

		Keyword Arguments:
			normalization: whether to batch-nosmalize or not
				default: no batch-normalization
			dropout: dropout factor applied on input of the dense layer
				default: half
		"""
		super(BaseLayer, self).__init__(
			name=name,
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

	#	layer to modify with normalization and dropout
		self.layer = layer

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

	#	layer to modify with normalization and dropout
		x = self.layer(x,
			training=training,
		)

		return x


class BaseDense(BaseLayer):
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
		#	regularizer: on the weights of the layer
		#	constraint: on the weights of the layer
			activation: of layer
				default: none
		"""
		super(BaseDense, self).__init__(
			dropout=dropout,
			normalization=normalization,
			name=name,
		**kwargs)

		assert units > 0
		self.dense = tensorflow.keras.layers.Dense(units,
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
		)

	def call(self, inputs,
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
		return self.dense(
			super().call(inputs,
				training=training,
			)
		)


"""What follows are special types of dense layers."""


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
		super(AttentionDense, self).__init__(1,
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
		return tensorflow.squeeze(
			super(AttentionDense, self).call(
				tensorflow.stack(inputs,
					axis=-1,
				)
			),
			axis=-1,
		)


class JaccardDense(tensorflow.keras.layers.Dense):
	"""A dense layer that perform the jaccard operation per input and kernel vector instead of a dot product.

	Such a modified layer has no bias explicitely.
	"""

	def __init__(self,
		activation: Callable | str | None = None,
		kernel_initializer: tensorflow.keras.initializers.Initializer | str = "glorot_uniform",
		name: str = "jaccard",
	**kwargs):
		"""Hyperparametrize recombination layer.

		Keyword arguments:
			activation: to apply on output of decision
			kernel_initializer: weight values to begin with
		"""
		super(JaccardDense, self).__init__(1,
			activation=activation,
			use_bias=False,
			kernel_initializer=kernel_initializer,
			bias_initializer="zeros",
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
		dot_product = super(JaccardDense, self).call(inputs)

	#	the norm of inputs vector
		inputs_norm = tensorflow.tensordot(
			inputs,
			inputs, 1
		)

	#	the norm of kernel vectors
		kernel_norm = tensorflow.linalg.trace(
			tensorflow.linalg.matmul(
				super(JaccardDense, self).kernel,
				super(JaccardDense, self).kernel,

			#	transpose_a=True,
				transpose_b=True,
			)
		)

		return dot_product / (
			inputs_norm +
			kernel_norm + dot_product
		)
