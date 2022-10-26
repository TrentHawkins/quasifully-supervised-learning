"""Custom keras layers for model building.

Compound dense layer containing:
-	batch normalization
-	dropout
-	dense sublayer

Basic attention layer assigning a trainable weight for each input in an array of inputs.
"""


import tensorflow

from ..seed import SEED


class BaseLayer(tensorflow.keras.layers.Layer):
	"""Custom base layer equipped with (optional) batch normalization and dropout.

	Attributes:
		dropout: layer with optionally fixed random seeding
		activation: optional activation to layer
		normalization: batch normalization of layer
	"""

	def __init__(self,
		normalization: bool = False,
		dropout: float = .5,
		name: str = "base_layer",
	**kwargs):
		"""Hyperparametrize custom base layer.

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
			#	beta_initializer='zeros',
			#	gamma_initializer='ones',
			#	moving_mean_initializer='zeros',
			#	moving_variance_initializer='ones',
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

		return x


class BaseDense(BaseLayer):
	"""Custom compound dense layer.

	Attribures:
		dense: the dense sublayer
	"""

	def __init__(self, units: int,
			activation: str | None = None,
			regularizer: str | None = None,
			normalization: bool = False,
			dropout: float = .5,
			name: str = "base_dense",
	**kwargs):
		"""Hyrparametrize base layer with dense topping.

		Arguments:
			units: number of neurons in layer

		Keyword arguments:
			regularizer: on the weights of the layer
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
		#	kernel_initializer='glorot_uniform',
		#	bias_initializer='zeros',
			kernel_regularizer=regularizer,
			bias_regularizer=regularizer,
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
		x = inputs

	#	base layer
		x = super().call(x,
				training=training,
		)

	#	dense layer
		x = self.dense(x)

		return x


class AttentionDense(tensorflow.keras.layers.Layer):
	"""Wrapper for dense layer operating on a stacks of input to recombine them with attention.

	Such a layer is expected to have no bias and be trainable with no dropout.
	Other dense features include activation only.

	Attributes:
		stack: stack inputs horizontally
		dense: one weight for each input
		squeeze: eliminate redudant dims on output
	"""

	def __init__(self,
		activation: str = "linear",
		name: str = "attention",
	**kwargs):
		"""Hyperparametrize recombination layer.

		Keyword arguments:
			activation: to apply on output of decision
		"""
		super(AttentionDense, self).__init__(
			name=name,
		**kwargs)

		self.dense = tensorflow.keras.layers.Dense(1,
			activation=activation,
			use_bias=False,
		#	kernel_initializer='glorot_uniform',
		#	bias_initializer='zeros',
		#	kernel_regularizer=regularizer,
		#	bias_regularizer=regularizer,
		#	activity_regularizer=regularizer,
		#	kernel_constraint=constraint,
		#	bias_constraint=constraint,
			name=name,  # None
		)

	def call(self, inputs):
		"""Call the model on new inputs.

		In this case call just reapplies all ops in the graph to the new inputs.

		Arguments:
			inputs: a tensor or list of tensors

		Returns:
			output of layer
		"""
		x = inputs

	#	recombination layer
		x = tensorflow.squeeze(self.dense(tensorflow.stack(x,
					axis=-1,
				)
			),
			axis=-1,
		)

		return x