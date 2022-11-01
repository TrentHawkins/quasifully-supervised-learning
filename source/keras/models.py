"""Custom keras models based on TensorFlow's functional API.

Dense layer stack.
-	Basically a sequential of dense layers with uniform settings across, apart for the top.

Dense layer stack array.
-	An colelction of dense sequentials used in parallel.
"""


from typing import Callable

import tensorflow

from .layers import BaseDense, AttentionDense
from ..numtools import divisors, hidden_dims


class DenseStack(tensorflow.keras.Model):
	"""A sequence of dense layers equiped with dropout and (optional) batch normalization and uniform activation throughout.

	The network complexity is defined in a specific way which based on inputs and output dimensionality:
	-	the hidden layers gradually change from one to the other with complexities defined by an integer divisor logic
	-	the depth of the change is adjustable
	"""

	def __init__(self,
		inputs_dim: int,
		output_dim: int,
		skip: int = 1,
		activation: Callable | str | None = None,
	#	regularizer: tensorflow.keras.regularizers.Regularizer | str | None = None,
	#	constraint: tensorflow.keras.constraints.Constraint | str | None = None,
		normalization: bool = False,
		dropout: float = .5,
		name: str = "dense_stack",
	**kwargs):
		"""Hyperparametrize dense layer stack.

		Arguments:
			inputs_dim: size of last inputs dimension
			output_dim: size of last output dimension

		keyword arguments:
			skip: the (inverse) depth of the dense layer stack
				default: no skipping (full depth)
			activation: of dense layers in dense layer stack
				default: none
			regularizer: on the weights of dense layers in dense layer stack
				default: Glorot uniform
			constraint: on the weights of dense layers in dense layer stack
				default: none
			normalization: whether to batch-nosmalize or not
				default: no batch-normalization
			dropout: dropout factor applied on input of dense layers in dense layer stack
				default: half
		"""
		super(DenseStack, self).__init__(
			name=name,
		**kwargs)

		self.denses = []

	#	Do not include the initial dimensionality of the input in hidden dimensionalities.
		for index, hidden_dim in enumerate(
			hidden_dims(
				inputs_dim,
				output_dim, skip=skip
			)[1:]
		):
			self.denses.append(
				BaseDense(hidden_dim,
					activation=activation,
				#	regularizer=regularizer,
				#	constraint=constraint,
					normalization=normalization,
					dropout=dropout,
					name=f"{name}_{index+1}",
				**kwargs)
			)

	def __len__(self):
		"""Get number of dense layers in model."""
		return len(self.denses)

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

	#	Mind that the dense layers are augmented with dropout and (optional) batch normalization.
		for dense in self.denses:
			x = dense(x,
				training=training,
			)

		return x


class DenseStackArray(tensorflow.keras.Model):
	"""A sequence of dense layer stacks equiped with dropout and (optional) batch normalization and uniform activation throughout.

	The number of dense layer stacks (threads) is defined by the inputs and output dimensionality.

	The network complexity is defined in a specific way which based on inputs and output dimensionality:
	-	the hidden layers gradually change from one to the other with complexities defined by an integer divisor logic
	-	the depth of the change is adjustable
	"""

	def __init__(self,
		inputs_dim: int,
		output_dim: int, threads: int | None = None,
		attention_activation: Callable | str | None = None,
		activation: Callable | str | None = None,
	#	regularizer: tensorflow.keras.regularizers.Regularizer | str | None = None,
	#	constraint: tensorflow.keras.constraints.Constraint | str | None = None,
		normalization: bool = False,
		dropout: float = .5,
		name: str = "dense_stack_array",
	**kwargs):
		"""Hyperparametrize dense layer stack.

		Arguments:
			inputs_dim: size of last inputs dimension
			output_dim: size of last output dimension

		keyword arguments:
			threads: the number of (parallel) dense layer stacks to build
				default: all of them
			attention_activation: activation applied on the attention dense combining the output from all stacks
		"""
		super(DenseStackArray, self).__init__(
			name=name,
		**kwargs)

		self.dense_stacks = []

		skips = divisors(len(hidden_dims(inputs_dim, output_dim)) - 1, reverse=True)

	#	Do not include the initial dimensionality of the input in hidden dimensionalities.
		for thread, skip in enumerate(skips[0:threads]):
			self.dense_stacks.append(
				DenseStack(
					inputs_dim,
					output_dim,
					skip=skip,
					activation=activation,
				#	regularizer=regularizer,
				#	constraint=constraint,
					normalization=normalization,
					dropout=dropout,
					name=f"{name}_{thread}",
				)
			)

	#	Attention dense to collate outputs from each dense layer stack.
		self.attention = AttentionDense(
			activation=attention_activation,
			name=f"{name}_attention",
		)

	def __len__(self):
		"""Get number of dense layer stacks in model."""
		return len(self.dense_stacks)

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

	#	Collate outputs from each dense layer stack.
		x = self.attention(
			tensorflow.stack(
				[
					dense_stack(x,
						training=training,
					) for dense_stack in self.dense_stacks
				],
				axis=-1,
			)
		)

		return x
