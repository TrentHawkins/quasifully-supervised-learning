"""Custom keras models based on TensorFlow's functional API.

Dense layer stack.
-	Basically a sequential of dense layers with uniform settings across, apart for the top.

Dense layer stack array.
-	An colelction of dense sequentials used in parallel.
"""


from typing import Callable

import tensorflow

from .layers import Dense, AttentionDense
from ..numtools import divisors, hidden_dims


def DenseStack(
	inputs_dim: int,
	output_dim: int,
	skip: int = 1,
	activation: Callable | str | None = None,
#	regularizer: tensorflow.keras.regularizers.Regularizer | str | None = None,
#	constraint: tensorflow.keras.constraints.Constraint | str | None = None,
	normalization: bool = False,
	dropout: float = .5,
	name: str = "dense_stack",
) -> tensorflow.keras.Model:
	"""Sequence dense layers equiped with dropout and (optional) batch normalization and uniform activation throughout.

	The network complexity is defined in a specific way which based on inputs and output dimensionality:
	-	the hidden layers gradually change from one to the other with complexities defined by an integer divisor logic
	-	the depth of the change is adjustable

	Arguments:
		inputs_dim: size of last inputs dimension
		output_dim: size of last output dimension

	keyword arguments:
		skip: the (inverse) depth of the dense layer stack
			default: no skipping (full depth)
		activation: of dense layers in dense layer stack
			default: linear
		regularizer: on the weights of dense layers in dense layer stack
			default: Glorot uniform
		constraint: on the weights of dense layers in dense layer stack
			default: none
		normalization: whether to batch-nosmalize or not
			default: no batch-normalization
		dropout: dropout factor applied on input of dense layers in dense layer stack
			default: half

	Returns:
		Sequential model with predefined dense layers
	"""
	model = tensorflow.keras.models.Sequential(name=name)

#	Input specification:
	model.add(
		tensorflow.keras.layers.InputLayer(
			input_shape=(
				inputs_dim,
			),
		#	batch_size=None,
		#	dtype=None,
		#	input_tensor=None,
		#	sparse=None,
		#	name=None,
		#	ragged=None,
		#	type_spec=None,
		)
	)

#	Layers:
	for index, hidden_dim in enumerate(
		hidden_dims(
			inputs_dim,
			output_dim, skip=skip
		)[1:]
	):
		model.add(
			Dense(hidden_dim,
				activation=activation,
			#	regularizer=regularizer,
			#	constraint=constraint,
				normalization=normalization,
				dropout=dropout,
				name=f"{name}_{index+1}",
			)
		)

	return model


def DenseStackArray(
	inputs_dim,
	output_dim,
	threads: int | None = None,
	attention_activation: Callable | str | None = None,
	activation: Callable | str | None = None,
#	regularizer: tensorflow.keras.regularizers.Regularizer | str | None = None,
#	constraint: tensorflow.keras.constraints.Constraint | str | None = None,
	normalization: bool = False,
	dropout: float = .5,
	name: str = "dense_stack_array",
) -> tensorflow.keras.Model:
	"""Sequence dense layer stacks equiped with dropout and (optional) batch normalization and uniform activation throughout.

	The number of dense layer stacks (threads) is defined by the inputs and output dimensionality.

	The network complexity is defined in a specific way which based on inputs and output dimensionality:
	-	the hidden layers gradually change from one to the other with complexities defined by an integer divisor logic
	-	the depth of the change is adjustable

	Arguments:
		inputs_dim: size of last inputs dimension
		output_dim: size of last output dimension

	keyword arguments:
		threads: the number of (parallel) dense layer stacks to build
			default: all of them
		attention_activation: activation applied on the attention dense combining the output from all stacks
			default: linear

	Returns:
		Funnctional model with predefined Sequential threads
	"""
	sequentials = []

#	Thread specification:
	skips = divisors(len(hidden_dims(inputs_dim, output_dim)) - 1, reverse=True)

#	Input specification:
	inputs = tensorflow.keras.Input(
		shape=(
			inputs_dim,
		),
	#	batch_size=None,
	#	name=None,
	#	dtype=None,
	#	sparse=None,
	#	tensor=None,
	#	ragged=None,
	#	type_spec=None,
		name=f"{name}_input"
	)

#	Threads:
	for thread, skip in enumerate(skips[0:threads]):
		sequentials.append(
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
	attention = AttentionDense(
		activation=attention_activation,
		name=f"{name}_attention",
	)

#	Call:
	outputs = attention(
		tensorflow.stack([sequential(inputs) for sequential in sequentials],
			axis=-1,
			name=f"{name}_collation",
		)
	)

	return tensorflow.keras.Model(
		inputs=inputs,
		outputs=outputs,
		name=name,
	)
