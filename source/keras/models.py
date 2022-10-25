from ..numtools import divisors
from .layers import AttentionDense
import tensorflow as tf

from ..numtools import hidden_dims
from .layers import BaseDense


class DenseStack(tf.keras.Model):
    """	Stack of dense layers.

    The last layer is omitted, to allow external control on the stack's topping.
    The first layer is also omitted to avoid repeating the input dimensionality plus making the dense layer simpler.
    This means that inputs_dim and output_dim do not have intermediate divisors the resulting model will be empty (None).
    Always make sure to top this stack with a dense layer to make sure you connect to something.
    The topping is delegated to the caller, just in case they wish to use one dense layer for several dense stacks in parallel.
    This allows for avoiding output dimensionality incompatibilities too in case of a non-divisor skip.

    Attributes:
            denses: list of layers to be stacked
    """

    def __init__(self, inputs_dim: int, output_dim: int,
            skip: int = 1,
            regularizer: str = None,
            activation: str = None,
            dropout: float = None,
            normalization: bool = False,
            name: str = "feedforward_stack",
    **kwargs):
        """	Hyperparametrize stack of dense layers.

        Arguments:
                inputs_dim: size of input
                output_dim: size of output
                skip: the lower the index the deeper the model in general
                        default: deepest allowable model
        """

    #	skip must be a divisor of the full input-to-output architecture
        assert skip in divisors(len(hidden_dims(inputs_dim, output_dim)) - 1)

        super(DenseStack, self).__init__(
            name=name,
        **kwargs)

        self.hidden_dims = hidden_dims(inputs_dim, output_dim,
                skip=skip,
                                       )
        self.denses = []

        for index, (hidden_inputs_dim, hidden_output_dim) in enumerate(zip(
            self.hidden_dims[:-1],
            self.hidden_dims[1:],
        )
        ):
            self.denses.append(BaseDense(hidden_output_dim,
                            regularizer=regularizer,
                            activation=activation,
                            dropout=dropout,
                            normalization=normalization,
                            name=f"{name}_{index+1}",
                            input_size=hidden_inputs_dim,
                                         )
                               )

    #	self.layers=self.denses

    #	mock-call to calculate shape
        self(tf.keras.Input(
            shape=(
                inputs_dim,
            ),
            #	batch_size=None,
            name="input",
                #	dtype=None,
                #	sparse=None,
                #	tensor=None,
                #	ragged=None,
                #	type_spec=None,
        )
        )

    def __len__(self):
        """ Count stacks on array, giving array width.
        """

        return len(self.denses)  # accountin for the missing top layer

    def call(self, inputs,
            training: bool = False,
             ):
        """	Calls the model on new inputs.

        In this case call just reapplies all ops in the graph to the new inputs.

        Arguments:
                inputs: a tensor or list of tensors
                training: boolean indicating whether to run the Network in training mode or inference mode

        Returns:
                outputs of layer
        """

        x = inputs

        for dense in self.denses:
            x = dense(x,
                    training=training,  # mind to delegate this flag to operations that change on inference mode
                      )

        return x


class DenseStackArray(tf.keras.Model):
    """	Array of stacks of dense layers.

    The recombination of threads is enhanced from plain summation to a "lateral" `Dense`, much like an attention mechanism.

    Attributes:
            dense_stacks: list of stacks of dense layers to be parallelized
            topping: last dense layer connected to all threads
    """

    def __init__(self, inputs_dim: int, output_dim: int,
            threads: int = None,
            regularizer: str = None,
            activation: str = None,
            top_activation: str = None,
            dropout: float = None,
            normalization: bool = False,
            name: str = "feedforward_stack_array",
    **kwargs):
        """	Hyperparametrize array of stacks of dense layers.

        Arguments:
                inputs_dim: size of input
                output_dim: size of output
                threads: how many threads to use starting from the simplest
        """

        super(DenseStackArray, self).__init__(
            name=name,
        **kwargs)

        self.skips = divisors(len(hidden_dims(inputs_dim, output_dim)) - 1)
        self.dense_stacks = []

        for index, skip in enumerate(list(reversed(self.skips))[0:threads]):
            self.dense_stacks.append(
                DenseStack(inputs_dim, output_dim,
                            skip=skip,
                            regularizer=regularizer,
                            activation=activation,
                            dropout=dropout,
                            normalization=normalization,
                            name=f"{name}_{index+1}"
                           )
            )

        self.top = AttentionDense(
            activation=top_activation,
            name=f"{name}_top"
        )

    #	self.layers=self.dense_stacks+[self.top]

    #	mock-call to calculate shape
        self(tf.keras.Input(
            shape=(
                inputs_dim,
            ),
            #	batch_size=None,
            name="input",
                #	dtype=None,
                #	sparse=None,
                #	tensor=None,
                #	ragged=None,
                #	type_spec=None,
        )
        )
    #	self.top([tf.keras.Input(
    #				shape=(
    #					output_dim,
    #				),
    #			#	batch_size=None,
    #				name="input",
    #			#	dtype=None,
    #			#	sparse=None,
    #			#	tensor=None,
    #			#	ragged=None,
    #			#	type_spec=None,
    #			)
    #		]*len(self)
    #	)

    def __len__(self):
        """ Count stacks on array, giving array width.
        """

        return len(self.dense_stacks)

    def call(self, inputs,
            training: bool = False,
             ):
        """	Calls the model on new inputs.

        In this case call just reapplies all ops in the graph to the new inputs.

        Arguments:
                inputs: a tensor or list of tensors
                training: boolean indicating whether to run the Network in training mode or inference mode

        Returns:
                outputs of layer
        """

        x = inputs

        x = self.top([dense_stack(x,
                                training=training,  # mind to delegate this flag to operations that change on inference mode
                                  ) for dense_stack in self.dense_stacks
                      ]
                     )

        return x
