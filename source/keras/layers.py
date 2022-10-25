import tensorflow as tf

from ..seed import SEED


class BaseLayer(tf.keras.layers.Layer):
    """	Base layer foundation equipped with dropout, activation and batch normalization.

    Attributes:
            dropout: layer with optionally fixed random seeding
            activation: optional activation to layer
            normalization: batch normalization of layer
    """

    def __init__(self,
            dropout: float = .5,
            normalization: bool = False,
            name: str = "base_layer",
    **kwargs):
        """	Hyperparametrize custom layer foundation.

        Keyword Arguments:
                dropout: dropout factor applied on input of the dense layer
                normalization: whether to batch-nosmalize or not
                        default: no batch-normalization
                name: optional name layers distinctifying this layer
        """

        super(BaseLayer, self).__init__(
            name=name,
        **kwargs)

    #	dropout
        assert dropout >= 0. and dropout <= 1.
        self.dropout = tf.keras.layers.Dropout(dropout,
        #	noise_shape=None,
                seed=SEED,  # None,
                name=f"dropout_{name}",  # None
                                                )
#	self.layers.append(self.dropout)

    #	batch-normalization
        if normalization:
            self.normalization = tf.keras.layers.BatchNormalization(
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
            )
        #	self.layers.append(self.normalization)

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

        try:
            x = self.dropout(x,
                    training=training,  # mind to delegate this flag to operations that change on inference mode
                             )

        except AttributeError:
            pass

        try:
            x = self.normalization(x,
                    training=training,  # mind to delegate this flag to operations that change on inference mode
                                   )

        except AttributeError:
            pass

        return x


class BaseDense(BaseLayer):
    """	Dense layer.

    Attribures:
            dense: the dense layer topping
    """

    def __init__(self, units: int,
            regularizer: str = None,
            activation: str = None,
            dropout: float = None,
            normalization: bool = False,
            name: str = "base_feedforward",
            input_size: int = None,
    **kwargs):
        """	Hyrparametrize layer foundation with dense topping.

        Arguments:
                units: number of neurons in layer
        """

    #	number of units
        assert units > 0

        super(BaseDense, self).__init__(
            dropout=dropout,
            normalization=normalization,
            name=name,
        **kwargs)

        self.dense = tf.keras.layers.Dense(units,
                activation=activation,
        #	use_bias=True,
        #	kernel_initializer='glorot_uniform',"
        #	bias_initializer='zeros',
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
        #	activity_regularizer=regularizer,
        #	kernel_constraint=constraint,
        #	bias_constraint=constraint,
                name=f"{name}",  # None
                                           )
    #	self.layers.append(self.dense)

    #	mock-call to calculate shape
        if input_size:
            self(tf.keras.Input(
                shape=(
                    input_size,
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
        x = super(BaseDense, self).call(x,
                training=training,  # mind to delegate this flag to operations that change on inference mode
                                        )
        x = self.dense(x)

        return x


class AttentionDense(tf.keras.layers.Layer):
    """ Wrapper for Dense layer operating on a stacks of input to recombine them with attention.
ProbabilitySum(
                            axis=0,
                    )
    Such a layer is expected to have no bias and be trainable with no dropout.
    Other dense features include activation only.

    Attributes:
            stack: stack the multiple inputs
            dense: apply weights on each input
            squeeze: eliminate redudant dims on output
    """

    def __init__(self,
            activation: str = None,
            name: str = "attention",
    **kwargs):
        """ Hyperparametrize recombination layer.

        Arguments:
                activation: to apply on output of decision
        """

        super(AttentionDense, self).__init__(
            name=name,
        **kwargs)

        self.dense = tf.keras.layers.Dense(1,
                activation=activation,
                use_bias=False,
        #	kernel_initializer='glorot_uniform',"
        #	bias_initializer='zeros',
        #	kernel_regularizer=regularizer,
        #	bias_regularizer=regularizer,
        #	activity_regularizer=regularizer,
        #	kernel_constraint=constraint,
        #	bias_constraint=constraint, # obsolete because `use_bias=False`
                name=f"{name}",  # None
                                           )

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
        x = tf.squeeze(self.dense(tf.stack(x,
                                axis=-1,
                                           )
                                  ),
                axis=-1,
                       )

        return x
