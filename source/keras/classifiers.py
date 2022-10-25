from typing import Union

import math

import numpy as np
import pandas as pd
import tensorflow as tf


class BaseClassifier:
    """	Base TensorFlow classifier (model wrapper).

    Methods (impementable):
            model: build model if not provided
            compile: compilation wrapper(loptimizer, loss, metrics)
            fit: training wrapping
            predict: predicting wrapping
            evaluate: evaluation wrapping
    """

    def __init__(self,
            model: tf.keras.Model = None,
                 ):
        """	Instantiate base classifier.

        The model is deferred to subclassing the base classifier, to allow flexibility.

        Arguments:
                model: a`tf.keras.Model`
                        default: delegate initialization to model loading
        """

        self.model = model

    def compile(self, *args, **kwargs):
        """	Configures the model for training.
        """

        raise NotImplementedError

    def fit(self, *args, **kwargs):
        """	Trains the model for a fixed number of epochs (iterations on a dataset).
        """

        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """	Generates output predictions for the input samples.

        Computation is done in batches.
        This method is designed for performance in large scale inputs
        For small amount of inputs that fit in one batch, directly using `__call__` is recommended for faster execution,
        e.g., `model(x)`, or `model(x, training=False)`
        if you have layers such as `tf.keras.layers.BatchNormalization` that behaves differently during inference.
        Also, note the fact that test loss is not affected by regularization layers like noise and dropout.
        """

        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        """	Returns the loss value & metrics values for the model in test mode.

        Computation is done in batches (see the batch_size arg.)
        """

        raise NotImplementedError

    def save(self, filepath):
        """	Saves all layer weights in Tensorflow format.

        When saving in TensorFlow format,
        all objects referenced by the network are saved in the same format as `tf.train.Checkpoint`,
        including any `Layer` instances or `Optimizer` instances assigned to object attributes.
        For networks constructed from inputs and outputs using `tf.keras.Model(inputs, outputs)`,
        `Layer` instances used by the network are tracked/saved automatically.
        For user-defined classes which inherit from `tf.keras.Model`,
        `Layer` instances must be assigned to object attributes,
        typically in the constructor.
        See the documentation of `tf.train.Checkpoint` and `tf.keras.Model` for details.

        While the formats are the same,
        do not mix `save_weights` and `tf.train.Checkpoint`.
        Checkpoints saved by `Model.save_weights` should be loaded using `Model.load_weights`.
        Checkpoints saved using `tf.train.Checkpoint.save` should be restored using the corresponding `tf.train.Checkpoint.restore`.
        Prefer `tf.train.Checkpoint` over `save_weights` for training checkpoints.

        The TensorFlow format matches objects and variables by starting at a root object,
        `self` for save_weights,
        and greedily matching attribute names.
        For `Model.save` this is the Model,
        and for `Checkpoint.save` this is the `Checkpoint` even if the `Checkpoint` has a model attached.
        This means saving a `tf.keras.Model` using `save_weights`
        and loading into a `tf.train.Checkpoint` with a `Model` attached (or vice versa)
        will not match the `Model`'s variables.
        See the guide to training checkpoints for details on the TensorFlow format.

        Arguments:
                filepath: path to the file to save the weights to which is the prefix used for checkpoint files
        """

        self.model.save_weights(filepath,
        #	overwrite=True,
        #	save_format=None,
        #	options=None,
                                )

    def load(self, filepath):
        """	Loads all layer weights from a TensorFlow weight file.

        Arguments:
                filepath: path to the weights file to load which is the same file prefix passed to save_weights
        """

        self.model.load_weights(filepath,
        #	by_name=False,
        #	skip_mismatch=False,
        #	options=None,
                                )

    def summary(self, **kwargs):
        """ Prints a string summary of the network.
        """

        self.model.summary(**kwargs)

    @property
    def name(self):
        """ Returns model name.
        """

        return self.model._name


class BaseSequenceClassifier(BaseClassifier):
    """	Base TensorFlow classifier (model wrapper).
    """

    def __init__(self, sequence,
            model=None,
                 ):
        """	Instantiate base classifier.

        The model is deferred to subclassing the base classifier, to allow flexibilit

        Arguments:
                sequence: a `tf.keras.utils.Sequence` subclass name
                model: a`tf.keras.Model` subclass name
                        default: let the build function handle it
        """

        self.sequence = sequence

        super(BaseSequenceClassifier, self).__init__(model)

    def generator(self, data: pd.DataFrame,
    **sequence_kwargs):
        """	Build a TensorFlow sequence from reading a file with pandas.

        File is read with `squeeze=False` in case sample weights are present.
        File is expected to list x/path values in first column.

        Arguments:
                data: either stored in memory (array) or on disk (path)

        Returns:
                an instantiated `tf.keras.utils.Sequence`
        """

        return self.sequence(data,
        **sequence_kwargs)
