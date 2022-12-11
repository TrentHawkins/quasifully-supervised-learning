"""Custom classifier wrappers for keras models."""


from dataclasses import dataclass, field
from math import ceil, sqrt
from os import PathLike

import tensorflow

from .utils.layer_utils import print_separator


@dataclass
class Classifier:
	"""Custom classifier wrappers for keras models.

	Attributes:
		train: Data to be used for training.
		devel: Data to be used for development (training regularization).
		valid: Data to be used for validation (evaluation/testing).
		model: To be operated. Assumed it is built.
		loss: Custom function to use in training.
	"""

#	Model to operate:
	model: tensorflow.keras.Model = field()

#	Dataset splits:
	train: tensorflow.data.Dataset = field()
	devel: tensorflow.data.Dataset = field()
	valid: tensorflow.data.Dataset = field()

#	Global verbosity:
	seed: int = field(default=0, kw_only=True)

#	Global verbosity:
	verbose: str | int = field(default="auto", kw_only=True)

#	Classifier name overriding model name:
	name: str | None = field(default=None, kw_only=True)

	def __post_init__(self):
		"""Set classifier name to model name unless any."""
		self.name = self.name or self.model.name

	def compile(self,
		optimizer: tensorflow.keras.optimizers.Optimizer | str,
		loss: tensorflow.keras.losses.Loss | str,
		metrics: list[tensorflow.keras.metrics.Metric | str] | None = None,
	):
		"""Configure the model for training.

		Arguments:
			optimizer: String (name of optimizer) or `tensorflow.keras.optimizers.Optimizer` instance
				See `tensorflow.keras.optimizers`.

			loss: Loss function.
				May be a string (name of loss function), or a `tensorflow.keras.losses.Loss` instance.
				See `tensorflow.keras.losses`.

				A loss function is any callable with the signature `loss=fn(y_true, y_pred)`, where
					`y_true` are the ground truth values with shape `(batch_size,d0,...,dN)`, and
					`y_pred` are the model's predictions with shape `(batch_size,d0,...,dN)`.

				The loss function should return a `float` tensor.

				If a custom `Loss` instance is used and reduction is set to `None`,
				return value has shape `(batch_size, d0, .. dN-1)`,
				i.e. per-sample or per-timestep loss values;
				otherwise, it is a scalar.

				If the model has multiple outputs,
				you can use a different loss on each output by passing a dictionary or a list of losses.

				The loss value that will be minimized by the model will then be the sum of all individual losses,
				unless loss_weights is specified.

			metrics: List of metrics to be evaluated by the model during training and testing.
				Each of this can be a string (name of a built-in function),
				function or a `tensorflow.keras.metrics.Metric` instance.
				See tensorflow.keras.metrics.

				Typically you will use:
					```python
					metrics=[
						'accuracy'
					]
					```

				A function is any callable with the signature `result=fn(y_true,y_pred)`.
				To specify different metrics for different outputs of a multi-output model,
				you could also pass a dictionary, such as:
					```python
					metrics={
						'output_a':'accuracy',
						'output_b':[
							'accuracy',
							'mse',
						],
					}
					```
				You can also pass a list to specify a metric or a list of metrics for each output, such as:
					```python
					metrics=[
						[
							'accuracy',
						],
						[
							'accuracy',
							'mse',
						],
					]
					```
				or:
					```python
					metrics=[
						'accuracy',
						[
							'accuracy',
							'mse',
						],
					]
					```
				When you pass the strings 'accuracy' or 'acc', we convert this to one of:
				-	`tensorflow.keras.metrics.BinaryAccuracy`,
				-	`tensorflow.keras.metrics.CategoricalAccuracy`,
				-	`tensorflow.keras.metrics.SparseCategoricalAccuracy`,
				based on the loss function used and the model output shape.

				We do a similar conversion for the strings 'crossentropy' and 'ce' as well.

				The metrics passed here are evaluated without sample weighting;
				if you would like sample weighting to apply,
				you can specify your metrics via the weighted_metrics argument instead.
		"""
		self.model.compile(
			optimizer=optimizer,
			loss=loss,
			metrics=metrics,
		#	loss_weights=None,
		#	weighted_metrics=None,
		#	run_eagerly=None,
		#	steps_per_execution=None,
		#	jit_compile=None,
		)

	def fit(self,
		train: tensorflow.data.Dataset | None = None,
		devel: tensorflow.data.Dataset | None = None,
		*,
		epochs: int = 1,
	):
		"""Train the model for a fixed number of epochs (iterations on a dataset).

		Unpacking behavior for iterator-like inputs:
			A common pattern is to pass a
				`tensorflow.data.Dataset`,
				`generator`, or
				`tensorflow.keras.utils.Sequence`
			which will in fact yield not only features but optionally targets and sample weights.

			Keras requires that the output of such iterator-likes be unambiguous.

		In this wrapper `tensorflow.data.Dataset` are used.

		Arguments:
			train: Input data as a `tensorflow.data dataset`.
				Should return a `tuple` of either `(inputs, targets)` or `(inputs, targets, sample_weights)`.

				Defaults to the `tensorflow.data dataset` set at initialization.

			devel: Data as a `tensorflow.data dataset` to evaluate the loss and any model metrics at the end of each epoch on.
				The model will not be trained on this data.
				Thus, note the fact that the validation loss is not affected by regularization layers like noise and dropout.

				Defaults to the `tensorflow.data dataset` set at initialization.

		Keyword Arguments:
			epochs: Integer.
				Number of epochs to train the model.

				An epoch is an iteration over the entire data provided.

				Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch".
				The model is not trained for a number of iterations given by `epochs`,
				but merely until the epoch of index epochs is reached.

				Defaults to 1.

			callbacks: List of `tensorflow.keras.callbacks.Callback` instances.
				List of callbacks to apply during training.
				See tensorflow.keras.callbacks.

				Note:
				-	`tensorflow.keras.callbacks.ProgbarLogger` and
				-	`tensorflow.keras.callbacks.History`
				callbacks are created automatically and need not be passed into `model.fit`.

				`tensorflow.keras.callbacks.ProgbarLogger` is created or not based on verbose argument to `model.fit`.

				Callbacks with batch-level calls are currently unsupported with:
				-	`tensorflow.distribute.experimental.ParameterServerStrategy`,
				and users are advised to implement epoch-level calls instead with an appropriate `steps_per_epoch` value.

		Returns:
			A `History` object:
				Its `History.history` attribute is a record of training loss values and metrics values at successive epochs,
				as well as validation loss values and validation metrics values (if applicable).

		Raises:
			RuntimeError: If the model was never compiled or `model.fit` is wrapped in `tensorflow.function`.

			ValueError:	If mismatch between the provided input data and what the model expects or when the input data is empty.
		"""
		print_separator(3, f"{self.model.name}: fitting")

		patience_early_stopping = ceil(sqrt(epochs))
		patience_reduce_learning_rate_on_plateau = ceil(sqrt(patience_early_stopping))

	#	Callbacks:
	#	checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(f"./models/{self.model.name}",
	#		monitor="val_loss",
	#	#	verbose=0,
	#		save_best_only=True,
	#		save_weights_only=True,
	#		mode="auto",
	#		save_freq="epoch",
	#	#	options=None,
	#	#	initial_value_threshold=None,
	#	)
		reduce_learning_rate_on_plateau = tensorflow.keras.callbacks.ReduceLROnPlateau(
		#	monitor="val_loss",
		#	factor=1e-1,
			patience=patience_reduce_learning_rate_on_plateau,
		#	verbose=0,
		#	mode="auto",
		#	min_delta=1e-4,
			cooldown=patience_reduce_learning_rate_on_plateau,
		#	min_lr=0,
		)
		early_stopping = tensorflow.keras.callbacks.EarlyStopping(
			monitor="val_loss",
		#	min_delta=0,
			patience=patience_early_stopping,
		#	verbose=0,
		#	mode="auto",
		#	baseline=None,
			restore_best_weights=True,
		)

		history = self.model.fit(train or self.train,
		#	batch_size=None,  # batches generated from dataset
			epochs=epochs,
			verbose=1,  # type: ignore  # incorrectly type-hinted in Tensorflow
			callbacks=[
			#	checkpoint,
				reduce_learning_rate_on_plateau,
				early_stopping,
			],
		#	validation_split=0.,
			validation_data=devel or self.devel,
		#	shuffle=True,  # NOTE: make sure dataset is shuffled with reshuffling enabled
		#	class_weight=None,
		#	sample_weight=None,
		#	initial_epoch=0,
		#	steps_per_epoch=None,
		#	validation_steps=None,
		#	validation_batch_size=None,  # batches generated from dataset
		#	validation_freq=1,
		#	max_queue_size=10,
		#	workers=1,
		#	use_multiprocessing=False
		)

		print_separator(3)

		return history

	def predict(self,
		valid: tensorflow.data.Dataset | None = None,
	):
		"""Generate output predictions for the input samples.

		Computation is done in batches.

		This method is designed for batch processing of large numbers of inputs.
		It is not intended for use inside of loops that iterate over your data and process small numbers of inputs at a time.

		Also, note the fact that test loss is not affected by regularization layers like noise and dropout.

		Arguments:
			valid: Input samples as a `tensorflow.data dataset`.
				Defaults to the `tensorflow.data dataset` set at initialization.

		Returns:
			Numpy array(s) of predictions.

		Raises:
			RuntimeError: If `model.predict` is wrapped in a `tensorflow.function`.

			ValueError: In case of mismatch between the provided input data and the model's expectations,
				or in case a stateful model receives a number of samples that is not a multiple of the batch size.
		"""
		print_separator(3, f"{self.model.name}: predicting")

		predictions = self.model.predict(valid or self.valid,
		#	batch_size=None,  # batches generated from dataset
			verbose=1,  # type: ignore  # incorrectly type-hinted in Tensorflow
		#	steps=None,
		#	callbacks=None,
		#	max_queue_size=10,
		#	workers=1,
		#	use_multiprocessing=False
		)

		print_separator(3)

		return predictions

	def evaluate(self,
		valid: tensorflow.data.Dataset | None = None,
	):
		"""Return the loss value & metrics values for the model in test mode.

		Computation is done in batches if provided by the input data.

		Arguments:
			valid: Input samples as a `tensorflow.data dataset`.
				Defaults to the `tensorflow.data dataset` set at initialization.

		Returns:
			Scalar test loss (if the model has a single output and no metrics), or
			`dict` of scalars (if the model has multiple outputs and/or metrics).
			The attribute `model.metrics_names` will give you the display labels for the scalar outputs.

		Raises:
			RuntimeError: If `model.predict` is wrapped in a `tensorflow.function`.
		"""
		print_separator(3, f"{self.model.name}: evaluating")

		metrics = self.model.evaluate(valid or self.valid,
		#	batch_size=None,  # batches generated from dataset0
			verbose=1,  # type: ignore  # incorrectly type-hinted in Tensorflow
		#	sample_weight=None,
		#	steps=None,
		#	callbacks=None,
		#	max_queue_size=10,
		#	workers=1,
		#	use_multiprocessing=False,
			return_dict=True,  # to store as `.json` later
		)

		print_separator(3)

		return metrics

	def save(self, filepath: str | PathLike):
		"""Save the model weights in classifier to Tensorflow `SavedModel`.

		Argumentss:
			filepath: Path to SavedModel or H5 file to save the model.

		Keyword Arguments (fixed):
			overwrite: Whether to silently overwrite any existing file at the target location,
				or provide the user with a manual prompt.

			include_optimizer: If True, save optimizer's state together.

			save_format: Either `"tf"` or `"h5"`, indicating whether to save the model to Tensorflow `SavedModel` or HDF5.
				Defaults to "tf" in TensorFlow 2.X, and "h5" in TensorFlow 1.X.

			signatures: Signatures to save with the SavedModel.
				Applicable to the "tf" format only.
				Please see the `signatures` argument in `tensorflow.saved_model.save` for details.

			options: (only applies to `SavedModel` format)
				`tensorflow.saved_model.SaveOptions` object that specifies options for saving to `SavedModel`.

			save_traces: (only applies to `SavedModel` format)
				When enabled, the `SavedModel` will store the function traces for each layer.
				This can be disabled, so that only the configs of each layer are stored.

				Defaults to `True`.

				Disabling this will decrease serialization time and reduce file size,
				but it requires that all custom layers/models implement a `get_config()` method.

		Example:
		```python
		from keras.models import load_model
		model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
		del model  # deletes the existing model
		model = load_model('my_model.h5')  # returns a compiled model identical to the previous one
		```
		"""
		self.model.save(filepath,
			overwrite=True,
			include_optimizer=True,
		#	save_format=None,
		#	signatures=None,
		#	options=None,
			save_traces=True,
		)

	@classmethod
	def load(cls, filepath: str | PathLike, *args, **kwargs):
		"""Load a model to the classifier saved via `Classifier.save()`.

		Usage:
		```
		>>> model = tensoflow.keras.Sequential([
		...     tensorflow.keras.layers.Dense(5, input_shape=(3,)),
		...     tensorflow.keras.layers.Softmax()])
		>>> model.save('/tmp/model')
		>>> loaded_model = tensorflow.keras.models.load_model('/tmp/model')
		>>> x = tensorflow.random.uniform((10, 3))
		>>> assert np.allclose(model.predict(x), loaded_model.predict(x))
		```

		NOTE: The model weights may have different scoped names after being loaded.
		Scoped names include the model/layer names, such as `"dense_1/kernel:0"`.
		It is recommended that you use the layer properties to access specific variables,
		e.g. `model.get_layer("dense_1").kernel`.

		Argumentss:
			filepath: Path to the saved model.

			*args: Positional arguments passed to classifier initializer, apart from `model`.

		Keyword Arguments (fixed):
			custom_objects: Optional dictionary mapping names to custom classes or functions to be considered in deserialization.

			compile: Whether to compile the model after loading.

			options: Optional `tensorflow.saved_model.LoadOptions` object that specifies options for loading from `SavedModel`.

			**kwargs: Keyword arguments passed to classifier initializer.

		Returns:
			A Keras model instance.

			If the original model was compiled, and saved with the optimizer,then the returned model will be compiled.
			Otherwise, the model will be left uncompiled.
			In the case that an uncompiled model is returned, a warning is displayed if the `compile` argument is set to `True`.

		Raises:
			IOError: In case of an invalid savefile.
		"""
		model = tensorflow.keras.models.load_model(filepath,
		#	custom_objects=None,
			compile=True,
		#	options=None,
		)

		return cls(model, *args, **kwargs)  # type: ignore  # type hinting in`load_model` is messed up

	def summary(self, **kwargs):
		"""Print a string summary of the network, with an added separator.

		Keyword Arguments:
			line_length: Total length of printed lines (e.g. set this to adapt the display to different terminal window sizes).
				Defaults to terminal width.

			positions: Relative or absolute positions of log elements in each line.
				If not provided, defaults to `[.33, .55, .67, 1.]`.

			print_fn: Print function to use.
				Defaults to `print`.

				It will be called on each line of the summary.
				You can set it to a custom function in order to capture the string summary.

			expand_nested: Whether to expand the nested models.
				If not provided, defaults to `False`.

			show_trainable: Whether to show if a layer is trainable.
				If not provided, defaults to `False`.

			layer_range: a list or tuple of 2 strings,
				which is the starting layer name and ending layer name (both inclusive)
				indicating the range of layers to be printed in summary.

				It also accepts regex patterns instead of exact	name.
				In such case,
				-	start predicate will be the first element it matches to `layer_range[0]` and
				-	the end predicate will be the last element it matches to `layer_range[1]`.

				By default `None` which considers all layers of model.

		Raises:
			ValueError: if `summary()` is called before the model is built.
		"""
		kwargs.update(
			{
				"expand_nested": False,
				"show_trainable": True,
			}
		)
		self.model.summary(**kwargs)
