"""Custom classifier wrappers for keras models."""


from dataclasses import dataclass
from math import sqrt

import tensorflow


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

#	Dataset splits:
	train: tensorflow.data.Dataset
	devel: tensorflow.data.Dataset
	valid: tensorflow.data.Dataset

#	Model to operate:
	model: tensorflow.keras.Model

	def compile(self, loss: tensorflow.keras.losses.Loss,
		learning_rate: float = 1e-3,
	):
		"""Configure the model for training with custom loss.

		Arguments:
			loss: Loss function.
				May be a string (name of loss function), or a `tensorflow.keras.losses.Loss instance`.
				See `tensorflow.keras.losses`.
				A loss function is any callable with the signature `loss = fn(y_true, y_pred)`, where
					`y_true` are the ground truth values with shape `(batch_size, d0, .. dN)`, and
					`y_pred` are the model's predictions with shape `(batch_size, d0, .. dN)`.
				The loss function should return a `float` tensor.
				If a custom `Loss` instance is used and reduction is set to `None`,
					return value has shape `(batch_size, d0, .. dN-1)`,
					i.e. per-sample or per-timestep loss values;
					otherwise, it is a scalar.
				If the model has multiple outputs,
					you can use a different loss on each output by passing a dictionary or a list of losses.
				The loss value that will be minimized by the model will then be the sum of all individual losses,
					unless loss_weights is specified.
			learning_rate: A `Tensor`, floating point value,
				or a schedule that is a `tensorflow.keras.optimizers.schedules.LearningRateSchedule`,
				or a callable that takes no arguments and returns the actual value to use, the learning rate.
				Defaults to 0.001.
				To be used with the `tensorflow.keras.optimizers.Adam` optimizer.
		"""
		self.model.compile(
			optimizer=tensorflow.keras.optimizers.Adam(
				learning_rate=learning_rate,
			#	beta_1=.9,
			#	beta_2=.999,
			#	epsilon=1e-7,
				amsgrad=True,
			#	name="Adam",
			),
			loss=loss,
			metrics=[
				tensorflow.keras.metrics.Accuracy,
			],
		#	loss_weights=None,
		#	weighted_metrics=None,
		#	run_eagerly=None,
		#	steps_per_execution=None,
		#	jit_compile=None,
		)

	def fit(self,
		train: tensorflow.data.Dataset | None = None,
		devel: tensorflow.data.Dataset | None = None,
		epochs: int = 1,
	):
		"""Train the model for a fixed number of epochs (iterations on a dataset).

		Unpacking behavior for iterator-like inputs:
			A common pattern is to pass a
				`tf.data.Dataset`,
				`generator`, or
				`tf.keras.utils.Sequence`
			which will in fact yield not only features but optionally targets and sample weights.
			Keras requires that the output of such iterator-likes be unambiguous.

		In this wrapper `tf.data.Dataset` are used.

		Arguments:
			train: Input data as a `tensorflow.data dataset`.
				Should return a `tuple` of either `(inputs, targets)` or `(inputs, targets, sample_weights)`.
				Defaults to the `tensorflow.data dataset` set at initialization.
			devel: Data as a `tensorflow.data dataset` to evaluate the loss and any model metrics at the end of each epoch on.
				The model will not be trained on this data.
				Thus, note the fact that the validation loss is not affected by regularization layers like noise and dropout.
				Defaults to the `tensorflow.data dataset` set at initialization.
			epochs: Integer.
				Number of epochs to train the model.
				An epoch is an iteration over the entire data provided.
				Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch".
				The model is not trained for a number of iterations given by `epochs`,
					but merely until the epoch of index epochs is reached.
				Defaults to 1.

		Returns:
			A `History` object:
				Its `History.history` attribute is a record of training loss values and metrics values at successive epochs,
					as well as validation loss values and validation metrics values (if applicable).

		Raises:
			RuntimeError: If the model was never compiled or `model.fit` is wrapped in `tensorflow.function`.
			ValueError:	If mismatch between the provided input data and what the model expects or when the input data is empty.
		"""
		patience_early_stopping = int(sqrt(epochs)) + 1
		patience_reduce_learning_rate_on_plateau = int(sqrt(sqrt(epochs))) + 1

		return self.model.fit(train or self.train,
		#	batch_size=None,  # batches generated from dataset
			epochs=1,
		#	verbose="auto",  # means 1 in an interactive enviromnemt which is desired
			callbacks=[
				tensorflow.keras.callbacks.ReduceLROnPlateau(
				#	monitor="val_loss",
				#	factor=1e-1,
					patience=patience_reduce_learning_rate_on_plateau,
				#	verbose=0,
				#	mode="auto",
				#	min_delta=1e-4,
					cooldown=patience_reduce_learning_rate_on_plateau,
				#	min_lr=0,
				),
				tensorflow.keras.callbacks.EarlyStopping(
					monitor="val_loss",
				#	min_delta=0,
					patience=patience_early_stopping,
				#	verbose=0,
				#	mode="auto",
				#	baseline=None,
					restore_best_weights=True,
				)
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
		return self.model.predict(valid or self.valid,
		#	batch_size=None,  # batches generated from dataset
		#	verbose="auto",  # means 1 in an interactive enviromnemt which is desired
		#	steps=None,
		#	callbacks=None,
		#	max_queue_size=10,
		#	workers=1,
		#	use_multiprocessing=False
		)

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
		return self.model.evaluate(valid or self.valid,
		#	batch_size=None,  # batches generated from dataset
			verbose="auto",  # means 1 in an interactive enviromnemt which is desired
		#	sample_weight=None,
		#	steps=None,
		#	callbacks=None,
		#	max_queue_size=10,
		#	workers=1,
		#	use_multiprocessing=False,
			return_dict=True,  # to store as `.json` later
		)
