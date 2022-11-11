"""Tests for models on zeroshot problems defined here."""


class TestModels:
	"""Tests for specific latent-embedding model based on EfficientNet and Dense."""

	from random import randint

	batch_size = randint(1, 50)
	image_size = (
		224,
		224, 3
	)

	input_shape = (batch_size, *image_size)

	def test_efficientnet_dense(self):
		"""Test proper model instantiation and output."""
		import numpy
		import tensorflow

		import source.keras.applications.efficientnet

		from source.dataset.animals_with_attributes import Dataset
		from source.keras.layers import JaccardDense
		from source.zeroshot.models import EfficientNetDense

		images = tensorflow.random.uniform(self.input_shape,
			minval=0,
			maxval=255,
			dtype=tensorflow.int64,
		)
		predicates = Dataset().alphas().transpose().to_numpy()

		softmaxModel = EfficientNetDense(
			visual=tensorflow.keras.applications.efficientnet.EfficientNetB0(),
			semantic_matrix=tensorflow.constant(predicates, dtype=tensorflow.float32),
		)
		jaccardModel = EfficientNetDense(
			visual=tensorflow.keras.applications.efficientnet.EfficientNetB0(),
			semantic_matrix=tensorflow.constant(predicates, dtype=tensorflow.float32),
			semantic_class=JaccardDense,
		)

	#   Assert semantic weight are properly initialized:
		assert numpy.allclose(softmaxModel.layers[-1].kernel.numpy(), predicates)
		assert numpy.allclose(jaccardModel.layers[-1].kernel.numpy(), predicates)

	#	Assert softmax output:
		assert tensorflow.reduce_sum(softmaxModel(images)) == self.batch_size

	#	Assert sigmoid-like output:
		assert tensorflow.math.reduce_all(0 <= jaccardModel(images))  # type: ignore
		assert tensorflow.math.reduce_all(1 >= jaccardModel(images))  # type: ignore
