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

	def test_metric_dense(self):
		"""Test proper model instantiation and output."""
		import numpy
		import tensorflow

		from source.dataset.animals_with_attributes import Dataset
		from source.keras.applications.efficientnet import EfficientNet
		from source.keras.layers import JaccardDense
		from source.zeroshot.models import GeneralizedZeroshotModel

		images = tensorflow.random.uniform(self.input_shape,
			minval=0,
			maxval=255,
			dtype=tensorflow.int64,
		)
		predicates = Dataset().alphas().transpose().to_numpy()

		softmaxModel = GeneralizedZeroshotModel(
			visual=EfficientNet.B0(),
			semantic_matrix=tensorflow.constant(predicates, dtype=tensorflow.float32),
		)
		jaccardModel = GeneralizedZeroshotModel(
			visual=EfficientNet.B0(),
			semantic_matrix=tensorflow.constant(predicates, dtype=tensorflow.float32),
			semantic_class=JaccardDense,
		)

	#   Assert semantic weight are properly initialized:
		assert numpy.allclose(softmaxModel.layers[-1].kernel.numpy(), predicates)
		assert numpy.allclose(jaccardModel.layers[-1].kernel.numpy(), predicates)

	#	Assert softmax output:
		assert numpy.isclose(tensorflow.reduce_sum(softmaxModel(images)), self.batch_size)

	#	Assert sigmoid-like output:
		assert tensorflow.math.reduce_all(0 <= jaccardModel(images))  # type: ignore
		assert tensorflow.math.reduce_all(1 >= jaccardModel(images))  # type: ignore


class TestClassifiers:
	"""Setup test experiments with one training epoch to test classifiers."""

	def test_quasifully_zeroshot_categorical_classifier(self):
		"""Test plain categorical classifier."""
		import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
		import numpy
		import tensorflow  # ; tensorflow.keras.backend.set_image_data_format("channels_first")

		import source.keras.utils.generic_utils
		import source.keras.utils.layer_utils

		from source.dataset.animals_with_attributes import TransductiveZeroshotDataset
		from source.keras.applications.convnext import ConvNeXt
		from source.keras.applications.efficientnet import EfficientNet
		from source.zeroshot.classifiers import QuasifullyZeroshotCategoricalClassifier
		from source.zeroshot.models import GeneralizedZeroshotModel

	#	Setup data:
		dataset = TransductiveZeroshotDataset()

	#	Setup model components:
		visual = ConvNeXt.Tiny()
	#	visual = EfficientNet.B0()
		semantic_matrix = tensorflow.convert_to_tensor(dataset.alphas().transpose(),
			dtype=tensorflow.float32,
		)

	#	Set up model:
		model = GeneralizedZeroshotModel(visual, semantic_matrix,
			name="quasifully_zeroshot_categorical",
		)

	#	Setup model pipeline (classifier):
		classifier = QuasifullyZeroshotCategoricalClassifier(model, *dataset.split(),
			dataset.labels("trainvalclasses.txt"),
			dataset.labels("testclasses.txt"),
		#	bias=1,
			verbose=2,
			name="quasifully_zeroshot_categorical",
		)

	#	Setup model pipeline (classifier):
		classifier = QuasifullyZeroshotCategoricalClassifier(model, *dataset.split(),
			dataset.labels("trainvalclasses.txt"),
			dataset.labels("testclasses.txt"),
		#	bias=1,
			verbose=2,
			name="quasifully_zeroshot_categorical",
		)

	#	Compile classifier:
		classifier.compile()
		classifier.summary()

	#	Learning cycle:
		history = classifier.fit(epochs=1)
		predict = classifier.predict()
		metrics = classifier.evaluate()

	#	Save compiled and trained model:
	#	classifier.save(f"./models/{classifier.name}")

	#	Load compiled pre-trained model:
	#	reloaded_classifier = QuasifullyZeroshotCategoricalClassifier.load(f"./models/{classifier.name}", *dataset.split(),
	#		dataset.labels("trainvalclasses.txt"),
	#		dataset.labels("testclasses.txt"),
	#	#	bias=1,
	#		verbose=1,
	#		name="quasifully_zeroshot_categorical",
	#	)

	#	Assert models are the same:
	#	assert numpy.allclose(reloaded_classifier.predict(), predict)
