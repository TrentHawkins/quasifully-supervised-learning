"""Main."""


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

if __name__ == "__main__":
	"""Test."""

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
		verbose=1,
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
	classifier.save(f"./models/{classifier.name}")

#	Load compiled pre-trained model:
	reloaded_classifier = QuasifullyZeroshotCategoricalClassifier.load(f"./models/{classifier.name}", *dataset.split(),
		dataset.labels("trainvalclasses.txt"),
		dataset.labels("testclasses.txt"),
	#	bias=1,
		verbose=1,
		name="quasifully_zeroshot_categorical",
	)

#	Assert compiled elements are the same:
	assert reloaded_classifier.model.get_config() == classifier.model.get_config()

	predict_reloaded = classifier.predict()
	predict_relolution = reloaded_classifier.predict()

#	Assert seed is trully fixed:
	assert numpy.allclose(predict_reloaded, predict)

#	Assert seed is trully fixed:
	assert numpy.allclose(predict_relolution, predict)
