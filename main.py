"""Main."""

import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow

import source.keras.applications
import source.keras.utils.generic_utils
import source.keras.utils.layer_utils

from source.dataset.animals_with_attributes import Dataset
from source.keras.classifiers import Classifier
from source.keras.models import DenseStackArray
from source.zeroshot.embedding import EfficientNetDense

if __name__ == "__main__":
	"""Learning cycle."""

#	Setup data:
	dataset = Dataset()

#	Setup model components:
	visual = tensorflow.keras.applications.efficientnet.EfficientNetB0()
	semantic_matrix = tensorflow.convert_to_tensor(dataset.alphas().transpose(),
			dtype=tensorflow.float32,
		)

#	Setup model:
	model = EfficientNetDense(visual, semantic_matrix)

#	Setup model pipeline:
	classifier = Classifier(*dataset.split(), model)
	classifier.compile("categorical_crossentropy")
	classifier.model.summary()

#	Learning cycle:
	history = classifier.fit()
	predict = classifier.predict()
	metrics = classifier.evaluate()
