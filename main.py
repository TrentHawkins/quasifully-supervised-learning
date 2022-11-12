"""Main."""

import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow

import source.keras.utils.generic_utils
import source.keras.utils.layer_utils

from source.dataset.animals_with_attributes import Dataset
from source.keras.applications.efficientnet import EfficientNet
from source.zeroshot.classifiers import QuasifullyZeroshotCategoricalClassifier
from source.zeroshot.models import EfficientNetDense

if __name__ == "__main__":
	"""Learning cycle."""

#	Setup data:
	dataset = Dataset()

#	Setup model components:
	visual = EfficientNet.B0()
	semantic_matrix = tensorflow.convert_to_tensor(dataset.alphas().transpose(),
			dtype=tensorflow.float32,
		)

#	Setup model:
	model = EfficientNetDense(visual, semantic_matrix)

#	Setup model pipeline:
	classifier = QuasifullyZeroshotCategoricalClassifier(*dataset.split(), model,
		dataset.labels("trainvalclasses.txt"),
		dataset.labels("testclasses.txt"),
	)
	classifier.compile()
	classifier.summary()

#	Learning cycle:
	history = classifier.fit()
	predict = classifier.predict()
	metrics = classifier.evaluate()
