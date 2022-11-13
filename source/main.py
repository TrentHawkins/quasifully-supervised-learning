#!/bin/python3 --


"""A sample learning cycle.

Initialize a dataset, setting all its metadata necessary to generate splits.

Initialize a model with the defined components:
-	Choose a visual component, in this case an EfficientNet model.
-	Choose a predicate matrix from the dataset to use as kernel for the semantic component.
-	Choose (optionally) a type of semantic component (to potentially use manifestly non-linear layers).

Initialize a classifier with said dataset and model:
-	Compile it with its predefined:
	-	optimizer
	-	loss
	-	metrics
-	Summarize model to see if things go according to plan.

Perform learning cycle:
-	Fit the model on training data as set by the dataset.
-	Generate predictions on test data as set by the dataset, to be used in generating confusion plots.
-	Test model on predefined metrics.
"""


import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow

import source.keras.utils.generic_utils
import source.keras.utils.layer_utils

from source.dataset.animals_with_attributes import Dataset
from source.keras.applications.efficientnet import EfficientNet
from source.zeroshot.classifiers import QuasifullyZeroshotCategoricalClassifier
from source.zeroshot.models import EfficientNetDense

if __name__ == "__main__":
	"""This will become the main executable script where experiments will be loaded."""

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
