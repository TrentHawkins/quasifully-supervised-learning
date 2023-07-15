"""Main."""


from __future__ import annotations

import numpy
import torch

import pytorch.globals
import pytorch.torch.nn
import pytorch.torch.utils.data
import pytorch.torchvision.datasets

if __name__ == "__main__":
	dataset = pytorch.torchvision.datasets.TransductiveZeroshotAnimalsWithAttributesDataset()
	dataloader = pytorch.torch.utils.data.AnimalsWithAttributesDataLoader(dataset, pytorch.globals.generator, 2)

if __name__ == "__main__" and False:
	"""Test."""

	import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
	import numpy
	import tensorflow  # ; tensorflow.keras.backend.set_image_data_format("channels_first")

	import source.keras.utils.generic_utils
	import source.keras.utils.layer_utils

	from source.dataset.animals_with_attributes import TransductiveZeroshotDataset
	from source.keras.applications.convnext import ConvNeXt
	from source.keras.applications.efficientnet import EfficientNet
	from source.zeroshot.classifiers import QuasifullyGeneralizedZeroshotCategoricalClassifier
	from source.zeroshot.models import GeneralizedZeroshotModel

#	Set all seeds:
	tensorflow.keras.utils.set_random_seed(0)

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
	classifier = QuasifullyGeneralizedZeroshotCategoricalClassifier(model, *dataset.split(),
		dataset.labels("trainvalclasses.txt"),
		dataset.labels("testclasses.txt"),
		bias=1,
	#	seed=0,
	#	verbose=1,
	#	name="quasifully_zeroshot_categorical",
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
	reloaded_classifier = QuasifullyGeneralizedZeroshotCategoricalClassifier.load(f"./models/{classifier.name}", *dataset.split(),
		dataset.labels("trainvalclasses.txt"),
		dataset.labels("testclasses.txt"),
		bias=2,
	#	seed=0,
	#	verbose=1,
	#	name="quasifully_zeroshot_categorical",
	)

#	Assert compiled elements are the same:
	assert reloaded_classifier.model.get_config() == classifier.model.get_config()

#	Assert predictions are the same with the original model:
	assert numpy.allclose(reloaded_classifier.predict(), predict)

#	Assert metrics are the same with the original model:
	assert numpy.allclose(list(reloaded_classifier.evaluate().values()), list(metrics.values()))
