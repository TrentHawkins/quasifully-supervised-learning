"""Latent embedding models.

The model consists of 3 main component:
-	a visual model encoding images into visual features
-	a visual-semantic model encoding visual features into semantic features that can be compared to known embeddings
-	a semantic (usually frozen) classifier that encodes semantic features into classes having known semantic features
"""


from __future__ import annotations

from typing import Iterable

import tensorflow

from ..keras.layers import MetricDense
from ..keras.models import DenseStackArray


def Model(
	visual: tensorflow.keras.Model | tensorflow.keras.layers.Layer,
	encoder: tensorflow.keras.Model | tensorflow.keras.layers.Layer,
	semantic: tensorflow.keras.Model | tensorflow.keras.layers.Layer,
	*,
	trainable: bool = False,
	name: str = "generalized_zeroshot_embedding_model"
) -> tensorflow.keras.Model:
	"""Build a generic latent embedding model.

	Arguments:
		visual: a visual model encoding images into visual features
		encoder: a visual-semantic model encoding visual features into semantic features that can be compared to known embeddings
		semantic: a semantic (frozen) classifier that encodes semantic features into classes having known semantic features

	Keyword Arguments:
		trainable: whether to freeze the visual and semantic compoments
			Freezing the visual component prevents information loss if it is prettrained.
			Freezing the semantic component prevents overfitting in favor of seen class labels.
	"""
	visual.trainable = trainable
	semantic.trainable = trainable

	return tensorflow.keras.models.Sequential(
		layers=[
			visual,
			encoder,
			semantic,
		],
		name=name,
	)


def GeneralizedZeroshotModel(
	visual: tensorflow.keras.Model,
	semantic_matrix: tensorflow.Tensor | Iterable[Iterable[float]],
	semantic_class: type = MetricDense,
	*,
	name: str = "efficientnet_zeroshot_embedding_model",
):
	"""Build a specific latent embedding model based on EfficientNet for visual featuress and Dense encoding.

	Arguments:
		visual: a pretrained visual model encoding images into visual features
		semantic_matrix: a kernel for the (frozen) semantic encoder

	Keyword Arguments:
		semantic_class: suptype of `tensorflow.keras.layers.Dense` to use in building a semantic component
			default: simple Dense with no bias (naturally)
	"""
	visual._name = "visual"
	kernel = tensorflow.constant(semantic_matrix, dtype=tensorflow.float32)

	return Model(
		visual=visual,
		encoder=DenseStackArray(
			visual.output.shape[-1],  # type: ignore  # output shall be known for pretrained models
			kernel.shape[0],
			attention_activation="sigmoid",
			activation="swish",
			name="encoder",
		),
		semantic=semantic_class(
			kernel.shape[1],
			activation="softmax" if semantic_class == MetricDense else None,
			kernel=semantic_matrix,
			name="semantic"
		),
		name=name,
	)
