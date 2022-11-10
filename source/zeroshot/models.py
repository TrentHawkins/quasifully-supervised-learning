"""Latent embedding models.

The model consists of 3 main component:
-	a visual model encoding images into visual features
-	a visual-semantic model encoding visual features into semantic features that can be compared to known embeddings
-	a semantic (usually frozen) classifier that encodes semantic features into classes having known semantic features
"""


import tensorflow

from ..keras.layers import MetricDense
from ..keras.models import DenseStackArray


def Model(
	visual: tensorflow.keras.Model | tensorflow.keras.layers.Layer,
	encoder: tensorflow.keras.Model | tensorflow.keras.layers.Layer,
	semantic: tensorflow.keras.Model | tensorflow.keras.layers.Layer,
	*,
	freeze: bool = True,
	name: str = "generalized_zeroshot_embedding_model"
) -> tensorflow.keras.Model:
	"""Build a generic latent embedding model.

	Arguments:
		visual: a visual model encoding images into visual features
		encoder: a visual-semantic model encoding visual features into semantic features that can be compared to known embeddings
		semantic: a semantic (frozen) classifier that encodes semantic features into classes having known semantic features

	Keyword Arguments:
		freeze: whether to freeze the visual and semantic compoments
			Freezing the visual component prevents information loss if it is prettrained.
			Freezing the semantic component prevents overfitting in favor of seen class labels.
	"""
	visual.trainable = not freeze
	semantic.trainable = not freeze

	return tensorflow.keras.models.Sequential(
		layers=[
			visual,
			encoder,
			semantic,
		],
		name=name,
	)


def EfficientNetDense(
	visual: tensorflow.keras.Model,
	semantic_matrix: tensorflow.Tensor,
	*,
	semantic_class: type = MetricDense,
):
	"""Build a specific latent embedding model based on EfficientNet for visual featuress and Dense encoding.

	Arguments:
		visual: a pretrained visual model encoding images into visual features
		semantic_matrix: a kernel for the (frozen) semantic encoder

	Keyword Arguments:
		semantic_class: suptype of `tensorflow.keras.layers.Dense` to use in building a semantic component
			default: simple Dense with no bias (naturally)
	"""
	return Model(
		visual=visual,
		encoder=DenseStackArray(
			visual.output.shape[-1],  # type: ignore
			semantic_matrix.shape[0],
			attention_activation="sigmoid",
			activation="swish",
			name="visual_semantic",
		),
		semantic=semantic_class(
			semantic_matrix.shape[1],
			activation="softmax",
			kernel_initializer=tensorflow.keras.initializers.Constant(semantic_matrix),  # type: ignore
			name="semantic"
		),
	)
