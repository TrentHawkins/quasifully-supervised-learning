"""Latent embedding models.

The model consists of 3 main component:
-	a visual model encoding images into visual features
-	a visual-semantic model encoding visual features into semantic features that can be compared to known embeddings
-	a semantic (usually frozen) classifier that encodes semantic features into classes having known semantic features
"""


import tensorflow

from ..keras.applications.efficientnet import EfficientNet
from ..keras.layers import Jaccard
from ..keras.models import DenseStackArray


def Model(
	input,
	visual: tensorflow.keras.Model,
	encoder: tensorflow.keras.Model,
	semantic: tensorflow.keras.Model,
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
	visual.trainable = freeze
	semantic.trainable = freeze

	return tensorflow.keras.Model(input, semantic(encoder(visual(input))), name)


def EfficientNetDense(
	input_shape: tensorflow.TensorShape,
	visual: tensorflow.keras.Model,
	semantic_matrix: tensorflow.Tensor,
	jaccard: bool = False,
):
	"""Build a specific latent embedding model based on EfficientNet for visual featuress and Dense encoding.

	Arguments:
		input_shape: a tuple or tensor shape with image width, height and channels in proper order
		visual: a pretrained visual model encoding images into visual features
		semantic_matrix: a kernel for the (frozen) semantic encoder
		jaccard: whether to apply semantic non-linearity akin to the jaccard similarity metric replaceing the dot product of Dense

	Keyword Arguments:
		
	"""
	encoder = DenseStackArray(visual.output.shape[-1], semantic_matrix.shape[0],  # type: ignore
		attention_activation="sigmoid",
		activation="swish",
		name="visual_semantic",
	)

	if jaccard:
		semantic=Jaccard(encoder.output[-1],