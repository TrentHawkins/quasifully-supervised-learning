"""Latent embedding model based on dense encoding.

The model consists of 3 main component:
-	a visual model encoding images into visual features
-	a visual-semantic model encoding visual features into semantic features that can be compared to known embeddings
-	a semantic (usually frozen) classifier that encodes semantic features into classes having known semantic features
"""


import tensorflow


def Model(
	input: tensorflow.keras.Input,
	visual: tensorflow.keras.Model,
	encoder: tensorflow.keras.Model,
	semantic: tensorflow.keras.Model,
	*,
	freeze: bool = True,
	name: str = "generalized_zeroshot_embedding_model"
) -> tensorflow.keras.Model:
	"""Build a latent embedding model based on dense encoding.

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
