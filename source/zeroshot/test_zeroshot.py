"""Tests for models on zeroshot problems defined here."""


class TestEfficientNetDense:
    """Tests for specific latent-embedding model based on EfficientNet and Dense."""

    def test_models(self):
        """Test proper instantiation of model."""
        import numpy
        import tensorflow

        from source.dataset.animals_with_attributes import Dataset
        from source.keras.applications.efficientnet import EfficientNet, inputs_size
        from source.keras.layers import Jaccard
        from source.zeroshot.embedding import EfficientNetDense

        input_shape = tensorflow.TensorShape(
            (
                inputs_size["B0"],
                inputs_size["B0"], 3
            )
        )

        softmaxModel = EfficientNetDense(
            input_shape=input_shape,
            visual=EfficientNet.B0.value(),
            semantic_matrix=tensorflow.constant(Dataset().predicates().transpose().to_numpy(), dtype=tensorflow.float32),
        )
        jaccardModel = EfficientNetDense(
            input_shape=input_shape,
            visual=EfficientNet.B0.value(),
            semantic_matrix=tensorflow.constant(Dataset().predicates().transpose().to_numpy(), dtype=tensorflow.float32),
            semanticModel=Jaccard,
        )

    #   Assert semantic weight are properly initialized
        assert numpy.allclose(softmaxModel.layers[-1].kernel.numpy(), Dataset().predicates().transpose().to_numpy())
        assert numpy.allclose(jaccardModel.layers[-1].kernel.numpy(), Dataset().predicates().transpose().to_numpy())
