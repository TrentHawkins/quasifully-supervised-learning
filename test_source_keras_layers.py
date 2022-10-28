"""Unit tests for composite layers."""


class TestTensorFlowLinearOperations:
    """Tests for TensorFlow bsic linear operations to be used inside call methods and functional models."""

    def test_tensordot(self):
        """Test tensordot as a vector dot product on the last dimension of tensors."""
        from tensorflow import constant

        self.x = constant(
            [
                [
                    [10, 11, 12, 13, 14],
                    [13, 14, 15, 16, 17],
                    [16, 17, 18, 19, 20],
                ],
                [
                    [21, 22, 23, 24, 25],
                    [24, 25, 26, 27, 28],
                    [27, 28, 29, 30, 31],
                ],
            ],
        )

        assert self.x.shape == (2, 3, 5)
