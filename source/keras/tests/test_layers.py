"""Unit tests for composite layers."""


class TestTensorFlowLinearOperations:
	"""Tests for TensorFlow bsic linear operations to be used inside call methods and functional models."""

	from tensorflow import constant

#	A small sample tensor to test linear operations.
	x = constant(
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
		]
	)

	def test_vector_dot_product(self):
		"""Test tensordot as a vector dot product on the last dimension of tensors."""
		from tensorflow import constant, einsum
		from tensorflow import math

	#   Use prime numbers for chape of tensor for better dot compatibility testing.
		assert self.x.shape == (2, 3, 5)

	#	Assume x is a (2,3)-shaped batch of 5-sized feature vectors. The norm should be (2,3) then.
		assert einsum("...i, ...i -> ...", self.x, self.x).shape == (2, 3)
		assert math.reduce_all(
			einsum("...i, ...i -> ...", self.x, self.x) == constant([[j @ j for j in i] for i in self.x.numpy()])
		)

	def test_matrix_multiplication(self):
		"""Test tensordot as a vector dot product on the last dimension of tensors."""
		from tensorflow import constant, einsum
		from tensorflow import math

	#   Use prime numbers for chape of tensor for better dot compatibility testing.
		assert self.x.shape == (2, 3, 5)

	#	Assume x is a 2-sized batch of (3,5)-shaped kernel weights. The norm should be (2,3) again, via the diagonal dots.
		assert einsum("...ji, ...ji -> ...j", self.x, self.x).shape == (2, 3)
		assert math.reduce_all(
			einsum("...i, ...i -> ...", self.x, self.x) == constant([(i @ i.transpose()).diagonal() for i in self.x.numpy()])
		)
