"""Unit tests for composite layers."""


class TestLinearOperations:
	"""Tests for TensorFlow bsic linear operations to be used inside call methods and functional models."""

	from tensorflow import constant

#	A small sample tensor to test linear operations.
	inputs = constant(
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
		assert self.inputs.shape == (
			2,
			3,
			5,
		)

	#	Assume x is a (2,3)-shaped batch of 5-sized feature vectors. The norm should be (2,3) then.
		assert einsum("...i, ...i -> ...", self.inputs, self.inputs).shape == (2, 3)
		assert math.reduce_all(einsum("...i, ...i -> ...", self.inputs, self.inputs)
			== constant([[j @ j for j in i] for i in self.inputs.numpy()]))

	def test_matrix_multiplication(self):
		"""Test tensordot as a vector dot product on the last dimension of tensors."""
		from tensorflow import constant, einsum
		from tensorflow import math

	#   Use prime numbers for chape of tensor for better dot compatibility testing.
		assert self.inputs.shape == (
			2,
			3,
			5,
		)

	#	Assume x is a 2-sized batch of (3,5)-shaped kernel weights. The norm should be (2,3) again, via the diagonal dots.
		assert einsum("...ji, ...ji -> ...j", self.inputs, self.inputs).shape == (2, 3)
		assert math.reduce_all(einsum("...i, ...i -> ...", self.inputs, self.inputs)
			== constant([(i @ i.transpose()).diagonal() for i in self.inputs.numpy()]))


class TestMetricDense:
	"""Test the custom Jaccard layer as a sinlge layer model."""

	from tensorflow import random

#	A small sample tensor of sigmoids to test the Jaccard layer.
	input = random.uniform(
		(
			2,
			3,
			5,
		)
	)

	def test_jaccard(self):
		"""Test if operations are carried out successfully over a batch of inputs."""
		import tensorflow

		from source.keras.layers import JaccardDense

		testDense = JaccardDense(
			self.input.shape[-1],
			kernel_initializer=tensorflow.keras.initializers.Constant(
				tensorflow.random.uniform(
					(
						5,
						5,
					)
				)
			)
		)

	#	Assert `JaccardDense` forwards successfully and with the same shapes as a normal `Dense`.
		assert testDense(self.input).shape == tensorflow.keras.layers.Dense(self.input.shape[-1])(self.input).shape  # type: ignore

	#	Assert the range of values of a `JaccardDense` pass is 0 to 1.
		assert tensorflow.math.reduce_all(0 <= testDense(self.input))  # type: ignore
		assert tensorflow.math.reduce_all(1 >= testDense(self.input))  # type: ignore

	def test_cosine(self):
		"""Test if operations are carried out successfully over a batch of inputs."""
		import tensorflow

		from source.keras.layers import CosineDense

		testDense = CosineDense(
			self.input.shape[-1],
			kernel_initializer=tensorflow.keras.initializers.Constant(
				tensorflow.random.uniform(
					(
						5,
						5,
					)
				)
			)
		)

	#	Assert `JaccardDense` forwards successfully and with the same shapes as a normal `Dense`.
		assert testDense(self.input).shape == tensorflow.keras.layers.Dense(self.input.shape[-1])(self.input).shape  # type: ignore

	#	Assert the range of values of a `CosineDense` pass is 0 to 1.
		assert tensorflow.math.reduce_all(0 <= testDense(self.input))  # type: ignore
		assert tensorflow.math.reduce_all(1 >= testDense(self.input))  # type: ignore


class TestBaseDense:
	"""Test the custom Jaccard layer as a sinlge layer model."""

	from tensorflow import random

#	A small sample tensor of sigmoids to test the Jaccard layer.
	input = random.normal(
		(
			2,
			3,
			5,
		)
	)

	def test_forward_pass(self):
		"""Test if operations are carried out successfully over a batch of inputs."""
		import tensorflow

		from source.keras.layers import Dense

		testDense = Dense(self.input.shape[-1],
			activation="sigmoid",
			normalization=True,
		)

	#	Assert `JaccardDense` forwards successfully and with the same shapes as a normal `Dense`.
		assert testDense(self.input).shape == tensorflow.keras.layers.Dense(self.input.shape[-1])(self.input).shape  # type: ignore

	#	Assert the range of values of a `JaccardDense` pass is 0 to 1.
		assert tensorflow.math.reduce_all(0 <= testDense(self.input))  # type: ignore
		assert tensorflow.math.reduce_all(1 >= testDense(self.input))  # type: ignore
