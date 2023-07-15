"""Unit tests for composite layers."""


class TestTorchOperations:
	"""Tests for torch bsic linear operations to be used inside call methods and functional models."""

	from torch import Tensor, set_default_device
	from torch.cuda import is_available

	set_default_device("cuda" if is_available() else "cpu")  # use CUDA if available

#	A small sample tensor to test linear operations.
	inputs = Tensor(
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
		from torch import Tensor
		from torch import all, einsum, eq

	#   Use prime numbers for chape of tensor for better dot compatibility testing.
		assert self.inputs.size() == (
			2,
			3,
			5,
		)

	#	Assume x is a (2,3)-shaped batch of 5-sized feature vectors. The norm should be (2,3) then.
		assert einsum("...i, ...i -> ...", self.inputs, self.inputs).size() == (2, 3)
		assert all(
			eq(
				einsum("...i, ...i -> ...", self.inputs, self.inputs),
				Tensor([[j @ j for j in i] for i in self.inputs.numpy()])
			)
		)

	def test_matrix_multiplication(self):
		"""Test tensordot as a vector dot product on the last dimension of tensors."""
		from torch import Tensor
		from torch import all, einsum, eq

	#   Use prime numbers for chape of tensor for better dot compatibility testing.
		assert self.inputs.size() == (
			2,
			3,
			5,
		)

	#	Assume x is a 2-sized batch of (3,5)-shaped kernel weights. The norm should be (2,3) again, via the diagonal dots.
		assert einsum("...ji, ...ji -> ...j", self.inputs, self.inputs).size() == (2, 3)
		assert all(
			eq(
				einsum("...i, ...i -> ...", self.inputs, self.inputs),
				Tensor([(i @ i.transpose()).diagonal() for i in self.inputs.numpy()])
			)
		)


class TestMetricLinear:
	"""Test the custom Jaccard layer as a sinlge layer model."""

	from torch import Tensor, rand, set_default_device
	from torch.cuda import is_available

	set_default_device("cuda" if is_available() else "cpu")  # use CUDA if available

#	A small sample tensor of sigmoids to test the Jaccard layer.
	input = rand(
		2,
		3,
		5,
	)

	def template_test_metric(self, typeMetricLinear: type):
		"""Test if operations are carried out successfully over a batch of inputs."""
		from torch import all, rand, ge, le
		from torch.nn import Linear

		testLinear = typeMetricLinear(
			kernel=rand(
				5,
				5,
			)
		)

	#	Assert `CosineLinear` forwards successfully and with the same shapes as a normal `Linear`.
		assert testLinear(self.input).size() == Linear(
			self.input.size()[-1],
			self.input.size()[-1],
		)(self.input).size()

	#	Assert the range of values of a `CosineLinear` pass is 0 to 1.
		assert all(ge(testLinear(self.input), 0))
		assert all(le(testLinear(self.input), 1))

	def test_cosine(self):
		"""Test if operations are carried out successfully over a batch of inputs."""
		from pytorch.torch.nn import CosineLinear

		self.template_test_metric(CosineLinear)

	def test_jaccard(self):
		"""Test if operations are carried out successfully over a batch of inputs."""
		from pytorch.torch.nn import JaccardLinear

		self.template_test_metric(JaccardLinear)

	def test_dice(self):
		"""Test if operations are carried out successfully over a batch of inputs."""
		from pytorch.torch.nn import DiceLinear

		self.template_test_metric(DiceLinear)


class TestDropoutLinear:
	"""Test the custom Jaccard layer as a sinlge layer model."""

	from torch import Tensor, rand, set_default_device
	from torch.cuda import is_available

	set_default_device("cuda" if is_available() else "cpu")  # use CUDA if available

#	A small sample tensor of sigmoids to test the Jaccard layer.
	input = rand(
		2,
		3,
		5,
	)

	def test_forward_pass(self):
		"""Test if operations are carried out successfully over a batch of inputs."""
		from torch.nn import Linear
		from pytorch.torch.nn import DropoutLinear

		testLinear = DropoutLinear(
			self.input.size()[-1],
			self.input.size()[-1],
		)

	#	Assert `DropoutLinear` forwards successfully and with the same shapes as a normal `Linear`.
		assert testLinear(self.input).size() == Linear(
			self.input.size()[-1],
			self.input.size()[-1],
		)(self.input).size()
