"""Custom `torch.nn` modules.

Layers:
	Customized `torch.nn.Linear` modules for prototyping simple Deep Neural Networks.

	`DropoutLinear`: a `torch.nn.Linear` module augmented with
		a `torch.nn.Dropout` input dropout module
		a `torch.nn.SiLU` pre-activation module

	`AttentionLinear`: a `torch.nn.Linear` module linearly combining multiple outputs (stacked in a higher dimension tensor) with:
		a `torch.nn.Sigmoid` hard-coded activation

	`MetricLinear`: a `torch.nn.Linear` module with a non-linear modification based on a similarity metric
	`CosineLinear`: a `MetricLinear` based on the cosine similarity
	`JaccardLinear`: a `MetricLinear` based on the Jaccard similarity
	`DiceLinear`: a `MetricLinear` based on the Dice index

Models:
	Custom combinations of aforementioned custom layers.

	`LinearStack`: a pyramid-like `torch.nn.Sequential` module made of `DropoutLinear` submodules in a stack
	`LinearStackArray`: several `LinearStack` submodules in parallel combining output with an `AttentionLinear` submodule
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import scipy.special
import tensorflow
import torch

from ..numtools import divisors, hidden_sizes


class DropoutLinear(torch.nn.Linear):
	"""A `torch.nn.Linear` module augmented with input dropout and pre-activation.

	Submodules:
		`dropout`: a `torch.nn.Dropout` input dropout module
		`activation`: a `torch.nn.SiLU` pre-activation module
	"""

	def __init__(self,
		inputs_size: int,
		output_size: int, dropout: float = .0,
	**kwargs):
		"""Hyrparametrize `DropoutLinear` module.

		Arguments:
			`inputs_size`: size of each inputs sample
			`output_size`: size of each output sample

		Keyword arguments:
			`dropout`: logit of probability of an element to be zeroed (default half probability)
		"""
		super(DropoutLinear, self).__init__(
			inputs_size,
			output_size, bias=True,
		**kwargs)

	#	dropout:
		self.dropout = torch.nn.Dropout(scipy.special.expit(dropout))

	#	activation:
		self.activation = torch.nn.SiLU()  # hardcoded

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		"""Define the computation performed at every call.

		Call stack:
			`torch.nn.SiLU`
			`torch.nn.Dropout`
			`torch.nn.Linear`
		"""
		return super(DropoutLinear, self).forward(self.dropout(self.activation(inputs)))


class AttentionLinear(torch.nn.Linear):
	"""A `torch.nn.Linear` module linearly combining multiple outputs (stacked in a higher dimension tensor).

	Attributes:
		activation: a `torch.nn.Sigmoid` hard-coded activation
	"""

	def __init__(self, threads: int, **kwargs):
		"""Hyperparametrize `AttentionLinear` module.

		Arguments:
			`threads`: number of submodule outputs to linearly combine
		"""
		super(AttentionLinear, self).__init__(threads, 1, bias=False, **kwargs)

		self.activation = torch.nn.Sigmoid()  # hardcoded

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		"""Define the computation performed at every call.

		Call stack:
			`torch.nn.Linear` (on stacked output)
			`torch.nn.Sigmoid`
		"""
		return self.activation(super(AttentionLinear, self).forward(inputs).squeeze(dim=-1))


class MetricLinear(torch.nn.Linear):
	"""A `torch.nn.Linear` module with a non-linear modification based on a similarity metric

	A `MetricLinear` has no bias, no activation (it is nonlinear anyway) and is frozen as an explicit comparator comparator.
	"""

	def __init__(self, kernel: torch.Tensor, **kwargs):
		"""Hyperparametrize `MetricLinear` module.

		Arguments:
			`kernel`: weight values
		"""
		super(MetricLinear, self).__init__(*kernel.size(), bias=False, **kwargs)

	#	frozen semantic kernel:
		assert self.weight.size() == kernel.transpose(0, 1).size()
		self.weight = kernel.transpose(0, 1)
		self.weight.requires_grad = False


class CosineLinear(MetricLinear):
	"""A non-linear module that performs the cosine operation per input and kernel vector instead of a dot product.

	Such a modified module has no bias explicitely.
	"""

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		"""Define the computation performed at every call.

		Formula:
		"""
		inputs_kernel = super(CosineLinear, self).forward(inputs)

	#	the norms of inputs vectors:
		inputs_inputs = torch.einsum("...i, ...i -> ...",
			inputs,
			inputs,
		).unsqueeze(-1)

	#	the norms of kernel vectors (diagonal):
		kernel_kernel = torch.einsum("...ji, ...ji -> ...i",
			self.weight,
			self.weight,
		).expand(inputs_kernel.size())

		return inputs_kernel / torch.sqrt(inputs_inputs * kernel_kernel)


class JaccardLinear(MetricLinear):
	"""A non-linear module that performs the Jaccard operation per input and kernel vector instead of a dot product.

	Such a modified module has no bias explicitely.
	"""

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		"""Define the computation performed at every call.

		Should be overridden by all subclasses.

		NOTE: Although the recipe for forward pass needs to be defined within this function,
		one should call the `Module` instance afterwards instead of this,
		since the former takes care of running the registered hooks while the latter silently ignores them.
		"""
		inputs_kernel = super(JaccardLinear, self).forward(inputs)

	#	the norms of inputs vectors:
		inputs_inputs = torch.einsum("...i, ...i -> ...",
			inputs,
			inputs,
		).unsqueeze(-1)

	#	the norms of kernel vectors (diagonal):
		kernel_kernel = torch.einsum("...ji, ...ji -> ...i",
			self.weight,
			self.weight,
		).expand(inputs_kernel.size())

		return inputs_kernel / (inputs_inputs + kernel_kernel - inputs_kernel)


class DiceLinear(MetricLinear):
	"""A non-linear module that performs the Dice operation per input and kernel vector instead of a dot product.

	Such a modified module has no bias explicitely.
	"""

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		"""Define the computation performed at every call.

		Should be overridden by all subclasses.

		NOTE: Although the recipe for forward pass needs to be defined within this function,
		one should call the `Module` instance afterwards instead of this,
		since the former takes care of running the registered hooks while the latter silently ignores them.
		"""
		inputs_kernel = super(DiceLinear, self).forward(inputs)

	#	the norms of inputs vectors:
		inputs_inputs = torch.einsum("...i, ...i -> ...",
			inputs,
			inputs,
		).unsqueeze(-1)

	#	the norms of kernel vectors (diagonal):
		kernel_kernel = torch.einsum("...ji, ...ji -> ...i",
			self.weight,
			self.weight,
		).expand(inputs_kernel.size())

		return inputs_kernel / ((inputs_inputs + kernel_kernel) / 2)


def LinearStack(
	inputs_size: int,
	output_size: int, skip: int = 1, dropout: float = .0
) -> torch.nn.Sequential:
	"""Sequence of linear modules equipped with dropout and uniform activation throughout.

	The network complexity is defined in a specific way which is based on inputs and output sizes:
	-	the hidden layers gradually change from one to the other with complexities defined by an integer divisor logic
	-	the depth of the change is adjustable

	Arguments:
		inputs_dim: size of last inputs dimension
		output_dim: size of last output dimension

	keyword arguments:
		skip: the (inverse) depth of the dense layer stack
			default: no skipping (full depth)
		dropout: dropout factor applied on input of dense layers in dense layer stack
			default: half

	Returns:
		Sequential module with predefined linear submodules
	"""
	sizes = hidden_sizes(
		inputs_size,
		output_size, skip=skip
	)

	return torch.nn.Sequential(
		*[
			DropoutLinear(
				inputs_size,
				output_size, dropout=dropout,
			) for inputs_size, output_size in zip(sizes[:-1], sizes[1:])
		]
	)


class LinearStackArray(torch.nn.Module):
	"""Array of linear stacks equipped with dropout and uniform activation throughout.

	The number of linear stacks (threads) is defined by the inputs and output sizes.

	The network complexity is defined in a specific way which based on inputs and output dimensionality:
	-	the hidden layers gradually change from one to the other with complexities defined by an integer divisor logic
	-	the depth of the change is adjustable

	The linear stack array recombines the linear stacks (threads) with attention.
	"""

	def __init__(self,
		inputs_size: int,
		output_size: int, threads: int, dropout: float = .0,
	):
		"""Hyperparametrize the linear stack array.

		Arguments:
			inputs_dim: size of last inputs dimension
			output_dim: size of last output dimension

		keyword arguments:
			threads: the number of (parallel) dense layer stacks to build
				default: base
			dropout: dropout factor applied on input of dense layers in dense layer stack
				default: half
		"""
		self.array = [
			LinearStack(
				inputs_size,
				output_size, skip=skip, dropout=dropout,
			) for skip in divisors(len(hidden_sizes(inputs_size, output_size)) - 1, reverse=True)[0:threads]
		]

		self.attention = AttentionLinear(threads + 1)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		"""Define the computation performed at every call.

		Should be overridden by all subclasses.

		NOTE: Although the recipe for forward pass needs to be defined within this function,
		one should call the `Module` instance afterwards instead of this,
		since the former takes care of running the registered hooks while the latter silently ignores them.
		"""
		return self.attention(torch.stack([stack(inputs) for stack in self.array], dim=-1))
