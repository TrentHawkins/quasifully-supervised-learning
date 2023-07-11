"""Custom `torch.nn` modules.

Layers:
	Customized `torch.nn.Linear` modules for prototyping simple Deep Neural Networks.

	`DropoutLinear`: a `torch.nn.Linear` module augmented with
		a `torch.nn.Dropout` input dropout module
		a `torch.nn.SiLU` pre-activation module

	`AttentionLinear`: a `torch.nn.Linear` module linearly combining multiple outputs (stacked in a higher dimension tensor) with:
		a `torch.nn.Sigmoid` hard-coded activation

	`MetricLinear`: a `torch.nn.Linear` module with a non-linear modification based on a similarity metric
	`CosineLinear`: a `MetricLinear` based on the cosine similarity index
	`JaccardLinear`: a `MetricLinear` based on the Jaccard similarity index
	`DiceLinear`: a `MetricLinear` based on the Dice similarity index

Models:
	Custom combinations of aforementioned custom layers.

	`LinearStack`: a pyramid-like `torch.nn.Sequential` module made of `DropoutLinear` submodules in a stack
	`LinearStackArray`: several `LinearStack` submodules in parallel combining output with an `AttentionLinear` submodule
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import scipy.special
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
			`kernel`: initial weight values (frozen)
		"""
		super(MetricLinear, self).__init__(*kernel.size(), bias=False, **kwargs)

	#	frozen kernel:
		assert self.weight.size() == kernel.size()
		self.weight = kernel  # no need to transpose `kernel` as `self.weight` is already transposed
		self.weight.requires_grad = False

	#	the norms of kernel vectors (diagonal):
		self.kernel_norms = torch.einsum("ij, ij -> i",
			self.weight,
			self.weight,
		)


class CosineLinear(MetricLinear):
	"""A `MetricLinear` based on the cosine similarity index.

	Such a modified module has no bias explicitely and is frozen.
	"""

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		"""Define the computation performed at every call.

		Formula:
			`torch.nn.Linear`: $y_{i}=\\sum_{j}w_{ij}x_{j}$
			$$y_{i}=\\dfrac{\\sum_{j}w_{ij}x_{j}}{\\sum_{j}w_{ij}w_{ij}\\sum_{j}x_{j}x_{j}}$$
		"""
		output = super(CosineLinear, self).forward(inputs)

	#	the norms of inputs vectors:
		inputs_norms = torch.einsum("...i, ...i -> ...",
			inputs,
			inputs,
		)

		return output / torch.sqrt(inputs_norms.unsqueeze(-1) * self.kernel_norms.expand(output.size()))


class JaccardLinear(MetricLinear):
	"""A `MetricLinear` based on the Jaccard similarity index.

	Such a modified module has no bias explicitely.
	"""

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		"""Define the computation performed at every call.

		Formula:
			`torch.nn.Linear`: $y_{i}=\\sum_{j}w_{ij}x_{j}$
			$$y_{i}=\\dfrac{\\sum_{j}w_{ij}x_{j}}{\\sum_{j}w_{ij}w_{ij}+\\sum_{j}x_{j}x_{j}-\\sum_{j}w_{ij}x_{j}}$$
		"""
		output = super(JaccardLinear, self).forward(inputs)

	#	the norms of inputs vectors:
		inputs_norms = torch.einsum("...i, ...i -> ...",
			inputs,
			inputs,
		)

		return output / (inputs_norms.unsqueeze(-1) + self.kernel_norms.expand(output.size()) - output)


class DiceLinear(MetricLinear):
	"""A `MetricLinear` based on the Dice similarity index.

	Such a modified module has no bias explicitely.
	"""

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		"""Define the computation performed at every call.

		Formula:
			`torch.nn.Linear`: $y_{i}=\\sum_{j}w_{ij}x_{j}$
			$$y_{i}=\\dfrac{\\sum_{j}w_{ij}x_{j}}{\\sum_{j}w_{ij}w_{ij}+\\sum_{j}x_{j}x_{j}-\\sum_{j}w_{ij}x_{j}}$$
		"""
		output = super(DiceLinear, self).forward(inputs)

	#	the norms of inputs vectors:
		inputs_norms = torch.einsum("...i, ...i -> ...",
			inputs,
			inputs,
		)

		return output / ((inputs_norms.unsqueeze(-1) + self.kernel_norms.expand(output.size())) / 2)


def LinearStack(
	inputs_size: int,
	output_size: int, skip: int = 1, dropout: float = .0
) -> torch.nn.Sequential:
	"""A pyramid-like `torch.nn.Sequential` module made of `DropoutLinear` submodules in a stack.

	The network complexity is defined in a specific way which is based on inputs and output sizes:
	-	the hidden sizes gradually change from one to the other with complexities defined by an integer divisor logic
	-	the depth of the change is adjustable

	Arguments:
		`inputs_dim`: size of last inputs dimension
		`output_dim`: size of last output dimension

	keyword arguments:
		`skip`: the (inverse) depth of the linear layer stack (default full depth)
		`dropout`: logit of dropout factor applied on input of dense layers in dense layer stack (default half probability)

	Returns:
		`torch.nn.Sequential` module with predefined linear submodules
	"""
	sizes = hidden_sizes(
		inputs_size,
		output_size, skip=skip
	)

#	linear stack:
	stack = torch.nn.Sequential(
		*[
			DropoutLinear(
				inputs_size,
				output_size, dropout=dropout,
			) for inputs_size, output_size in zip(sizes[:-1], sizes[1:])
		]
	)

	return stack


class LinearStackArray(torch.nn.Module):
	"""Several `LinearStack` submodules in parallel combining output with an `AttentionLinear` submodule

	The number of linear stacks (threads) is defined by the inputs and output sizes.
	The linear stack array recombines the linear stacks (threads) with attention.
	"""

	def __init__(self,
		inputs_size: int,
		output_size: int, threads: Optional[int] = None, dropout: float = .0,
	**kwargs):
		"""Hyperparametrize the `LinearStackArray` module.

		Arguments:
			`inputs_dim`: size of last inputs dimension
			`output_dim`: size of last output dimension

		keyword arguments:
			`threads`: the number of (parallel) dense layer stacks to build (default base thread only)
			`dropout`: logit of dropout factor applied on input of dense layers in dense layer stack (default half probability)
		"""
		super(LinearStackArray, self).__init__(**kwargs)

	#	architecture metadata:
		skips = divisors(len(hidden_sizes(inputs_size, output_size)) - 1, reverse=True)
		threads = len(skips) if threads is None else threads

	#	linear stack array:
		assert threads <= len(skips)
		self.array = [
			LinearStack(
				inputs_size,
				output_size, skip=skip, dropout=dropout,
			) for skip in divisors(len(hidden_sizes(inputs_size, output_size)) - 1, reverse=True)[0:threads]
		]

	#	attention:
		self.attention = AttentionLinear(threads)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		"""Define the computation performed at every call.

		Call stack:
				`LinearStack` of base depth (parallel thread)
				...
				`LinearStack` of full depth (parallel thread)
			`AttentionLinear` (on all threads stacked)
		"""
		return self.attention(torch.stack([stack(inputs) for stack in self.array], dim=-1))
