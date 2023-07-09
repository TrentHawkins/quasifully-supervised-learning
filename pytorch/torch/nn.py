"""Custom modules.

Compound dense layer containing:
-	dropout
-	dense sublayer

Basic attention layer assigning a trainable weight for each input in an array of inputs.

Metric sigmoid-like non-linear dense layers using similarity metrics in their definition:
-	based on the jaccard similarity metric (sigmoid-like only on sigmoid-like features)
-	based on the cosine similarity metric (sigmoid-like always)

Linear layer stack.
-	Basically a sequential of dense layers with uniform settings across, apart for the top.

Linear layer stack array.
-	An colelction of dense sequentials used in parallel.
"""

from __future__ import annotations

import scipy.special
import torch


class DropoutLinear(torch.nn.Linear):
	"""Custom compound dense layer.

	Attribures:
		dense: the dense sublayer
	"""

	def __init__(self,
		inputs_shape: int,
		output_shape: int, dropout_rate: float = .0,
	**kwargs):
		"""Hyrparametrize base layer with dense topping.

		Arguments:
			units: number of neurons in layer

		Keyword arguments:
			dropout: logit of dropout factor applied on input of the layer
				default: half
		"""
		super(DropoutLinear, self).__init__(
			inputs_shape,
			output_shape, bias=True,
		**kwargs)

	#	dropout:
		self.dropout = torch.nn.Dropout(scipy.special.expit(dropout_rate))

	#	activation:
		self.activation = torch.nn.SiLU()  # hardcoded best internal neuron activator

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		"""Call the model on new inputs.

		In this case call just reapplies all ops in the graph to the new inputs.

		Arguments:
			inputs: a tensor or list of tensors

		Returns:
			output of layer
		"""
		return self.activation(super(DropoutLinear, self).forward(self.dropout(inputs)))


"""What follows are special types of dense layers."""


class AttentionLinear(torch.nn.Linear):
	"""Wrapper for dense layer operating on a stacks of input to recombine them with attention.

	Such a layer is expected to have no bias and be trainable with no dropout.
	Other dense features include activation only.
	Make sure the batch dimensions include the threads themselves:
		`len(shape(batch)) > 1`

	Call:
		stack: stack inputs horizontally
		dense: one weight for each input
		squeeze: eliminate redudant dims on output
	"""

	def __init__(self, threads: int, **kwargs):
		"""Hyperparametrize recombination layer.

		Keyword arguments:
			activation: to apply on output of decision
		"""
		super(AttentionLinear, self).__init__(threads, 1, bias=False, **kwargs)

		self.activation = torch.nn.Sigmoid()  # hardcoded semantic features renormalization

	def call(self, inputs: torch.Tensor) -> torch.Tensor:
		"""Call the model on new inputs.

		In this case call just reapplies all ops in the graph to the new inputs.

		Arguments:
			inputs: a tensor or list of tensors

		Returns:
			output of layer
		"""
		return self.activation(super(AttentionLinear, self).forward(inputs).squeeze(dim=-1))


class MetricLinear(torch.nn.Linear):
	"""A non-linear dense layer emulating the action of a metric and outputing in sigmoid range.

	Such a modified layer has no bias explicitely.
	"""

	def __init__(self, kernel: torch.Tensor, **kwargs):
		"""Hyperparametrize recombination layer.

		No activation is needed as these metric layers are manifestly non-linear to begin with.
		Option is left (and ignored) for compatibility with other dense-like layers.

		Arguments:
			kernel: weight values to begin with
		"""
		super(MetricLinear, self).__init__(*kernel.size(), bias=False, **kwargs)

	#	frozen semantic kernel:
		assert self.weight.dim() == kernel.transpose(0, 1).dim()
		self.weight = kernel.transpose(0, 1)
		self.weight.requires_grad = False

	#	probabilistic activation:
		self.activation = torch.nn.Softmax()


class CosineLinear(MetricLinear):
	"""A dense layer that perform the cosine operation per input and kernel vector instead of a dot product.

	Such a modified layer has no bias explicitely.
	"""

	def forward(self, inputs):
		"""Call the model on new inputs.

		In this case call just reapplies all ops in the graph to the new inputs.

		Arguments:
			inputs: a tensor or list of tensors

		Returns:
			output of layer
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
	"""A dense layer that perform the Jaccard operation per input and kernel vector instead of a dot product.

	Such a modified layer has no bias explicitely.
	"""

	def call(self, inputs):
		"""Call the model on new inputs.

		In this case call just reapplies all ops in the graph to the new inputs.

		Arguments:
			inputs: a tensor or list of tensors

		Returns:
			output of layer
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
	"""A dense layer that perform the Dice operation per input and kernel vector instead of a dot product.

	Such a modified layer has no bias explicitely.
	"""

	def call(self, inputs):
		"""Call the model on new inputs.

		In this case call just reapplies all ops in the graph to the new inputs.

		Arguments:
			inputs: a tensor or list of tensors

		Returns:
			output of layer
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
