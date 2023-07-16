"""Custom `torch.nn` modules.

Losses:
	`ZeroshotBCELoss`:
	`QuasifullyZeroshotBCEloss`:
"""


from __future__ import annotations

from typing import Iterable, Optional

import torch


class ZeroshotBCELoss(torch.nn.BCEWithLogitsLoss):
	"""This loss combines a `Sigmoid` layer and the `BCELoss` in one single class.

	This version is more numerically stable than using a plain `Sigmoid` followed by a `BCELoss` as,
	by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.

	The unreduced (i.e. with `reduction` set to `"none"`) loss can be described as:
		$$\\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top,$$
		$$l_n = - w_n \\left[ y_n \\cdot \\log \\sigma(x_n) + (1 - y_n) \\cdot \\log (1 - \\sigma(x_n)) \\right],$$
	where $N$ is the batch size.

	If `reduction` is not `"none"` (default `"mean"`), then
		$$\\ell(x, y) = \\begin{cases}
			\\operatorname{mean}(L), & \\text{if reduction} = \\text{`mean";}\\\\
			\\operatorname{sum}(L),  & \\text{if reduction} = \\text{`sum".}
		\\end{cases}$$

	This is used for measuring the error of a reconstruction in for example an auto-encoder.
	Note that the targets $y_n$ should be numbers between 0 and 1.

	It is possible to trade off recall and precision by adding weights to positive examples.
	In the case of multi-label classification the loss can be described as:
		$$\\ell_c(x, y) = L_c = \\{l_{1,c},\\dots,l_{N,c}\\}^\\top,$$
		$$l_{n,c} = - w_{n,c} \\left[ p_c y_{n,c} \\cdot \\log \\sigma(x_{n,c})
		+ (1 - y_{n,c}) \\cdot \\log (1 - \\sigma(x_{n,c})) \\right],$$
	where
	-	$c$ is the class number,
		-	$c > 1$ for multi-label binary classification,
		-	$c = 1$ for single-label binary classification;
	-	$n$ is the number of the sample in the batch, and
	-	$p_c$ is the weight of the positive answer for the class $c$.
		-	$p_c > 1$ increases the recall,
		-	$p_c < 1$ increases the precision.

	For example, if a dataset contains 100 positive and 300 negative examples of a single class,
	then `pos_weight` for the class should be equal to $\\frac{300}{100}=3$.
	The loss would act as if the dataset contains $3\\times 100=300$ positive examples.

	Examples
	```
		>>> target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
		>>> output = torch.full([10, 64], 1.5)  # A prediction (logit)
		>>> pos_weight = torch.ones([64])  # All weights are equal to 1
		>>> criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
		>>> criterion(output, target)  # -log(sigmoid(1.5)) tensor(0.20...)
	```

	Argumentss:
	Shape:
		- Input: `(*)`, where `*` means any number of dimensions.
		- Target: `(*)`, same shape as the input.
		- Output: scalar. If `reduction` is `"none"`, then `(*)`, same
			shape as input.

	Examples
	```
		>>> loss = nn.BCEWithLogitsLoss()
		>>> input = torch.randn(3, requires_grad=True)
		>>> target = torch.empty(3).random_(2)
		>>> output = loss(input, target)
		>>> output.backward()
	```
	"""

	def __init__(self, source: Iterable[int], pos_weight: Optional[torch.Tensor] = None):
		"""Provide loss with a source labels filter.

		Arguments:
			`pos_weight`: a weight of positive examples.
				Must be a vector with length equal to the number of classes.
			`source`: filter
		"""
		super(ZeroshotBCELoss, self).__init__(
		#	weight=None,
		#	size_average=None,
		#	reduce=None,
		#	reduction="mean",
			pos_weight=pos_weight,
		)

	#	Transform `pandas.Series` or other source label iterable to integer tensor:
		self._source = torch.Tensor(source).long()

	def forward(self, y_pred, y_true):
		"""Clip all samples with a target label from evaluation.

		Arguments:
			`y_pred`: prediction logit of any label
			`y_true`: true probability of any label

		Returns:
			`torch.nn.BCEWithLogitsLoss` on the samples with a source label
		"""
		return super(ZeroshotBCELoss, self).forward(
			torch.gather(y_pred, -1, self._source.expand(*self._source.size()[:-1], -1)),  # clip target label contributions
			torch.gather(y_true, -1, self._source.expand(*self._source.size()[:-1], -1)),  # clip target label contributions
		)
