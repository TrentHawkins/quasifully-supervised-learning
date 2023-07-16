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

	[https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html]

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
	>>>	source = [1, 2, 3, 5, 7]
	>>>	target = [0, 4, 6, 8, 9]
	>>>	y_true = torch.ones([10, 10], dtype=torch.float32)  # 10 classes, batch size = 10
	>>>	y_pred = torch.full([10, 10], 1.5)  # a prediction (logit)
	>>>	pos_weight = torch.ones([10])  # all weights are equal to 1
	>>>	criterion = ZeroshotBCELoss(
	>>>		source,
	>>>		target, pos_weight=pos_weight
	>>>	)
	>>>	criterion(
	>>>		y_pred,  # `y_pred_source.size() == (10, 5)`
	>>>		y_true,  # `y_true_source.size() == (10, 5)`
	>>>	)
	```

	NOTE: This modification applies binary cross-entropy loss with logits filtered by a subcollection of labels (source).
	"""

	def __init__(self,
		source: Iterable[int], pos_weight: Optional[torch.Tensor] = None,
	**kwargs):
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
		**kwargs)

	#	Transform `pandas.Series` or other source label iterable to integer tensor:
		self._source = torch.Tensor(source).long()

	def forward(self, y_pred, y_true):
		"""Clip all samples with a target label from evaluation.

		Arguments:
			`y_pred`: prediction logit of any label
			`y_true`: true probability of any label

		Shape:
			- Input: `(*)`, where `*` means any number of dimensions.
			- Target: `(*)`, same shape as the input.
			- Output: scalar. If `reduction` is `"none"`, then `(*)`, same
				shape as input.

		Returns:
			`torch.nn.BCEWithLogitsLoss` on the samples with a source label

		Examples
		```
		>>>	loss = ZeroshotBCELoss(
		>>>		source,
		>>>	)
		>>>	y_pred = torch.randn(3, requires_grad=True)
		>>>	y_true = torch.empty(3).random_(2)
		>>>	output = loss(
		>>>		y_pred,
		>>>		y_true,
		>>>	)
		>>>	output.backward()
		```
		"""
		y_pred_source = torch.gather(y_pred, -1, self._source.expand(*self._source.size()[:-1], -1))  # clip target labels
		y_true_source = torch.gather(y_true, -1, self._source.expand(*self._source.size()[:-1], -1))  # clip target labels

		return super(ZeroshotBCELoss, self).forward(
			y_pred_source,
			y_true_source,
		)


class QuasifullyZeroshotBCELoss(ZeroshotBCELoss):
	"""This loss combines a `Sigmoid` layer and the `BCELoss` in one single class.

	[https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html]

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
	>>>	source = [1, 2, 3, 5, 7]
	>>>	target = [0, 4, 6, 8, 9]
	>>>	y_true = torch.ones([10, 10], dtype=torch.float32)  # 10 classes, batch size = 10
	>>>	y_pred = torch.full([10, 10], 1.5)  # a prediction (logit)
	>>>	pos_weight = torch.ones([10])  # all weights are equal to 1
	>>>	criterion = QuasifullyZeroshotBCELoss(
	>>>		source,
	>>>		target, pos_weight=pos_weight
	>>>	)
	>>>	criterion(
	>>>		y_pred,  # `y_pred_source.size() == (10, 5)` and `y_pred_target.size() == (10, 5)`
	>>>		y_true,  # `y_true_source.size() == (10, 5)`
	>>>	)
	```

	NOTE: This modification applies binary cross-entropy loss with logits filtered by a subcollection of labels (source).
	It also applied a modified bias to loss accounting on unlabelled samples (with a target label unknown in training).
	"""

	def __init__(self,
		source: Iterable[int],
		target: Iterable[int], pos_weight: Optional[torch.Tensor] = None,
	**kwargs):
		"""Provide loss with a source labels filter.

		Arguments:
			`pos_weight`: a weight of positive examples. Must be a vector with length equal to the number of classes.
			`source`: filter
		"""
		super(QuasifullyZeroshotBCELoss, self).__init__(source,
			pos_weight=pos_weight,
		**kwargs)

	#	Transform `pandas.Series` or other source label iterable to integer tensor:
		self._target = torch.Tensor(target).long()

	#	Quasifully loss bias operations:
		self._reduce = torch.sum  # inside a logarithm sum and mean differ by an additive fixed (batch size) constant
		self._log = torch.log  # use the same log in `torch.nn.BCELoss`

	def forward(self, y_pred, y_true):
		"""Clip all samples with a target label from evaluation.

		Arguments:
			`y_pred`: prediction logit of any label
			`y_true`: true probability of any label

		Shape:
			- Input: `(*)`, where `*` means any number of dimensions.
			- Target: `(*)`, same shape as the input.
			- Output: scalar. If `reduction` is `"none"`, then `(*)`, same shape as input.

		Returns:
			`torch.nn.BCEWithLogitsLoss` on the samples with a source label plus a term depending on unlabelled samples.

		Examples
		```
		>>>	loss = QuasifullyZeroshotBCELoss(
		>>>		source,
		>>>		target,
		>>>	)
		>>>	y_pred = torch.randn(3, requires_grad=True)
		>>>	y_true = torch.empty(3).random_(2)
		>>>	output = loss(
		>>>		y_pred,
		>>>		y_true,
		>>>	)
		>>>	output.backward()
		```
		"""
		y_pred_target = torch.gather(y_pred, -1, self._target.expand(*self._target.size()[:-1], -1))  # clip source labels
		y_pred_target = torch.nn.functional.sigmoid(y_pred_target)  # turn to a probability

		return super(QuasifullyZeroshotBCELoss, self).forward(
			y_pred,
			y_true,
		) + self._log(self._reduce(y_pred_target, -1)) + self._log(self._reduce(1. - y_pred_target, -1)) / len(y_pred_target)
