"""Custom $torch.nn$ modules.

Losses:
	$ZeroshotBCELoss$:
	$QuasifullyZeroshotBCEloss$:
"""


from __future__ import annotations

from typing import Iterable, Optional

import torch


class ZeroshotBCELoss(torch.nn.BCELoss):
	r"""Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities:
	NOTE: This modification applies binary cross-entropy loss with logits filtered by a subcollection of labels (source).

	[https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html]

	The unreduced (i.e. with `reduction` set to `"none"`) loss can be described as:
		$$\ell(x, y) = L = \{l_1,\dots,l_N\}^\top,$$
		$$l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],$$
	where $N$ is the batch size. If `reduction` is not `"none"` (default `"mean"`), then
		$$\ell(x, y) = \begin{cases}
			\operatorname{mean}(L), & \text{if reduction} = \text{"mean";}\\
			\operatorname{sum}(L),  & \text{if reduction} = \text{"sum".}
		\end{cases}$$

	This is used for measuring the error of a reconstruction in for example an auto-encoder.
	Note that the targets $y$ should be numbers between 0 and 1.

	Notice that if $x_n$ is either 0 or 1, one of the log terms would be mathematically undefined in the above loss equation.
	PyTorch chooses to set $\log (0) = -\infty$, since $\lim_{x\to 0} \log (x) = -\infty$.
	However, an infinite term in the loss equation is not desirable for several reasons.

	For one, if either $y_n = 0$ or $(1 - y_n) = 0$, then we would be multiplying 0 with infinity.
	Secondly, if we have an infinite loss value, then we would also have an infinite term in our gradient,
	since $\lim_{x\to 0} \frac{d}{dx} \log (x) = \infty$.
	This would make the backward method of `torch.nn.BCELoss` nonlinear with respect to $x_n$,
	and using it for things like linear regression would not be straight-forward.

	Our solution is that `torch.nn.BCELoss` clamps its log function outputs to be greater than or equal to -100.
	This way, we can always have a finite loss value and a linear backward method.

	Examples
	```
	>>>	source = [1, 2, 3, 5, 7]
	>>>	y_true = torch.ones([10, 10], dtype=torch.float32)  # 10 classes, batch size = 10
	>>>	y_pred = torch.full([10, 10], 1.5)  # a prediction (logit)
	>>>	criterion = ZeroshotBCELoss(
	>>>		source,
	>>>	)
	>>>	criterion(
	>>>		y_pred,  # `y_pred_source.size() == (10, 5)`
	>>>		y_true,  # `y_true_source.size() == (10, 5)`
	>>>	)
	```
	"""

	def __init__(self,
		source: Iterable[int],
	**kwargs):
		"""Provide loss with a source labels filter.

		Arguments:
			$source$: filter
		"""
		super(ZeroshotBCELoss, self).__init__(**kwargs)

	#	Transform $pandas.Series$ or other source label iterable to integer tensor:
		self._source = torch.Tensor(source).long()

	def forward(self, y_pred, y_true):
		"""Clip all samples with a target label from evaluation.

		Arguments:
			`y_pred`: prediction logit of any label
			`y_true`: true probability of any label

		Shape:
			`y_pred`: `(*)`, where `*` means any number of dimensions.
			`y_true`: `(*)`, same shape as the input.
			`output`: scalar. If `reduction` is `"none"`, then `(*)`, same shape as input.

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
	r"""Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities:
	NOTE: This modification applies binary cross-entropy loss with logits filtered by a subcollection of labels (source).
	It also applied a modified bias to loss accounting on unlabelled samples (with a target label unknown in training).

	[https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html]

	The unreduced (i.e. with `reduction` set to `"none"`) loss can be described as:
		$$\ell(x, y) = L = \{l_1,\dots,l_N\}^\top,$$
		$$l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],$$
	where $N$ is the batch size. If `reduction` is not `"none"` (default `"mean"`), then
		$$\ell(x, y) = \begin{cases}
			\operatorname{mean}(L), & \text{if reduction} = \text{"mean";}\\
			\operatorname{sum}(L),  & \text{if reduction} = \text{"sum".}
		\end{cases}$$

	This is used for measuring the error of a reconstruction in for example an auto-encoder.
	Note that the targets $y$ should be numbers between 0 and 1.

	Notice that if $x_n$ is either 0 or 1, one of the log terms would be mathematically undefined in the above loss equation.
	PyTorch chooses to set $\log (0) = -\infty$, since $\lim_{x\to 0} \log (x) = -\infty$.
	However, an infinite term in the loss equation is not desirable for several reasons.

	For one, if either $y_n = 0$ or $(1 - y_n) = 0$, then we would be multiplying 0 with infinity.
	Secondly, if we have an infinite loss value, then we would also have an infinite term in our gradient,
	since $\lim_{x\to 0} \frac{d}{dx} \log (x) = \infty$.
	This would make the backward method of `torch.nn.BCELoss` nonlinear with respect to $x_n$,
	and using it for things like linear regression would not be straight-forward.

	Our solution is that `torch.nn.BCELoss` clamps its log function outputs to be greater than or equal to -100.
	This way, we can always have a finite loss value and a linear backward method.

	Examples
	```
	>>>	source = [1, 2, 3, 5, 7]
	>>>	target = [0, 4, 6, 8, 9]
	>>>	y_true = torch.ones([10, 10], dtype=torch.float32)  # 10 classes, batch size = 10
	>>>	y_pred = torch.full([10, 10], 1.5)  # a prediction (logit)
	>>>	pos_weight = torch.ones([10])  # all weights are equal to 1
	>>>	criterion = QuasifullyZeroshotBCELoss(
	>>>		source,
	>>>		target,
	>>>	)
	>>>	criterion(
	>>>		y_pred,  # `$`y_pred_source.size() == (10, 5)` and `y_pred_target.size() == (10, 5)`
	>>>		y_true,  # `$`y_true_source.size() == (10, 5)`
	>>>	)
	```
	"""

	def __init__(self,
		source: Iterable[int],
		target: Iterable[int], pos_weight: Optional[torch.Tensor] = None,
	**kwargs):
		"""Provide loss with a source labels filter.

		Arguments:
			$pos_weight$: a weight of positive examples. Must be a vector with length equal to the number of classes.
			$source$: filter
		"""
		super(QuasifullyZeroshotBCELoss, self).__init__(source,
			pos_weight=pos_weight,
		**kwargs)

	#	Transform $pandas.Series$ or other source label iterable to integer tensor:
		self._target = torch.Tensor(target).long()

	#	Quasifully loss bias operations:
		self._reduce = torch.sum  # inside a logarithm sum and mean differ by an additive fixed (batch size) constant
		self._log = torch.log  # use the same log in $torch.nn.BCELoss$

	def forward(self, y_pred, y_true):
		"""Clip all samples with a target label from evaluation.

		Arguments:
			`y_pred`: prediction logit of any label
			`y_true`: true probability of any label

		Shape:
			`y_pred`: `(*)`, where `*` means any number of dimensions.
			`y_true`: `(*)`, same shape as the input.
			`output`: scalar. If `reduction` is `"none"`, then `(*)`, same shape as input.

		Returns:
			`torch.nn.BCEWithLogitsLoss` on the samples with a source label plus bias on unlabelled samples

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
