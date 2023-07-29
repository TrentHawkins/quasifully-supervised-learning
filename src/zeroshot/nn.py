"""Custom $torch.nn$ modules.

Losses:
	$ZeroshotBCELoss$:
	$QuasifullyZeroshotBCEloss$:
"""


from __future__ import annotations

from typing import Iterable

import torch


class ZeroshotLoss(torch.nn.Module):
	r"""Apply given loss filtered by a subcollection of labels (source).

	Example:
	```
	>>>	source = [1, 2, 3, 5, 7]
	>>>	y_true = torch.ones([10, 10], dtype=torch.float32)  # 10 classes, batch size = 10
	>>>	y_pred = torch.full([10, 10], 1.5)  # a prediction (logit)
	>>>	criterion = ZeroshotLoss(torch.nn.BCELoss(),
	>>>		source,
	>>>	)
	>>>	criterion(
	>>>		y_pred,  # `y_pred_source.size() == (10, 5)`
	>>>		y_true,  # `y_true_source.size() == (10, 5)`
	>>>	)
	```
	"""

	def __init__(self, loss: torch.nn.Module,
		source: Iterable[int],
	**kwargs):
		"""Provide loss with a source labels filter.

		Arguments:
			source: filter
		"""
		super(ZeroshotLoss, self).__init__(**kwargs)

	#	Loss:
		self.loss = loss

	#	Transform `pandas.Series` or other source label iterable to integer tensor:
		self._source = torch.Tensor(source).long()

	def forward(self, y_pred, y_true):
		"""Clip all samples with a target label from evaluation.

		Arguments:
			y_pred: prediction logit of any label
			y_true: true probability of any label

		Shape:
			y_pred: `(*)`, where `*` means any number of dimensions.
			y_true: `(*)`, same shape as the input.
			output: scalar. If `reduction` is `"none"`, then `(*)`, same shape as input.

		Returns:
			loss on the samples with a source label

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

		return self.loss.forward(
			y_pred_source,
			y_true_source,
		)


class QuasifullyZeroshotLoss(ZeroshotLoss):
	r"""Apply given loss with logits filtered by a subcollection of labels (source) plus bias accounting for unlabelled samples.

	Example:
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

	def __init__(self, loss: torch.nn.Module,
		source: Iterable[int],
		target: Iterable[int],
	**kwargs):
		"""Provide loss with a source labels filter.

		Arguments:
			source: filter
			target: filter
		"""
		super(QuasifullyZeroshotLoss, self).__init__(loss, source, **kwargs)

	#	Transform $pandas.Series$ or other source label iterable to integer tensor:
		self._target = torch.Tensor(target).long()

	#	Quasifully loss bias operations:
		self._reduce = torch.sum  # inside a logarithm sum and mean differ by an additive fixed (batch size) constant
		self._log = torch.log  # use the same log in $torch.nn.BCELoss$

	def forward(self, y_pred, y_true):
		"""Clip all samples with a target label from evaluation.

		Arguments:
			y_pred: prediction logit of any label
			y_true: true probability of any label

		Shape:
			y_pred: `(*)`, where `*` means any number of dimensions.
			y_true: `(*)`, same shape as the input.
			output: scalar. If `reduction` is `"none"`, then `(*)`, same shape as input.

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
	#	y_true_target = torch.gather(y_true, -1, self._target.expand(*self._target.size()[:-1], -1))  # clip source labels

		return super(QuasifullyZeroshotLoss, self).forward(
			y_pred,
			y_true,
		) + self._log(self._reduce(y_pred_target, -1)) + self._log(self._reduce(1. - y_pred_target, -1)) / len(y_pred_target)
