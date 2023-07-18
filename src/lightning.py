"""Pytorch lightning wrappers.

Includes:
	`AnimalsWithAttributesDataModule`: based on `AnimalsWithAttribulesDataLoader` on a `AnimalsWithAttributesDataset`
	`AnimalsWithAttributesModule`: composing several `torch.nn.Module` with a loss and an optimizer
"""


from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from .chartools import separator
from .globals import generator

import torch
import torch.utils.data
import torchmetrics
import pytorch_lightning

from src.torch.utils.data import AnimalsWithAttributesDataLoader
from src.torchvision.datasets import AnimalsWithAttributesDataset


torch.utils.data.ConcatDataset.__str__ = torch.utils.data.Dataset.__str__


class AnimalsWithAttributesDataModule(pytorch_lightning.LightningDataModule):
	"""A lightning data module encapsulating datasplitting and loading depending on context.

	Contexts:
		normal training:
		-	all labels known during training
		-	all labels known during  testing

		generalized_zeroshot training:
		-	only source labels known during training
		-	all         labels known during  testing

		generalized_zeroshot training in the transductive setting:
		-	source labels     known during training
		-	target labels not known during training but samples with such are known and unlabelled
		-	all    labels     known during  testing
	"""

	def __init__(self,
		labels_path: str = "allclasses.txt",
		source_path: str = "trainvalclasses.txt",
		target_path: str = "testclasses.txt",
		batch_size: int = 1,
		generalized_zeroshot: bool = False,
		transductive_setting: bool = False,
	**kwargs):
		"""Set `AnimalsWithAttributesDataModule` metadata.

		Arguments:
			`source_path`: path to source labels (usually known during training)
			`target_path`: path to target labels (usually known during  testing)

			`batch_size`: for training/validating/testing/predicting

			`generalized_zeroshot`: set to `True` to separate dataset based on source/target labels
			`transductive_setting`: set to `True` to allocate samples with a target label as unlabelled during training
		"""
		super(AnimalsWithAttributesDataModule, self).__init__(**kwargs)

	#	Label subset paths:
		self.labels_path = labels_path
		self.source_path = source_path
		self.target_path = target_path

	#	Batch size:
		self.batch_size = batch_size

	#	Setting:
		self.generalized_zeroshot = generalized_zeroshot
		self.transductive_setting = transductive_setting if self.generalized_zeroshot else False

	def prepare_data(self):
		"""Create datasets for accessing Animals with Attributes data based on label.

		Attributes:
			`source`: subset containing examples of source label
			`target`: subset containing examples of target label

			`totals`: a concatenated view of `source` and `target` for access as a whole dataset
		"""
		separator(3)
		separator(2, f"Animals with Attributes 2: reading")

	#	Images with source label:
		self.source = AnimalsWithAttributesDataset(
			labels_path=self.source_path,
		)

		separator(1, str(self.source))

	#	Images with target label:
		self.target = AnimalsWithAttributesDataset(
			labels_path=self.target_path,
		)

		separator(1, str(self.target))

	#	Images with anyall label:
		self.totals = AnimalsWithAttributesDataset(
			labels_path=self.labels_path,
		)
		separator(2, str(self.totals))

	def setup(self, stage: str):
		"""Create subsets for training/validating/testing/predicting.

		Arguments:
			`stage`: the staging/`stage`/name correpsondence is:
			-	training  :`'fit'`     :`train`
			-	validation:`'validate'`:`devel`
			-	testing   :`'test'`    :`valid`
			-	prediction:`'predict'` :`valid`
		"""
		separator(2, f"Animals with Attributes 2: spliting")

	#	If training with unknown labels:
		if self.generalized_zeroshot:
			(
				source_train_images,
				source_devel_images,
				source_valid_images,
			) = self.source.random_split(
				len(self.source),
				len(self.target),
			)
			(
				target_train_images,
				target_devel_images,
				target_valid_images,
			) = self.target.random_split(
				len(self.source),
				len(self.target),
			)

		#	If unlabelled samples allowed during training:
			if self.transductive_setting:
				separator(3, f"Animals with Attributes 2: data to {stage} in generalized zeroshot transductive setting")

				if stage == "fit":
					self.train_images = torch.utils.data.ConcatDataset(
						[
							source_train_images,
							target_train_images,
						]
					)

				if stage == "validate":
					self.devel_images = torch.utils.data.ConcatDataset(
						[
							source_devel_images,
							target_devel_images,
						]
					)

				if stage == "test" or stage == "predict":
					self.valid_images = torch.utils.data.ConcatDataset(
						[
							source_valid_images,
							target_valid_images,
						]
					)

		#	if only source labels allowed during training:
			else:
				separator(3, f"Animals with Attributes 2: data to {stage} in generalized zeroshot")

				if stage == "fit":
					self.train_images = torch.utils.data.ConcatDataset(
						[
							source_train_images,
						]
					)

				if stage == "validate":
					self.devel_images = torch.utils.data.ConcatDataset(
						[
							source_devel_images,
						]
					)

				if stage == "test" or stage == "predict":
					self.valid_images = torch.utils.data.ConcatDataset(
						[
							target_train_images,
							target_devel_images,
							source_valid_images,
							target_valid_images,
						]
					)

	#	If training normally:
		else:
			(
				train_images,
				devel_images,
				valid_images,
			) = self.totals.random_split(
				len(self.source),
				len(self.target),
			)

			separator(3, f"Animals with Attributes 2: data to {stage}")

			if stage == "fit":
				self.train_images = train_images

			if stage == "validate":
				self.devel_images = devel_images

			if stage == "test" or stage == "predict":
				self.valid_images = valid_images

	def train_dataloader(self) -> AnimalsWithAttributesDataLoader:
		"""Create dataloader for fitting.

		Returns:
			`AnimalsWithAttributesDataLoader(torch.utils.dataDataloader)` on a subset
		"""
		return AnimalsWithAttributesDataLoader(self.train_images, generator, batch_size=self.batch_size)

	def val_dataloader(self) -> AnimalsWithAttributesDataLoader:
		"""Create dataloader for validating.

		Returns:
			`AnimalsWithAttributesDataLoader(torch.utils.dataDataloader)` on a subset
		"""
		return AnimalsWithAttributesDataLoader(self.devel_images, generator, batch_size=self.batch_size)

	def test_dataloader(self) -> AnimalsWithAttributesDataLoader:
		"""Create dataloader for testing.

		Returns:
			`AnimalsWithAttributesDataLoader(torch.utils.dataDataloader)` on a subset
		"""
		return AnimalsWithAttributesDataLoader(self.valid_images, generator, batch_size=self.batch_size)

	def predict_dataloader(self) -> AnimalsWithAttributesDataLoader:
		"""Create dataloader for predicting.

		Returns:
			`AnimalsWithAttributesDataLoader(torch.utils.dataDataloader)` on a subset
		"""
		return AnimalsWithAttributesDataLoader(self.valid_images, generator, batch_size=self.batch_size)


class GeneralizedZeroshotModule(pytorch_lightning.LightningModule):
	"""Full stack model for training on any visual `pytorch_lightning.DataModule` in a generalized zeroshot setting.

	Submodules:
		`visual`: translate images into visual features
		`visual_semantic`: translate visual features into semantic features
		`semantic`: translate semantic features into (fuzzy similarity) labels
		`loss`: compare fuzzy sigmoid predicitons to "many-hot" binary multi-label truths
	"""

	def __init__(self,
		visual: torch.nn.Module,
		visual_semantic: torch.nn.Module,
		semantic: torch.nn.Module,
		loss: torch.nn.Module,
	):
		"""Instansiate model stack with given subcomponents.

		Arguments:
			`visual`: translate images into visual features
			`visual_semantic`: translate visual features into semantic features
			`semantic`: translate semantic features into (fuzzy similarity) labels
		"""
		super().__init__()

		self.visual = visual
		self.visual_semantic = visual_semantic
		self.semantic = semantic
		self.loss = loss

	#	Accuracy monitoring:
		self.accuracy: torchmetrics.Metric = torchmetrics.Accuracy("multilabel",
			threshold=0.5,
			num_classes=None,
			num_labels=None,
			average="micro",
			multidim_average="global",
			top_k=1,
			ignore_index=None,
			validate_args=True,
		)

	#	Dictionary of metrics:
		self.metrics = {}

	def configure_optimizers(self) -> torch.optim.Optimizer:
		"""Choose what optimizers and learning-rate schedulers to use in your optimization.

		Normally you’d need one.
		But in the case of GANs or similar you might have multiple.
		Optimization with multiple optimizers only works in the manual optimization mode.

		Returns:
			`torch.optim.Optimizer` with default settings
		"""
		return torch.optim.Adam(
			self.parameters(),
		#	lr=0.001,
		#	betas=(
		#		0.9,
		#		0.999,
		#	),
		#	eps=1e-08,
		#	weight_decay=0,
		#	amsgrad=False,
		#	foreach=None,
		#	maximize=False,
		#	capturable=False,
		#	differentiable=False,
		#	fused=None,
		)

	def _shared_eval_step(self, batch: torch.Tensor, batch_idx: int, stage: str) -> dict[str, torchmetrics.Metric]:
		"""Do calculations shared across different stages.

		Base function for:
			`training_step`
			`validation_step`
			`test_step

		Arguments:
			`batch`: current batch
			`batch_idx`: index of current batch

		Returns:
			dictionary of metrics including loss
		"""
		x, y_true = batch

	#	Model forward:
		y_pred = self.semantic(self.visual_semantic(self.visual(x)))

	#	Update metrics:
		self.metrics.update(
			{
				f"{stage}_loss": self.loss(
					y_pred,
					y_true,
				),
				f"{stage}_accuracy": self.accuracy(
					y_pred,
					y_true,
				)
			}
		)

		return self.metrics

	def training_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
		"""Here you compute and return the training loss and some additional metrics for e.g. the progress bar or
		logger.

		Args:
			batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
				The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
			batch_idx (``int``): Integer displaying index of this batch

		Return:
			Any of.

			- :class:`~torch.Tensor` - The loss tensor
			- ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``
			- ``None`` - Training will skip to the next batch. This is only for automatic optimization.
				This is not supported for multi-GPU, TPU, IPU, or DeepSpeed.

		In this step you'd normally do the forward pass and calculate the loss for a batch.
		You can also do fancier things like multiple forward passes or something model specific.

		Example::

			def training_step(self, batch, batch_idx):
				x, y, z = batch
				out = self.encoder(x)
				loss = self.loss(out, x)
				return loss
		"""
		return self._shared_eval_step(batch, batch_idx, "fit")
