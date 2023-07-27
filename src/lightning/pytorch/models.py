"""Pytorch lightning wrappers.

Includes:
	`AnimalsWithAttributesDataModule`: based on `AnimalsWithAttribulesDataLoader` on a `AnimalsWithAttributesDataset`
	`AnimalsWithAttributesModule`: composing several `torch.nn.Module` with a loss and an optimizer
"""


import torch
import torch.utils.data
import torchmetrics
import lightning.pytorch
import lightning.pytorch.callbacks


class GeneralizedZeroshotModule(lightning.pytorch.LightningModule):
	"""Full stack model for training on any visual `lightning.pytorch.DataModule` in a generalized zeroshot setting.

	[https://lightning.ai/docs/pytorch/latest/common/lightning_module.html]

	Submodules:
		`visual`: translate images into visual features
		`latent`: translate visual features into semantic features
		`alphas`: translate semantic features into (fuzzy similarity) labels
		`loss_f`: compare fuzzy sigmoid predicitons to "many-hot" binary multi-label truths
	"""

	def __init__(self,
		visual: torch.nn.Module,
		latent: torch.nn.Module,
		alphas: torch.nn.Module,
		loss_f: torch.nn.Module,
		*,
		patience: int = 3,
		learning_rate: float = 1e-3,
	):
		"""Instansiate model stack with given subcomponents.

		Arguments:
			`visual`: translate images into visual features
			`latent`: translate visual features into semantic features
			`alphas`: translate semantic features into (fuzzy similarity) labels
			`loss_f`: compare fuzzy sigmoid predicitons to "many-hot" binary multi-label truths

		Keyword Arguments:
			`patience`: of `lightning.pytorch.callbacks.EarlyStopping` callback
			`learning_rate`: of `torch.optim.Adam` optimizer
		"""
		super().__init__()

		self.visual = visual
		self.latent = latent
		self.alphas = alphas
		self.loss_f = loss_f

	#	Accuracy monitoring:
		self.accuracy: torchmetrics.Metric = torchmetrics.Accuracy("multilabel",
		#	threshold=0.5,
		#	num_classes=None,
			num_labels=50,  # HACK
		#	average="micro",
		#	multidim_average="global",
		#	top_k=1,
		#	ignore_index=None,
		#	validate_args=True,
		)

	#	Early stopping:
		self.patience = patience

	#	Adam optimizer:
		self.learning_rate = learning_rate

	def configure_callbacks(self) -> list[lightning.pytorch.Callback]:
		"""Configure model-specific callbacks.

		When the model gets attached, e.g., when `.fit()` or `.test()` gets called, the list or a callback returned here
		will be merged with the list of callbacks passed to the Trainer's `callbacks` argument.

		If a callback returned here has the same type as one or several callbacks
		already present in the Trainer's `callbacks` list, it will take priority and replace them.

		In addition, Lightning will make sure `lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint` callbacks run last.

		Returns:
			a list of callbacks which will extend the list of callbacks in the Trainer

		Example:
		```
		def configure_callbacks(self):
			early_stop = EarlyStopping(monitor="val_acc", mode="max")
			checkpoint = ModelCheckpoint(monitor="val_loss")
			return [early_stop, checkpoint]
		```
		"""
		return [
			lightning.pytorch.callbacks.EarlyStopping("devel_loss",
			#	min_delta=0,
				patience=self.patience,
				verbose=True,
			#	mode="min",
			#	strict=True,
			#	check_finite=True,
			#	stopping_threshold=None,
			#	divergence_threshold=None,
			#	check_on_train_epoch_end=None,
			#	log_rank_zero_only=False,
			),
		]

	def configure_optimizers(self) -> torch.optim.Optimizer:
		"""Choose what optimizers and learning-rate schedulers to use in your optimization.

		[https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers]

		Normally you would need one.
		But in the case of GANs or similar you might have multiple.
		Optimization with multiple optimizers only works in the manual optimization mode.

		Returns:
			`torch.optim.Optimizer` with default settings
		"""
		return torch.optim.Adam(self.parameters(),
			lr=self.learning_rate,
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

		[https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#child-modules]

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
		y_pred, y_true = batch

	#	Model forward:
		y_pred = self.visual(y_pred)
		y_pred = self.latent(y_pred)
		y_pred = self.alphas(y_pred)

	#	Update metrics:
		metrics = {
			f"{stage}_loss": self.loss_f(
				y_pred,
				y_true,
			),
			f"{stage}_accuracy": self.accuracy(
				y_pred,
				y_true,
			)
		}

	#	Log metrics:
		self.log_dict(metrics)

		return metrics

	def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torchmetrics.Metric]:
		"""Compute and return the training loss and some additional metrics for e.g. the progress bar or logger.

		[https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#training-step]

		Arguments:
			`batch`: the tensor output of your `torch.utils.data.DataLoader`
			`batch_idx`: integer displaying index of this batch

		Returns:
			a dictionary that can include any keys, but must include the key `"loss"` with `torch.Tensor` loss value

		In this step you would normally do the forward pass and calculate the loss for a batch.
		You can also do fancier things like multiple forward passes or something model specific.

		Example
		``
		def training_step(self, batch, batch_idx):
			x, y, z = batch
			out = self.encoder(x)
			loss = self.loss(out, x)
			return loss
		``
		"""
		return self._shared_eval_step(batch, batch_idx, "train")

	def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torchmetrics.Metric]:
		"""Operates on a single batch of data from the validation set.

		In this step you would might generate examples or calculate anything of interest like accuracy.

		[https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#training-step]

		Arguments:
			`batch`: the output of your data iterable normally a `torch.utils.data.DataLoader`
			`batch_idx`: the index of this batch

		Returns:
			a dictionary that can include any keys, but must include the key `"loss"` with `torch.Tensor` loss value

		Example:
		``
		def validation_step(self, batch, batch_idx):
			x, y = batch

		#	implement your own
			out = self(x)
			loss = self.loss(out, y)

		#	log 6 example images or generated text... or whatever
			sample_imgs = x[:6]
			grid = torchvision.utils.make_grid(sample_imgs)
			self.logger.experiment.add_image("example_images", grid, 0)

		#	calculate accuracy
			labels_hat = torch.argmax(out, dim=1)
			val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

		#	log the outputs!
			self.log_dict({"val_loss": loss, "val_acc": val_acc})
		``

		NOTE: If you don't need to validate you don't need to implement this method.

		NOTE: When the `validation_step` is called, the model has been put in eval mode and PyTorch gradients have been disabled.
		At the end of validation, the model goes back to training mode and gradients are enabled.

		NOTE: This is where early stopping regularization occurs.
		Dropout regularization is engraved in the model.
		"""
		return self._shared_eval_step(batch, batch_idx, "devel")

	def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torchmetrics.Metric]:
		r"""Operates on a single batch of data from the test set.

		[https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#test-step]

		In this step you would normally generate examples or calculate anything of interest such as accuracy.

		Arguments:
			`batch`: the output of your data iterable, normally a `torch.utils.data.DataLoader`
			`batch_idx`: the index of this batch

		Return:
			a dictionary that can include any keys, but must include the key `"loss"` with `torch.Tensor` loss value

		Example:
		``
		def test_step(self, batch, batch_idx):
			x, y = batch

		#	implement your own
			out = self(x)
			loss = self.loss(out, y)

		#	log 6 example images or generated text... or whatever
			sample_imgs = x[:6]
			grid = torchvision.utils.make_grid(sample_imgs)
			self.logger.experiment.add_image('example_images', grid, 0)

		#	calculate acc
			labels_hat = torch.argmax(out, dim=1)
			test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

		#	log the outputs!
			self.log_dict({'test_loss': loss, 'test_acc': test_acc})
		``

		NOTE: If you don't need to test you don't need to implement this method.

		NOTE: When the `test_step` is called, the model has been put in eval mode and PyTorch gradients have been disabled.
		At the end of the test epoch, the model goes back to training mode and gradients are enabled.
		"""
		return self._shared_eval_step(batch, batch_idx, "valid")
