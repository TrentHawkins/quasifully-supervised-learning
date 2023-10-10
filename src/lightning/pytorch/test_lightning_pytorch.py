"""Tests for lightning lifecycle."""


class TestAnimalsWithAttributesDataModule:
	"""Test Animals with Attributes lightning data module."""

	def test_splitting(self):
		"""Test Animals with Attributes splitting on various settings."""
		from src.lightning.pytorch.data import AnimalsWithAttributesDataModule

	#	Normal:
		data = AnimalsWithAttributesDataModule(
			generalized_zeroshot=False,
			transductive_setting=False,
		)
		data.prepare_data()
		data.setup("fit"); data.train_dataloader()
		data.setup("validate"); data.val_dataloader()
		data.setup("test"); data.test_dataloader()
		data.setup("predict"); data.predict_dataloader()

	#	Zeroshot:
		data = AnimalsWithAttributesDataModule(
			generalized_zeroshot=True,
			transductive_setting=False,
		)
		data.prepare_data()
		data.setup("fit"); data.train_dataloader()
		data.setup("validate"); data.val_dataloader()
		data.setup("test"); data.test_dataloader()
		data.setup("predict"); data.predict_dataloader()

	#	Transductive:
		data = AnimalsWithAttributesDataModule(
			generalized_zeroshot=True,
			transductive_setting=True,
		)
		data.prepare_data()
		data.setup("fit"); data.train_dataloader()
		data.setup("validate"); data.val_dataloader()
		data.setup("test"); data.test_dataloader()
		data.setup("predict"); data.predict_dataloader()


class TestGeneralizedZeroshotModule:
	"""Test generalized zeroshot lightning module."""

	def test_model(self):
		"""Test preparation of module wit data."""
		from torch import from_numpy
		from torch.nn import BCELoss
		from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
		from lightning.pytorch import Trainer
		from torchmetrics.classification import MultilabelAccuracy

		from src.torch.nn import JaccardLinear, LinearStackArray
		from src.zeroshot.nn import QuasifullyZeroshotLoss
		from src.lightning.pytorch.data import AnimalsWithAttributesDataModule
		from src.lightning.pytorch.models import GeneralizedZeroshotModule

	#	Data:
		datamodule = AnimalsWithAttributesDataModule(
			generalized_zeroshot=True,
			transductive_setting=True,
		)
		datamodule.prepare_data()
		datamodule.setup("fit")
		datamodule.setup("validate")
		datamodule.setup("test")
		datamodule.setup("predict")

		model = GeneralizedZeroshotModule(
			visual=efficientnet_b0(
				weights=EfficientNet_B0_Weights.IMAGENET1K_V1,
			),
			visual_semantic=LinearStackArray(1280, 85),
			semantic=JaccardLinear(from_numpy(datamodule.totals.alphas().to_numpy())),
			loss=QuasifullyZeroshotLoss(BCELoss(),
				datamodule.source.labels(),
				datamodule.target.labels(),
			),
			metrics={
				"accuracy": MultilabelAccuracy(len(datamodule.totals.labels()),
					threshold=0.5,
					average='macro',
					multidim_average='global',
					ignore_index=None,
					validate_args=True,
				)
			},
		)

		classifier = Trainer(
		#	accelerator="auto",
		#	strategy="auto",
		#	devices="auto",
		#	num_nodes=1,
		#	precision="32-true",
		#	logger=None,
		#	callbacks=None,
		#	fast_dev_run=True,  # False
			max_epochs=1,  # none
		#	min_epochs=None,
		#	max_steps=-1,
		#	min_steps=None,
		#	max_time=None,
		#	limit_train_batches=None,
		#	limit_val_batches=None,
		#	limit_test_batches=None,
		#	limit_predict_batches=None,
		#	overfit_batches=0.0,
		#	val_check_interval=None,
		#	check_val_every_n_epoch=1,
		#	num_sanity_val_steps=None,
		#	log_every_n_steps=None,
		#	enable_checkpointing=None,
		#	enable_progress_bar=None,
		#	enable_model_summary=None,
		#	accumulate_grad_batches=1,
		#	gradient_clip_val=None,
		#	gradient_clip_algorithm=None,
			deterministic=True,  # None
		#	benchmark=None,
		#	inference_mode=True,
		#	use_distributed_sampler=True,
		#	profiler=None,
		#	detect_anomaly=False,
		#	barebones=False,
		#	plugins=None,
		#	sync_batchnorm=False,
		#	reload_dataloaders_every_n_epochs=0,
			default_root_dir="models",
		)

		classifier.fit(
			model=model,
			datamodule=datamodule,
		)
