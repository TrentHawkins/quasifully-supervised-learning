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

	def test_model_prep(self):
		"""Test preparation of module wit data."""
		from torch import from_numpy
		from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

		from src.torch.nn import JaccardLinear, LinearStackArray
		from src.zeroshot.nn import QuasifullyZeroshotBCELoss
		from src.lightning.pytorch.data import AnimalsWithAttributesDataModule
		from src.lightning.pytorch.models import GeneralizedZeroshotModule

	#	Data:
		data = AnimalsWithAttributesDataModule(
			generalized_zeroshot=True,
			transductive_setting=True,
		)
		data.prepare_data()
		data.setup("fit")
		data.setup("validate")
		data.setup("test")
		data.setup("predict")

		model = GeneralizedZeroshotModule(
			visual=efficientnet_b0(
				weights=EfficientNet_B0_Weights.IMAGENET1K_V1,
			),
			latent=LinearStackArray(1280, 85),
			alphas=JaccardLinear(from_numpy(data.totals.alphas().to_numpy())),
			loss_f=QuasifullyZeroshotBCELoss(
				data.source.labels(),
				data.target.labels(),
			),
		)
