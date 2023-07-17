"""Tests for dataset-related code."""


class TestAnimalsWithAttributesDataModule:
	"""Test Animals with Attributes lightning data module."""

	def test_splitting(self):
		"""Test Animals with Attributes splitting on various settings."""
		from src.lightning import AnimalsWithAttributesDataModule

		data = AnimalsWithAttributesDataModule(
			generalized_zeroshot=False,
			transductive_setting=False,
		)
		data.prepare_data()
		data.setup("fit"); data.train_dataloader()
		data.setup("validate"); data.val_dataloader()
		data.setup("test"); data.test_dataloader()
		data.setup("predict"); data.predict_dataloader()

		data = AnimalsWithAttributesDataModule(
			generalized_zeroshot=True,
			transductive_setting=False,
		)
		data.prepare_data()
		data.setup("fit"); data.train_dataloader()
		data.setup("validate"); data.val_dataloader()
		data.setup("test"); data.test_dataloader()
		data.setup("predict"); data.predict_dataloader()

		data = AnimalsWithAttributesDataModule(
			generalized_zeroshot=True,
			transductive_setting=True,
		)
		data.prepare_data()
		data.setup("fit"); data.train_dataloader()
		data.setup("validate"); data.val_dataloader()
		data.setup("test"); data.test_dataloader()
		data.setup("predict"); data.predict_dataloader()
