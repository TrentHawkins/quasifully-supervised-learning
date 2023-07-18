"""Main."""


from __future__ import annotations

import numpy
import torch

import src.globals
import src.torch.nn
import src.torch.utils.data
import src.torchvision.datasets
import src.lightning


if __name__ == "__main__":
	""" Main testing."""

#	Normal:
	data = src.lightning.AnimalsWithAttributesDataModule(
		generalized_zeroshot=False,
		transductive_setting=False,
	)
	data.prepare_data()
	data.setup("fit"); data.train_dataloader()
	data.setup("validate"); data.val_dataloader()
	data.setup("test"); data.test_dataloader()
	data.setup("predict"); data.predict_dataloader()

#	Zeroshot:
	data = src.lightning.AnimalsWithAttributesDataModule(
		generalized_zeroshot=True,
		transductive_setting=False,
	)
	data.prepare_data()
	data.setup("fit"); data.train_dataloader()
	data.setup("validate"); data.val_dataloader()
	data.setup("test"); data.test_dataloader()
	data.setup("predict"); data.predict_dataloader()

#	Transductive:
	data = src.lightning.AnimalsWithAttributesDataModule(
		generalized_zeroshot=True,
		transductive_setting=True,
	)
	data.prepare_data()
	data.setup("fit"); data.train_dataloader()
	data.setup("validate"); data.val_dataloader()
	data.setup("test"); data.test_dataloader()
	data.setup("predict"); data.predict_dataloader()
