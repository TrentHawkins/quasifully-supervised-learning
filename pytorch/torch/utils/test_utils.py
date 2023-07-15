"""Tests for dataloaders."""


class TestDataLoader:
	"""Test Animals with Attributes dataloader."""

	def test_data_loader(self):
		"""Test Animals with Attributes dataloader."""
		import pytorch.globals

		from torchvision.transforms import CenterCrop, Compose, Resize
		from pytorch.torchvision.datasets import (
			AnimalsWithAttributesDataset,
			ZeroshotAnimalsWithAttributesDataset,
			TransductiveZeroshotAnimalsWithAttributesDataset,
		)
		from pytorch.torch.utils.data import AnimalsWithAttributesDataLoader

	#	Transform to be used on images:
		size = 224
		transform = Compose(
			[
				Resize(size),
				CenterCrop(size),
			]
		)

	#	Batch size:
		batch_size = 64

	#	Plain dataloader:
		dataloader = AnimalsWithAttributesDataLoader(
			AnimalsWithAttributesDataset(
				transform=transform,
			), pytorch.globals.generator, batch_size)
		image, label = next(iter(dataloader))
		assert image.size()[0] == batch_size
		assert label.size()[0] == batch_size

	#	Zeroshot dataloader:
		dataloader = AnimalsWithAttributesDataLoader(
			ZeroshotAnimalsWithAttributesDataset(
				transform=transform,
			), pytorch.globals.generator, batch_size)
		image, label = next(iter(dataloader))
		assert image.size()[0] == batch_size
		assert label.size()[0] == batch_size

	#	Transductive zeroshot dataloader:
		dataloader = AnimalsWithAttributesDataLoader(
			TransductiveZeroshotAnimalsWithAttributesDataset(
				transform=transform,
			), pytorch.globals.generator, batch_size)
		image, label = next(iter(dataloader))
		assert image.size()[0] == batch_size
		assert label.size()[0] == batch_size
