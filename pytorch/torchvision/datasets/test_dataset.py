"""Tests for dataset-related code."""


class TestAnimalsAttributes:
	"""Test Animals with Attributes dataset."""

	def test_plots(self):
		"""Generate plots akin to the Animals with Attributes dataset."""
		import numpy

		from pytorch.torchvision.datasets.animals_with_attributes import Dataset
		from pytorch.similarities import cosine, jaccard, dice

		dataset = Dataset()

	#	Plot basics:
		dataset.plot_labels()
		dataset.plot_alphas()
		dataset.plot_alphas(binary=True)

	#	Plot everything:
		for alter_dot in (
			numpy.dot,
			cosine,
			jaccard,
			dice,
		):
			dataset.plot_label_correlation(alter_dot,
				binary=False,
				logits=False,
				softmx=False,
			)

	def test_split(self):
		"""Test Animals with Attributes splitting on various settings."""
		from pytorch.torchvision.datasets.animals_with_attributes import Dataset, ZeroshotDataset, TransductiveZeroshotDataset

		(
			train_subset,
			devel_subset,
			valid_subset,
		) = Dataset().random_split()
		(
			zeroshot_train_subset,
			zeroshot_devel_subset,
			zeroshot_valid_subset,
		) = ZeroshotDataset().random_split()
		(
			quasifully_zeroshot_train_subset,
			quasifully_zeroshot_devel_subset,
			quasifully_zeroshot_valid_subset,
		) = TransductiveZeroshotDataset().random_split()

	#	Assert no-zeroshot and transductive settings approximately match (accounting for rounding errors):
		assert abs(len(train_subset) - len(quasifully_zeroshot_train_subset)) <= 1  # type: ignore
		assert abs(len(devel_subset) - len(quasifully_zeroshot_devel_subset)) <= 1  # type: ignore
		assert abs(len(valid_subset) - len(quasifully_zeroshot_valid_subset)) <= 1  # type: ignore

	#	Assert semi-transductive setting shifts training examples to validation (lacking unlabelled target examples):
		assert len(zeroshot_train_subset) <= len(quasifully_zeroshot_train_subset)  # type: ignore
		assert len(zeroshot_devel_subset) <= len(quasifully_zeroshot_devel_subset)  # type: ignore
		assert len(zeroshot_valid_subset) >= len(quasifully_zeroshot_valid_subset)  # type: ignore

	#	Assert all splits sum-up to the full dataset (by counting):
		assert 37322 == (
			len(train_subset) +  # type: ignore
			len(devel_subset) +  # type: ignore
			len(valid_subset)    # type: ignore
		)
		assert 37322 == (
			len(zeroshot_train_subset) +  # type: ignore
			len(zeroshot_devel_subset) +  # type: ignore
			len(zeroshot_valid_subset)    # type: ignore
		)
		assert 37322 == (
			len(quasifully_zeroshot_train_subset) +  # type: ignore
			len(quasifully_zeroshot_devel_subset) +  # type: ignore
			len(quasifully_zeroshot_valid_subset)    # type: ignore
		)
