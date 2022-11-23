"""Tests for dataset-related code."""


class TestAnimalsAttributes:
	"""Test Animals with Attributes dataset."""

	def test_plots(self):
		"""Generate plots akin to the Animals with Attributes dataset."""
		import numpy

		from source.dataset.animals_with_attributes import Dataset
		from source.similarities import cosine, jaccard, dice, rand

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
			rand,
		):
			dataset.plot_label_correlation(alter_dot,
				binary=False,
				logits=False,
				softmx=False,
			)

	def test_split(self):
		"""Test Animals with Attributes splitting on various settings."""
		from source.dataset.animals_with_attributes import Dataset, ZeroshotDataset, TransductiveZeroshotDataset

		(
			train_subset,
			devel_subset,
			valid_subset,
		) = Dataset().split()
		(
			zeroshot_train_subset,
			zeroshot_devel_subset,
			zeroshot_valid_subset,
		) = ZeroshotDataset().split()
		(
			quasifully_zeroshot_train_subset,
			quasifully_zeroshot_devel_subset,
			quasifully_zeroshot_valid_subset,
		) = TransductiveZeroshotDataset().split()

	#	Assert no-zeroshot and transductive settings approximately match (accounting for rounding errors):
		assert abs(len(train_subset) - len(quasifully_zeroshot_train_subset)) <= 1
		assert abs(len(devel_subset) - len(quasifully_zeroshot_devel_subset)) <= 1
		assert abs(len(valid_subset) - len(quasifully_zeroshot_valid_subset)) <= 1

	#	Assert semi-transductive setting shifts training examples to validation (lacking unlabelled target examples):
		assert len(zeroshot_train_subset) <= len(quasifully_zeroshot_train_subset)
		assert len(zeroshot_devel_subset) <= len(quasifully_zeroshot_devel_subset)
		assert len(zeroshot_valid_subset) >= len(quasifully_zeroshot_valid_subset)

	#	Assert all splits sum-up to the full dataset (by counting):
		assert 37322 == \
			len(train_subset) + \
			len(devel_subset) + \
			len(valid_subset)
		assert 37322 == \
			len(zeroshot_train_subset) + \
			len(zeroshot_devel_subset) + \
			len(zeroshot_valid_subset)
		assert 37322 == \
			len(quasifully_zeroshot_train_subset) + \
			len(quasifully_zeroshot_devel_subset) + \
			len(quasifully_zeroshot_valid_subset)
