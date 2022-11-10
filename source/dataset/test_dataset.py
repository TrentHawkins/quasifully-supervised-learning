"""Tests for dataset-related code."""


class TestAA:
	"""Test Animals with Attributes dataset."""

	def test_plots(self):
		"""Generate pltos akin to the Animals with Attributes dataset."""
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
