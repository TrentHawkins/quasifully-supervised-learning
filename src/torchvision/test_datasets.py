"""Tests for dataset-related code."""


class TestAnimalsWithAttributes:
	"""Test Animals with Attributes dataset."""

	def test_plots(self):
		"""Generate plots akin to the Animals with Attributes dataset."""
		import src.globals
		import numpy

		from src.torchvision.datasets import AnimalsWithAttributesDataset
		from src.similarities import cosine, jaccard, dice

		dataset = AnimalsWithAttributesDataset()

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
