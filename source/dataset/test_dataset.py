"""Tests for dataset-related code."""


class TestAA:
	"""Test Animals with Attributes dataset."""

	def test_plots(self):
		"""Generate pltos akin to the Animals with Attributes dataset."""
		import numpy

		from source.dataset.animals_with_attributes import Dataset
		from source.similarity import cosine, jaccard

		dataset = Dataset()

	#	Plot basics:
		dataset.plot_labels()
		dataset.plot_alphas()
		dataset.plot_alphas(binary=True)

	#	Plot everything:
		for binary in (True, False):
			for logits in (True, False):
				for softmx in (True, False):
					for alter_dot in (cosine, jaccard, numpy.dot):
						dataset.plot_label_correlation(alter_dot,
							binary,
							logits,
							softmx,
						)
