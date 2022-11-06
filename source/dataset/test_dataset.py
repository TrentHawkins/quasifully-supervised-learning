"""Tests for dataset-related code."""


class TestAA:
	"""Test Animals with Attributes dataset."""

	def test_plots(self):
		"""Generate pltos akin to the Animals with Attributes dataset."""
		from source.dataset.animals_with_attributes import Dataset
		from source.similarity import cosine, jaccard

		dataset = Dataset()

		dataset.plot_labels()
		dataset.plot_predicates()
		dataset.plot_predicates(binary=True)

		kwargs = {
		#	"binary": True,
		#	"logits": True,
		#	"softmx": True,
		}
		dataset.plot_label_correlation(**kwargs)
		dataset.plot_label_correlation(**kwargs, alter_dot=jaccard)
		dataset.plot_label_correlation(**kwargs, alter_dot=cosine)
