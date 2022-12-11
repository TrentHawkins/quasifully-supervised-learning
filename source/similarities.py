"""Vectorizable utilites."""


from typing import Callable

import numpy
import pandas


def cosine(
	a: numpy.ndarray,
	b: numpy.ndarray,
):
	"""Customize dot product emulating the cossine similarity metric.

	The cosine similarity of two vectors x and y:
		cosine(x, y) == x · y / √((x · x)(y · y))

	Returns:
		either cosine scalar of vectors or contracted array of cosines
	"""
	if len(a.shape) == len(b.shape) == 1:
		return numpy.inner(a, b) / numpy.sqrt(numpy.inner(a, a) * numpy.inner(b, b))

	if len(a.shape) == len(b.shape) >= 2:
		return numpy.array([[cosine(i, j) for i in a] for j in b.transpose()])  # type: ignore  # NOTE: After PyLance update.


def jaccard(
	a: numpy.ndarray,
	b: numpy.ndarray,
):
	"""Customize dot product emulating the Jaccard similarity metric.

	The Jaccard similarity of two sets A and B:
		jaccard(A, B) == |A ∩ B| / |A ∪ B| == |A ∩ B| / (|A| + |B| - |A ∩ B|)

	If the sets are repesented by {0,1}-valued vectors a and b, the Jaccard similarity becomes:
		jaccard(a, b) == a · b / (a · a + b · b - a · b)

	This formula can be generalized for fuzzy (0,1)-valued vextors as it.

	Returns:
		either Jaccard scalar of vectors or contracted array of jaccards
	"""
	if len(a.shape) == len(b.shape) == 1:
		return numpy.inner(a, b) / (numpy.inner(a, a) + numpy.inner(b, b) - numpy.inner(a, b))

	if len(a.shape) == len(b.shape) >= 2:
		return numpy.array([[jaccard(i, j) for i in a] for j in b.transpose()])  # type: ignore  # NOTE: After PyLance update.


def dice(
	a: numpy.ndarray,
	b: numpy.ndarray,
):
	"""Customize dot product emulating the Sørensen–Dice similarity metric.

	The Sørensen–Dice similarity of two sets A and B:
		dice(A, B) == 2 |A ∩ B| / (|A| + |B|)

	If the sets are repesented by {0,1}-valued vectors a and b, the Sørensen–Dice similarity becomes:
		dice(a, b) == 2 (a · b) / (a · a + b · b)

	This formula can be generalized for fuzzy (0,1)-valued vextors as it.

	Returns:
		either Dice scalar of vectors or contracted array of Dices
	"""
	if len(a.shape) == len(b.shape) == 1:
		return 2 * numpy.inner(a, b) / (numpy.inner(a, a) + numpy.inner(b, b))

	if len(a.shape) == len(b.shape) >= 2:
		return numpy.array([[dice(i, j) for i in a] for j in b.transpose()])  # type: ignore  # NOTE: After PyLance update.


def rand(
	a: numpy.ndarray,
	b: numpy.ndarray,
):
	"""Customize dot product emulating the Rand similarity metric.

	The Rand similarity of two sets A and B:
		rand(A, B) == (|A ∩ B| + |¬A ∩ ¬B|) / (|A ∪ B| + |¬A ∩ ¬B|) == (|¬(A ⊕ B)|) / |¬∅|

	If the sets are repesented by {0,1}-valued vectors a and b, the Rand similarity becomes:
		rand(a, b) == a · b + (1 - a) · (1 - b) / dimensionality

	This formula can be generalized for fuzzy (0,1)-valued vextors as it.

	Returns:
		either Rand scalar of vectors or contracted array of Rands
	"""
	if len(a.shape) == len(b.shape) == 1:
		return (numpy.inner(a, b) + numpy.inner(1 - a, 1 - b)) /\
			(numpy.inner(a, a) + numpy.inner(b, b) - numpy.inner(a, b) + numpy.inner(1 - a, 1 - b))

	if len(a.shape) == len(b.shape) >= 2:
		return numpy.array([[rand(i, j) for i in a] for j in b.transpose()])  # type: ignore  # NOTE: After PyLance update.


def dotDataFrame(
	a: pandas.DataFrame,
	b: pandas.DataFrame, alter_dot: Callable = numpy.dot
):
	"""Modify dot product for dataframes.

	Keyword Arguments:
		alter_dot: customized dot product to replace the NumPy original

	Returns:
		dataframe of customized dot product results
	"""
	numpy_dot = numpy.dot  # save-keep NumPy original dot product
	numpy.dot = alter_dot  # replace   NumPy original dot product with custom

	_dot = a.dot(b)

	numpy.dot = numpy_dot  # load-back NumPy original dot product

	return _dot
