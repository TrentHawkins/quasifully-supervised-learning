"""Basic string encoding/decoding utilities."""


from typing import Iterable, Type


def to_string(objects: Iterable | map,
	type: Type = int,
	separator: str = " ",
) -> str:
	"""Encode objects to strings based on their representation.

	Arguments:
		objects: A tuple of mixed type objects or other homogenous iterable of objects to encode into a string.

	Keyword Arguments:
		type: The way to parse objects (all assumed of the same type).
		separator: A character sequence for separating entries in the string encoding.

	Returns
		A string with the representation of each object separated by a set of characters.
	"""
	return separator.join(map(repr, map(type, objects)))


def from_string(string: str,
	type: Type = int,
	separator: str = " ",
) -> map:
	"""Encode objects to strings based on their representation.

	Arguments:
		string: A string contatinating representations of a homogenous iterable of objects to be decoded.

	Keyword Arguments:
		type: The way to parse objects (all assumed of the same type).
		separator: A character sequence for separating entries in the string encoding.

	Returns
		A map of the objects encoded in the string.
		Apply an iterable cast or iterate through.
	"""
	return map(type, string.split(separator))
