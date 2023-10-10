"""Basic integer numeric utilities.

Includes:
	lcm: `math.lcm` is unavailable to `python < 3.9`

	divisors: of an integer
	is_prime: is an integer

	hidden_sizes: devise hidden sizes of a pyramid-like MLP based on divisor logic given inputs and output size

	mettalic: `a(n) = m * a(n - 1) + a(n - 2)'
		golden: `m = 1` (Fibonacci)
		silver: `m = 2`
		bronze: `m = 3`
"""


from __future__ import annotations

from math import gcd

import numpy


def lcm(a, b):
	"""`python3.8` compatibility `lcm` function.

	Arguments:
		a: number
		b: number

	Returns:
		number
	"""
	return (a * b) // gcd(a, b)


def divisors(dividee: int, reverse: bool = False) -> list[int]:
	"""Get all divisors of dividee.

	Arguments:
		dividee: the number to divide

	Keyword Arguments:
		reverse: the order of devisors

	Returns:
		list` of divisors
	"""
	_divisors = set()

	for divisor in range(1, int(numpy.sqrt(dividee)) + 1):
		if dividee % divisor == 0:
			_divisors.add(divisor)
			_divisors.add(dividee // divisor)

	return sorted(_divisors, reverse=reverse)


def is_prime(dividee: int) -> bool:
	"""Check if dividee is prime or not.

	Arguments:
		dividee: the number to check

	Returns:
		True` if dividee is prime else `False`
	"""
#	return dividee > 1 and all(dividee % divisor for divisor in range(2, int(numpy.sqrt(dividee)) + 1))
	return dividee > 1 and len(divisors(dividee)) == 1


def hidden_sizes(
	inputs_size: int,
	output_size: int, skip: int = 1, less: bool = True,
) -> list[int]:
	"""Get all possible layers given by a divisor logic.

	Take inputs_dim*outputs_dim and find all divisors in range(inputs_dim, outputs_dim+1, skip).

	Arguments:
		inputs_size: the inputs dimension for the requested architecture
		output_size: the output dimension for the requested architecture
		skip: subsample proposed architecture uniformly by skipping layers with a divisor of the depth (default no skip)
		less: whether to thinout results by selection of multiples of the GCD only (default thinout)

	Returns:
		reversed` `list` of divisors as proposed layer sizes
	"""
	_lcm = lcm(inputs_size, output_size) if less else inputs_size * output_size
	_gcd = gcd(inputs_size, output_size) if less else 1

#	shrinking model
	if inputs_size > output_size:
		return [x for x in divisors(_lcm) if inputs_size >= x >= output_size and x % _gcd == 0][::-skip]

#	expanding model
	if inputs_size < output_size:
		return [x for x in divisors(_lcm) if inputs_size <= x <= output_size and x % _gcd == 0][::+skip]

#	projecting model
	return [inputs_size]


def metallic(n: int, m: int) -> int:
	"""Metallic sequence of rank `m`.

	metallic(n)=m*metallic(n-1)+metallic(n-2)
	metallic(1)=1
	metallic(0)=0

	Arguments:
		n: input
		m: rank

	Returns:
		n-th term in the sequence starting from 0
	"""
	a, b = 0, 1

	for i in range(n):
		a, b = b, a + b * m

	return a


def golden(n: int) -> int:
	"""Golden ratio (`m=1`, Fibonacci) sequence.

	Arguments:
		n: input

	Returns:
		n-th term in the sequence starting from 0
	"""
	return metallic(n, 1)


def silver(n: int) -> int:
	"""	Silver ratio (`m=2`) sequence.

	Arguments:
		n: input

	Returns:
		n-th term in the sequence starting from 0
	"""
	return metallic(n, 2)


def bronze(n: int) -> int:
	"""	Bronze ratio (`m=3`) sequence.

	Arguments:
		n: input

	Returns:
		n-th term in the sequence starting from 0
	"""
	return metallic(n, 3)
