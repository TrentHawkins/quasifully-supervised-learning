"""Basic integer numeric utilities."""


import math

import numpy


def divisors(dividee: int, reverse: bool = False) -> list[int]:
	"""	Get all divisors of dividee.

	Arguments:
		dividee: the number to divide

	Keyword Arguments:
		reverse: the order of devisors

	Returns:
		list of divisors
	"""
	_divisors = set()

	for divisor in range(1, int(numpy.sqrt(dividee)) + 1):
		if dividee % divisor == 0:
			_divisors.add(divisor)
			_divisors.add(dividee // divisor)

	return sorted(_divisors, reverse=reverse)


def is_prime(dividee: int) -> bool:
	"""	Check if dividee is prime or not.

	Arguments:
		dividee: the number to check

	Returns:
		True if dividee is prime
	"""
#	return dividee > 1 and all(dividee % divisor for divisor in range(2, int(numpy.sqrt(dividee)) + 1))
	return dividee > 1 and len(divisors(dividee)) == 1


def hidden_dims(
	inputs_dim: int,
	output_dim: int, skip: int = 1, less: bool = True,
) -> list[int]:
	"""Get all possible layers given by a divisor logic.

	Take inputs_dim*outputs_dim and find all divisors in range(inputs_dim, outputs_dim+1, skip).

	Arguments:
		inputs_dim: the inputs dimension for the requested architecture
		output_dim: the output dimension for the requested architecture
		skip: subsample proposed architecture uniformly by skipping layers with (possibly) a divisor of the depth
			default: no skip
		less: whether to thin out results by selection of multiples of the GCD only
			default: thin out

	Returns:
		reversed list of divisors as proposed layer sizes
	"""
	_lcm = math.lcm(inputs_dim, output_dim) if less else inputs_dim * output_dim
	_gcd = math.gcd(inputs_dim, output_dim) if less else 1

#	shrinking model
	if inputs_dim > output_dim:
		return [x for x in divisors(_lcm) if inputs_dim >= x >= output_dim and x % _gcd == 0][::-skip]

#	expanding model
	if inputs_dim < output_dim:
		return [x for x in divisors(_lcm) if inputs_dim <= x <= output_dim and x % _gcd == 0][::+skip]

#	projecting model
	return [inputs_dim]


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
	"""Golden ratio (m=1, Fibonacci) sequence.

	Arguments:
		n: input

	Returns:
		n-th term in the sequence starting from 0
	"""
	return metallic(n, 1)


def silver(n: int) -> int:
	"""	Silver ratio (m=2) sequence.

	Arguments:
		n: input

	Returns:
		n-th term in the sequence starting from 0
	"""
	return metallic(n, 2)


def bronze(n: int) -> int:
	"""	Bronze ratio (m-3) sequence.

	Arguments:
		n: input

	Returns:
		n-th term in the sequence starting from 0
	"""
	return metallic(n, 3)
