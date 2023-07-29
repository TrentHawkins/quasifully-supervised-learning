"""Tools for input and output formatting and other string operations.

Includes:
	print_separator: repeats character across full terminal width
"""


from shutil import get_terminal_size
from typing import Optional, Union


def separator(char: Union[str, int] = 0,
	title: Optional[str] = None,
):
	"""Repeat character across full terminal width.

	Arguments:
		char: single character to repeat
			0: " "
			1: "─"
			2: "═"
			3: "━"

	Keyword Arguments:
		title: optional text to display before separator
	"""
	if isinstance(char, int):
		char = {
			0: " ",
			1: "─",
			2: "═",
			3: "━",
		}[char]

	if title is not None:
		print(title)

	print(char * get_terminal_size((96, 96)).columns)
