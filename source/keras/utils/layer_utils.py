"""Utilities related to layer/model functionality.

┌─┬─┐ ╒═╤═╕ ┍━┯━┑ ╓─╥─╖ ╔═╦═╗ ┎─┰─┒ ┏━┳━┓ ╭─┬─╮
│ │ │ │ │ │ │ │ │ ║ ║ ║ ║ ║ ║ ┃ ┃ ┃ ┃ ┃ ┃ │ │ │
├─┼─┤ ╞═╪═╡ ┝━┿━┥ ╟─╫─╢ ╠═╬═╣ ┠─╂─┨ ┣━╋━┫ ├─┼─┤
│ │ │ │ │ │ │ │ │ ║ ║ ║ ║ ║ ║ ┃ ┃ ┃ ┃ ┃ ┃ │ │ │
└─┴─┘ ╘═╧═╛ ┕━┷━┙ ╙─╨─╜ ╚═╩═╝ ┖─┸─┚ ┗━┻━┛ ╰─┴─╯

NOTE: THIS FILE IS MONKEY-PATCHING THE ORIGINAL

Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
"""


from re import match
from shutil import get_terminal_size
from typing import Callable

import numpy
import tensorflow

import keras.utils.layer_utils


def count_params(weights):
	"""Count the total number of scalars composing the weights.

	Argumentss:
		weights: An iterable containing the weights on which to compute params

	Returns:
		The total number of scalars composing the weights
	"""
	unique_weights = {id(w): w for w in weights}.values()

#	Ignore TrackableWeightHandlers, which will not have a shape defined.
	unique_weights = [w for w in unique_weights if hasattr(w, "shape")]
	weight_shapes = [w.shape.as_list() for w in unique_weights]
	standardized_weight_shapes = [[0 if w_i is None else w_i for w_i in w] for w in weight_shapes]
	return int(sum(numpy.prod(p) for p in standardized_weight_shapes))  # type: ignore


def get_layer_index_bound_by_layer_name(model,
	layer_range: list[str] | tuple[str, str] | None = None,
):
	"""Get the layer indexes from the model based on layer names.

	The layer indexes can be used to slice the model into sub models for display.

	Arguments:
		model: `tf.keras.Model` instance.
		layer_names: a list or tuple of 2 strings, the starting layer name and ending layer name (both inclusive) for the result.
			All layers will	be included when `None` is provided.

	Returns:
		The index value of layer based on its unique name (layer_names).
		Output will be `[first_layer_index, last_layer_index + 1]`.
	"""
	if layer_range is not None:
		if len(layer_range) != 2:
			raise ValueError(
				"layer_range must be a list or tuple of length 2. Received: "
				f"layer_range = {layer_range} of length {len(layer_range)}"
			)

		if not isinstance(layer_range[0], str) or not isinstance(layer_range[1], str):
			raise ValueError(
				"layer_range should contain string type only. "
				f"Received: {layer_range}"
			)

	else:
		return [0, len(model.layers)]

	lower_index = [
		idx
		for idx, layer in enumerate(model.layers)
		if match(layer_range[0], layer.name)
	]
	upper_index = [
		idx
		for idx, layer in enumerate(model.layers)
		if match(layer_range[1], layer.name)
	]

	if not lower_index or not upper_index:
		raise ValueError(
			"Passed layer_names do not match the layer names in the model. "
			f"Received: {layer_range}"
		)

	if min(lower_index) > max(upper_index):
		return [min(upper_index), max(lower_index) + 1]

	return [min(lower_index), max(upper_index) + 1]


def print_summary(model,
	line_length: int | None = None,
	positions: list[int] | list[float] | None = None,  # type: ignore
	print_fn: Callable[[str], None] | None = None,  # type: ignore
	expand_nested: bool = False,
	show_trainable: bool = False,
	layer_range: list[str] | tuple[str, str] | None = None,  # type: ignore
):
	"""Print a summary of a model.

	Arguments:
		model: Keras model instance.
		line_length: Total length of printed lines (e.g. set this to adapt the display to different terminal window sizes).
		positions: Relative or absolute positions of log elements in each line.
			If not provided, defaults to `[.33, .55, .67, 1.]`.
		print_fn: Print function to use.
			It will be called on each line of the summary.
			You can set it to a custom function in order to capture the string summary.
			It defaults to `print` (prints to stdout).
		expand_nested: Whether to expand the nested models.
			If not provided, defaults to `False`.
		show_trainable: Whether to show if a layer is trainable.
			If not provided, defaults to `False`.
		layer_range: List or tuple containing two strings,
			the starting layer name and ending layer name (both inclusive),
			indicating the range of layers to be printed in the summary.
			The strings could also be regexes instead of an exact name.
			In this case, the starting layer will be the first layer that matches `layer_range[0]`
			and the ending layer will be the last element that matches `layer_range[1]`.
			By default (`None`) all layers in the model are included in the summary.
	"""
	if print_fn is None:
		print_fn: Callable[[str], None] = keras.utils.io_utils.print_msg  # type: ignore

	if model.__class__.__name__ == "Sequential":
		sequential_like = True

#	We treat subclassed models as a simple sequence of layers, for logging purposes.
	elif not model._is_graph_network:
		sequential_like = True

	else:
		sequential_like = True
		nodes_by_depth = model._nodes_by_depth.values()
		nodes = []

	#	If the model has multiple nodes, or if the nodes have multiple inbound_layers, the model is no longer sequential.
		for v in nodes_by_depth:
			if (len(v) > 1) or (len(v) == 1 and len(tensorflow.nest.flatten(v[0].keras_inputs)) > 1):
				sequential_like = False

				break

			nodes += v

	#	Search for shared layers,
		if sequential_like:
			for layer in model.layers:
				flag = False

				for node in layer._inbound_nodes:
					if node in nodes:
						if flag:
							sequential_like = False

							break

						else:
							flag = True

				if not sequential_like:
					break

	if sequential_like:
		line_length = line_length or get_terminal_size().columns  # 65
		positions = positions or [
			0.50,  # 0.45
			0.75,  # 0.85
			1.00,  # 1.00
		]  # type: ignore

	#	Header names for the different log elements:
		to_display = [
			"Layer (type)",
			"Output Shape",
			"Param #",
		]

	else:
		line_length = line_length or get_terminal_size().columns  # 98
		positions = positions or [
			0.333,  # 0.33
			0.500,  # 0.55
			0.667,  # 0.67
			1.000,  # 1.00
		]  # type: ignore

	#	Header names for the different log elements:
		to_display = [
			"Layer (type)",
			"Output Shape",
			"Param #",
			"Connected to",
		]
		relevant_nodes = []

		for v in model._nodes_by_depth.values():
			relevant_nodes += v

	if positions[-1] <= 1:
		positions: list[int] = [int(line_length * p) for p in positions]

	if show_trainable:
		line_length += 2
		positions.append(line_length)
		to_display.append("T")

	layer_range: list[int] = get_layer_index_bound_by_layer_name(model, layer_range)

	def print_row(fields: list[str], positions: list[int],
		nested_level: int = 0,
	):
		left_to_print = [str(x) for x in fields]

		while any(left_to_print):
			line = ""

			for col in range(len(left_to_print)):
				if col > 0:
					start_pos = positions[col - 1]

				else:
					start_pos = 0

				end_pos = positions[col]

			#	Leave room for 2 spaces to delineate columns we don't need any if we are printing the last column.
				space = 2 if col != len(positions) - 1 else 0
				cutoff = end_pos - start_pos - space
				fit_into_line = left_to_print[col][:cutoff]

			#	For nicer formatting we line-break on seeing end of `tuple`/`dict` etc.
				line_break_conditions = (
					"),",
					"},",
					"],",
					"',",
				)
				candidate_cutoffs = [fit_into_line.find(x) + len(x) for x in line_break_conditions if fit_into_line.find(x) >= 0]

				if candidate_cutoffs:
					cutoff = min(candidate_cutoffs)
					fit_into_line = fit_into_line[:cutoff]

				if col == 0:
					line += "│" * nested_level + " "

				line += fit_into_line
				line += " " * space if space else ""
				left_to_print[col] = left_to_print[col][cutoff:]

			#	Pad out to the next position.
				if nested_level:
					line += " " * (positions[col] - len(line) - nested_level)

				else:
					line += " " * (positions[col] - len(line))

			line += "│" * nested_level

			print_fn(line)

	print_fn(f"Model: \"{model.name}\"")
	print_fn("━" * line_length)

	print_row(to_display, positions)
	print_fn("═" * line_length)

	def print_layer_summary(layer,
		nested_level: int = 0,
	):
		"""Print a summary for a single layer.

		Arguments:
			layer: target layer
			nested_level: level of nesting of the layer inside its parent layer
				(e.g. 0 for a top-level layer, 1 for a nested layer)
		"""
		try:
			output_shape = layer.output_shape

		except AttributeError:
			output_shape = "multiple"

		except RuntimeError:  # output_shape unknown in Eager mode.
			output_shape = "?"

		name = layer.name
		cls_name = layer.__class__.__name__

	#	If a subclassed model has a layer that is not called in `Model.call`,
	#	the layer will not be built and we cannot call `layer.count_params()`.
		if not layer.built and not getattr(layer, "_is_graph_network", False):
			params = "0 (unused)"

		else:
			params = layer.count_params()

		fields = [
			f"{name} ({cls_name})",
			output_shape,
			params,
		]

		if show_trainable:
			fields.append("+" if layer.trainable else "-")

		print_row(fields, positions, nested_level)

	def print_layer_summary_with_connections(layer,
		nested_level: int = 0,
	):
		"""Print a summary for a single layer (including topological connections).

		Arguments:
			layer: target layer
			nested_level: level of nesting of the layer inside its parent layer
				(e.g. 0 for a top-level layer, 1 for a nested layer)
		"""
		try:
			output_shape = layer.output_shape

		except AttributeError:
			output_shape = "multiple"

		connections = []

	#	Node is not part of the current network.
		for node in layer._inbound_nodes:
			if relevant_nodes and node not in relevant_nodes:
				continue

			for (inbound_layer, node_index, tensor_index, _) in node.iterate_inbound():
				connections.append(f"{inbound_layer.name}[{node_index}][{tensor_index}]")

		name = layer.name
		cls_name = layer.__class__.__name__
		fields = [
			f"{name} ({cls_name})",
			output_shape,
			layer.count_params(),
			connections,
		]

		if show_trainable:
			fields.append("+" if layer.trainable else "-")

		print_row(fields, positions, nested_level)

	def print_layer(layer,
		nested_level: int = 0,
		is_nested_last: bool = False,
	):
		if sequential_like:
			print_layer_summary(layer, nested_level)

		else:
			print_layer_summary_with_connections(layer, nested_level)

		if expand_nested and hasattr(layer, "layers") and layer.layers:
			print_fn('│' * nested_level + '╭' + '─' * (line_length - 2 * nested_level - 2) + '╮' + '│' * nested_level)

			nested_layer = layer.layers
			is_nested_last = False

			for i in range(len(nested_layer)):
				if i == len(nested_layer) - 1:
					is_nested_last = True

				print_layer(nested_layer[i], nested_level + 1, is_nested_last)

			print_fn('│' * nested_level + '╰' + '─' * (line_length - 2 * nested_level - 2) + '╯' + '│' * nested_level)

	#	if not is_nested_last:
	#		print_fn('│' * nested_level + ' ' * (line_length - 2 * nested_level) + '│' * nested_level)

	for layer in model.layers[layer_range[0]:layer_range[1]]:
		print_layer(layer)

	print_fn("═" * line_length)

	if hasattr(model, "_collected_trainable_weights"):
		trainable_count = count_params(model._collected_trainable_weights)

	else:
		trainable_count = count_params(model.trainable_weights)

	non_trainable_count = count_params(model.non_trainable_weights)

	print_fn(f"Total params: {trainable_count + non_trainable_count}")
	print_fn(f"Trainable params: {trainable_count}")
	print_fn(f"Non-trainable params: {trainable_count}")
	print_fn("━" * line_length)


keras.utils.layer_utils.print_summary = print_summary  # NOTE: MONKEYPATCH
