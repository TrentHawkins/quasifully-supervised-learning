"""Python utilities required by Keras.

▁▂▃▄▅▆▇█
███████▉
███████▊
███████▋
███████▌
███████▍
███████▎
███████▏

NOTE: THIS FILE IS MONKEY-PATCHING THE ORIGINAL

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
"""


import sys
import time

import numpy as numpy

from tensorflow.keras.utils import Progbar  # type: ignore


def update(
	self: Progbar,
	current: int,
	values: list[tuple] | None = None,
	finalize: bool | None = None
):
	"""Update the progress bar.

	Arguments:
		current: Index of current step.
		values: List of tuples: `(name, value_for_last_step)`.
			If `name` is in `stateful_metrics`, `value_for_last_step` will be displayed as-is.
			Else, an average of the metric over time will be displayed.
		finalize: Whether this is the last update for the progress bar.
			If `None`, defaults to `current >= self.target`.
	"""
	if finalize is None:
		if self.target is None:
			finalize = False

		else:
			finalize = current >= self.target

	values = values or []

	for k, v in values:
		if k not in self._values_order:
			self._values_order.append(k)

	#	In the case that progress bar doesn"t have a target value in the first epoch,
	#	both on_batch_end and on_epoch_end will be called,
	#	which will cause `current` and `self._seen_so_far` to have the same value. Force
	#	the minimal value to 1 here, otherwise stateful_metric will be 0s.
		if k not in self.stateful_metrics:
			value_base = max(current - self._seen_so_far, 1)

			if k not in self._values:
				self._values[k] = [v * value_base, value_base]

			else:
				self._values[k][0] += v * value_base
				self._values[k][1] += value_base

	#	Stateful metrics output a numeric value.
	#	This representation means "take an average from a single value" but keeps the numeric formatting.
		else:
			self._values[k] = [v, 1]

	self._seen_so_far = current

#	Custom time formatter:
	def _format_time(time):
		return f"{int(time) // 3600:02d}:{(int(time) % 3600) // 60:02d}:{int(time) % 60:02d}"

	now = time.time()
	info = f"  {_format_time(now - self._start):s}"

	if current == self.target:
		self._time_at_epoch_end = now

	if self.verbose == 1:
		if now - self._last_update < self.interval and not finalize:
			return

		prev_total_width = self._total_width

		if self._dynamic_display:
			sys.stdout.write("\b" * prev_total_width)
			sys.stdout.write("\r")

		else:
			sys.stdout.write("\n")

		if self.target is not None:
			numdigits = 7  # int(numpy.log10(self.target)) + 1
			bar = (f"%{str(numdigits)}d/%{str(numdigits)}d |") % (current, self.target)
			prog = float(current) / self.target
			prog_width = int(self.width * prog * 8)

			bar += ('█' * (prog_width // 8))  # prog_width-1

			if current < self.target:
				if prog_width % 8 == 0:
					bar += ' '

				if prog_width % 8 == 1:
					bar += '▏'

				if prog_width % 8 == 2:
					bar += '▎'

				if prog_width % 8 == 3:
					bar += '▍'

				if prog_width % 8 == 4:
					bar += '▌'

				if prog_width % 8 == 5:
					bar += '▋'

				if prog_width % 8 == 6:
					bar += '▊'

				if prog_width % 8 == 7:
					bar += '▉'

			bar += " " * (self.width - prog_width // 8 - 1)
			bar += "|"

		else:
			bar = "%7d/Unknown" % current

		self._total_width = len(bar)
		sys.stdout.write(bar)

		time_per_unit = self._estimate_step_duration(current, now)

		if self.target is None or finalize:
			pass
		#	info += f"  {_format_time(time_per_unit):s}"

		else:
			info = f"  {_format_time(time_per_unit * (self.target - current)):s}"

		for k in self._values_order:
			info += f"  {k:s}:"

			if isinstance(self._values[k], list):
				info += f" {numpy.mean(self._values[k][0] / max(1, self._values[k][1])):6.4f}"

			else:
				info += f" {self._values[k]:6s}"

		self._total_width += len(info)

		if prev_total_width > self._total_width:
			info += (" " * (prev_total_width - self._total_width))

		if finalize:
			info += "\n"

		sys.stdout.write(info)
		sys.stdout.flush()

	elif self.verbose == 2:
		if finalize:
			numdigits = 7  # int(numpy.log10(self.target)) + 1
			count = (f"%{str(numdigits)}d/%{str(numdigits)}d |") % (current, self.target)
			info = count + info

			for k in self._values_order:
				info += f"  {k:s}:"
				info += f" {numpy.mean(self._values[k][0] / max(1, self._values[k][1])):6.4f}"

			if self._time_at_epoch_end:
				time_per_epoch = self._time_at_epoch_end - self._time_at_epoch_start
				avg_time_per_step = time_per_epoch / self.target
				self._time_at_epoch_start = now
				self._time_at_epoch_end = None
				info += f"  {_format_time(time_per_epoch)}"
				info += f"  {_format_time(avg_time_per_step)}"
				info += "\n"

			sys.stdout.write(info)
			sys.stdout.flush()

	self._last_update = now


Progbar.update = update  # NOTE: MONKEYPATCH
