"""ConvNeXt models for Keras.

NOTE: THIS FILE IS MOKNEYPATCHING THE ORIGINAL

References:
-	[A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) (CVPR 2022)

Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
"""


from enum import Enum, IntEnum
from functools import partial

import tensorflow


class ConvNeXt:
	"""ConvNeXt topless with average pooling."""

#	Benchmark model:
	Tiny = partial(tensorflow.keras.applications.convnext.ConvNeXtTiny, include_top=False, pooling="avg")

#	Full depth models:
	Small = partial(tensorflow.keras.applications.convnext.ConvNeXtSmall, include_top=False, pooling="avg")
	Base = partial(tensorflow.keras.applications.convnext.ConvNeXtBase, include_top=False, pooling="avg")
	Large = partial(tensorflow.keras.applications.convnext.ConvNeXtLarge, include_top=False, pooling="avg")
	XLarge = partial(tensorflow.keras.applications.convnext.ConvNeXtXLarge, include_top=False, pooling="avg")


class inputs_size(IntEnum):
	"""ConvNeXt default sizes."""

#	Benchmark model:
	Tiny = 224

#	Full depth models:
	Small = 224
	Base = 224
	Large = 224
	XLarge = 224


class output_size(IntEnum):
	"""ConvNeXt default sizes."""

#	Benchmark model:
	Tiny = 768

#	Full depth models:
	Small = 768
	Base = 1024
	Large = 1536
	XLarge = 2048
