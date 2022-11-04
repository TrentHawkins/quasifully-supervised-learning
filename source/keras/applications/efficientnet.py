"""EfficientNet models for Keras.

NOTE: THIS FILE IS MOKNEYPATCHING THE ORIGINAL

References:
-	[EfficientNet  : Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)
-	[EfficientNetV2: Smaller Models and Faster Training                        ](https://arxiv.org/abs/2104.00298) (ICML 2021)

Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
"""


from enum import Enum
from functools import partial


import tensorflow


class EfficientNetV2(Enum):
	"""Enumerate the various EfficientNetV2 models specifically with their default input sizes applied."""

#	Small EfficientNetV2 models:
	B0 = partial(tensorflow.keras.applications.efficientnet_v2.EfficientNetV2B0, include_top=False, pooling="avg")
	B1 = partial(tensorflow.keras.applications.efficientnet_v2.EfficientNetV2B1, include_top=False, pooling="avg")
	B2 = partial(tensorflow.keras.applications.efficientnet_v2.EfficientNetV2B2, include_top=False, pooling="avg")
	B3 = partial(tensorflow.keras.applications.efficientnet_v2.EfficientNetV2B3, include_top=False, pooling="avg")

#	Large EfficientNetV2 models:
	S = partial(tensorflow.keras.applications.efficientnet_v2.EfficientNetV2S, include_top=False, pooling="avg")
	M = partial(tensorflow.keras.applications.efficientnet_v2.EfficientNetV2M, include_top=False, pooling="avg")
	L = partial(tensorflow.keras.applications.efficientnet_v2.EfficientNetV2L, include_top=False, pooling="avg")


class EfficientNet(Enum):
	"""Enumerate the various EfficientNet models with their default input sizes applied."""

#	EfficientNet models:
	B0 = partial(tensorflow.keras.applications.efficientnet.EfficientNetB0, include_top=False, pooling="avg")
	B1 = partial(tensorflow.keras.applications.efficientnet.EfficientNetB1, include_top=False, pooling="avg")
	B2 = partial(tensorflow.keras.applications.efficientnet.EfficientNetB2, include_top=False, pooling="avg")
	B3 = partial(tensorflow.keras.applications.efficientnet.EfficientNetB3, include_top=False, pooling="avg")
	B4 = partial(tensorflow.keras.applications.efficientnet.EfficientNetB4, include_top=False, pooling="avg")
	B5 = partial(tensorflow.keras.applications.efficientnet.EfficientNetB5, include_top=False, pooling="avg")
	B6 = partial(tensorflow.keras.applications.efficientnet.EfficientNetB6, include_top=False, pooling="avg")
	B7 = partial(tensorflow.keras.applications.efficientnet.EfficientNetB7, include_top=False, pooling="avg")

#	EfficientNetV2 models:
	V2 = EfficientNetV2

	@classmethod
	def models(cls):
		"""Iterate over all EfficientNet models."""
		for old_model in cls:
			if old_model == cls.V2:
				for new_model in old_model.__class__.V2.value:
					yield new_model.value

			else:
				yield old_model.value


inputs_size = {
	"B0": 224,
	"B1": 240,
	"B2": 260,
	"B3": 300,
	"B4": 380,
	"B5": 456,
	"B6": 528,
	"B7": 600,

	"V2": {
		"B0": 224,
		"B1": 240,
		"B2": 260,
		"B3": 300,

		"S": 384,
		"M": 480,
		"L": 480,
	}
}

output_size = {
	"B0": 1280,
	"B1": 1280,
	"B2": 1408,
	"B3": 1536,
	"B4": 1792,
	"B5": 2048,
	"B6": 2304,
	"B7": 2560,

	"V2": {
		"B0": 1280,
		"B1": 1280,
		"B2": 1408,
		"B3": 1536,

		"S": 1280,
		"M": 1280,
		"L": 1280,
	}
}


if __name__ == "__main__":
	for model in EfficientNet.models():
		print(f"Fecthing {model().name} (if not already).")
