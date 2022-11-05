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


from functools import partial

import tensorflow


tensorflow.keras.applications.efficientnet.EfficientNetB0 = partial(
tensorflow.keras.applications.efficientnet.EfficientNetB0, include_top=False, pooling="avg")
tensorflow.keras.applications.efficientnet.EfficientNetB1 = partial(
tensorflow.keras.applications.efficientnet.EfficientNetB1, include_top=False, pooling="avg")
tensorflow.keras.applications.efficientnet.EfficientNetB2 = partial(
tensorflow.keras.applications.efficientnet.EfficientNetB2, include_top=False, pooling="avg")
tensorflow.keras.applications.efficientnet.EfficientNetB3 = partial(
tensorflow.keras.applications.efficientnet.EfficientNetB3, include_top=False, pooling="avg")
tensorflow.keras.applications.efficientnet.EfficientNetB4 = partial(
tensorflow.keras.applications.efficientnet.EfficientNetB4, include_top=False, pooling="avg")
tensorflow.keras.applications.efficientnet.EfficientNetB5 = partial(
tensorflow.keras.applications.efficientnet.EfficientNetB5, include_top=False, pooling="avg")
tensorflow.keras.applications.efficientnet.EfficientNetB6 = partial(
tensorflow.keras.applications.efficientnet.EfficientNetB6, include_top=False, pooling="avg")
tensorflow.keras.applications.efficientnet.EfficientNetB7 = partial(
tensorflow.keras.applications.efficientnet.EfficientNetB7, include_top=False, pooling="avg")

tensorflow.keras.applications.efficientnet_v2.EfficientNetV2B0 = partial(
tensorflow.keras.applications.efficientnet_v2.EfficientNetV2B0, include_top=False, pooling="avg")
tensorflow.keras.applications.efficientnet_v2.EfficientNetV2B1 = partial(
tensorflow.keras.applications.efficientnet_v2.EfficientNetV2B1, include_top=False, pooling="avg")
tensorflow.keras.applications.efficientnet_v2.EfficientNetV2B2 = partial(
tensorflow.keras.applications.efficientnet_v2.EfficientNetV2B2, include_top=False, pooling="avg")
tensorflow.keras.applications.efficientnet_v2.EfficientNetV2B3 = partial(
tensorflow.keras.applications.efficientnet_v2.EfficientNetV2B3, include_top=False, pooling="avg")

tensorflow.keras.applications.efficientnet_v2.EfficientNetV2S = partial(
tensorflow.keras.applications.efficientnet_v2.EfficientNetV2S, include_top=False, pooling="avg")
tensorflow.keras.applications.efficientnet_v2.EfficientNetV2M = partial(
tensorflow.keras.applications.efficientnet_v2.EfficientNetV2M, include_top=False, pooling="avg")
tensorflow.keras.applications.efficientnet_v2.EfficientNetV2L = partial(
tensorflow.keras.applications.efficientnet_v2.EfficientNetV2L, include_top=False, pooling="avg")


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
