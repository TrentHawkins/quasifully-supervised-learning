"""Zeroshot metrics."""


import tensorflow


@tensorflow.keras.utils.register_keras_serializable("source>zeroshot>metrics")
class ZeroshotCategoricalAccuracy(tensorflow.keras.metrics.CategoricalAccuracy):
	"""Calculate how often predictions match one-hot labels.

	You can provide logits of classes as `y_pred`, since argmax of logits and probabilities are same.

	This metric creates two local variables, `total` and `count`,
	that are used to compute the frequency with which `y_pred` matches `y_true`.

	This frequency is ultimately returned as `categorical accuracy`:
		an idempotent operation that simply divides `total` by `count`.

	`y_pred` and `y_true` should be passed in as vectors of probabilities, rather than as labels.
	If necessary, use `tf.one_hot` to expand `y_true` as a vector.

	If `sample_weight` is `None`, weights default to 1.
	Use `sample_weight` of 0 to mask values.

	Argumentss:

	Standalone usage:
	```
	>>>	m = tf.keras.metrics.CategoricalAccuracy()
	>>>	m.update_state(
	...		[
	...			[0, 0, 1],
	...			[0, 1, 0]
	...		],[
	...			[0.1 , 0.9 , 0.8],
	...			[0.05, 0.95, 0  ],
	...		]
	...	)
	>>>	m.result().numpy()
	0.5
	>>>	m.reset_state()
	>>>	m.update_state(
	...		[
	...			[0, 0, 1],
	...			[0, 1, 0]
	...		],[
	...			[0.1 , 0.9 , 0.8],
	...			[0.05, 0.95, 0  ],
	...		],
	...		sample_weight=[
	...			0.7,
	...			0.3,
	...		]
	...	)
	>>>	m.result().numpy()
	0.3
	```

	Usage with `compile()` API:
	```python
	model.compile(
		optimizer="sgd",
		loss="mse",
		metrics=[
			tf.keras.metrics.CategoricalAccuracy()
		]
	)
	```
	"""

	...
