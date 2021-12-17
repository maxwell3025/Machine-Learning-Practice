import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

print(model.evaluate(x_test,  y_test, verbose=2))

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

np.set_printoptions(suppress=True)
def do_test(index):
  sample = x_test[index:index+1]
  plt.imshow(sample[0])
  print(f"predicted: {np.argmax(probability_model(sample).numpy()[0])} with {np.max(probability_model(sample).numpy()[0])}% confidence")
  print(f"correct answer: {y_test[index]}")
  plt.show(block=False)

for index in range(0,100):
  do_test(index)
  plt.pause(4)