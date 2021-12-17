
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import tensorflow as tf
import time
mnist = tf.keras.datasets.mnist

#load up data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#define model structure
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

#train the NN
model.fit(x_train, y_train, epochs=5)


model.save("saved_models/number_classifier")
