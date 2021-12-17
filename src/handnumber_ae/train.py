import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses
import matplotlib.pyplot as plt

SQUEEZE = 100

PLT_W = 5
PLT_H = 4

#load up data
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = x_train, x_test

BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train)).shuffle(60000).batch(256)

encoder = models.Sequential([
	layers.Reshape((28, 28, 1), input_shape=(28, 28)),
	layers.Conv2D(8, 3, activation = "relu"),
	layers.MaxPool2D(),
	layers.Conv2D(16, 3, activation = "relu"),
	layers.MaxPool2D(),
	layers.Conv2D(32, 3, activation = "relu"),
	layers.Flatten(),
	layers.Dense(128, activation = "relu"),
	layers.Dense(SQUEEZE, activation = "relu")
], name = "encoder")

decoder = models.Sequential([
	layers.Dense(128, input_shape=(SQUEEZE,), activation = "relu"),
	layers.Dense(128, activation = "relu"),
	layers.Dense(288, activation = "relu"),
	layers.Reshape((3,3,32)),
	layers.Conv2DTranspose(16, (3,3), activation = "relu"),
	layers.UpSampling2D(),
	layers.Conv2DTranspose(16, (3,3), activation = "relu"),
	layers.UpSampling2D(),
	layers.Conv2DTranspose(8, (3,3), activation = "relu"),
	layers.Conv2DTranspose(1, (3,3), activation = "relu"),
	layers.Reshape((28,28)),
], name = "decoder")

autoencoder = models.Sequential([
	encoder,
	decoder
], name = "autoencoder")

autoencoder.compile(
	optimizer = 'adam',
	loss = losses.MeanSquaredError(),
	metrics=['accuracy']
)
autoencoder.fit(train_dataset, epochs=20)

pred = autoencoder(x_test[0:PLT_W]).numpy()
rand_input = tf.random.normal((PLT_W,28,28))
rand = autoencoder(rand_input).numpy()

plt.figure(figsize=(10, 10))
for i in range(PLT_W):
    ax = plt.subplot(PLT_H, PLT_W, i + 1)
    plt.imshow(x_test[i])
    plt.title("encoded")
    plt.axis("off")
    ax = plt.subplot(PLT_H, PLT_W, i + 6)
    plt.imshow(pred[i])
    plt.title("decoded")
    plt.axis("off")
    ax = plt.subplot(PLT_H, PLT_W, i + 11)
    plt.imshow(rand_input[i])
    plt.title("random input")
    plt.axis("off")
    ax = plt.subplot(PLT_H, PLT_W, i + 16)
    plt.imshow(rand[i])
    plt.title("random")
    plt.axis("off")

plt.show()

