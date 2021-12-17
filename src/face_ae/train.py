import tensorflow as tf
import os
import glob
import PIL
import PIL.Image
from tensorflow.keras import models, layers, datasets, preprocessing, losses, optimizers, utils
import numpy as np
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from shutil import copyfile

IMG_WIDTH = 64
IMG_HEIGHT = 64
BATCH_SIZE = 32

print("loading data...")
np_data = []
for url in glob.glob('data/face_data/*.png'):
    image = PIL.Image.open(url)
    tensor = np.array(image)
    tensor = tensor/127.5-1
    np_data.append(tensor)

np_data = np.stack(np_data)
# training_data = tf.data.Dataset.from_tensor_slices(np.stack(np_data)).shuffle(1000).batch(128)
training_data = tf.data.Dataset.from_tensor_slices((np_data, np_data)).shuffle(1000).batch(BATCH_SIZE)
print("done")
#create model

ENCODED_SIZE = 256

def create_encoder():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(8, 3, activation = "relu", input_shape=(IMG_WIDTH,IMG_HEIGHT,3)))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(16, 3, activation = "relu"))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(32, 3, activation = "relu"))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(64, 3, activation = "relu"))
    model.add(layers.MaxPool2D())
    model.add(layers.Reshape((256,)))
    model.add(layers.Dense(256, activation = "relu"))
    model.add(layers.Dense(256, activation = "relu"))
    model.add(layers.Dense(ENCODED_SIZE, activation = "tanh"))
    return model

encoder = create_encoder()

def create_decoder():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_shape=(ENCODED_SIZE,), activation = "relu"))
    model.add(layers.Dense(576, activation = "relu"))
    model.add(layers.Reshape((3,3,64)))
    model.add(layers.Conv2DTranspose(64, 3, activation = "relu"))
    model.add(layers.Conv2DTranspose(64, 3, activation = "relu"))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2DTranspose(32, 3, activation = "relu"))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2DTranspose(16, 3, activation = "relu"))
    model.add(layers.Conv2D(8, 3, activation = "relu"))
    model.add(layers.Conv2DTranspose(8, 3, activation = "relu"))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2DTranspose(8, 3, activation = "relu"))
    model.add(layers.Conv2D(8, 3, activation = "relu"))
    model.add(layers.Conv2D(8, 3, activation = "relu"))
    model.add(layers.Conv2D(3, 3, activation = "tanh"))
    return model

decoder = create_decoder()

autoencoder = models.Sequential([
    encoder,
    decoder
], name = "autoencoder")


autoencoder.compile(
    optimizer = 'adam',
    loss = losses.MeanSquaredError(),
    metrics=['accuracy']
)
autoencoder.fit(training_data, epochs=200)

def showface(n):
    k = autoencoder(np_data[n:n+1])[0]
    k = k+1
    k = k/2
    plt.imshow(k)
    plt.show()

def showrandom():
    k = decoder(tf.random.normal((1,256)))
    k = k+1
    k = k/2
    k = k[0]
    plt.imshow(k)
    plt.show()