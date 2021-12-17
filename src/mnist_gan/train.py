import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, losses, optimizers
import matplotlib.pyplot as plt
import time
import glob

#load data
(train_images, _), (_, _) = datasets.mnist.load_data()
BATCH_SIZE = 256
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(256)

#define model constants
NOISE_DIM = 100

def gen_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

def dis_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def dis_loss(real, fake):
    real_loss = cross_entropy(tf.ones_like(real), real)
    fake_loss = cross_entropy(tf.zeros_like(fake), fake)
    return real_loss + fake_loss

def gen_loss(fake):
    return cross_entropy(tf.ones_like(fake), fake)

gen_optimizer = optimizers.Adam(1e-4)
dis_optimizer = optimizers.Adam(1e-4)

EPOCHS = 25

generator = gen_model()
discriminator = dis_model()

@tf.function()
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        generated_images = generator(noise, training = True)

        real = discriminator(images, training = True)
        fake = discriminator(generated_images, training = True)

        g_loss = gen_loss(fake)
        d_loss = dis_loss(real, fake)

    gen_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
    dis_grads = dis_tape.gradient(d_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    dis_optimizer.apply_gradients(zip(dis_grads, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


train(train_dataset, EPOCHS)

def gen_image():
    noise = tf.random.normal((1,100))
    image = generator(noise)
    image = image.numpy()       
    image = image.reshape(28,28)
    return image

def show_image(img):
    plt.imshow(img)
    plt.show()