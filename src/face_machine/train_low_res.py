import tensorflow as tf
import numpy as np
import PIL
import PIL.Image
from tensorflow.keras import datasets, layers, models, losses, optimizers
import matplotlib.pyplot as plt
import time
import glob

BATCH_SIZE = 32

# print("loading data...")
# np_data = []
# for url in glob.glob('data/face_data/*.png'):
#     image = PIL.Image.open(url)
#     image = image.resize((32,32))
#     tensor = np.array(image)
#     tensor = tensor/127.5-1
#     np_data.append(tensor)
# np_data = np.stack(np_data)
np_data = np.load("data/numpy_datasets/face_tensor_low_res.npy")
training_data = tf.data.Dataset.from_tensor_slices(np_data).shuffle(1000).batch(BATCH_SIZE)
print("done")
#create model

NOISE_DIM = 100

def make_gen_model():
    model = tf.keras.Sequential()
    #NOISE_DIM

    model.add(layers.Dense(4*4*32, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((4, 4, 32)))
    #4x4

    model.add(layers.UpSampling2D(size=(2, 2), interpolation="nearest"))
    #8x8

    model.add(layers.Conv2D(64, (3, 3), padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #8x8

    model.add(layers.UpSampling2D(size=(2, 2), interpolation="nearest"))
    #16x16

    model.add(layers.Conv2D(32, (3, 3), padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #16x16

    model.add(layers.UpSampling2D(size=(2, 2), interpolation="nearest"))
    #32x32

    model.add(layers.Conv2D(32, (3, 3), padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #32x32

    model.add(layers.Conv2D(3, (3, 3), padding = 'same', activation='tanh'))
    #32x32

    return model

def make_dis_model():
    model = tf.keras.Sequential()
    #32x32

    model.add(layers.Conv2D(8, (3, 3), padding = 'same', input_shape = (32, 32, 3)))
    model.add(layers.LeakyReLU())
    #32x32

    model.add(layers.MaxPool2D())
    #16x16

    model.add(layers.Conv2D(16, (3, 3), padding = 'same'))
    model.add(layers.LeakyReLU())
    #16x16

    model.add(layers.MaxPool2D())
    #8x8

    model.add(layers.Conv2D(32, (3, 3), padding = 'same'))
    model.add(layers.LeakyReLU())
    #8x8

    model.add(layers.MaxPool2D())
    #4x4

    model.add(layers.Conv2D(64, (3, 3), padding = 'same'))
    model.add(layers.LeakyReLU())
    #4x4

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

generator = make_gen_model()
discriminator = make_dis_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# def dis_loss(real, fake):
#     real_loss = cross_entropy(tf.ones_like(real), real)
#     fake_loss = cross_entropy(tf.zeros_like(fake), fake)
#     return real_loss+fake_loss

# def gen_loss(fake):
#     return cross_entropy(tf.ones_like(fake),fake)

# wasserstein loss
def dis_loss(real, fake):
    return tf.reduce_mean(fake)-tf.reduce_mean(real)

def gen_loss(fake):
    return -tf.reduce_mean(fake)

gen_optimizer = optimizers.Adam(1e-4)
dis_optimizer = optimizers.Adam(1e-4)

EPOCHS = 4

@tf.function()
def train_step(images):
    noise = tf.random.normal((BATCH_SIZE, NOISE_DIM))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        generated_images = generator(noise)

        real = discriminator(images)
        fake = discriminator(generated_images)

        d_loss = dis_loss(real, fake)
        g_loss = gen_loss(fake)

    gen_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
    dis_grads = dis_tape.gradient(d_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    dis_optimizer.apply_gradients(zip(dis_grads, discriminator.trainable_variables))

def test_step():
    images = next(iter(training_data))
    noise = tf.random.normal((BATCH_SIZE, NOISE_DIM))
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        generated_images = generator(noise)
        real = discriminator(images)
        fake = discriminator(generated_images)
        d_loss = dis_loss(real, fake)
        g_loss = gen_loss(fake)
    print('discriminator loss: {}'.format(d_loss))
    print('generator loss: {}'.format(g_loss))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        test_step()

def test():
    noise = tf.random.normal((1,100))
    image = generator(noise)
    image = (image.numpy()+1)/2
    image = image.reshape(32,32,3)
    plt.imshow(image)
    plt.show()