import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'

def download(url):
  name = url.split('/')[-1]
  image_path = tf.keras.utils.get_file(name, origin=url)
  img = PIL.Image.open(image_path).resize((32,32))
  return np.array(img)

#load model
animal_classifier = models.load_model("saved_models/animal_classifier")

print(animal_classifier(np.array([download(url)])))