import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras.layers import Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

[train_ds, test_ds], ds_info = tfds.load("cifar10", split = ["train", "test"], as_supervised = True, with_info = True)
#print(ds_info)

""" Information about the dataset: 

1. Consists of 60000 32x32x3 colored images in 10 different classes.
2. There are 50000 training images and 10000 test images.
3. Pixel values range from 0 to 255.
4. Labels are N x !.

"""

#tfds.show_examples(train_ds, ds_info)
def prepare_cifar10_data(cifar10):
    # Convert uint8 datatype to tf.floa32 values
    cifar10 = cifar10.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # Reshape and flatten the 32x32 images
    cifar10 = cifar10.map(lambda img, target: (tf.reshape(img, (-1,)), target))
    # Normalize the inputs by bringing the image values to [0, 1]
    cifar10 = cifar10.map(lambda img, target: ((img / 128) -1., target))
    # One-hot encoding for labels
    cifar10 = cifar10.map(lambda img, target: (img, tf.one_hot(target, depth = 10)))
    
    cifar10 = cifar10.cache()

    # Shuffle the data, 1000 elements are loaded into the memory for shuffling at a time
    cifar10 = cifar10.shuffle(1000)
    # Create 32 number of samples (image - label pairs)
    cifar10 = cifar10.batch(32)
    # Prefetch to make 20 elements ready for the next interation
    cifar10 = cifar10.prefetch(20)

    return cifar10

train_dataset = train_ds.apply(prepare_cifar10_data)
test_dataset = test_ds.apply(prepare_cifar10_data)