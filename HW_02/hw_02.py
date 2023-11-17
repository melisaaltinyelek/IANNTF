import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras.layers import Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# Splitting the data into training and test datasets.
(train_ds, test_ds), ds_info = tfds.load("mnist", split=["train", "test"], as_supervised=True, with_info=True)

#print(ds_info)

"""
How many training/test images are there? --> 60000 and 10000, respectively.
What’s the image shape? --> Image(shape=(28, 28, 1)
What range are pixel values in? --> [0, 255]
"""

# Image data type
# for image,label in train_ds.take(1):
    # print(f"Image data type: {image.dtype}")

# The range of pixel values
# min_pixel_value = 255
# max_pixel_value = 0

# for image, label in train_ds.take(1000):  # Take a subset for efficiency
#     min_pixel_value = min(min_pixel_value, tf.reduce_min(image))
#     max_pixel_value = max(max_pixel_value, tf.reduce_max(image))

# print("Min Pixel Value:", min_pixel_value.numpy())
# print("Max Pixel Value:", max_pixel_value.numpy())

# tfds.show_examples(train_ds, ds_info)

def prepare_mnist_data(mnist):
    # Covert uint8 datatype to tf.float values
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # Reshape and flatten 28x28 images
    mnist = mnist.map(lambda img, target: (tf.reshape(img, (-1,)), target))
    # Input normalization by bringing image values from [0, 255] to [-1, 1]
    mnist = mnist.map(lambda img, target: ((img / 128.) - 1., target))
    # One-hot encode labels for targets
    mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth=10)))

    mnist = mnist.cache()

    # Shuffle the data, 1000 elements are loaded into the memory for shuffling at a time.
    mnist = mnist.shuffle(1000)
    # Create 32 number of samples (image - label pairs).
    mnist = mnist.batch(32)
    # Prefetch to make 20 elements ready for the next interation.
    mnist = mnist.prefetch(20)

    return mnist

train_dataset = train_ds.apply(prepare_mnist_data)
test_dataset = test_ds.apply(prepare_mnist_data)


# Prepare the model

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    # Define the forward pass of the model.
    @tf.function
    def call(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.output_layer(x)
        return x

###################### Train the Model ###################### 

def train_model(model, train_dataset, test_dataset, loss_function, optimizer, num_epochs):
    
    train_loss_final = []
    test_loss_final = []


    for epoch in range(num_epochs):

        train_loss_aggregator = []
        test_loss_aggregator = []

        print(f"Epoch: {str(epoch)} Test data loss: {test_loss_final}")
        
        for input, target in train_dataset:
            with tf.GradientTape() as tape:
                prediction = model(input)
                loss = loss_function(target, prediction)
                train_loss_aggregator.append(loss)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            

        for input, target in test_dataset:
            prediction = model(input)
            sample_test_loss = loss_function(target, prediction)
            test_loss_aggregator.append(sample_test_loss.numpy())


        test_loss = tf.reduce_mean(test_loss_aggregator)
        train_loss = tf.reduce_mean(train_loss_aggregator)
        test_loss_final.append(test_loss)
        train_loss_final.append(train_loss)

    return train_loss_final, test_loss_final

###################### Application ###################### 

tf.keras.backend.clear_session()

train_dataset = train_dataset.take(1000)
test_dataset = test_dataset.take(100)

# Initialize the model
model = MyModel()
# Initialize the loss
cross_entrophy_loss = tf.keras.losses.CategoricalCrossentropy()
# Initialize the optimuzer
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)
train_loss_final, test_loss_final = train_model(model, train_dataset, test_dataset, cross_entrophy_loss, optimizer, 20)

# Visualize loss for training and test data

plt.figure()
line1, = plt.plot(train_loss_final)
line2, = plt.plot(test_loss_final)
plt.xlabel("num of epochs")
plt.ylabel("Loss")
plt.legend((line1,line2),("train loss","test loss"))
plt.show()