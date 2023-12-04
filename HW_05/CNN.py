import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras.layers import Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

[train_ds, test_ds], ds_info = tfds.load("cifar10", split = ["train", "test"], as_supervised = True, with_info = True)
# Visualize a sample of the dataset
#fig = tfds.show_examples(train_ds, ds_info)

""" Information about the dataset: 

1. Consists of 60000 32x32x3 colored images in 10 different classes.
2. There are 50000 training images and 10000 test images.
3. Pixel values range from 0 to 255.
4. Labels are N x !.

"""

def prepare_cifar10_data(cifar10):
    # Convert uint8 datatype to tf.float32 values
    cifar10 = cifar10.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # Normalize the inputs by bringing the image values to [-1, 1]
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

#####################################################################

# Define the class named BasicConv that inherit from tf.keras.Model
class BasicConv(tf.keras.Model):
    # Initialize the class constructor
    def __init__(self):
        super(BasicConv, self).__init__()
        
        # Filters define the features (24) and kernel size defines the filter size (3x3)
        self.convlayer1 = tf.keras.layers.Conv2D(filters = 24, kernel_size = 3, padding = "same", activation = "relu")
        self.convlayer2 = tf.keras.layers.Conv2D(filters = 25, kernel_size = 3, padding = "same", activation = "relu")
        # Pooling size is 2x2 and stride (step size) is 2
        self.pooling = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2)

        self.convlayer3 = tf.keras.layers.Conv2D(filters = 48, kernel_size = 3, padding = "same", activation = "relu")
        self.convlayer4 = tf.keras.layers.Conv2D(filters = 48, kernel_size = 3, padding = "same", activation = "relu")
        self.global_pool = tf.keras.layers.GlobalAvgPool2D()

        self.out = tf.keras.layers.Dense(10, activation = "softmax")

    def call(self, x):
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.pooling(x)
        x = self.convlayer3(x)
        x = self.convlayer4(x)
        x = self.global_pool(x)
        x = self.out(x)
        return x
    
#####################################################################

def train_model(model, train_dataset, test_dataset, loss_function, optimizer, num_epochs):
    
    # Store mean losses for each epoch in the train and test datasets
    train_loss_final = []
    test_loss_final = []

    # Store mean accuracies for each epoch in the train and test datasets
    train_accuracy_final = []
    test_accuracy_final = []

    for epoch in range(num_epochs):
        # Store losses and accuracies for each batch in the train and test datasets.
        train_loss_aggregator = []
        test_loss_aggregator = []
        train_accuracy_aggregator = []
        test_accuracy_aggregator = []

        if epoch >=1:
            # -1 accesses the most recent loss stored in test_loss_final and test_accuracy_final
            print(f'Epoch: {str(epoch)} >>> Test data loss: {test_loss_final[-1]}, Accuracy: {test_accuracy_final[-1]}')
        
        # Iterate over the batches in the training dataset.
        for input, target in train_dataset:
            with tf.GradientTape() as tape:
                # Forward pass to get predictions for the given input batch
                prediction = model(input)
                # Compute the loss between the target and predicted output
                loss = loss_function(target, prediction)
                # Append the computed loss to the train_loss_aggregator
                train_loss_aggregator.append(loss)

        
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Calculate training accuracy
            train_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
            train_accuracy_metric.update_state(target, prediction)
            train_accuracy = train_accuracy_metric.result().numpy()
            train_accuracy_aggregator.append(train_accuracy)

        train_accuracy_metric.reset_states()
            
        # Iterate over the batches in the testing dataset.
        for input, target in test_dataset:
            prediction = model(input)
            sample_test_loss = loss_function(target, prediction)
            test_loss_aggregator.append(sample_test_loss.numpy())

            # Calculate test accuracy
            test_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
            test_accuracy_metric.update_state(target, prediction)
            test_accuracy = test_accuracy_metric.result().numpy()
            test_accuracy_aggregator.append(test_accuracy)

        test_accuracy_metric.reset_states()


        # Calculate the mean and append losses and accuracies
        test_loss = tf.reduce_mean(test_loss_aggregator)
        train_loss = tf.reduce_mean(train_loss_aggregator)

        test_acc = tf.reduce_mean(test_accuracy_aggregator)
        train_acc = tf.reduce_mean(train_accuracy_aggregator)

        test_loss_final.append(test_loss)
        train_loss_final.append(train_loss)

        test_accuracy_final.append(test_acc)
        train_accuracy_final.append(train_acc)

    return train_loss_final, test_loss_final, test_accuracy_final, train_accuracy_final

#####################################################################

# Define hyperparameters: loss function, optimizer, learning rate, momentum

tf.keras.backend.clear_session()
# Take subsets from both train and test datasets
train_dataset = train_dataset.take(1000)
test_dataset = test_dataset.take(100)
# Define the loss
cce_loss = tf.keras.losses.CategoricalCrossentropy()
# Initialize the model
model = BasicConv()
# Define the optimizer along with learning rate and momentum
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate = 0.01, momentum = 0.9)

train_loss_final, test_loss_final, train_accuracy_final, test_accuracy_final = train_model(model, train_dataset, test_dataset, cce_loss, optimizer, 15)


#####################################################################

# Visualize loss, accuracy for training and test data

plt.figure()
line1, = plt.plot(train_loss_final)
line2, = plt.plot(test_loss_final)
line3, = plt.plot(train_accuracy_final)
line4, = plt.plot(test_accuracy_final)

plt.xlabel("Number of epochs")
plt.ylabel("Metric")
plt.legend((line1, line2, line3, line4), ("train loss", "test loss", "train accuracy", "test accuracy"))
plt.show()