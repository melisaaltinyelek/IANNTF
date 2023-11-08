from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import random

# Loading the dataset
digits = load_digits()

# Accessing the data (input) and target (output)
data, target = digits.data, digits.target

# Iterating over the dataset and creating a tuple of input and output
input_output_tuple = [(data[i], target[i]) for i in range (len(data))]
#print(input_output_tuple[0])

# Plotting the first 5 images in the dataset
num_images_to_show = 5
for n in range(num_images_to_show):
    # Forms one row with five columns, each representing the image of the digit.
    plt.subplot(1, num_images_to_show, n + 1)
    plt.imshow(data[n].reshape(8, 8), cmap = "gray")
    plt.title(f"Digits: {target[n]}")
    # No labels and ticks for x axis
    plt.axis("off")

#plt.show()

# Reshaping the images to 1D array
reshaped_images = []
for image in data:
    reshaped_image = image.reshape(64,)
    reshaped_images.append(reshaped_image)

reshaped_image_array = np.array(reshaped_images)
print(reshaped_image_array)

# Converting the data type of the array to float32.
reshaped_data = reshaped_image_array.astype(np.float32)
print(reshaped_data)

# Scale the data to the range of 0 to 1
reshaped_data /= 255.0

# One hot encoding the target
num_classes = 10
one_hot_targets = np.eye(num_classes)[target]

# Create (input, target) tuples with reshaped data
reshaped_input_target_tuple = list(zip(reshaped_data, one_hot_targets))
######################################################

def generator(input_target_data, minibatch_size):
    """Shuffles the input - target pairs and yields minibatches"""

    # Packing the input and target data into a list of tuples
    input_target_pairs = list(zip(input_target_data))
    # Shuffling the input - target pairs
    random.shuffle(input_target_pairs)
    # Calculating the number of minibatches
    num_minibatches = len(input_target_pairs) // minibatch_size

    # Iterating over the minibatches and yielding the input and target data.
    for j in range(num_minibatches):
        # Calculating the start and end indices of a minibatch
        start_index = j * minibatch_size
        end_index = start_index + minibatch_size
        minibatch = input_target_data[start_index:end_index]

        # Unpacking the minibatch into input and target data
        minibatch_input, minibatch_target = zip(*minibatch)
        minibatch_input = np.array(minibatch_input)
        minibatch_target = np.array(minibatch_target)

        yield minibatch_input, minibatch_target


class sigmoid():
    def __call__(self, x):
        return 1 / 1 + np.exp(-x)
    
class Softmax():
    def __Call__(self, x):
        return np.exp / np.sum(np.exp(x), axis = 1, keepdims = True)

class MLP_Layer():
    def __init__(self, num_inputs, num_neurons):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons

        self.weights = np.zeros(self.num_neurons, self,num_inputs) 
        self.bias = np.zeros((self.num_neurons))

    def set_weights(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def call(self, x):
        pre_activations = self.weights @ x + np.transpose(self.bias)
        activations = sigmoid(pre_activations) 
        return activations 
