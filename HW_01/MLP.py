from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import random

# n_class: the digits from 0 to 9
# return_X_y = False: (data, target) as a tuple
# as_frame = False: returns (data, target) as a numpy array
digits_dataset = load_digits(n_class = 10, return_X_y = False, as_frame = False)

# Accessing the data (input) and target (output)
data = digits_dataset.data
target = digits_dataset.target

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
data_as_float = reshaped_image_array.astype(np.float32)
print(data_as_float)

# Scale the data to the range of 0 to 1
data_as_float /= 255.0

# Creating an empty list to store the one hot encoded labels
one_hot_encoded_labels = []

# Iterating over the target and creating a one hot encoded label for each target
for t in target:
    one_hot_encoded = np.zeros(10, dtype = np.float32)
    # Setting the index of the label to 1
    one_hot_encoded[t] = 1
    one_hot_encoded_labels.append(one_hot_encoded)

one_hot_encoded_labels_array = np.array(one_hot_encoded_labels)  
print(one_hot_encoded_labels)

def generator(input_data, target_data, batch_size):
    """Shuffles the input - target pairs"""

    input_target_pairs = list(zip(input_data, target_data))
    random.shuffle(input_target_pairs)

    total_samples = len(input_target_pairs)
    mini_batches = total_samples // batch_size

    #for m in range(mini_batches):

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
        pre_activations = self.weights @ x + self.bias
        activations = sigmoid(pre_activations) 
        return activations 
