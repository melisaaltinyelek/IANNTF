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

#####################################################################

class Sigmoid():
    def __call__(self, x):
        return 1 / 1 + np.exp(-x)
    
    def backward(self, output):
        return output * (1 - output)
    
#####################################################################
    
class Softmax_Activation():
    def __call__(self, x):
        return np.exp / np.sum(np.exp(x), axis = 1, keepdims = True)
    
    def backward(self, output):
        return output * (1 - output)
    
#####################################################################
    
class CCE_Loss():
    def __call__(self, predicted_probs, true_labels):
    
        # Avoid the division by zero
        predicted_probs = np.clip(predicted_probs, 1e-15, 1 - 1e-5)

        # Calculate the loss (the formula: - Σ (true_labels * log(predicted_probs))
        loss = - np.sum(true_labels * np.log(predicted_probs), axis = 1)
        return loss

    def backward(self, predicted_probs, true_labels):
        # Calculate the error signal by subtracting the predicted values from the true labels
        error_signal = predicted_probs - true_labels

        return error_signal


#####################################################################

class MLP_Layer():
    def __init__(self, activation_function, num_inputs, num_neurons):
        self.activation_function = activation_function
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons

        # Iniziliazing the wights and the bias
        self.weights = np.random.normal(loc = 0.0, scale = 1.0, size = (self.num_neurons, self.num_inputs))
        self.bias = np.zeros((1, self.num_neurons))

        # Store input, pre-activation, and activation for backpropagation
        self.input = None
        self.pre_activation = None
        self.activation = None

    def forward(self, input_data):
        # Storing the input data
        self.input = input_data
        
        # Computing pre-activation values
        self.pre_activation = self.weights @ input_data + np.transpose(self.bias)

        # Applying activation function
        if isinstance(self.activation_function, Sigmoid):
            self.activation = self.activation_function(self.pre_activation)
        elif isinstance(self.activation_function, Softmax_Activation):
            self.activation = self.activation_function(self.pre_activation)
        else:
            raise ValueError("Invalid activation function. Please choose Sigmoid or Softmax.")
         
        return self.activation
    
    def backward(self, error_signal, learning_rate):

        if isinstance(self.activation_function, Sigmoid):
            activation_gradient = self.activation_function.backward(self.activation)
        elif isinstance(self.activation_function, Softmax_Activation):
            activation_gradient = self.activation_function.backward(self.activation)
        else:
            raise ValueError("Invalid activation function. Please choose Sigmoid or Softmax.")
        
        delta = error_signal * activation_gradient

        # Adjusting weights and biases
        self.weights -= learning_rate * delta @ np.transpose(self.input) #np.dot(self.input.T, delta)
        self.bias -= learning_rate * np.sum(delta, axis = 0, keepdims = True)

        # Calculating the error signal for the next layer
        next_layer_error_signal = delta @ np.transpose(self.weights)

        return next_layer_error_signal
    

def train_MLP(mlp, input_target_data, num_epochs, minibatch_size, learning_rate):
    cce_loss = CCE_Loss()
    loss_history = []

    # Exctracting the input and target data from the input_target_data
    input_data, true_labels = zip(*input_target_data)
    input_data = np.array(input_data)
    true_labels = np.array(true_labels)

    for epoch in range(1, num_epochs + 1):
        total_loss = 0,0

        # Shuffle the training data for each epoch
        indices = np.arange(len(input_data))
        np.random.shuffle(indices)

        for i in range(0, len(input_data), minibatch_size):
            minibatch_indices = indices[i:i + minibatch_size]
            minibatch_data = input_data[minibatch_indices]
            minibatch_labels = true_labels[minibatch_indices]

            # Forward pass
            current_input = minibatch_data
            for layer in mlp:
                current_input = layer.forward(current_input)

            # Calculate loss
            loss = cce_loss(current_input, minibatch_labels)
            total_loss += loss

            # Backward pass
            error_signal = cce_loss.backward(current_input, minibatch_labels)
            for layer in reversed(mlp):
                error_signal = layer.backward(error_signal, learning_rate)

        # Calculate average loss for the epoch
        average_loss = total_loss / (len(input_data) / minibatch_size)
        loss_history.append(average_loss)

        print(f"Epoch {epoch}/{num_epochs}, Average Loss: {average_loss}")
    
     # Plot the loss history
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()


layer_sizes = [64, 32, 10]
activation_functions = [Sigmoid(), Softmax_Activation()]

# Create MLP
mlp = []
for i in range(len(layer_sizes) -1):
    layer = MLP_Layer(activation_functions[i], layer_sizes[i + 1], layer_sizes[i])
    mlp.append(layer)

# training
train_MLP(mlp, reshaped_input_target_tuple, num_epochs=1000, minibatch_size = 32, learning_rate = 0.01)    