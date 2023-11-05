
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

# Load the digits dataset from the link
digits = load_digits()

# Extract the data into (input, target) tuples
data, target = digits.data, digits.target

########################################################

# Function to plot a digit

def plot_digit(data, target):
    plt.figure(figsize=(3, 3))
    plt.imshow(data.reshape(8, 8), cmap='gray')
    plt.title(f"Digit: {target}")
    plt.show()


##################################################################
# normalizing data
normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
###############################################################
# Reshape all image vectors to (64,)
reshaped_data = np.reshape(normalized_data, (normalized_data.shape[0], -1))
#########################################
# one-hot encoding the target
num_classes = 10
one_hot_targets = np.eye(num_classes)[target]
####################################################
# Create (input, target) tuples with reshaped data
reshaped_input_target_tuples = list(zip(reshaped_data, one_hot_targets))
######################################################

# Visualize the first few digits in the normalized and reshaped dataset (checked)
# for i in range(min(5, len(reshaped_input_target_tuples))):
#     data, target = reshaped_input_target_tuples[i]
#     plot_digit(data, target)

################################################################
# generator function

# def shuffle_generator(input_target_tuples):
#     # Convert input_target_tuples to a numpy array for easy shuffling
#     data_array = np.array(input_target_tuples, dtype=object)
#
#     # Infinite loop for continuous shuffling
#     while True:
#         # Shuffle the array along the first axis
#         np.random.shuffle(data_array)
#
#         # Yield each shuffled (input, target) pair
#         for pair in data_array:
#             yield pair

############################################
# modified generator function
def minibatch_generator(input_target_tuples, minibatch_size):
    # Convert input_target_tuples to a numpy array for easy shuffling
    data_array = np.array(input_target_tuples, dtype=object)

    # Calculate the number of minibatches
    num_minibatches = len(data_array) // minibatch_size

    # Infinite loop for continuous minibatch generation
    while True:
        # Shuffle the array along the first axis
        np.random.shuffle(data_array)

        # Reshape the data array to (num_minibatches, minibatch_size, -1)
        reshaped_data = data_array[:num_minibatches * minibatch_size].reshape(
            num_minibatches, minibatch_size, -1
        )

        # Yield each minibatch
        for minibatch in reshaped_data:
            input_batch, target_batch = zip(*minibatch)

            # Convert batches to numpy arrays
            input_batch = np.array(input_batch)
            target_batch = np.array(target_batch)

            yield input_batch, target_batch

#####################################################################
# sigmoid activation function
class SigmoidActivation:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    # its gradient w.r.t. its output
    def backward(self, output):
        return output * (1 - output)

#######################################################

class SoftmaxActivation:
    def __call__(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # its gradient w.r.t. its output
    def backward(self, output):
        return output * (1 - output)

##############################################################

# CCE loss function
class CategoricalCrossEntropyLoss:
    def __call__(self, predicted_probs, true_labels):
        # Clip predicted probabilities to avoid log(0) issues
        predicted_probs = np.clip(predicted_probs, 1e-15, 1 - 1e-15)

        # Calculate categorical cross-entropy loss
        loss = -np.sum(true_labels * np.log(predicted_probs)) / len(true_labels)

        return loss

    def backward(self, predicted_probs, true_labels):
        # Compute the error signal for categorical cross-entropy
        error_signal = predicted_probs - true_labels

        return error_signal

##########################################################

class MLPLayer:
    def __init__(self, activation_function, num_units, input_size):
        self.activation_function = activation_function
        self.num_units = num_units
        self.input_size = input_size

        # Initialize weights and biases
        self.weights = np.random.normal(loc=0.0, scale=0.2, size=(input_size, num_units))
        self.biases = np.zeros((1, num_units))

        # Store input, pre-activation, and activation for backpropagation
        self.input = None
        self.pre_activation = None
        self.activation = None

    def forward(self, input_data):
        self.input = input_data
        self.pre_activation = np.dot(input_data, self.weights) + self.biases

        # Apply activation function
        if isinstance(self.activation_function, SigmoidActivation):
            self.activation = self.activation_function(self.pre_activation)
        elif isinstance(self.activation_function, SoftmaxActivation):
            self.activation = self.activation_function(self.pre_activation)
        else:
            raise ValueError("Invalid activation function. Choose Sigmoid or Softmax.")

        return self.activation

    def backward(self, error_signal, learning_rate):
        # Compute gradients for weights and biases
        if isinstance(self.activation_function, SigmoidActivation):
            activation_gradient = self.activation_function.backward(self.activation)
        elif isinstance(self.activation_function, SoftmaxActivation):
            activation_gradient = self.activation_function.backward(self.activation)
        else:
            raise ValueError("Invalid activation function. Choose Sigmoid or Softmax.")

        delta = error_signal * activation_gradient

        # Update weights and biases
        self.weights -= learning_rate * np.dot(self.input.T, delta)
        self.biases -= learning_rate * np.sum(delta, axis=0, keepdims=True)

        # Calculate error signal for the next layer
        next_layer_error_signal = np.dot(delta, self.weights.T)

        return next_layer_error_signal

##############################################################

def train_mlp(mlp, input_target_tuples, num_epochs, minibatch_size, learning_rate):
    cce_loss = CategoricalCrossEntropyLoss()
    loss_history = []

    # Extract input data and true labels from input_target_tuples
    input_data, true_labels = zip(*input_target_tuples)
    input_data = np.array(input_data)
    true_labels = np.array(true_labels)

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0

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

####################### Application #################################

# MLP architecture
layer_sizes = [64, 32, 10]
activation_functions = [SigmoidActivation(), SoftmaxActivation()]

# Create MLP
mlp = []
for i in range(len(layer_sizes) -1):
    layer = MLPLayer(activation_functions[i], layer_sizes[i + 1], layer_sizes[i])
    mlp.append(layer)

# training
train_mlp(mlp, reshaped_input_target_tuples, num_epochs=1000, minibatch_size=32, learning_rate=0.01)
