from sklearn.datasets import load_digits

# n_class: the digits from 0 to 9
# return_X_y = False: (data, target) as a tuple
# as_frame = False: returns (data, target) as a numpy array
digits_dataset = load_digits(n_class = 10, return_X_y = False, as_frame = False)

# Accessing the data (input) and target (output)
data = digits_dataset.data
target = digits_dataset.target

# Iterating over the dataset and creating a tuple of input and output
input_output_tuple = [(data[i], target[i]) for i in range (len(data))]
print(input_output_tuple[0])