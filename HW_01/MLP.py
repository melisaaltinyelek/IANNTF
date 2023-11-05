from sklearn.datasets import load_digits

# n_class: the digits from 0 to 9
# return_X_y = False: (data, target) as a tuple
# as_frame = False: returns (data, target) as a numpy array
digits_dataset = load_digits(n_class=10, return_X_y=False, as_frame=False)
print(digits_dataset)