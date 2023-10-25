# 2.1 Objects and modules

# Defining a Cat class
class Cat:
    def __init__(self, name):
        self.name = name

# Creating greet method, which returns a print statement, to make the cats greet each other.
    def greet(self, other_cat):
        print(f"Hello, I am {self.name}! I see you are also a cool fluffy {other_cat.name}, let’s together purr at the human, so that they shall give us food”. ")

