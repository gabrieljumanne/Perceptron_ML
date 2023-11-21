import numpy as np

# using class generators is more important and usefull
# rgen = np.random.RandomState(1)
# numbers = rgen.rand(10)
# print(numbers)
#
# x = [1, 2, 3, 4, 5]
# y = ["a", "b", "c", "d", "e"]
#
# z = zip(x, y)
# for pair in z:

# application of the mapping function
def square(x):
    return x**2

numbers=[1, 2, 3, 4, 5, 6]
# applying square function to each number in the list
squared_number = map(square, numbers)
structure = list(squared_number)
print(structure)

