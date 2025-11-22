import numpy as np


a = np.zeros((2, 3, 3))
# all 0s matrix
print(np.zeros((2, 3, 3)))
print('\n')

# all 1s matrix
print(np.ones((3, 4), dtype="int32"))
print('\n')

# any other number
print(np.full((2, 2), 99))
print('\n')

# any other number (full like)
print(np.full_like(a, 4))
print('\n')

# random decimal numbers
print(np.random.rand(4, 2))
print('\n')

# random integer put 4,7 it mean (4 -> 7) values or can put 7 it mean (0 -> 7)
print(np.random.randint(4, 7, size=(3, 3)))
print('\n')

# The identity matrix
print(np.identity(5))
print('\n')