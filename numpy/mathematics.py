import numpy as np

a = np.array([1, 2, 3, 4])
print(a)
print(a + 2)

b = np.array([1, 2, 3, 4])
print(a + b)

# take the cos or the sin
print(np.sin(a))
print(np.cos(a))
print('\n\n')



## Linear algebra

a = np.ones((2, 3))
print(a)

b = np.full((3, 2), 2)
print(b)

print(np.matmul(a, b))

# Find the determinant
c = np.identity(3)
print(np.linalg.det(c))
print('\n\n')


## Statistics

stats = np.array([[1, 2, 3], [4, 5, 6]])
print(stats)
print(np.min(stats))
print(np.max(stats, axis=1))
print(np.sum(stats, axis=0))