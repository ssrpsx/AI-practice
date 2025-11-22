import numpy as np

# Indexing: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.indexing.html
# Array Creation Routines: https://numpy.org/doc/stable/reference/routines.array-creation.html
# Math Routines Docs: https://numpy.org/doc/stable/reference/routines.math.html
# Linear Algebra Docs: https://numpy.org/doc/stable/reference/routines.linalg.html

a = np.array([1, 2, 3], dtype='int64')
print(a)
print('\n')

b = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]])
print(b)
print('\n')

# get dimension
print(a.ndim)
print('\n')

# get shape
print(b.shape)
print('\n')

# get type
print(a.dtype)
print('\n')

# get size คือจำนวน byte ของข้อมูลเช่น A คือ int64 ก็คือ 8 byte
print(a.itemsize)
print('\n')

# get total size
print(a.size * a.itemsize)
print('\n')

# get a specific element [row, column], example 4.0 in b array
print(b[1, 2])
print('\n')

# get a specific row
print(b[1, :])
print('\n')

# get a specific column
print(b[:, 0])
print('\n')

# make value in all row and column[0] is 10.
b[:, 0] = 10
print(b[:, 0])
print('\n')

c = np.array([[[1, 2],[3, 4]], [[5, 6], [7, 8]]])
# get specific element (work outside in)
print(c[:, 1, :])
print('\n')
print(c)
print('\n')

# replace
c[:, 1, :] = [[8,8], [9,9]]
print(c[:, 1, :])
print('\n\n\n')


## Reorganizing Arrays
before = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(before)
print("\n")

after = before.reshape((2,2,2))
print(after)
print("\n")

v1 = np.array([1, 2, 3, 4])
v2 = np.array([5, 6, 7, 8])

print(np.vstack([v1, v2, v1, v2]))
print('\n')
print(np.hstack([v1, v2, v1, v2]))

# u can change type A to type B by variable.astype('tpye')

# axis คือ แนวทาง (dimension) ที่ NumPy จะทำ operation
# 
# NumPy array เป็น multi-dimensional array

# 2D
# 2D → [[1,2,3], [4,5,6]]
# Axis = แกนของ array
# axis=0 → แนวตั้ง (rows)
# axis=1 → แนวนอน (columns)
# ใน array 3D → axis=0/1/2 ตามลำดับ dimension
# [[1, 2, 3],
#  [4, 5, 6]]
# axis=0 → ข้าม row → sum per column [1+4, 2+5, 3+6]
# axis=1 → ข้าม column → sum per row [1+2+3, 4+5+6]

# 3D
# array (depth × rows × columns)
# ลองคิดเป็น “stack ของ 2D array”
# axis=0 → ข้าม depth (stack) → sum per pixel position
# axis=1 → ข้าม rows → sum per row in each depth
# axis=2 → ข้าม columns → sum per column in each depth
# depth 0:  [[1,2],
#            [3,4]]
# depth 1:  [[5,6],
#            [7,8]]
# sum axis=0 → [[1+5,2+6],[3+7,4+8]] = [[6,8],[10,12]]