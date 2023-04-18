import numpy as np

A = np.array([[2, -2], [-2, 3]])
B = np.array([[2, -1], [-1, 2]])
H = np.matmul(A,B)+np.matmul(np.transpose(B),np.transpose(A))
print(H)
eigenvalues= np.linalg.eigvals(H)
print(eigenvalues) 