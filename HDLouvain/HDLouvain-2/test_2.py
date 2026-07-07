import numpy as np
import louvain_module

matrix = np.random.rand(100, 50)  # your distance matrix
k = 15

labels = louvain_module.run_louvain(matrix, k)
print(labels)

