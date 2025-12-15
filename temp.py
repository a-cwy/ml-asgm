import utils
import numpy as np

x = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
]

print(x)
print(np.swapaxes(x, 0, 1))
print(np.swapaxes(np.cumsum(x, axis = 0), 0, 1))
