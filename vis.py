import matplotlib.pyplot as plt
import numpy as np
import math

print("Loading dataset...")
with open('x_train.npy', 'rb') as f:
    x_train = np.load(f)
with open('y_train.npy', 'rb') as f:
    y_train = np.load(f)
print("Dataset loaded")
print("Dataset Length: " + str(len(x_train)))

plt.hist(y_train, bins=np.arange(np.min(y_train), np.max(y_train), 0.01))
plt.show()