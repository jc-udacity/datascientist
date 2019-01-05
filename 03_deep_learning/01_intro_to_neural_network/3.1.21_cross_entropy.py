import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    ouanes = np.ones(len(Y))
    return -np.sum(np.multiply(Y, np.log(P)) + np.multiply(ouanes - Y, np.log(ouanes - P)))
