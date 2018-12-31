import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    result = []
    for i in L:
        result.append(np.exp(i)/np.sum(np.exp(L)))

    return result
