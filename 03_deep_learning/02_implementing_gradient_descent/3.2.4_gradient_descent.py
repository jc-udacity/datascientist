import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    """
    # Derivative of the sigmoid function
    """
    return sigmoid(x) * (1 - sigmoid(x))

learnrate = 0.5
x = np.array([1, 2, 3, 4])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5, 0.3, 0.1])

### Calculate one gradient descent step for each weight
### Note: Some steps have been consolidated, so there are
###       fewer variable names than in the above sample code

# TODO: Calculate the node's linear combination of inputs and weights
# h = sum(w_i * x_i)
h = np.sum(w * x)

# TODO: Calculate output of neural network
# y_hat = sigmoid(h)
nn_output = sigmoid(h)

# TODO: Calculate error of neural network
# E = 1/2 * sum(y - y_hat)**2
# error = 0.5 * np.sum((y - nn_output)**2)
error = y - nn_output

# TODO: Calculate the error term
#       Remember, this requires the output gradient, which we haven't
#       specifically added a variable for.
# delta = (y - y_hat) * f'(h)  with f the activation function (sigmoid)
error_term = (y - nn_output) * sigmoid_prime(h)

# TODO: Calculate change in weights
# delta weight is product of learning rate eta, error term delta and inputs x
# deltaW = eta * delta * x
del_w = learnrate * error_term * x

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)
