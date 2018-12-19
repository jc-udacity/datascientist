import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

data = pd.read_csv("2.3.9_data.csv", names = ['x1', 'x2', 'y'])
X = data[['x1', 'x2']].values
print(np.shape(X))
#print(X)
y = data[['y']].values
print(np.shape(y))
#print(y)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y_hat - y[i] > 0:
            W[0] = W[0] - X[i][0] * learn_rate
            W[1] = W[1] - X[i][1] * learn_rate
            b = b - learn_rate
        if y_hat - y[i] < 0:
            W[0] = W[0] + X[i][0] * learn_rate
            W[1] = W[1] + X[i][1] * learn_rate
            b = b + learn_rate

    return W, b

# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

regression_coef = trainPerceptronAlgorithm(X, y, 0.01, 50)

plt.figure()
X_min = X[:,0].min()
X_max = X[:,0].max()
Y_min = X[:,1].min()
Y_max = X[:,1].max()
print(X_min, X_max, Y_min, Y_max)
counter = len(regression_coef)
for W, b in regression_coef:
    counter -= 1
    color = [1 - 0.92 ** counter for _ in range(3)]
    plt.plot([X_min, X_max],[X_min * W + b, X_max * W + b], color = color)

for i in range(len(y)):
    if y[i] == 0:
        plt.scatter(X[i,0], X[i,1], c = 'blue', zorder = 3)
    else:
        plt.scatter(X[i,0], X[i,1], c = 'red', zorder = 3)

plt.show()
