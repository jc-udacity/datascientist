import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def score(x1, x2):
    return 4*x1 + 5*x2 - 9

values = [(1, 1), (2, 4), (5, -5), (-4, 5)]
for x1, x2 in values:
    print(sigmoid(score(x1, x2)))
