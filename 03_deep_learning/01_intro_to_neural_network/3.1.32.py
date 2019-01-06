import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def final_proba(w1, w2, b):
    return sigmoid(w1*0.4 + w2*0.6 + b)

L = ([2, 6, -2], [3, 5, -2.2], [5, 4, -3])

for w1, w2, b in L:
    print('[{},{},{}] : {}'.format(w1, w2, b, final_proba(w1, w2, b)))
