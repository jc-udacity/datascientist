w1 = 3
w2 = 4
b = -10
learning_rate = 0.1
point = [1,1]
steps = 0

linear_combination = w1 * point[0] + w2 * point[1] + b
output = int(linear_combination >= 0)

while output <= 0:
    steps += 1
    w1 = w1 + point[0] * learning_rate
    w2 = w2 + point[1] * learning_rate
    b = b + 1 * learning_rate
    linear_combination = w1 * point[0] + w2 * point[1] + b
    output = int(linear_combination >= 0)
    print(w1, w2, b, linear_combination, output)

print(steps)
