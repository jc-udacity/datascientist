import math
bucket = [8, 3, 2]
entropy = 0

for i in range(len(bucket)):
    p = bucket[i] / sum(bucket)
    entropy -= p * math.log(p,2)

print(entropy)
