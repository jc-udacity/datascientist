import math
m = 4   # red balls
n = 10  # blue balls
entropy = (-m/(n+m))*math.log(m/(n+m),2) - (n/(m+n))*math.log(n/(n+m),2)
print(entropy)
