import math
import pandas as pd

data = pd.read_csv("ml-bugs.csv")
#print(data)

# determine entropy of parent
# retrieve the number of lobug and Mobug when color = brown
# they are n and m
brown = data['Color'] == 'Brown'
brown_insects = data[brown]

m = len(brown_insects[data['Species']=='Mobug'])
n = len(brown_insects[data['Species']=='Lobug'])
entropy_parent = (-m/(n+m))*math.log(m/(n+m),2) - (n/(m+n))*math.log(n/(n+m),2)
print('parent entropy =', entropy_parent)

# calculate entropy for the 2 childs
# one child is color brown
# second child is color Blue and Green
#nb_colors = data.Color.value_counts()



for i in range(len(nb_colors)):
    p = nb_colors[i] / sum(nb_colors)
    entropy_parent -= p * math.log(p,2)
for i in nb_colors.index:
    color, nb_occur =  i, nb_colors[i]
    entropy.append((color, nb_occur/sum(nb_colors)))



print(entropy)
e_color = 0
"""
# determine entropy of childs for color = Brown



# determine entropy of child length < 17.0 mm


# determine entropy of child length < 20.0 mm
