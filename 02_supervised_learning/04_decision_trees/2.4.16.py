import pandas as pd
import numpy as np

data = pd.read_csv("ml-bugs.csv")

def getEntropy(m, n):
    return (-m/(m+n))*np.log2(m/(n+m)) - (n/(m+n))*np.log2(n/(n+m))

species = data['Species'].value_counts()
parent_entropy = getEntropy(species['Mobug'], species['Lobug'])

list_color = ['Brown', 'Blue', 'Green']
list_length = [17.0, 20.0]
info_gain = []

for color in list_color:
    b = data['Color'] == color
    entropy_child1 = getEntropy(data[b]['Species'].value_counts()['Mobug'],
                                data[b]['Species'].value_counts()['Lobug'])
    notb = data['Color'] != color
    entropy_child2 = getEntropy(data[notb]['Species'].value_counts()['Mobug'],
                                data[notb]['Species'].value_counts()['Lobug'])
    m = len(data[b])
    n = len(data[notb])
    entropy = parent_entropy - m/(m+n)*entropy_child1 - n/(m+n)*entropy_child2
    info_gain.append((color, entropy))

for length in list_length:
    b = data['Length (mm)'] < length
    entropy_child1 = getEntropy(data[b]['Species'].value_counts()['Mobug'],
                                data[b]['Species'].value_counts()['Lobug'])
    notb = data['Length (mm)'] >= length
    entropy_child2 = getEntropy(data[notb]['Species'].value_counts()['Mobug'],
                                data[notb]['Species'].value_counts()['Lobug'])
    m = len(data[b])
    n = len(data[notb])
    entropy = parent_entropy - m/(m+n)*entropy_child1 - n/(m+n)*entropy_child2
    info_gain.append((length, entropy))

for info in info_gain:
    print(info)
