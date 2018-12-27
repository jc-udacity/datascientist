import numpy as np

for nb_correct, nb_total in [(7, 8), (4, 8), (2, 8)]:
    acc = nb_correct / nb_total
    weight = np.log(acc/(1-acc))
    print("weight = %.2f" % weight)
