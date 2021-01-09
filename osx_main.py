from random_forest import randomForest
from random_forest import loadData
from random_forest import sobelFilter
from imagery_to_data import data_gen

import numpy as np

p = data_gen("/Library/Data" , "/Users/eugenemiller/Desktop/kralj-lab.tmp/Data", 1, 5)

[X, y] = loadData(p)

X = np.array(X)
y = np.array(y)

X = sobelFilter(X)

print(X.shape[0])
print(X.shape[1])
print(y.shape)
dict = randomForest(X, y)

print("acc", np.mean(dict["accuracy"]))
print("f1",np.mean(dict["f1"]))
print("precision", np.mean(dict["precision"]))
print("recall", np.mean(dict["recall"]))