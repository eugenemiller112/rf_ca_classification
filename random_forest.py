from __future__ import print_function

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import sys
import os



def randomForest(data, response):
    X = data    # n x p x p (n = num samples, p = 256)            (this will eventually be convs)
    y = response    # n x 1 (n = num samples, 1 = res, 0 = sus)
    X_nu = np.zeros(shape=(X.shape[0], (X.shape[1])**2))
    print(X_nu.shape)
    for i in range(X.shape[0]):
         X_nu[i,:] = X[i,:,:].flatten()
    #print(y)
    X_train, X_test, y_train, y_test = train_test_split(X_nu, y, test_size=0.2)
    #print(y_train)
    lb = LabelBinarizer()
    y_train = np.array([number[0] for number in lb.fit_transform(y_train)])
    rf = RandomForestClassifier(n_estimators=100, max_features="sqrt")
    rf.fit(X_train, y_train)

    recall = cross_val_score(rf, X_test, y_test, cv=5, scoring='recall')
    precision = cross_val_score(rf, X_test, y_test, cv=5, scoring='precision')
    accuracy = cross_val_score(rf, X_test, y_test, cv=5, scoring='accuracy')
    f1_score = cross_val_score(rf, X_test, y_test, cv=5, scoring='f1_macro')

    return {'accuracy': accuracy, 'f1': f1_score, 'precision': precision, 'recall': recall}

def loadData(dir):
    i = 0
    X = []
    y = []
    for cat in os.listdir(dir):
        print(cat, "is class:", i)
        for datum in os.listdir(os.path.join(dir, cat)):
            im = Image.open(os.path.join(os.path.join(dir,cat),datum))
            im = ImageOps.grayscale(im)
            im = im.resize((256,256))
            arr = np.asarray(im)
            X.append(arr)
            y.append(i)
        i += 1
    print("Fin")
    return [X, y]

[X, y] = loadData(r'C:\Users\eugmille\Desktop\rf_test_data')
X = np.array(X)
y = np.array(y)
print(X.shape[0])
print(X.shape[1])
print(y.shape)
dict = randomForest(X, y)

print("acc", np.mean(dict["accuracy"]))
print("f1",np.mean(dict["f1"]))
print("precision", np.mean(dict["precision"]))
print("recall", np.mean(dict["recall"]))


